# Tesla-style Queryable Occupancy Network Architecture

이 문서는 Tesla AI Day 2022의 Occupancy Network 공개 구조를 참고하되, 실제 구현을 위해 REO, ViewFormer, PanoOcc, BEVFormer/BEVDet4D 계열 아이디어를 섞은 아키텍처 초안이다.

현재 설계의 핵심은 다음이다.

```text
최종 해상도:
  30cm voxel (occupied 표면)

최종 feature (2-해상도 hybrid):
  free/unknown 빈 공간: coarse dense @ 0.6m
  occupied 표면:        sparse deconv + 프루닝으로 30cm까지
  (30cm dense volume feature는 만들지 않는다)

최종 출력:
  Tesla-style 4개 Volume Head를 붙인다.

4 Volume Heads:
  1. Occupancy Head
  2. Occupancy Flow Head
  3. Sub-Voxel Shape Information Head
  4. 3D Semantics Head
```

이 문서에서 "refine"이라고 부르던 기능은 별도 예측 경로가 아니라, **Sub-Voxel Shape Information Head** 안으로 들어간다. 즉 30cm voxel 자체의 occupied/free/unknown 여부는 Occupancy Head가 만들고, 30cm voxel 내부에서 실제 표면이 어디를 지나가는지, 어떤 face가 노출되어 있는지, normal과 offset이 무엇인지는 Sub-Voxel Shape Information Head가 만든다.

여러 논문/레포에서 가져오는 핵심 아이디어는 다음과 같다.

| Source | 가져오는 핵심 아이디어 | 이 문서에서 쓰는 위치 |
|---|---|---|
| REO | calibration-free / calibration-light **vanilla attention 방식** (projection-first deformable attention 대신) | 1.2m spatial lifting의 attention 종류. 단 REO는 이 attention을 BEV query에 적용하지만, 본 문서는 Tesla처럼 3D voxel query에 적용한다 |
| ViewFormer | z-squeeze voxel->BEV temporal, streaming memory(N=4), ego-motion alignment, BEV-level occupancy flow | 0.6m 단일 BEV temporal+flow stage (ViewFormer voxel query ~0.8m에 근접) |
| BEVFormer / BEVDet4D | BEV feature ego-motion alignment, temporal fusion 위치에 대한 교훈 | pre-temporal refinement, coarse temporal fusion |
| PanoOcc | coarse-to-fine **sparse deconvolution + occupancy 프루닝**, unified occupancy representation | 1.2m -> 0.6m -> 0.3m sparse decoder (단계마다 점유 후보만 keep) |
| SparseOcc / sparse execution 계열 | 중요한 위치만 더 비싼 연산을 실행하는 runtime control | sparse decoder의 프루닝 + Sub-Voxel Shape Head의 masked / face-aware execution |
| RoadBEV / FastRSR | BEV에서 road surface elevation을 직접 예측하는 관점 | 별도 Surface Outputs head(dense BEV)에서 z_surface 직접 회귀, geometry only (Section 11) |

특히 spatial lifting은 **vanilla cross-attention**을 기본으로 한다. projection-first deformable attention을 기본 구조로 두지 않고, 1.2m 3D voxel query가 multi-camera image tokens를 `Q/K/V` attention으로 읽는 방식이다.

여기서 두 논문의 역할을 정확히 구분한다.

```text
attention 종류 (from REO):
  projection-first deformable attention이 아니라
  calibration-light vanilla cross-attention을 쓴다.
  REO는 이 vanilla attention을 BEV query에 적용한다.

query 타깃 (from Tesla):
  REO와 달리, BEV가 아니라 3D voxel query에 직접 적용한다.
  Tesla AI Day 그림의 Spatial Query가 3D latent를 만드는 방식을 따른다.
```

즉 이 spatial lifting은 "REO의 vanilla attention 방식 + Tesla의 3D voxel query 타깃"의 의도된 hybrid다. 3D voxel query(50 x 17 x 5 = 4,250개)는 BEV query(850개)보다 약 5배 비싸지만, 그 대가로 lifting 단계에서 **높이(Z) 정보를 직접 보존**한다. 이는 속도 대비 정확도를 위한 의식적 선택이다.

Reference visual style: [Screenshot from 2026-06-05 22-38-02.png](<Screenshot from 2026-06-05 22-38-02.png>)

---

## 0. Target Space and Whole Architecture

이 아키텍처가 고려하는 3D 공간은 ego vehicle / robot 기준의 전방 중심 local occupancy volume이다.

```text
X range:
  rear 10m ~ front 50m
  total 60m

Y target range:
  left 10m ~ right 10m
  total 20m

Z target range:
  -2m ~ +4m
  total 6m
  (z축은 차체가 아니라 중력에 정렬, Section 20)
  (격자=타깃: 6m = 20 cells로 딱 떨어지므로 Z는 padding/mask가 없다)

Final voxel size:
  0.3m x 0.3m x 0.3m
```

Z 범위 근거 (약 2m 높이 로봇 기준):

```text
위 +4m:
  로봇 키(~2m) + 윗마진 ~2m -> 낮은 천장 / 육교 / 돌출물 통과 판단에 충분.
  그 이상(>4m)은 2m 로봇 충돌/통과와 무관하므로 두지 않는다.

아래 -2m:
  내리막 / 경사로 하행 / 연석 아래 / 구멍(negative obstacle) /
  pitch 시 원거리 바닥을 담기 위해 아래쪽을 1m 더 확보.
```

30cm는 60m X축과 6m Z축에는 딱 떨어지지만, 20m Y축에는 딱 나누어떨어지지 않는다(20.4m로 padding). 따라서 모델 내부 tensor는 Y만 padding된 aligned grid를 쓰고, Y의 valid ROI는 mask로 crop한다. X·Z는 격자=타깃이라 mask가 필요 없다.

```text
Internal 30cm grid:
  X: 200 cells = 60.0m   (격자=타깃)
  Y:  68 cells = 20.4m   (타깃 20m, 0.4m padding)
  Z:  20 cells =  6.0m   (격자=타깃, -2m ~ +4m)

0.3m 표현:
  이 200 x 68 x 20 grid 위에서 sparse하게 유지한다.
  dense하게 채우지 않고, sparse deconv + 프루닝으로
  occupied/boundary 후보 voxel만 활성화한다.

Target valid ROI:
  X: rear 10m ~ front 50m  (전체)
  Y: left/right 10m 영역   (68칸 중 20m만 valid)
  Z: -2m ~ +4m 영역        (전체)
```

가장 중요한 설계 구분 (2-해상도 hybrid):

```text
Coarse dense occupancy / visibility:
  0.6m에서 dense하게 만든다 (temporal stage와 같은 해상도).
  occupied / free / unknown / visibility를 담당한다.
  빈 공간은 크고 매끄러우므로 30cm 정밀도가 필요 없다.

Sparse fine 30cm output:
  PanoOcc식 sparse deconv + occupancy 프루닝으로
  occupied / boundary surface voxel만 남긴다.
  표면 정밀 Occupancy / Flow / Sub-Voxel Shape / Semantics는
  이 살아남은 sparse voxel 위에서 계산한다.

별도 30cm dense 3D feature volume:
  만들지 않는다. 프루닝으로 sparse하게 유지한다.

Sub-Voxel Shape Information:
  4개 volume head 중 하나로 예측한다.
  sparse 표면 voxel 중에서도 비싼 local query는
  exposed face가 있는 voxel에만 masked 실행한다.
```

핵심 원리:

```text
free / unknown (빈 공간의 속성):
  저주파, 큰 덩어리 -> coarse dense로 충분

occupied surface (물체 경계 / 형상):
  고주파, 정밀 필요 -> sparse fine 30cm

-> dense 0.3m volume 비용을 피하면서
   Tesla 4 volume head를 모두 유지한다.
```

이 분리의 근거는 Tesla 데모 영상이다. Tesla occupancy 데모의 voxel은 빈 공간이 아니라 **표면에만** 존재한다. 즉 최종 표현 자체가 surface-centric(사실상 sparse)이며, free space는 planner용으로 별도/coarse하게 들고 있는 것으로 보인다. 본 문서의 2-해상도 hybrid는 이 관찰을 그대로 따른 것이다.

Tesla 발표 그림과 비슷한 블록 흐름으로 보면 다음과 같다.

```text
+-----------------------+
| Multi-camera Images   |
| front / side / rear   |
+-----------+-----------+
            |
            v
+-----------------------+
| Canonical-view Input  |
| undistort / rectify   |
| virtual camera layout |
| canonical ray grid    |
+-----------+-----------+
            |
            v
+-----------------------+
| Image Featurizers     |
| backbone + FPN/BiFPN  |
| ray positional emb.   |
+-----------+-----------+
            |
            v
+-------------------------------+
| Spatial Attention at 1.2m     |
| REO vanilla attention 방식    |
| Q: 1.2m 3D voxel queries(Tesla)|
| K,V: image tokens             |
+---------------+---------------+
                |
                v
+-------------------------------+
| 1.2m Spatial Feature          |
| 50 x 17 x 5 x C               |
+---------------+---------------+
                |
                v
+-------------------------------+
| Dense Deconv 1.2m -> 0.6m     |
| F_0.6 dense 100 x 34 x 10     |
+---------------+---------------+
                |
                v
+-------------------------------+
| 0.6m Refinement (pre-temporal)|
| BEVDet4D 교훈: temporal 전 인코딩|
+---------------+---------------+
                |
                v
+-------------------------------+       +--------------------------+
| ViewFormer-style Temporal@0.6m|<----->| Streaming BEV Memory     |
| z-squeeze -> B_0.6            |       | ego-motion aligned       |
| 경량 gated conv (NOT full attn)|       | N=3~4 history keyframes  |
| BEV-level flow                |       +--------------------------+
| unsqueeze -> inject to F_0.6  |
+---------------+---------------+
                |
                v
+-------------------------------+       +--------------------------+
| Coarse Dense Occupancy @0.6m  |       | Sparse Deconv + Prune    |
| occupied/free/unknown/vis     |       | 0.6m -> 0.3m             |
| (빈 공간 free/unknown 담당)   |       | keep occupied 후보만     |
| (프루닝 guide도 담당)         |       +------------+-------------+
+---------------+---------------+                    |
                |                                    v
                |                       +--------------------------+
                |                       | Sparse 0.3m Surface Feat |
                |                       | kept voxels only         |
                |                       +------------+-------------+
                |                                    |
                v                                    v
+--------------------------------------------------------------+
| Tesla-style 4 Volume Heads (= Tesla "Volume Outputs")        |
| 1. Occupancy Head (coarse dense free/unknown +               |
|                    sparse fine occupied surface)             |
| 2. Occupancy Flow Head        (0.3m sparse 출력,             |
|                                motion은 0.6m temporal에서)   |
| 3. Sub-Voxel Shape Info Head  (sparse + exposed-face masked) |
| 4. 3D Semantics Head          (sparse, kept voxels)          |
+------+----------------------------------------+--------------+
       |                                        |
       v                                        v
+---------------------------+        +-------------------------------+
| Surface Outputs (별도 head)|       | Queryable Output Interface    |
| = Tesla Road Surface       |       | kept -> sparse feat+shape query|
|   Geometry (geometry only) |       | pruned -> 부모 0.6m coarse로   |
| dense BEV: z_surface,      |       |   보수적 판정 (free일 때만 free)|
| slope, step, uncertainty  |        |   occupied/uncertain -> occupied|
+---------------------------+        +-------------------------------+
```

축약하면:

```text
Canonical-view vanilla attention (REO 방식) on 3D voxel query (Tesla 타깃)
+ dense deconv 1.2m -> 0.6m
+ 0.6m pre-temporal refinement (BEVDet4D 교훈)
+ ViewFormer-style single BEV temporal @ 0.6m (경량 conv, BEV flow)
+ coarse dense occupancy/visibility @ 0.6m (free/unknown 담당)
+ PanoOcc-style sparse deconv + 프루닝 0.6m -> 0.3m
+ Tesla-style 4 Volume Heads (sparse 표면 voxel 위)
+ Surface Outputs head (Tesla처럼 별도 브랜치)
+ face-aware Sub-Voxel Shape Information
+ queryable occupancy interface
```

핵심 목표는 다음과 같다.

- occupied surface는 30cm sparse로, free/unknown 빈 공간은 coarse dense로 만든다 (2-해상도 hybrid).
- 30cm dense volume feature는 만들지 않는다. sparse deconv + 프루닝으로 점유 후보만 남긴다.
- volume output은 Tesla 그림처럼 4개 head로 나눈다. 3개(Flow / Sub-Voxel Shape / Semantics)는 occupied voxel에서만 의미가 있어 sparse가 자연스럽다.
- Occupancy Head만 두 갈래로 나뉜다: free/unknown은 coarse dense, occupied 표면 정밀 경계는 sparse fine.
- Surface(road geometry)는 Tesla 그림처럼 **별도 출력 브랜치(Surface Outputs head)** 로 둔다. volume head readout이 아니다 (Section 11).
- Sub-Voxel Shape Information은 별도 예측 경로가 아니라 head다.
- "보이는 면만 refine"은 Sub-Voxel Shape Head 내부의 exposed-face mask와 masked query execution으로 처리한다.
- temporal은 **단일 stage @ 0.6m** (ViewFormer-style)로 한다. deconv 1.2m -> 0.6m 직후, deconv 0.6m -> 0.3m 이전의 coarse latent에서 처리한다.
- Flow는 0.3m sparse로 출력하되, motion 정보는 0.6m temporal에서 온다 (ViewFormer의 BEV-level flow -> voxel 매핑).
- 1.2m 별도 temporal stage는 두지 않는다 (과보수적, ViewFormer보다도 거칢).
- full 3D temporal memory, full global temporal attention(0.6m/0.3m), 0.3m temporal, all-voxel heavy local query는 v1에서 금지한다.

---

## 1. Prediction Range and Output Grid

예측 영역:

```text
X axis: rear 10m ~ front 50m = 60m target = 60.0m grid (격자=타깃)
Y axis: left 10m ~ right 10m = 20m target, 20.4m internal padded
Z axis: -2m ~ +4m = 6m target = 6.0m grid (격자=타깃, padding 없음)
```

Z를 -2m ~ +4m로 잡은 이유 (약 2m 높이 로봇 기준):

```text
위 +4m:
  로봇 키(~2m) 위로 충분한 마진. 낮은 천장 / 육교 / 돌출물의
  "통과 가능?" 판단에 필요. >4m는 2m 로봇과 무관해 두지 않는다.

아래 -2m (0이 아니라):
  내리막 경사 / 경사로 하행 / 연석 아래 / 도로 침하 / 구멍처럼
  ego 기준면보다 낮은 표면과 negative obstacle을 표현해야 한다.
  pitch만으로도 원거리 바닥이 z=0 아래로 내려간다.
  -1m보다 1m 더 내려 경사/드롭 마진을 확보했다.

grid의 z축은 차체가 아니라 VIO의 중력 추정에 정렬한다 (Section 20).
6m = 20 cells로 딱 떨어지므로 Z는 격자=타깃이며 padding/mask가 없다.
```

최종 voxel 크기:

```text
voxel_size = 0.3m
```

내부 grid 크기:

```text
0.3m: 200 x 68 x 20 = 272,000 voxels
0.6m: 100 x 34 x 10 =  34,000 voxels
1.2m:  50 x 17 x  5 =   4,250 voxels
```

valid target ROI는 internal grid 안에서 mask로 관리한다.

```text
model output 좌표계:
  200 x 68 x 20 nominal grid
  (occupied 표면은 이 중 점유 후보만 sparse 활성,
   free/unknown은 coarse dense에서 관리)

valid target region:
  원래 목표 범위에 해당하는 cell만 loss / metric / planner에 사용

padded region:
  ignore mask 또는 low-weight background로 처리
```

30cm feature 비용 감각 (dense였다면):

```text
만약 F_0.3를 dense로 만들면, C=64, fp16:
  200 x 68 x 20 x 64 x 2 bytes
  ~= 34.8 MB
  (272,000 voxel 전부에 3D conv -> Orin 20Hz 위협)
```

본 문서는 dense를 만들지 않는다. sparse deconv + 프루닝으로 점유 후보만 남긴다:

```text
주행 장면의 occupied 비율은 대략 수~십수 %.
keep ratio 0.4 -> 0.5 적용 시 0.3m 단계 활성 voxel은
  대략 수만 voxel 수준 (272k의 일부).
-> 메모리와 3D conv 연산이 dense 대비 크게 감소.
```

free / unknown 빈 공간은 coarse dense @ 0.6m(100 x 34 x 10 = 34,000 voxel)에서만 dense로 다룬다. 0.6m dense는 voxel 수가 작아(34k) Orin에서 부담 없고, 여기서 temporal(Section 7)과 coarse occupancy를 함께 처리한다. dense 비용이 위험한 곳은 0.3m(272k)뿐이며, 그 단계만 sparse로 만든다.

Jetson AGX Orin 20Hz를 목표로 하면 C는 처음에 48 또는 64로 시작하고, head는 1x1x1 (sparse) conv / light MLP 중심으로 둔다.

---

## 2. Overall Architecture

추천 전체 구조:

```text
multi-camera images
-> canonical-view preprocessing
-> image backbone + neck
-> canonical ray positional embedding
-> calibration-light vanilla spatial attention at 1.2m
-> 1.2m spatial 3D feature

-> dense deconv: 1.2m -> 0.6m
-> 0.6m pre-temporal refinement block (BEVDet4D 교훈)
-> ViewFormer-style single BEV temporal @ 0.6m
   z-squeeze -> streaming memory(N=3~4) -> 경량 gated conv
   -> BEV-level flow -> unsqueeze/inject back to 0.6m 3D
-> temporally enhanced 0.6m 3D feature

-> coarse dense occupancy / visibility head @ 0.6m (free/unknown + 프루닝 guide)

-> sparse deconv + 프루닝: 0.6m -> 0.3m (keep occupied 후보)
-> 30cm sparse surface feature (kept voxels only)

-> Occupancy Head (sparse fine 표면 정밀)
-> Occupancy Flow Head (0.3m sparse 출력, motion은 0.6m temporal에서)
-> Sub-Voxel Shape Information Head (sparse + exposed-face masked)
-> 3D Semantics Head (sparse)

-> Surface Outputs head (Tesla처럼 별도 브랜치, dense BEV)
-> queryable output interface
```

해상도별 역할:

```text
1.2m:
  image-to-3D spatial lifting (vanilla attention)
  coarse scene / occlusion / long-term context

0.6m (dense):
  single ViewFormer-style temporal + BEV flow
  coarse dense occupancy / free / unknown / visibility
  occupancy 프루닝 guide
  mid-level geometry refinement

0.3m (sparse):
  final sparse surface feature (kept voxels only)
  Tesla-style 4 volume heads
  Surface Outputs head / queryable occupancy의 feature source
```

`1.2m`에서 spatial attention을 수행하면 `0.6m`에서 attention을 수행하는 것보다 query 수가 크게 줄어든다.

```text
1.2m query count:
  50 x 17 x 5 = 4,250

0.6m query count:
  100 x 34 x 10 = 34,000
```

따라서 image cross-attention은 1.2m에서 수행하고, 30cm occupied 표면 feature는 sparse deconvolution + 프루닝으로 복원한다.

---

## 2.1 Core Volume Heads and Supporting Runtime Modules

이 문서의 core volume heads는 정확히 네 개다(= Tesla "Volume Outputs"). Surface는 별도 브랜치(Section 11), Queryable은 인터페이스(Section 10)다.

| Head | 출력 위치 | 역할 |
|---|---|---|
| Occupancy Head | coarse dense `100 x 34 x 10` @0.6m (free/unknown/vis) + sparse fine kept voxels (occupied 표면) | occupied / free / unknown / visibility logits |
| Occupancy Flow Head | 0.3m sparse kept voxels (motion은 0.6m BEV temporal) | dynamic probability, occupancy flow, velocity |
| Sub-Voxel Shape Information Head | sparse kept voxels + exposed-face masked local query | 30cm voxel 내부의 exposed face, surface offset, normal, shape code, uncertainty |
| 3D Semantics Head | sparse kept voxels | voxel semantic class |

별도 출력 브랜치 (Tesla 그림 기준):

| 출력 | 위치 | 역할 |
|---|---|---|
| Surface Outputs head | dense BEV (Section 11) | z_surface / slope / step / uncertainty |
| Queryable Outputs | MLP 인터페이스 (Section 10) | 임의 좌표 occupancy / semantic |

아래 항목들은 head가 아니라 runtime/control mechanism이다.

```text
Exposed-face mask:
  Sub-Voxel Shape Head가 어느 face를 자세히 볼지 정한다.

Active / priority mask:
  planner path, uncertainty, dynamic, visibility frontier 등으로
  expensive local query budget을 배분한다.

Queryable MLP:
  Sub-Voxel Shape Head의 shape code와 F_0.3 feature를 사용해
  임의 좌표 x,y,z의 occupancy / semantic을 평가하는 interface다.
```

즉 구조적으로는 다음처럼 분리한다.

```text
Prediction heads:
  1. Occupancy
  2. Occupancy Flow
  3. Sub-Voxel Shape Information
  4. 3D Semantics

Runtime controls:
  exposed-face mask
  active / priority mask
  query budget scheduler
  packed query execution
```

이 구분이 중요하다. `Sub-Voxel Shape Information`은 별도 예측 경로가 아니라 head다. 다만 모든 voxel의 모든 내부 point를 같은 비용으로 query하지 않기 위해, head 내부에서 mask와 budget을 사용한다.

---

## 3. Canonical-view REO-style Input

입력은 multi-camera image다. raw camera를 그대로 attention에 넣기보다, camera별 이미지를 canonical virtual camera layout으로 정리한다.

```text
raw cameras
-> undistort / rectify
-> canonical virtual views
-> shared image backbone
```

목표:

- camera intrinsics / extrinsics 변화에 덜 민감하게 만든다.
- REO-style calibration-light attention이 학습하기 쉬운 입력을 만든다.
- hardware-specific camera id에 과하게 묶이지 않게 한다.

canonical view는 완전한 calibration-free가 아니다. 하지만 model 내부 attention이 raw projection table에 직접 묶이지 않도록, 입력을 hardware-flexible한 virtual camera space로 옮긴다.

---

## 4. Image Feature and Positional Embedding

이미지 backbone은 Tesla 공개 발표의 RegNet + BiFPN 방향을 참고할 수 있지만, 실제 구현에서는 다음 정도로 시작한다.

```text
camera images
-> 2D backbone
-> FPN/BiFPN neck
-> image tokens
```

각 image token에는 positional embedding을 넣는다.

추천 positional embedding:

```text
normalized image coordinate: u, v
canonical ray direction: rx, ry, rz
virtual camera/view id
optional timestamp / frame offset
```

피하고 싶은 방식:

```text
raw pixel index에만 의존하는 learned positional embedding
camera id에 과하게 overfit되는 embedding
```

목적:

- REO-style vanilla attention의 초기 수렴을 안정화한다.
- image token이 어느 방향의 ray에서 온 feature인지 알려준다.
- 하드웨어 변경 시 raw camera geometry보다 canonical ray geometry에 의존하게 만든다.

### Auxiliary Depth Supervision (Training-only)

vanilla cross-attention은 projection 힌트 없이 3D 대응을 데이터로만 학습하므로, 수렴이 느리고 데이터 요구량이 커질 수 있다. 이를 완화하기 위해 학습 시에만 image feature 위에 작은 per-pixel depth head를 둔다.

```text
학습 시:
  image feature -> small depth head -> per-pixel depth
  L_depth를 전체 loss에 추가

depth GT 소스:
  dense:     Section 19 pipeline의 metric pointmap depth
  sparse:    Section 20 Basalt landmark 재투영 depth
             (landmark covariance로 가중)
  synthetic: 시뮬레이션 depth

추론 시:
  depth head 제거
  추론 비용 증가 0
```

효과: depth를 맞추려면 image feature가 "이 픽셀이 얼마나 먼 표면인지"를 인코딩해야 하고, 그 feature는 cross-attention의 3D 대응 학습을 직접 돕는다.

---

## 5. Coarse Spatial Attention at 1.2m

spatial attention은 최종 30cm가 아니라 `1.2m` coarse grid에서 수행한다.

이 단계의 attention은 명시적으로 **vanilla cross-attention**이다.

```text
기본으로 쓰는 방식:
  VanillaCrossAttention(Q, K, V)

기본으로 쓰지 않는 방식:
  projection-first deformable attention
  camera calibration reference point 기반 local sampling
  Lift-Splat-Shoot style depth distribution splatting
```

단, canonical ray positional embedding과 virtual camera layout은 사용한다. 따라서 이 구조는 완전한 무기하 attention이라기보다, hardware-flexible한 `calibration-light vanilla attention`에 가깝다.

```text
coarse_query_grid = 50 x 17 x 5
num_queries = 4,250
```

기본 형태:

```text
Q: 1.2m 3D voxel query positional embedding
K: multi-camera image tokens + canonical ray positional embedding
V: multi-camera image features

F_1.2_raw = VanillaCrossAttention(Q, K, V)
```

장점:

- 0.6m attention보다 query 수가 8배 적다.
- calibration projection에 과하게 의존하지 않는다.
- canonical ray positional embedding 덕분에 완전 무기하 attention보다 수렴이 안정적이다.

위험:

- raw 1.2m feature는 아직 image-to-3D lifting 직후라 noise가 있고, 주변 voxel context가 부족할 수 있다.
- 그래서 1.2m attention 직후 바로 temporal에 넣지 않는다. dense deconv로 0.6m까지 올린 뒤, 0.6m에서 pre-temporal refinement를 거쳐 temporal을 수행한다 (Section 6, 7).

권장 시작점:

```text
1.2m attention
-> dense deconv 1.2m -> 0.6m
-> 0.6m pre-temporal refinement
-> ViewFormer-style single BEV temporal @ 0.6m
-> coarse dense occupancy @ 0.6m
-> sparse deconv + prune 0.6m -> 0.3m
-> 4 volume heads (+ Surface Outputs head)
```

---

## 6. Pre-temporal Feature Refinement

pre-temporal refinement는 dense deconv로 0.6m까지 올린 feature를 바로 과거 feature와 align하지 않고, 현재 frame 내부에서 한 번 정리하는 작은 block이다. 위치는 **deconv(1.2m->0.6m) 직후, temporal 직전**이다.

```text
F_1.2 -> dense deconv -> F_0.6_raw
-> light 3D / BEV feature cleanup
-> F_0.6_refined
-> ViewFormer-style temporal @ 0.6m
```

왜 1.2m attention 직후가 아니라 0.6m에서 하는가:

```text
BEVDet4D 교훈:
  view transformer(여기서는 1.2m attention) 직후 feature는
  너무 coarse해서 temporal cue를 바로 쓰면 velocity error가 오른다(+11.9%).
  temporal 전에 extra encoder로 한 번 정리하는 것이 적절하다.

따라서:
  1.2m attention -> dense deconv 0.6m -> 0.6m refinement -> temporal
```

refinement 대상 F_0.6_raw의 문제:

```text
F_0.6_raw:
  1.2m attention feature를 deconv로 올린 것
  voxel 주변 context가 부족할 수 있음
  noise / ghost feature가 있을 수 있음
  temporal memory와 비교하기에 표현이 불안정할 수 있음
```

pre-temporal refinement는 큰 모듈이 아니다. 추가 camera attention이 아니라, 작은 3D/BEV conv block이다.

추천 최소 구현:

```text
F_0.6_raw: 100 x 34 x 10 x C

1x1x1 conv:
  C -> Cb

light 3D block:
  depthwise 3D conv 3x3x3
  + pointwise 1x1x1 conv
  + residual / gating

F_0.6_refined:
  100 x 34 x 10 x Cb
```

v1 추천:

```text
F_0.6_refined = F_0.6_raw + LightRes3DBlock(F_0.6_raw)
B_0.6 = ZPool(F_0.6_refined)   # temporal은 BEV에서
```

---

## 7. ViewFormer-style Single Temporal Stage at 0.6m

Tesla AI Day 2022 그림을 구조적으로 읽으면 temporal alignment는 deconvolution 이후의 고해상도 volume feature가 아니라, spatial attention 이후 deconvolution 이전의 coarse latent에서 수행된다. 본 문서도 그 원칙(temporal before final deconv)을 따른다. 다만 그 coarse latent를 **0.6m 단일 stage**로 둔다.

### 7.1 왜 0.6m 단일 stage인가 (1.2m이 아니라)

```text
Tesla:     ~4.8m latent에서 temporal (30cm final, ×16 deconv).
           거대 ROI + ×16 deconv + 커스텀 HW라서 가능.
ViewFormer: ~0.8m voxel query를 BEV로 squeeze해서 temporal.
           flow를 위해 fine-grained motion이 필요해 0.8m를 택함.
우리:      ROI 작고(Z 6m), deconv ×4뿐, Orin.
           Tesla의 4.8m는 Z가 붕괴(6/4.8≈1)해서 불가능.
           ViewFormer의 0.8m에 가장 가까운 Orin-feasible 지점이 0.6m.
```

판단:

```text
1.2m temporal:
  Tesla "coarsest" 원칙엔 맞지만, ViewFormer(0.8m)보다도 거칢.
  flow에서 motion이 한 cell(30cm 4x4)에 평균화됨. 과보수적.

0.6m temporal:
  ViewFormer ~0.8m에 근접. flow에 적합.
  Z=10층 확보(붕괴 없음). 여전히 deconv 이전 coarse latent.
  비용은 operator로 통제(아래) -> Orin 가능.

-> v1 main temporal은 0.6m 단일 stage. 1.2m 별도 temporal stage는 두지 않는다.
```

비용의 핵심은 해상도가 아니라 **operator**다.

```text
0.6m BEV = 100 x 34 = 3,400 cells

full self-attention:  3,400^2 = 11.56M  -> 금지
gated / depthwise separable conv:  cell 수에 선형 -> 매우 쌈
history N=4 메모리:  100 x 34 x 64 x 2B ≈ 0.44MB/frame -> 무시 가능

ViewFormer가 3090에서 4 FPS인 것은 0.8m라서가 아니라
temporal attention을 무겁게 했기 때문이다. 우리는 경량 conv로 한다.
```

### 7.2 ViewFormer-style 흐름 (z-squeeze -> temporal -> unsqueeze -> flow)

ViewFormer는 voxel query를 z축으로 squeeze해 BEV query로 만들고, BEV에서 streaming temporal을 한 뒤 다시 voxel로 unsqueeze해서 occupancy와 occupancy flow를 함께 예측한다. flow는 BEV-level로 예측하고 각 voxel cell에 매핑한다. 본 문서도 동일하게 한다.

```text
F_0.6_refined: 100 x 34 x 10 x C

B_0.6_t = ZPool(F_0.6_refined):
  100 x 34 x Cb   (z-squeeze to BEV)

Memory:
  [B_0.6_{t-1}, B_0.6_{t-2}, B_0.6_{t-3}]
  keyframe 간격 ~0.2s (0.6~0.8s 창)

Aligned memory:
  Warp(B_0.6_{t-k}, pose_{t-k -> t})   (ego-motion)

B_0.6_temporal:
  LightGatedTemporalFusion(B_0.6_t, aligned_memory)   (NOT full attention)

BEV-level flow:
  F_flow_bev = FlowHead_BEV(B_0.6_temporal)
  -> dynamic_logit, vx, vy, (vz)  per BEV cell

Unsqueeze / inject:
  F_0.6_temporal = Inject3D(F_0.6_refined, B_0.6_temporal, z_embedding)
  flow는 BEV cell -> 그 column의 voxel로 매핑 (최종 0.3m sparse voxel까지 broadcast)
```

### 7.3 temporal fusion operator (v1 / v1.5)

```text
권장 v1:
  ego-motion warp
  concatenate current + aligned history
  depthwise / separable BEV conv
  gated residual fusion

v1.5 이후 검토:
  local / windowed temporal attention (full global은 계속 금지)
  limited deformable sampling
```

추천 memory 길이:

```text
N_history = 3 keyframes
keyframe 간격 ~0.2s
```

20 FPS 연속 frame은 창이 0.15~0.2s로 occlusion 추론에 너무 짧다. 시간 간격 기반으로 띄엄띄엄 저장해 0.6~0.8s 창을 확보한다.

BEV grid 비교 (operator 비용 감각):

```text
1.2m BEV:  50 x 17 = 850 cells
0.6m BEV:  100 x 34 = 3,400 cells   <- 본 문서 temporal 위치
0.3m BEV:  200 x 68 = 13,600 cells  <- temporal 금지
```

금지:

```text
0.3m temporal (cell 4배, Orin 위협)
0.6m / 0.3m full global temporal attention
full 3D temporal memory (BEV squeeze로 대체)
```

---

## 8. Deconvolution to 30cm (0.6m dense -> 0.3m sparse)

해상도 단계를 비용에 맞게 나눈다. dense 비용이 위험한 곳은 0.3m(272k voxel)뿐이므로, **1.2m->0.6m는 dense로 두고(temporal/coarse occupancy가 여기 산다), 0.6m->0.3m만 sparse deconv + 프루닝**으로 만든다. PanoOcc의 coarse-to-fine sparse 업샘플 아이디어를 이 마지막 단계에 적용한다.

### 8.1 단계별 정리

```text
F_1.2 (dense, 50 x 17 x 5)
-> dense deconv -> F_0.6 (dense, 100 x 34 x 10)
-> 0.6m pre-temporal refinement (Section 6)
-> ViewFormer-style temporal @ 0.6m (Section 7) -> F_0.6_temporal (dense)

[여기서 두 갈래]

갈래 1) Coarse Dense Occupancy Head @ 0.6m:
  occupied / free / unknown / visibility logits (dense, 100 x 34 x 10)
  -> free vs unknown 구분의 dense source
  -> 동시에 0.3m sparse 프루닝을 guide

갈래 2) Sparse Deconv + Prune (0.6m -> 0.3m):
  F_0.6_temporal -> upsample -> per-voxel occupancy score
  -> keep_ratio만 keep -> F_0.3_sparse (occupied/boundary voxels only)
```

0.6m을 dense로 두는 이유:

```text
0.6m dense = 100 x 34 x 10 = 34,000 voxel.
3D conv가 가볍고, z-squeeze BEV temporal과 coarse dense occupancy를
여기서 한 번에 처리할 수 있다.
반면 0.3m dense = 272,000 voxel -> 이 단계만 sparse로 피한다.
```

4개 volume head는 `F_0.3_sparse`의 살아남은 voxel 위에 붙는다. Occupancy Head만 coarse dense 갈래(free/unknown)를 0.6m에서 추가로 갖는다.

### 8.2 Prune (keep ratio)

PanoOcc는 sparse deconv에서 keep ratio 0.2 / 0.5 / 0.5를 쓴다. 그러나 본 문서는 **front 50m 원거리 + camera-only**라 공격적으로 자르면 먼 물체나 얇은 구조물이 영구히 사라진다(프루닝은 되돌릴 수 없다).

```text
권장 keep ratio (PanoOcc보다 보수적):
  0.6m -> 0.3m: 0.5 (occupied recall 우선)

거리 적응:
  far-field(원거리)는 keep ratio를 더 높이거나
  occupancy score threshold를 낮춰 보호한다.
```

프루닝 결정은 0.6m coarse dense occupancy head의 occupied 확률로 한다(별도 score 헤드 불필요). multi-scale occupancy supervision을 받는다.

**Quota 기반 keep (v1.5).** 단일 keep ratio + top-k는 occupancy 점수만 보므로 **얇은 물체 / 원거리 / 저신뢰 occupied가 체계적으로 먼저 잘린다**(프루닝은 비가역 -> false negative 영구 손실). v1.5에서는 카테고리별 최소 quota를 둬 보호한다.

```text
keep quota (v1.5):
  near-field occupied candidates
  far-field low-confidence candidates
  dynamic candidates
  planner corridor
  thin-object / high-gradient candidates
  random exploration quota

-> 각 카테고리에 최소 keep을 보장해 단순 top-k가 놓치는 후보를 살린다.
   v1은 "far-field keep 상향"으로 단순 시작, v1.5에서 quota로 강화.
```

### 8.3 이전 버전과의 차이

```text
이전:
  dense feature를 0.3m까지 만들고, 4 head를 F_0.3 dense 위에 붙임

현재:
  0.6m까지 dense, 0.3m는 sparse deconv + 프루닝.
  free/unknown은 0.6m coarse dense, occupied 표면은 0.3m sparse fine.
  30cm dense feature volume은 만들지 않는다.
```

v1 권장 channel:

```text
F_1.2: C = 96 or 128   (dense)
F_0.6: C = 64 or 96    (dense, temporal/coarse occ)
F_0.3: C = 48 or 64    (sparse)
```

### 8.4 배포 주의 (sparse conv)

```text
sparse 3D conv (spconv / torchsparse / Minkowski 류):
  연산량은 dense보다 크게 줄지만
  TensorRT가 네이티브로 잘 못 돕는다.
  Orin에서는 커스텀 커널 / 플러그인이 필요하고
  FP16 최적화가 dense보다 까다롭다.

-> "TensorRT FP16 기본"(Section 13) 목표와 충돌하는 지점.
   엔지니어링 비용과 커널 최적화 리스크를 v1에서 감수 대상으로 명시한다.
   sparse 배포가 막히면 fallback은:
     coarse 해상도는 dense, 0.3m 단계만 masked dense (frustum/ROI crop)로 근사.
```

---

## 9. Tesla-style 4 Volume Heads

head는 `F_0.3_sparse`(살아남은 occupied/boundary 표면 voxel)와 coarse dense occupancy feature 두 source 위에 붙는다.

```text
Coarse dense (0.6m)
└── Occupancy Head (free / unknown / visibility 갈래)

F_0.3_sparse (kept voxels)
├── Occupancy Head (occupied 표면 정밀 갈래)
├── Occupancy Flow Head (0.3m 출력, motion은 0.6m temporal)
├── Sub-Voxel Shape Information Head
└── 3D Semantics Head

(Surface Outputs head는 별도 브랜치 - Section 11)
```

설계 원리: Flow / Sub-Voxel Shape / Semantics 세 head는 **occupied voxel에서만 의미**가 있으므로 sparse가 자연스럽다(빈 공간엔 flow도 표면도 semantic도 없다). Occupancy Head만 "빈 공간의 free/unknown"이라는 추가 책임이 있어 coarse dense 갈래를 함께 갖는다. Sub-Voxel Shape Information Head는 sparse 표면 voxel 중에서도 exposed face가 있는 voxel만 골라 masked execution을 사용한다.

### 9.1 Occupancy Head (2-갈래)

Occupancy Head는 두 갈래로 나뉜다. free/unknown은 빈 공간의 저주파 속성이라 coarse dense에서, occupied 표면의 정밀 경계는 sparse fine에서 예측한다.

```text
갈래 A: Coarse Dense Occupancy / Visibility (@ 0.6m)
  input:  F_0.6_temporal (dense, 100 x 34 x 10 x C)
  output: occupied / free / unknown / visibility logits
  역할:   빈 공간 free vs unknown 구분 (프루닝 / queryable의 기준)
  또한 이 occupancy score가 0.6m->0.3m sparse deconv 프루닝을 guide한다.

갈래 B: Sparse Fine Occupied Surface (@ 0.3m)
  input:  F_0.3_sparse (kept voxels)
  output: 살아남은 voxel의 occupied 확률 / 표면 경계 정밀화
  역할:   30cm 표면 정밀 occupancy
```

권장 시작:

```text
# 갈래 A (dense @ 0.6m)
O_occ_coarse = Conv1x1x1(F_0.6_temporal)
  -> 100 x 34 x 10 x (3 또는 occ+vis)

# 갈래 B (sparse @ 0.3m)
O_occ_fine = SparseConv1x1x1(F_0.3_sparse)
  -> [N_kept] x 1   (kept voxel별 occupied 정밀 logit)
```

unknown을 명시적으로 두는 이유:

```text
free:
  camera ray가 지나간 관측된 빈 공간

unknown:
  보이지 않았거나 occlusion 뒤라 판단 불가능한 공간

occupied:
  표면 / 물체가 있는 공간
```

free와 unknown을 구분하지 않으면, 카메라 뒤쪽 occluded 영역을 잘못 free로 학습할 위험이 크다.

### 9.2 Occupancy Flow Head

Occupancy Flow Head는 dynamic probability와 flow / velocity를 예측한다. ViewFormer 방식을 따른다: **motion 추정은 0.6m BEV temporal stage(Section 7)에서 BEV-level flow로 하고, 출력은 0.3m sparse voxel에 매핑**한다.

```text
motion 정보 (정보의 출처):
  Section 7의 0.6m temporal -> BEV-level flow
  (dynamic_logit, vx, vy, [vz]) per 0.6m BEV cell

출력 (다른 head와 같은 격자):
  0.3m sparse kept voxel별로 flow 벡터를 붙인다.
  0.6m BEV flow를 그 column의 0.3m voxel로 매핑(broadcast)하고,
  F_0.3_sparse의 fine geometry로 per-voxel 정제(선택).
```

왜 이렇게 나누나 (ViewFormer 근거):

```text
ViewFormer는 BEV-level flow를 예측하고 each voxel cell에 map한다.
flow의 motion granularity는 temporal을 한 해상도(0.6m)가 상한이다.
0.3m로 출력해도 0.6m 정보를 펼친 것이며, 0.3m에서 새 motion이 생기지 않는다.
강체(차량/보행자)는 한 BEV column이 같이 움직이므로 BEV-level로 충분.

-> 출력 격자 = 0.3m (다른 head와 일치),
   motion 해상도 = 0.6m (ViewFormer 수준, Orin 가능).
   0.3m temporal은 하지 않는다.
```

권장 출력:

```text
O_flow:
  [N_kept] x 5   (sparse 0.3m, kept voxel별)

channels:
  dynamic_logit
  vx
  vy
  vz
  flow_uncertainty
```

flow는 occupied voxel에서만 의미가 있으므로 프루닝된 빈 공간엔 두지 않는다. supervise도 차등 적용한다.

```text
occupied dynamic voxel:
  velocity / flow loss 적용

static occupied voxel:
  near-zero flow regularization

free / unknown voxel:
  flow loss 약하게 또는 ignore
```

motion 해상도 한계 (명시):

```text
v1: motion 해상도 상한 = 0.6m.
    0.3m flow output은 0.6m BEV flow를 sparse surface voxel에 매핑한 것이다.
    0.3m에서 새로운 motion 정보가 생기지 않는다.

v1.5 (필요 시): F_0.3_sparse 기반 small residual flow head를 추가해
    0.3m 수준 motion을 보정한다 (full 0.3m temporal은 여전히 금지).
```

### 9.3 Sub-Voxel Shape Information Head

Sub-Voxel Shape Information Head는 30cm voxel 내부의 local shape를 표현한다.

Occupancy Head가 말하는 것:

```text
이 30cm voxel은 occupied / free / unknown인가?
```

Sub-Voxel Shape Information Head가 말하는 것:

```text
occupied 또는 boundary voxel 내부에서
실제 표면이 어디를 지나가는가?
어떤 face가 free / unknown 공간과 맞닿아 있는가?
local surface normal은 무엇인가?
voxel 내부가 부분적으로만 차 있는가?
얇은 구조물인가?
shape prediction uncertainty는 얼마인가?
```

이 head는 "보이는 면만 refine" 원칙을 내부 동작으로 갖는다.

여기서 "보이는 면"의 기본 의미는 camera visibility가 아니라 **occupancy topology 기준의 exposed face**다.

```text
exposed face:
  occupied / boundary voxel의 6-neighbor 중
  free 또는 unknown neighbor와 맞닿은 face

camera visibility:
  query priority와 confidence를 조정하는 보조 신호
```

기본 exposed-face 계산:

```text
for each kept (occupied/boundary) voxel v:
  for face in [+x, -x, +y, -y, +z, -z]:
    neighbor = voxel adjacent to face
    if neighbor가 kept(occupied surface):
      exposed_face[face] = false   (표면끼리 맞닿음)
    else:  # neighbor가 프루닝됨 = fine 표면 증거 없음
      exposed_face[face] = exposed candidate
      # 그 면이 free / unknown / uncertain인지는 부모 0.6m coarse occupancy로 구분
```

중요: **프루닝됨 = "fine 30cm 표면 증거 없음"이지 곧 free가 아니다.** 프루닝된 voxel에는 빈 공간뿐 아니라 **물체 내부(표면 뒤 꽉 찬 부분)** 도 섞인다. 따라서 neighbor가 프루닝되면 그 면은 "exposed **candidate**"로 두고, 실제 free / unknown / uncertain 여부는 **부모 0.6m coarse occupancy**로 판정한다.

```text
neighbor 프루닝 + 부모 coarse free      -> exposed-to-free (진짜 노출면)
neighbor 프루닝 + 부모 coarse unknown   -> exposed-to-unknown (occlusion 경계)
neighbor 프루닝 + 부모 coarse occupied  -> 내부 경계일 수 있음, 노출면 아님 (보수적)
```

이 head는 `F_0.3_sparse`의 kept voxel 위에서만 동작하므로, 모든 voxel을 dense하게 도는 비용이 처음부터 없다.

권장 출력:

```text
O_shape:
  exposed_face_logits: 6 channels
  face_surface_offset: 6 channels
  local_normal: 3 channels
  shape_uncertainty: 1 channel
  thinness_logit: 1 channel
  shape_code: Cs channels, e.g. 8 or 16
```

전체 channel 예시:

```text
Cs = 8:
  6 + 6 + 3 + 1 + 1 + 8 = 25 channels

Cs = 16:
  33 channels
```

face_surface_offset은 각 exposed face에서 voxel 내부 surface까지의 normalized distance로 둔다.

```text
offset range:
  [0, 1] 또는 [-0.5, 0.5]

예:
  +x face가 exposed
  offset = 0.2
  -> +x face에서 voxel 내부 0.2 * 0.3m 지점에 surface crossing 후보
```

local_normal은 voxel 내부 surface normal이다. 처음에는 voxel당 하나의 normal로 시작하고, 나중에 복잡한 corner / thin object가 필요하면 face별 normal로 확장한다.

#### Face-aware Execution

모든 occupied voxel 내부를 동일하게 촘촘히 query하지 않는다.

```text
fully free voxel:
  Sub-Voxel Shape loss / query 없음

fully occupied interior voxel:
  exposed face 없음
  shape query 거의 없음

surface / boundary voxel:
  exposed face가 있는 방향으로만 local shape query

corner / thin object candidate:
  여러 exposed face가 있으므로 query budget 증가
```

query budget 예시:

```text
M_i = f(
  number_of_exposed_faces,
  occupancy_uncertainty,
  shape_uncertainty,
  dynamic_score,
  planner_relevance,
  camera_visibility
)
```

face 수에 따른 동작:

```text
one exposed face:
  해당 face normal 방향으로만 offset / boundary query

two exposed faces:
  edge voxel로 보고 두 face 방향 query

three or more exposed faces:
  corner / thin object 후보
  더 많은 local query 사용

zero exposed face:
  heavy local query 생략
```

중요한 점:

```text
Sub-Voxel Shape Information Head는 core 4 head 중 하나다.
exposed-face mask와 query budget은 이 head의 실행 방식이다.
별도 sparse feature path가 아니다.
```

### 9.4 3D Semantics Head

3D Semantics Head는 각 30cm voxel의 semantic class를 예측한다.

권장 초기 class:

```text
free / unknown은 Occupancy Head가 담당하므로 semantics에서는 제외하거나 ignore 처리

semantic classes:
  road / ground
  vehicle
  pedestrian
  cyclist / two-wheeler
  curb / barrier
  vegetation
  building / wall
  other static
```

출력:

```text
O_sem:
  [N_kept] x N_class   (sparse, kept voxel별)
```

semantic은 occupied voxel에서만 의미가 있으므로 sparse 표면 voxel 위에서 계산하는 것이 자연스럽다. semantic loss는 occupied 또는 surface-near voxel 중심으로 적용한다. free / unknown 영역에 semantic label을 강제로 주면 noisy supervision이 될 수 있다.

---

## 10. Queryable Occupancy Interface

Queryable output은 별도 volume head가 아니다. 4개 head가 만든 sparse surface feature / shape code / semantic output과 coarse dense occupancy를 사용해 임의 좌표를 평가하는 interface다.

```text
query point x, y, z
-> 30cm voxel index 찾기
-> 이 voxel이 kept(살아남음)?
     yes -> 정밀 경로:
            local coord u + sample F_0.3_sparse + shape_code
            -> small Queryable MLP -> continuous occupancy / semantic / uncertainty
     no(프루닝됨) -> 부모 0.6m coarse occupancy를 읽어 보수적 판정 (MLP 생략):
            high-confidence free  -> free
            unknown / occluded    -> unknown
            occupied / uncertain  -> occupied (내부일 수 있음, free로 답하지 않음)
```

핵심: **프루닝됨은 곧 free가 아니다.** 프루닝된 칸에는 빈 공간과 물체 내부가 섞여 있으므로, 부모 coarse가 **확실히 free일 때만 free**로 답하고, occupied/uncertain이면 보수적으로 occupied(또는 fine-level unknown)로 답한다. 이렇게 하면 차량 내부 등 solid interior를 free로 오답하는 충돌 위험을 막는다. 정밀 MLP는 kept(occupied 표면) voxel 근처 query에만 실행된다.

입력:

```text
F_query:
  trilinear_sample(F_0.3_sparse, x)   # kept voxel 주변에서만 유효

shape_context:
  shape_code
  exposed_face_mask
  face_surface_offset
  local_normal
  shape_uncertainty

local coordinate:
  u in [-0.5, 0.5]^3
```

출력:

```text
p_occupied(x)
p_unknown(x)
optional semantic(x)
optional signed distance / boundary probability
```

이 interface는 dense 5cm 또는 10cm grid를 만드는 것이 아니다. planner나 collision checker가 필요한 좌표만 물어보는 query path다.

packed execution:

```text
selected query points:
  Q x 3

MLP input:
  Q x D

MLP output:
  Q x output_dim
```

나쁜 방식:

```text
for voxel in selected_voxels:
  QueryableMLP(voxel_queries)
```

권장 방식:

```text
모든 local query point를 Q x D로 pack
QueryableMLP를 한 번 또는 큰 batch 몇 번으로 실행
```

---

## 11. Surface Outputs Head (Tesla-style 별도 브랜치, Geometry only)

Tesla AI Day 2022 그림에서 Surface Outputs는 Volume Outputs와 **나란한 별도 출력 브랜치**다(Queryable Outputs와도 분리). 본 문서도 이를 따라, road surface geometry를 4개 volume head의 readout 후처리가 아니라 **독립 head**로 둔다.

```text
Tesla 그림의 출력 그룹:
  Volume Outputs   = 4 volume heads (Occupancy/Flow/Sub-Voxel Shape/Semantics)
  Surface Outputs  = 별도 브랜치  <- 이 Section
  Queryable Outputs = MLP 인터페이스 (Section 10)
```

**Geometry only (Road Surface Semantics는 두지 않는다).** Tesla 그림의 Surface Outputs는 실제로 두 갈래 — `Road Surface Geometry`(높이/형상)와 `Road Surface Semantics`(차선/주행영역 같은 노면 평면 raster) — 다. 그러나 본 시스템은 **실내 및 실외(도로 아님)** 대상이라 차선/도로표시 같은 노면 semantic이 없다. 따라서 Road Surface Semantics 갈래는 두지 않고 **Geometry만** 둔다. (필요하면 v1.5에서 "traversable / non-traversable" 정도의 축소된 노면 semantic만 선택적으로 추가.)

### 11.1 입력과 출력

Surface는 바닥 높이장(z_surface)으로, BEV column마다 하나의 매끄러운 값이다. 따라서 sparse 표면 voxel이 아니라 **0.6m dense feature만 입력으로 받는 self-contained dense BEV head**로 둔다(RoadBEV식 BEV elevation regression). sparse head(3D Semantics, Sub-Voxel Shape) 출력을 입력으로 끌어오지 않는다 — sparse->dense 역류와 실행 순서 의존을 피하기 위함이다.

```text
input:
  F_0.6_temporal (dense, 100 x 34 x 10)  <- 이것만 사용 (self-contained)
  -> ZPool / BEV projection -> 100 x 34 x C_bev

output (dense BEV, 0.6m 격자 또는 0.3m로 upsample -> 수평 선명도 향상):
  z_surface        : 바닥 높이 (연속 회귀, 격자에 안 묶임 -> cm급 가능)
  valid            : 유효/관측 여부
  uncertainty      : 높이 불확실도
  slope / normal   : 경사
  step_height      : 연석 / 계단 높이
```

### 11.2 다른 head와의 관계 (입력 의존이 아니라 분업 + consistency loss)

Surface head는 **자립**한다. 다른 head 출력을 입력으로 받지 않는다. 대신 세부 바닥 기하를 두 경로로 나눠 담당하고, 둘을 consistency loss로 묶는다.

```text
Surface Outputs Head (dense, 0.6m feature):
  매끄러운 z_surface 전역 높이장. 경사 / 완만한 ramp / 관측 안 된 곳 보간.
  장점: 어디든 값이 있음. 한계: 날카로운 모서리는 다소 뭉개짐.

sparse 0.3m kept 바닥 voxel + Sub-Voxel Shape offset:
  연석 lip / 단차 edge 같은 날카로운 sub-30cm 기하.
  바닥은 occupied 표면이라 kept. offset/normal이 30cm 아래 표면 위치까지 잡음.

결합:
  consistency loss로 두 경로 z를 일치시킴 (입력 의존 아님).
  planner는 "Surface head 전역 높이장 + 날카로운 곳은 sparse offset"을 융합해 사용.
```

핵심: z_surface 수직 정밀도는 회귀라 0.3m 격자에 갇히지 않는다. 단 가장 날카로운 연석/단차의 위치 정밀도는 sparse sub-voxel offset이 authority다(Surface head는 매끄러운 base).

### 11.3 학습

```text
GT:
  Section 19의 ground/road segmentation + 연속 표면(SDF/mesh)에서
  per-BEV-cell z_surface / slope / step / valid 라벨 생성

loss:
  z_surface regression (valid mask)
  uncertainty: heteroscedastic 또는 ensemble
  consistency loss: Surface head z_surface <-> sparse kept 바닥 voxel의
    sub-voxel surface 높이가 같은 (x,y)에서 일치하도록 (soft coupling)
```

v1에서도 Tesla처럼 별도 head로 두되, 모듈은 가볍게(BEV conv 몇 layer) 시작한다. 3D Semantics에는 의존하지 않는다(Geometry only).

---

## 12. Runtime Masking and Multi-rate Scheduling

Active mask는 이제 별도 prediction head가 아니다. Sub-Voxel Shape Head와 Queryable Interface를 어디에 얼마나 비싸게 실행할지 정하는 runtime priority map이다.

priority score:

```text
S_query =
    exposed_face_score
  + occupancy_uncertainty
  + shape_uncertainty
  + dynamic_score
  + planner_relevance
  + visibility_frontier_score
```

사용처:

```text
1. Sub-Voxel Shape Head의 expensive local query 위치 선택
2. Queryable MLP의 point budget 배분
3. planner corridor / near-field collision region 강화
4. dynamic object boundary 보강
```

20 FPS v1에서는 모든 voxel에 heavy query를 수행하지 않는다.

```text
매 frame dense 실행 (0.6m dense):
  F_0.6 temporal trunk
  coarse occupancy / visibility @ 0.6m
  Surface Outputs head (dense BEV geometry)

매 frame sparse 실행 (0.3m sparse):
  sparse deconv + prune 0.6m -> 0.3m
  Occupancy fine head (occupied 표면)
  Occupancy Flow Head
  cheap Sub-Voxel Shape parameters
  3D Semantics Head

masked / budgeted 실행:
  expensive local sub-voxel query
  queryable MLP
  planner-triggered high-detail readout
```

권장 budget:

```text
K_surface_voxels:
  2048 ~ 8192 selected voxels

Q_total:
  16k ~ 64k local query points

start:
  K = 4096
  average M_i = 4~8
```

---

## 13. 20 FPS Runtime Target

Jetson AGX Orin 기준 v1 hard target은 20 FPS다.

```text
target FPS:
  20 FPS

frame budget:
  50ms / frame

latency target:
  p50 <= 45ms
  p95 <= 55~60ms
```

20 FPS implementation guardrails:

```text
Spatial lifting:
  1.2m vanilla cross-attention
  0.6m or 0.3m image cross-attention 금지

Temporal:
  단일 stage @ 0.6m BEV (ViewFormer-style), 경량 gated conv
  1.2m 별도 temporal stage 두지 않음
  full 3D temporal memory 금지
  0.6m/0.3m full global temporal attention 금지
  0.3m temporal 금지

Feature:
  1.2m -> 0.6m는 dense (temporal/coarse occupancy가 여기).
  coarse dense occupancy/visibility는 0.6m에서 dense.
  0.6m -> 0.3m만 sparse deconv + 프루닝.
  30cm dense volume feature는 만들지 않는다.
  C는 48 or 64로 시작한다.

Prune:
  0.6m -> 0.3m keep ratio는 PanoOcc(0.2)보다 보수적으로 (0.5, recall 우선).
  far-field는 더 높여 원거리 물체 보호.

Volume heads:
  1x1x1 (sparse) conv / light MLP 중심
  head 내부 large attention 금지

Sub-voxel shape:
  sparse 표면 voxel 위에서만 동작
  expensive local query는 exposed face / priority mask로 추가 제한

Precision / deployment:
  TensorRT FP16을 기본 목표
  단 sparse 3D conv(spconv/torchsparse)는 TensorRT 네이티브 지원이 약함.
  Orin 커스텀 커널/플러그인 필요, FP16 최적화 난이도 증가.
  sparse 배포가 막히면 fallback:
    coarse dense + 0.3m는 ROI/frustum crop된 masked dense로 근사.
  INT8은 v1.5 이후 검토
```

multi-rate schedule:

```text
20 FPS, every frame:
  image backbone
  1.2m spatial attention
  dense deconv 1.2m -> 0.6m
  ViewFormer-style temporal @ 0.6m (경량 conv)
  coarse dense occupancy / visibility @ 0.6m
  sparse deconv + prune 0.6m -> 0.3m
  4 volume heads (sparse) + Surface Outputs head
  small masked sub-voxel query

10 FPS, every 2 frames:
  larger queryable MLP budget
  wider dynamic boundary readout
  longer temporal history (N 증가)

planner-triggered:
  trajectory corridor fine query
  near-field collision detail
  uncertain exposed-face region
```

---

## 14. Final Inference Path

최종 inference 구조:

```text
1. Raw multi-camera images
2. Canonical-view preprocessing
3. Image backbone + neck
4. Canonical ray positional embedding
5. Vanilla spatial attention at 1.2m (3D voxel query)
6. Dense deconv: 1.2m -> 0.6m
7. 0.6m pre-temporal feature refinement
8. ViewFormer-style single BEV temporal @ 0.6m (z-squeeze, streaming, 경량 conv, BEV flow)
9. Unsqueeze / inject -> F_0.6_temporal
10. Coarse dense occupancy / visibility head @ 0.6m (free/unknown, 프루닝 guide)
11. Sparse deconv + prune: 0.6m -> 0.3m
12. Produce F_0.3 sparse surface feature (kept voxels)
13. Occupancy Head (sparse fine 표면 갈래)
14. Occupancy Flow Head (0.3m sparse 출력, motion은 step 8에서)
15. Sub-Voxel Shape Information Head (sparse)
16. 3D Semantics Head (sparse)
17. Surface Outputs head (별도 브랜치, dense BEV)
18. Build exposed-face / priority mask
19. Packed queryable MLP for selected local points (pruned 영역은 coarse로 즉답)
20. Planner readout
```

최종 구조를 한 줄로 정리하면:

```text
1.2m에서 이미지를 3D로 들어올리고,
0.6m에서 ViewFormer-style 시간 정렬 + flow + coarse dense free/unknown을 만들고,
sparse deconv + 프루닝으로 occupied 표면만 0.3m까지 올린 뒤,
Tesla-style 4개 volume head + Surface Outputs head를 붙인다.
```

---

## 15. Why This Architecture

### 1.2m Spatial Attention

0.6m image cross-attention보다 1.2m image cross-attention이 훨씬 싸다.

```text
1.2m queries:
  4,250

0.6m queries:
  34,000
```

attention 비용은 query 수, image token 수, memory movement에 민감하므로 embedded GPU에서는 coarse spatial lifting이 유리하다.

### Temporal at 0.6m, Before Final Deconvolution (ViewFormer-style)

Tesla 그림은 temporal alignment가 최종 deconvolution 이후 고해상도 feature가 아니라, 그 이전 coarse latent에 붙는 흐름으로 읽힌다. 본 문서는 그 coarse latent를 **0.6m 단일 stage**로 둔다.

```text
0.3m에서 temporal:
  cell 13,600개 -> Orin 위협, 금지

1.2m에서 temporal:
  Tesla coarsest 원칙엔 맞지만 ViewFormer(~0.8m)보다 거칢
  flow motion이 30cm 4x4에 평균화됨 -> 과보수적

0.6m에서 temporal (채택):
  ViewFormer voxel query ~0.8m에 근접 -> flow 적합
  Z=10층 유지(붕괴 없음), 여전히 최종 deconv 이전 latent
  cell 3,400개 + 경량 conv -> Orin 가능
```

ViewFormer 근거: voxel query를 z-squeeze해 BEV에서 streaming temporal을 하고 다시 voxel로 unsqueeze해 occupancy/flow를 낸다. 이 BEV-to-BEV temporal이 voxel-to-voxel보다 효율적이고 성능도 좋았다고 보고된다. 단, raw deconv feature는 날것일 수 있으므로 temporal 직전 0.6m pre-temporal refinement를 둔다(BEVDet4D 교훈).

### 30cm Sparse Surface Feature + Coarse Dense Occupancy

final feature를 전부 coarse하게 유지하면 Tesla식 4 volume head와 거리가 생기고, 반대로 0.3m를 완전 dense로 만들면 272,000 voxel 3D deconv가 Orin 20Hz를 위협한다. 그래서 둘로 나눈다.

```text
free / unknown (빈 공간, 저주파):
  coarse dense @ 0.6m

occupied surface (물체 경계, 고주파):
  sparse deconv + prune @ 0.3m
  -> F_0.3_sparse
  -> 4 volume heads
```

이 분리는 Tesla 데모(표면에만 voxel 존재 = surface-centric)와 맞고, dense 0.3m 비용을 피하면서 4 volume head를 그대로 유지한다. 빈 공간은 정밀도가 필요 없고, 정밀도가 필요한 표면만 30cm로 푼다.

### Sub-Voxel Shape as a Head

Sub-Voxel Shape Information은 성능 개선용 후처리가 아니다. Occupancy, Flow, Semantics와 같은 level의 volume output이다.

다만 그 내부에서:

```text
모든 voxel을 같은 비용으로 자세히 보지 않고,
exposed face / uncertainty / planner relevance에 따라
local query budget을 다르게 배분한다.
```

---

## 16. Implementation Notes

초기 구현 권장 순서:

```text
Stage 1:
  canonical-view preprocessing
  + 1.2m spatial attention
  + dense deconv 1.2m -> 0.6m
  + coarse dense occupancy / visibility head @ 0.6m (free/unknown)
  + 0.6m -> 0.3m decoder
    (학습 초기엔 dense로 프로토타입,
     occupancy supervision이 안정되면 sparse deconv + prune으로 전환)
  + Occupancy Head (표면 갈래)
  + 3D Semantics Head의 최소 class
  + auxiliary depth head (training-only)

Stage 2:
  0.6m pre-temporal refinement 추가
  ViewFormer-style single BEV temporal @ 0.6m 추가
  z-squeeze / streaming memory(N=3) / ego-motion warp / unsqueeze

Stage 3:
  Occupancy Flow Head 추가 (0.6m BEV-level flow -> 0.3m 매핑)
  dynamic probability / flow supervision 추가

Stage 4:
  Sub-Voxel Shape Information Head 추가
  exposed_face_mask / local offset / normal / uncertainty 학습

Stage 5:
  packed Queryable MLP 추가
  exposed-face / planner-aware query budget 추가

Stage 6:
  Surface Outputs head 추가 (z_surface / slope / step / uncertainty)
```

v1에서 명시적으로 제외:

```text
0.6m or 0.3m image cross-attention
0.3m full global temporal attention
full 3D temporal memory
30cm dense volume feature (sparse deconv + prune으로 대체)
all-voxel dense local sub-point query
large shape decoder per voxel
high-resolution local image re-query
```

training augmentation:

```text
camera extrinsic noise
intrinsic / FOV noise
virtual camera crop jitter
camera dropout
brightness / exposure augmentation
temporal frame dropout
history memory dropout
ego-motion pose noise
```

---

## 17. Current Recommended Architecture

현재 최종 추천:

```text
Input:
  multi-camera canonical views

Image encoder:
  backbone + FPN/BiFPN
  canonical ray positional embedding

Spatial lifting:
  1.2m 3D voxel queries
  vanilla cross-attention with image tokens

Dense deconv:
  1.2m -> 0.6m (dense)

Pre-temporal refinement @ 0.6m:
  light 3D/BEV feature cleanup (BEVDet4D 교훈)

Temporal module (ViewFormer-style, single stage @ 0.6m):
  z-squeeze 0.6m 3D feature to BEV
  streaming BEV memory queue (N=3~4)
  ego-motion alignment at 0.6m
  경량 gated temporal fusion (full attention 금지)
  BEV-level occupancy flow
  unsqueeze / inject temporal BEV context back into 0.6m 3D feature

Coarse dense occupancy:
  occupied / free / unknown / visibility @ 0.6m
  (free/unknown 담당 + 프루닝 guide)

Sparse decoder:
  sparse deconv + prune: 0.6m -> 0.3m (keep 0.5, recall 우선)

Final feature:
  F_0.3_sparse surface feature (kept occupied/boundary voxels)

Volume heads (= Tesla Volume Outputs):
  Occupancy Head (coarse dense free/unknown + sparse fine 표면)
  Occupancy Flow Head (0.3m sparse 출력, motion은 0.6m temporal)
  Sub-Voxel Shape Information Head (sparse)
  3D Semantics Head (sparse)

Surface Outputs head (= Tesla Surface Outputs, 별도 브랜치):
  dense BEV: z_surface / slope / step / uncertainty

Sub-voxel shape:
  exposed face mask
  face surface offset
  local surface normal
  shape uncertainty
  optional thinness
  shape code for queryable MLP

Queryable output:
  xyz + sampled F_0.3 + shape code + local coordinate
  -> continuous occupancy / semantic / uncertainty

Runtime controls:
  exposed-face mask
  planner-aware priority mask
  packed query execution
```

가장 중요한 설계 원칙:

```text
30cm는 occupied 표면의 결과 해상도다. 단 내부 표현은 dense가 아니라 sparse다.
free / unknown 빈 공간은 coarse dense에서 처리한다 (정밀도 불필요).
Sub-Voxel Shape Information은 30cm voxel 내부 형상을 말하는 head다.
보이는 / 노출된 face만 더 자세히 보는 것은 head 내부의 masked execution이다.
sparse deconv + 프루닝 = 빈 공간 제거, masked query = 표면 중 노출면만 정밀 -> 같은 철학의 두 단계.
```

---

## 18. Recommended Papers and Repositories

이 문서의 아키텍처를 구현하려면 논문을 모두 같은 깊이로 볼 필요는 없다. 목적별로 보는 순서를 나눈다.

### Core Architecture

1. [REO](https://github.com/ICEORY/REO)
   - calibration-free / calibration-light **vanilla attention 방식**과 query-based prediction 참고
   - 주의: REO는 이 vanilla attention을 BEV query에 적용한다. 본 문서는 그 attention "종류"만 가져오고,
     query "타깃"은 Tesla처럼 3D voxel로 둔다 (REO 전체 파이프라인을 모사하는 것이 아님)
   - queryable output interface에도 반영

2. [ViewFormer-Occ](https://github.com/viewformerocc/viewformer-occ)
   - z-squeeze voxel->BEV temporal, streaming memory(N=4), ego-motion alignment, BEV-level occupancy flow(FlowOcc3D) 참고
   - voxel query 100x100x8(~0.8m), GT 200x200x16(0.4m), ~4.1 FPS(RTX 3090)
   - 이 문서에서는 0.6m 단일 BEV temporal+flow stage와 Occupancy Flow Head(BEV-level flow -> voxel 매핑)에 반영

3. BEVFormer / BEVDet4D
   - BEV feature ego-motion alignment와 temporal fusion 위치에 대한 교훈 참고
   - 이 문서에서는 pre-temporal refinement와 deconv 전 temporal alignment에 반영

4. [PanoOcc](https://github.com/Robertwyq/PanoOcc)
   - coarse-to-fine **sparse deconvolution + occupancy sparsification(프루닝)** 과 unified occupancy representation 참고
   - PanoOcc는 3단계 sparse deconv에서 keep ratio 0.2/0.5/0.5로 점유 후보만 남겨 dense 고해상 3D conv를 회피한다
   - 이 문서에서는 1.2m -> 0.6m -> 0.3m sparse decoder + 프루닝에 그대로 반영 (keep ratio는 원거리 보호 위해 더 보수적으로)

5. SparseOcc / sparse execution examples
   - 모든 위치에 같은 비용을 쓰지 않고 중요한 위치만 더 비싼 연산을 쓰는 runtime idea 참고
   - 이 문서에서는 (1) sparse decoder의 프루닝 (2) Sub-Voxel Shape Head의 exposed-face / priority masked execution 두 곳에 반영

### Surface Geometry and Semantics

1. [RoadBEV](https://github.com/ztsrxh/RoadBEV)
   - BEV에서 road surface elevation을 직접 예측하는 기본 참고자료
   - 이 문서에서는 별도 Surface Outputs head(geometry only)가 0.6m dense feature에서 z_surface를 직접 회귀하는 방식에 반영 (Section 11)

2. [FastRSR](https://arxiv.org/abs/2504.09535)
   - RoadBEV 계열의 효율 개선 방향 참고
   - Jetson AGX Orin 같은 embedded inference를 생각하면 RoadBEV 다음으로 볼 것

3. [MapTR](https://github.com/hustvl/MapTR)
   - lane, divider, boundary 같은 online vectorized HD map element 예측 참고
   - 3D Semantics Head와 planner readout을 확장할 때 선택적으로 참고

4. [HDMapNet](https://arxiv.org/abs/2107.06307)
   - BEV semantic map learning의 기본 참고자료
   - road / lane / curb 같은 raster BEV semantics를 추가할 때 선택적으로 참고

5. [H3O](https://arxiv.org/abs/2503.04059)
   - depth, semantic segmentation, surface normal을 3D occupancy 학습에 보조 supervision으로 쓰는 방향 참고
   - surface normal / depth / semantic auxiliary loss가 필요할 때 선택적으로 참고

6. [UnScenes3D](https://www.nature.com/articles/s41597-025-05532-5)
   - 3D semantic occupancy와 road surface elevation reconstruction을 함께 정의한 데이터/태스크 참고
   - volume output과 surface geometry output을 함께 학습/평가하는 관점에서 참고

### Minimal Reading Order

최소로만 본다면 다음 순서를 추천한다.

```text
1. Tesla AI Day 2022 Occupancy Network 발표 자료
2. ViewFormer-Occ
3. BEVDet4D / BEVFormer temporal fusion sections
4. REO
5. PanoOcc
6. RoadBEV / FastRSR
7. Sparse execution examples
```

---

## 19. GT Data Pipeline (Camera-only Auto-labeling)

이 문서의 모든 head는 supervision이 필요하다. LiDAR가 없는 camera-only 환경에서는 DUSt3R 계열 3D foundation model을 기반으로 한 offline auto-labeling으로 GT를 만든다. offline이므로 추론 latency 제약과 무관하게 무거운 모델을 쓸 수 있다.

### Why Not Vanilla DUSt3R

DUSt3R 아이디어 자체는 출발점이지만, vanilla DUSt3R를 직접 쓰지 않는 이유는 네 가지다.

```text
1. scale 모호성:
   DUSt3R pointmap은 up-to-scale이다.
   30cm metric voxel GT에는 metric scale이 필수다.

2. pair 단위 추론 + 비싼 global alignment:
   DUSt3R는 이미지 쌍 단위로 추론하고,
   multi-view 통합에 최적화 기반 global alignment가 필요하다.
   수백~수천 frame의 주행 시퀀스에 비실용적이다.

3. 정적 장면 가정:
   동적 물체는 재구성 기하가 깨진다.
   별도의 dynamic 분리 없이는 GT 오염이 발생한다.

4. 기지 calibration을 활용하지 못함:
   우리는 rig extrinsics와 odometry를 이미 알고 있다.
   vanilla DUSt3R는 이를 입력으로 받지 못하고 전부 재추정한다.
   아는 정보를 버리는 만큼 정확도와 일관성을 손해 본다.
```

따라서 모델 선택:

```text
1순위: MapAnything
  metric scale 출력
  known intrinsics / pose를 입력으로 소비 가능
  multi-camera rig + odometry 보유 상황에 부합

대안:
  MASt3R / MASt3R-SfM
  VGGT

동적 분리 보조:
  MonST3R 또는 2D segmentation + tracking
```

### Two-stage Structure

DUSt3R 계열과 continuous reconstruction은 경쟁 관계가 아니라 순차 역할 분담이다.

```text
Stage A: 기하 + 포즈
  metric pointmap, camera pose, dense 초기 기하
  후속 정제 단계는 안정적인 metric pose와 초기 기하가 있어야 시작할 수 있다.

Stage B: 정제 + 연속 표면
  Stage A 출력을 초기화로 photometric / geometric 최적화
  표면 정제 후 mesh / SDF 추출
```

Stage B의 목적은 화질이 아니다. Stage A 점군을 voxelize하면 30cm 이산 GT까지만 나온다. Stage B가 만드는 연속 mesh/SDF가 Sub-Voxel Shape Information Head와 Queryable MLP를 학습시킬 **sub-voxel continuous GT**다.

### Pipeline

```text
1. 수집:      multi-camera 시퀀스 + Basalt VIO pose (Section 20) + 시각 동기화

2. 기하:      MapAnything (images + known intrinsics + pose prior)
              -> metric pointmaps, chunk 단위 처리

3. 동적 분리: SAM2 + tracking -> dynamic mask
              -> 정적 aggregation에서 동적 물체 제외

4. 정적:      static 점군 multi-frame aggregation -> TSDF fusion
              -> mesh / SDF
              -> 30cm voxelize -> volume occupancy GT
              -> ground / road segmentation -> Surface Outputs head GT (z_surface 등)
              -> 연속 SDF 보존 -> sub-voxel shape GT

5. 가시성:    per-frame camera ray casting -> free / unknown 분리
              (free != unknown 라벨의 유일한 출처)

6. 동적:      per-frame instance volume + track association
              -> dynamic occupancy + occupancy flow GT

7. 옵션:      neural SDF / NeRF refinement
              -> 표면 정밀화 + 연속 GT 강화

8. 검증:      같은 파이프라인을 nuScenes에 적용
              -> Occ3D LiDAR GT와 정량 비교
              -> 거리별 오차 프로파일 확보 후 자체 데이터에 적용
```

### Head별 GT 매핑

```text
Occupancy Head (coarse dense 갈래 @ 0.6m):
  step 4 voxelize + step 5 visibility
  -> 0.6m로 다운샘플한 free / unknown / occupied GT

Occupancy Head (sparse fine 갈래 @ 0.3m) + 프루닝:
  step 4 voxelize의 occupied GT
  -> 0.6m / 0.3m scale에서 multi-scale occupancy supervision
  -> 프루닝 keep/drop 학습 (occupied GT가 keep 타깃)
  주의: GT에서 occupied인 voxel이 프루닝되면 영구 손실 -> recall 우선 keep ratio

Occupancy Flow Head:
  step 6 tracking
  -> 0.6m BEV-level flow GT (ViewFormer FlowOcc3D 방식), 0.3m voxel로 매핑

Sub-Voxel Shape Information Head:
  step 4 / 7의 연속 SDF와 boundary samples
  exposed face / local offset / normal / uncertainty supervision

3D Semantics Head:
  static / dynamic segmentation
  road / vehicle / pedestrian / curb / wall 등 semantic labels

Surface Outputs head:
  ground / road segmentation + 연속 표면(SDF/mesh)
  -> per-BEV-cell z_surface / slope / step / valid / uncertainty
```

### Limitations

```text
원거리 정확도:
  camera 기반 depth 오차는 거리에 따라 급증한다.
  30cm GT 신뢰 범위도 원거리에서는 급격히 낮아진다.
  far-field는 GT uncertainty를 낮춰 loss weight를 줄이거나
  unknown으로 라벨한다.

검증 없는 적용 금지:
  자체 데이터에는 비교 기준이 없으므로,
  step 8의 nuScenes 검증으로 오차 수준을 먼저 정량화한다.

품질 보증:
  가능하면 GT 수집용으로만 LiDAR 1대를 일부 시퀀스에 장착해
  spot-check한다. 추론은 여전히 camera-only다.
```

### References

1. [DUSt3R](https://github.com/naver/dust3r) / [MASt3R](https://github.com/naver/mast3r)
   - pointmap 기반 unconstrained 3D 재구성의 출발점

2. [MapAnything](https://github.com/facebookresearch/map-anything)
   - metric feed-forward 재구성, known calibration/pose 입력 소비
   - 이 pipeline의 Stage A 1순위

3. [VGGT](https://github.com/facebookresearch/vggt)
   - 단일 forward로 pose + depth + pointmap 출력

4. [MonST3R](https://monst3r-project.github.io/)
   - 동적 장면 확장, dynamic mask 참고

5. [EmerNeRF](https://emernerf.github.io/)
   - self-supervised static/dynamic decomposition + flow
   - Stage B를 동적 장면까지 확장할 때 참고

---

## 20. VIO Integration (Basalt)

ego pose는 multi-camera Basalt VIO로 추정한다. Basalt의 "feature"는 학습된 embedding이 아니라 FAST corner + KLT patch tracking 기반의 기하 산출물이므로, image backbone과 feature를 공유하는 방식의 재활용은 불가능하다. 재활용 대상은 VIO 최적화기의 산출물이다.

```text
Basalt VIO 산출물:
  1. ego pose (IMU rate, metric scale)
  2. 중력 방향 (IMU)
  3. triangulated 3D landmark (sparse, metric, 수백 개/keyframe)
  4. landmark별 covariance
  5. outlier로 기각된 track
```

### A. Pose / 중력 정렬 (v1, 필수)

```text
pose:
  Section 7 temporal BEV memory의 ego-motion warp 입력

중력 방향:
  voxel grid의 z축을 차체가 아니라 중력에 정렬
  차체 pitch/roll에도 grid가 흔들리지 않음
  Section 1의 z range [-2m, +4m]는 이 중력 정렬 기준이다
```

### B. Landmark -> Sparse Depth Supervision (v1, 학습)

Section 4의 auxiliary depth supervision에 들어가는 온라인 GT 소스다.

```text
landmark를 이미지에 재투영
-> 해당 픽셀의 metric depth GT
-> depth head를 sparse하게 supervise
-> landmark covariance로 loss 가중
```

주의: landmark는 코너/텍스처 영역에만 찍히므로 분포가 편향된다. Section 19의 dense GT를 대체하지 않고 보완한다. 운영 중 공짜로 계속 쌓이는 것이 장점이다.

### C. Landmark 입력 주입 (v1.5)

supervision을 넘어 추론 시 입력 힌트로도 쓸 수 있다. known geometry 입력이 vanilla attention의 기하 힌트 부족을 보완한다.

```text
방법 1: voxel grid prior
  landmark 3D 위치를 1.2m voxel grid에 splat
  -> sparse 점유 prior 채널

방법 2: landmark token
  (위치, depth, covariance)를 추가 K/V 토큰으로
  cross-attention에 합류

필수 안전장치:
  학습 시 landmark 입력 전체를 확률적으로 dropout
  VIO 품질 저하(저텍스처, 급회전) 시에도
  perception이 단독 동작해야 한다
```

### D. Outlier Track -> Dynamic 힌트 (v1.5)

VIO가 reprojection / epipolar 불일치로 기각한 track은 동적 물체 위에 있을 가능성이 높다.

```text
기각된 track의 픽셀 / 방향
-> Occupancy Flow Head의 weak pseudo-label
-> dynamic boundary query priority 증가
```

노이즈가 많으므로 단독 라벨이 아니라 약한 보조 신호로만 쓴다.

### E. Landmark Ray -> Free Space 증거 (v1.5)

카메라에서 landmark까지 빛이 도달했다는 것은 그 사이 공간이 비어있다는 뜻이다.

```text
camera center -> landmark 구간의 voxel
-> 관측된 free 증거
-> free vs unknown 구분의 온라인 보조 신호
```

Section 19의 offline ray casting을 온라인 sparse로 보완한다.

### 우선순위

```text
v1:   A (pose / 중력 정렬)
      B (landmark sparse depth supervision)

v1.5: C (landmark 입력 주입 + dropout)
      D (outlier -> dynamic 힌트)
      E (landmark ray -> free space)
```
