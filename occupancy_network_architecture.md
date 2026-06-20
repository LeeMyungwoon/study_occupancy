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
| ViewFormer | streaming memory(N=4)로 학습 효율 / occupancy flow 데이터셋(FlowOcc3D) | streaming memory만 선택 차용. z-squeeze + temporal attention은 Tesla와 달라 폐기하고 PanoOcc식 3D temporal 채택 |
| BEVFormer / BEVDet4D | BEV feature ego-motion alignment, temporal fusion 위치에 대한 교훈 | pre-temporal refinement, coarse temporal fusion |
| PanoOcc | (1) coarse 3D voxel **temporal: align + concat + residual 3D conv** (z 유지, coarse ~40k voxel, 과거3+현재1) (2) coarse-to-fine **sparse deconvolution + occupancy 프루닝** | (1) 0.6m 3D 단일 temporal stage (z 유지, per-voxel flow, Tesla 그림 align+concat과 동일 계열) (2) 1.2m->0.6m->0.3m sparse decoder (점유 후보만 keep) |
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

참조 다이어그램: Tesla AI Day 2022 Occupancy Network — [tesla_ai_day.png](tesla_ai_day.png), [tesla_ai_day_2.png](tesla_ai_day_2.png)

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
| Tesla/PanoOcc Temporal@0.6m 3D|<----->| 3D Voxel Memory (선택:    |
| z 유지(no squeeze)            |       |   ViewFormer streaming)  |
| align + concat + 3D res conv  |       | ego-motion aligned 3D    |
| (NOT attention)               |       | 과거3+현재1 = 4 frame    |
| per-voxel flow (vx,vy,vz)     |       +--------------------------+
+---------------+---------------+
                |
                v
+-------------------------------+       +--------------------------+
| Coarse Dense Occupancy @0.6m  |       | Sparse Deconv (2단 게이트)|
| occupied/free/unknown/vis     |       | ①0.6m parent gate(연산)  |
| (빈 공간 free/unknown 담당)   |       | ②0.3m child prune(선명도) |
| (프루닝 게이트 기준도 담당)   |       +------------+-------------+
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
| 2. Occupancy Flow Head        (0.3m sparse 출력, vx/vy/vz,   |
|                                motion은 0.6m 3D temporal에서)|
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
+ Tesla/PanoOcc-style single 3D temporal @ 0.6m (align+concat+3D conv, per-voxel flow)
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
- temporal은 **단일 stage @ 0.6m 3D** (Tesla/PanoOcc-style, z-squeeze 없이 Z 유지)로 한다. deconv 1.2m -> 0.6m 직후, deconv 0.6m -> 0.3m 이전의 coarse latent에서 처리한다. 융합은 **trajectory align + concat + 3D residual conv** (attention 아님).
- Flow는 0.3m sparse로 출력하되, motion 정보는 0.6m 3D temporal에서 온다 (per-voxel vx/vy/vz -> voxel 매핑). Z 유지라 높이별 motion(vz)까지 잡는다.
- 1.2m 별도 temporal stage는 두지 않는다. 단 **1.2m 3D는 Orin 실측이 20Hz 예산 초과 시의 fallback**으로만 둔다.
- z-squeeze BEV temporal, temporal attention(ViewFormer식), full global temporal attention(0.6m/0.3m), 0.3m temporal, all-voxel heavy local query는 v1에서 금지한다 (ViewFormer streaming memory는 학습 효율용으로만 선택 차용).

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

valid target ROI는 internal grid 안에서 mask로 관리한다. 단 padding은 **Y축에만** 있다 (위 범위 참조: X 60m=200칸, Z 6m=20칸은 격자=타깃이라 여백 없음; Y만 20m 타깃 -> 20.4m=68칸이라 0.4m 여백).

```text
model output 좌표계:
  200 x 68 x 20 nominal grid  (X·Z는 타깃과 정확히 일치, Y만 +0.4m padded)
  (occupied 표면은 이 중 점유 후보만 sparse 활성,
   free/unknown은 coarse dense에서 관리)

valid target region:
  원래 목표 범위에 해당하는 cell만 loss / metric / planner에 사용

padded region (Y축 가장자리 0.4m 한정):
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
2단 게이트(0.6m parent + 0.3m child, keep ~0.5) 적용 시
  0.3m 단계 활성 voxel은 대략 수만 voxel 수준 (272k의 일부).
  + near-surface free shell(L2, 기본 ON)이 occupied 표면 대비 ~0.5~1배를 더해
    kept 총량이 대략 1.5~2배가 된다 (Section 8.2 N3 참조).
-> 그래도 272k dense 대비 메모리와 3D conv 연산은 크게 감소.
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
-> Tesla/PanoOcc-style single 3D temporal @ 0.6m (z 유지)
   3D voxel memory(과거3+현재1=4) -> ego-motion 3D warp(align)
   -> concat -> 3D residual conv(fuse, NOT attention)
   -> per-voxel flow (vx,vy,vz)
-> temporally enhanced 0.6m 3D feature

-> coarse dense occupancy / visibility head @ 0.6m (free/unknown + 프루닝 게이트 기준)

-> [surface 가지] 2단 게이트로 0.3m sparse 생성:
   게이트① @0.6m (연산량): coarse occupancy로 occupied/boundary parent만 남김
                          -> 남은 parent만 sparse 업샘플 (빈 영역 0.3m는 생성조차 안 함)
   게이트② @0.3m (선명도): 확장된 child 중 빈 child 제거 (표면 + 인접 free 1겹, anti-dilation)
-> 30cm sparse surface feature (kept voxels only)

-> Occupancy Head (sparse fine 표면 정밀)
-> Occupancy Flow Head (0.3m sparse 출력, vx/vy/vz, motion은 0.6m 3D temporal에서)
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
  single Tesla/PanoOcc-style 3D temporal (z 유지) + per-voxel flow
  coarse dense occupancy / free / unknown / visibility
  occupancy 프루닝 guide
  mid-level geometry refinement

0.3m (sparse):
  final sparse surface feature (kept voxels only)
  Tesla-style 4 volume heads
  queryable occupancy의 fine feature source
  (Surface Outputs head는 0.3m을 입력으로 받지 않음; consistency loss로만 연결)
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
| Occupancy Flow Head | 0.3m sparse kept voxels (motion은 0.6m 3D temporal) | dynamic probability, occupancy flow, velocity (vx/vy/vz) |
| Sub-Voxel Shape Information Head | sparse kept voxels + exposed-face masked local query | 30cm voxel 내부의 exposed face, surface offset, normal, shape code, uncertainty |
| 3D Semantics Head | sparse kept voxels | voxel semantic class |

별도 출력 브랜치 (Tesla 그림 기준):

| 출력 | 위치 | 역할 |
|---|---|---|
| Surface Outputs head | dense BEV (Section 11) | z_surface / slope / step / uncertainty |
| Queryable Outputs | MLP 인터페이스 (Section 10) | 임의 좌표 occupancy (occupancy 전용; semantic은 3D Semantics Head) |

아래 항목들은 head가 아니라 runtime/control mechanism이다.

```text
Exposed-face mask:
  Sub-Voxel Shape Head가 어느 face를 자세히 볼지 정한다.

Active / priority mask:
  planner path, uncertainty, dynamic, visibility frontier 등으로
  expensive local query budget을 배분한다.

Queryable MLP:
  kept voxel 근처는 Sub-Voxel Shape Head의 shape code와 F_0.3_sparse feature를,
  pruned 영역은 parent 0.6m coarse occupancy를 사용해
  임의 좌표 x,y,z의 occupancy를 평가하는 interface다 (occupancy 전용; semantic은 3D Semantics Head).
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
-> Tesla/PanoOcc-style single 3D temporal @ 0.6m (z 유지, align+concat+3D conv)
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
-> Tesla/PanoOcc-style 3D temporal @ 0.6m (z 유지)
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
# temporal은 3D에서 수행 (z-squeeze 없음, Z 유지 - Section 7)
```

---

## 7. Tesla/PanoOcc-style Single 3D Temporal Stage at 0.6m

Tesla AI Day 2022 그림을 구조적으로 읽으면 temporal alignment는 deconvolution 이후의 고해상도 volume feature가 아니라, spatial attention 이후 deconvolution 이전의 coarse latent에서 수행된다. 본 문서도 그 원칙(temporal before final deconv)을 따른다. 다만 그 coarse latent를 **0.6m 단일 stage**로 두고, **z-squeeze 없이 3D 부피를 유지**한다.

### 7.0 Tesla는 attention이 아니라 align + concat이다 (그림 판독)

Tesla 그림의 Temporal Alignment 열을 확대해 읽으면:

```text
- "Attention" 라벨은 그 앞 Spatial Attention 단계의 것이고,
  temporal 쪽엔 attention 기호가 없다.
- 과거 Spatial Features (t-0, t-1, t-2, t-3 ...)는 모두 3D 부피(슬래브)로
  그려진다 -> 높이 유지, z-squeeze 아님.
- 이들이 "Spatial Frame Alignment"(trajectory warp)로 정렬되어
  "Spatiotemporal Features"로 쌓인다(concat / stack).
- 캡션: "Trajectory used to align past features to current coordinate frame"
  -> 정렬 + concat 이지 attention이 아니다.
```

즉 Tesla temporal = **trajectory align + concat (3D 부피 유지)**. PanoOcc도 같은 계열이다("temporal align + temporal fuse": 3D voxel 공간 정렬 후 concat + residual 3D conv, 과거 3 + 현재 1 = 4 frame, coarse 50×50×16 = 40,000 voxel). 본 문서는 이 방식을 따른다. ViewFormer의 z-squeeze + temporal attention은 오히려 Tesla에서 벗어난 쪽이라 v1에서 폐기한다 (streaming memory만 학습 효율용으로 선택 차용).

### 7.1 왜 0.6m 단일 stage / 3D 유지인가

```text
Tesla:     coarse latent에서 3D 부피를 trajectory 정렬 + concat (z 유지).
PanoOcc:   coarse 50x50x16 = 40,000 voxel에서 3D align + concat + residual 3D conv,
           과거 3 + 현재 1 = 4 frame, 0.5s 간격.
우리:      0.6m coarse 3D = 100 x 34 x 10 = 34,000 voxel (PanoOcc 40k보다 적음).
           Z=10 유지 -> 높이별 motion / vz 가능. 여전히 deconv 이전 coarse latent.
```

판단:

```text
z-squeeze BEV (ViewFormer, 구버전 -> 폐기):
  100 x 34 = 3,400 cell로 싸지만 Z 소실 -> 높이별 motion / vz 불가, Tesla와 다름.

0.6m 3D (Tesla / PanoOcc, 채택):
  34,000 voxel. 융합이 attention이 아니라 concat + 3D conv라 비용은 voxel 수에 선형.
  PanoOcc가 40k에서 검증 -> Orin 가능. Z 유지로 vz까지 예측.

-> v1 main temporal은 0.6m 3D 단일 stage (align + concat + 3D residual conv).
   1.2m 3D는 Orin 실측이 20Hz 예산을 넘을 때의 fallback로만 둔다 (품질↓·비용↓,
   단계만 1.2m로 내리면 되어 구조 변경 없이 후퇴 가능).
```

비용의 핵심은 해상도가 아니라 **operator**다.

```text
0.6m 3D = 34,000 voxel

full self-attention:  34,000^2 -> 금지 (ViewFormer가 무겁던 진짜 이유)
align + concat + depthwise/separable 3D conv:  voxel 수에 선형 -> 감당 가능
history N=3 메모리:  100 x 34 x 10 x 64 x 2B ≈ 4.4MB/frame -> 무시 가능

ViewFormer가 3090에서 4 FPS인 것은 0.8m라서가 아니라 temporal attention 때문이다.
우리는 PanoOcc식 concat + 3D conv로 한다.
```

### 7.2 흐름 (3D align -> concat -> residual 3D conv -> per-voxel flow)

z-squeeze 없이 0.6m 3D feature를 그대로 trajectory 정렬해 concat한 뒤 residual 3D conv로 융합한다. flow는 per-voxel 3D로 예측한다 (Z 유지 -> vx, vy, vz).

```text
F_0.6_refined: 100 x 34 x 10 x C   (3D, Z 유지)

Memory (3D voxel feature):
  [F_0.6_{t-1}, F_0.6_{t-2}, F_0.6_{t-3}]
  keyframe 간격 ~0.2s (0.6~0.8s 창)

Aligned memory (3D voxel warp):
  Warp3D(F_0.6_{t-k}, pose_{t-k -> t})   (ego-motion, 3D 공간 정렬 - PanoOcc식)

Fuse (concat + residual 3D conv, NOT attention):
  X = concat(F_0.6_t, aligned F_0.6_{t-1..t-3})   # 채널 방향 stack
  F_0.6_temporal = F_0.6_t + Residual3DConv(X)

Per-voxel flow (Z 유지):
  F_flow = FlowHead3D(F_0.6_temporal)
  -> dynamic_logit, vx, vy, vz  per voxel (Z=10이라 높이별 motion 살아남)
  flow는 0.6m voxel -> 그 안의 0.3m sparse voxel로 broadcast
```

### 7.3 temporal fusion operator (v1 / v1.5)

```text
권장 v1:
  ego-motion 3D warp
  concatenate current + aligned history (채널 stack)
  depthwise / separable 3D conv
  gated residual fusion
  (선택) ViewFormer streaming memory: 학습 시 과거 feature를 재계산하지 않고
         캐시 -> 학습 효율↑, 추론 latency 변화 없음

v1.5 이후 검토:
  local / windowed temporal attention (full global은 계속 금지)
  limited deformable sampling
  장기 누적 기억 (Tesla "Temporal Context", 아래 T3)
```

추천 memory 길이:

```text
N_history = 3 keyframes (현재 포함 4 frame, PanoOcc와 동일)
keyframe 간격 ~0.2s
```

20 FPS 연속 frame은 창이 0.15~0.2s로 occlusion 추론에 너무 짧다. 시간 간격 기반으로 띄엄띄엄 저장해 0.6~0.8s 창을 확보한다.

장기 누적 기억 (Tesla Temporal Context, v1.5 옵션):

```text
Tesla 그림은 최근 N프레임 외에 "Temporal Context"라는 지속 상태를 따로 둔다
(오래 가려진 물체를 수 초 이상 기억하는 용도로 추정).
v1은 최근 3프레임(0.6~0.8s) 큐만 두므로, 수 초간 가려진 물체 추론은 약하다.

v1.5 옵션: ego-motion으로 정렬되는 가벼운 누적 BEV/voxel 상태(GRU/EMA식)를
  하나 추가해, N프레임 큐를 넘어선 장기 occlusion 기억을 보강한다.
  (큐는 단기 정밀, 누적 상태는 장기 잔존 — 역할 분리.)
```

3D grid 비교 (operator 비용 감각):

```text
1.2m 3D:  50 x 17 x 5   = 4,250 voxel    <- fallback (Orin 예산 초과 시)
0.6m 3D:  100 x 34 x 10 = 34,000 voxel   <- 본 문서 temporal 위치 (PanoOcc 40k 이내)
0.3m 3D:  200 x 68 x 20 = 272,000 voxel  <- temporal 금지
```

금지:

```text
0.3m temporal (voxel 8배, Orin 위협)
0.6m / 0.3m full global temporal attention
z-squeeze BEV temporal (Tesla / PanoOcc와 다름, 높이 motion 소실 -> 폐기)
```

---

## 8. Deconvolution to 30cm (0.6m dense -> 0.3m sparse)

해상도 단계를 비용에 맞게 나눈다. dense 비용이 위험한 곳은 0.3m(272k voxel)뿐이므로, **1.2m->0.6m는 dense로 두고(temporal/coarse occupancy가 여기 산다), 0.6m->0.3m만 sparse deconv + 프루닝**으로 만든다. PanoOcc의 coarse-to-fine sparse 업샘플 아이디어를 이 마지막 단계에 적용한다.

### 8.1 단계별 정리

```text
F_1.2 (dense, 50 x 17 x 5)
-> dense deconv -> F_0.6 (dense, 100 x 34 x 10)
-> 0.6m pre-temporal refinement (Section 6)
-> Tesla/PanoOcc-style 3D temporal @ 0.6m (Section 7, z 유지) -> F_0.6_temporal (dense)

[여기서 두 갈래]

갈래 1) Coarse Dense Occupancy Head @ 0.6m:
  occupied / free / unknown / visibility logits (dense, 100 x 34 x 10)
  -> free vs unknown 구분의 dense source
  -> 동시에 0.3m sparse 프루닝을 guide

갈래 2) Surface 가지: 2단 게이트로 0.3m sparse 생성
  게이트① @0.6m (parent gate):
    coarse occupancy로 occupied/boundary parent만 남기고
    free/unknown parent는 버린다.
    -> 남은 parent만 0.3m로 sparse 업샘플
    -> 빈 영역의 0.3m children은 "애초에 생성하지 않음" (272k 전체 materialize 회피)
  게이트② @0.3m (child prune):
    확장된 child 중 표면이 아닌 빈 child를 제거 (keep ~0.5)
    단, 표면 인접 free child 1겹은 함께 keep (near-surface 1-shell, 기본 ON)
    -> F_0.3_sparse (occupied/boundary surface + near-surface free 1-shell)
```

주의: dense 0.6m feature(temporal/free-unknown용)는 그대로 둔다. 게이트는 그 dense를 자르는 게 아니라, **0.3m로 내려보내는 surface 가지에만** 적용된다. 즉 "0.6m은 dense 유지" 원칙과 충돌하지 않는다.

0.6m을 dense로 두는 이유:

```text
0.6m dense = 100 x 34 x 10 = 34,000 voxel.
3D conv가 가볍고, 3D temporal(z 유지)과 coarse dense occupancy를
여기서 한 번에 처리할 수 있다 (PanoOcc coarse 40k voxel 이내).
반면 0.3m dense = 272,000 voxel -> 이 단계만 sparse로 피한다.
```

4개 volume head는 `F_0.3_sparse`의 살아남은 voxel 위에 붙는다. Occupancy Head만 coarse dense 갈래(free/unknown)를 0.6m에서 추가로 갖는다.

### 8.2 2단 게이트의 두 가지 목적 (연산량 + 선명도)

프루닝은 단순 비용 절감이 아니다. 두 게이트가 **서로 다른 목적**을 가진다.

```text
게이트① @0.6m (parent gate)  -> 목적: 연산량
  coarse occupancy로 free/unknown parent를 통째로 버려
  빈 공간의 0.3m children을 생성조차 안 한다.
  기준: 0.6m coarse occupancy의 occupied 확률 (별도 score head 불필요).

게이트② @0.3m (child prune)  -> 목적: 선명도 (anti-dilation)
  확장된 child 중 빈 child를 제거해 표면을 또렷하게 유지.
  단, 표면 인접 1겹 free child는 함께 keep한다 (near-surface 1-shell, 기본 ON).
  keep ratio ~0.5 (occupied recall 우선).
```

**왜 게이트②가 선명도에 중요한가 (anti-dilation):**

```text
dense 3D deconv(transposed conv)는 occupied 활성을 주변 빈 voxel로
"퍼뜨린다(dilation)":
  - 표면이 비대해짐, 좁은 틈이 메워짐, checkerboard
  - 층을 거칠수록 활성 영역이 계속 부풀어 오름 (submanifold dilation problem)
sparse + child prune은 빈 child를 제거해 활성을 표면에 집중시킨다:
  - 표면이 얇고 또렷하게 유지됨
  - Submanifold Sparse Conv가 고안된 핵심 이유와 같은 원리
-> 0.3m을 절대 dense로 만들면 안 되는 이유는 비용뿐 아니라 "표면 흐려짐"도 있다.
```

거리 적응:

```text
front 50m 원거리 + camera-only라 공격적으로 자르면 먼/얇은 물체가 영구 손실된다.
far-field는 keep ratio를 더 높이거나 occupancy score threshold를 낮춰 보호한다.
프루닝은 비가역이므로 recall 우선.
```

좁은 공간 보호 (near-surface 1-shell, **기본 ON**):

```text
근거: Tesla occupancy는 전부 dense(벽 앞 빈 공간도 0.3m)다. 표면 child만 남기면
      표면 바로 앞 free가 0.6m로만 해석되어, 2m 로봇이 좁은 통로를 "막힘"으로
      과소평가할 수 있다. 한정된 정밀 예산을 Tesla의 dense에 가장 가깝게 쓰는 길은
      "장애물 바로 앞 빈 공간"을 0.3m로 살리는 것이다 (그곳이 항법상 가장 중요).

동작: 게이트②에서 occupied 표면 child + 그 표면에 인접한 free child 1겹을 함께 keep.
      비용: 표면은 사실상 2D라 인접 free 1겹도 표면 voxel과 비슷한 규모다. 따라서
      kept voxel 수가 대략 1.5~2배로 늘어난다(공짜 아님) -> keep ratio / budget이
      이를 반영해야 한다(Section 1 cost 추정, Section 12 K_surface_voxels 참조).
      단 free shell voxel에는 무거운 Sub-Voxel Shape / Semantic head를 돌리지 않으므로
      (occupancy 해상도만 향상), 증가 비용은 sparse conv / 메모리 쪽에 한정된다.

안전: 안전은 빈칸을 통째로 막힘 처리해서 얻는 게 아니라, near-surface를 0.3m로 정밀히
      봐서 "진짜 막힘 / 통과 가능"을 정확히 구분해 얻는다.

config flag (near_surface_free_shell): Orin 실측이 20Hz 예산을 넘으면 끌 수 있다.
      끄면 표면-only로 돌아가며 near-surface free는 0.6m coarse로만 해석된다(보수적).
```

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
     단 dense fallback은 deconv dilation으로 표면이 다소 비대해질 수 있음
     (게이트②의 anti-dilation 이점을 일부 잃음).
```

---

## 9. Tesla-style 4 Volume Heads

head는 `F_0.3_sparse`(살아남은 kept voxel)와 coarse dense occupancy feature 두 source 위에 붙는다. 여기서 **kept voxel = occupied/boundary 표면 voxel + near-surface free shell voxel(L2)** 이다(occupied/free는 갈래 B의 fine occupancy flag로 구분). Flow / Sub-Voxel Shape / Semantics 같은 무거운 head는 **occupied 표면 voxel에서만** 동작하고, free shell voxel은 occupancy 해상도 향상에만 쓰인다.

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

Occupancy Flow Head는 dynamic probability와 flow / velocity를 예측한다. Tesla/PanoOcc 방식을 따른다: **motion 추정은 0.6m 3D temporal stage(Section 7)에서 per-voxel 3D flow로 하고, 출력은 0.3m sparse voxel에 매핑**한다. Z를 유지하므로 vx, vy에 더해 **vz(수직 속도)** 까지 예측한다.

```text
motion 정보 (정보의 출처):
  Section 7의 0.6m 3D temporal -> per-voxel 3D flow
  (dynamic_logit, vx, vy, vz) per 0.6m voxel (Z=10, 높이별 motion 유지)

출력 (다른 head와 같은 격자):
  0.3m sparse kept voxel별로 flow 벡터를 붙인다.
  0.6m voxel flow를 그 안의 0.3m voxel로 매핑(broadcast)하고,
  F_0.3_sparse의 fine geometry로 per-voxel 정제(선택).
```

왜 이렇게 나누나 (Tesla/PanoOcc 근거):

```text
Tesla/PanoOcc는 coarse 3D에서 motion을 추정하고 fine voxel에 펼친다.
flow의 motion granularity는 temporal을 한 해상도(0.6m)가 상한이다.
0.3m로 출력해도 0.6m 정보를 펼친 것이며, 0.3m에서 새 motion이 생기지 않는다.
강체(차량/보행자/카트)는 한 voxel column이 같이 움직이므로 0.6m로 충분.
3D 유지 덕에 높이별 다른 motion(예: 사람 다리/몸통, 포크)도 vz로 잡힌다.

-> 출력 격자 = 0.3m (다른 head와 일치),
   motion 해상도 = 0.6m 3D (Tesla/PanoOcc 수준, Orin 가능).
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
v1: motion 해상도 상한 = 0.6m (3D, vx/vy/vz).
    0.3m flow output은 0.6m 3D flow를 sparse surface voxel에 매핑한 것이다.
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

주의: kept voxel은 L2(near-surface free shell) 이후 **occupied 표면 voxel + 인접 free shell voxel** 두 종류가 섞여 있다. 따라서 노출 판정은 단순히 "neighbor가 kept인가"가 아니라 **"neighbor가 kept이면서 occupied인가"** 로 해야 한다(occupied/free 구분은 갈래 B의 fine occupancy flag로 한다). 그렇지 않으면 표면이 kept free shell과 맞닿는 면(= 진짜 노출면)을 "표면끼리 맞닿음"으로 오판해 노출에서 빠뜨린다.

```text
for each kept occupied/boundary voxel v:   # free shell voxel은 이 루프의 대상이 아님
  for face in [+x, -x, +y, -y, +z, -z]:
    neighbor = voxel adjacent to face
    if neighbor가 kept AND occupied:
      exposed_face[face] = false   (occupied 표면끼리 맞닿음)
    else:  # neighbor가 (a) kept free shell 이거나 (b) 프루닝됨
      exposed_face[face] = exposed candidate
      # 실제 free / unknown / uncertain 여부는 아래 규칙으로 구분
```

중요: **프루닝됨 = "fine 30cm 표면 증거 없음"이지 곧 free가 아니다.** 프루닝된 voxel에는 빈 공간뿐 아니라 **물체 내부(표면 뒤 꽉 찬 부분)** 도 섞인다. 따라서 neighbor가 프루닝되면 그 면은 "exposed **candidate**"로 두고, 실제 free / unknown / uncertain 여부는 **부모 0.6m coarse occupancy**로 판정한다. 반면 neighbor가 **kept free shell**이면 이미 fine-level에서 free로 확정된 면이므로 곧바로 exposed-to-free다.

```text
neighbor kept free shell                -> exposed-to-free (fine 확정 노출면)
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

## 10. Queryable Occupancy Interface (occupancy 전용)

Queryable output은 별도 volume head가 아니다. sparse surface feature / shape code와 coarse dense occupancy를 사용해 임의 좌표의 **점유(occupancy)** 를 평가하는 interface다.

**Tesla와의 의도적 차이 (occupancy 전용):** Tesla AI Day 그림의 Queryable Outputs는 MLP가 **두 개** — ① Continuous Occupancy Probability, ② Continuous 3D Semantics — 로 명시적으로 분리되어 있다. 본 시스템은 **임의 좌표의 연속 semantic 질의가 필요 없어** Queryable을 occupancy MLP **하나만** 둔다. semantic은 voxel-level **3D Semantics Head(Section 9.4)** 에서만 제공한다(연속 좌표 질의는 지원하지 않음). 필요해지면 v1.5에서 Tesla처럼 semantic MLP를 추가할 수 있다.

```text
query point x, y, z
-> 30cm voxel index 찾기
-> 이 voxel이 kept(살아남음)?
     yes -> 정밀 경로:
            local coord u + sample F_0.3_sparse + shape_code
            -> small Queryable MLP -> continuous occupancy / uncertainty (semantic 없음)
     no(프루닝됨) -> 부모 0.6m coarse occupancy를 읽어 보수적 판정 (MLP 생략):
            high-confidence free  -> free
            unknown / occluded    -> unknown
            occupied / uncertain  -> consumer mode에 따라 (아래)
```

프루닝된 칸이 occupied/uncertain일 때의 반환은 **소비 모드(consumer mode)** 로 일원화한다(같은 내부 상태를 모드별로 다르게 표현).

```text
planner collision mode:
  conservative occupied      (충돌 안전 -> 막힌 것으로 취급)

map / query probability mode:
  fine-level unknown with high occupancy risk  (확률맵용 -> 높은 점유 위험의 unknown)
```

핵심: **프루닝됨은 곧 free가 아니다.** 프루닝된 칸에는 빈 공간과 물체 내부가 섞여 있으므로, 부모 coarse가 **확실히 free일 때만 free**로 답한다. occupied/uncertain이면 위 모드 규칙을 따른다(절대 free 아님). 이렇게 하면 차량 내부 등 solid interior를 free로 오답하는 충돌 위험을 막는다. 정밀 MLP는 kept voxel(occupied 표면 + near-surface free shell) 근처 query에 실행된다 — free shell voxel 근처 query는 fine-level free로 즉답하므로 좁은 통로 통과 판단이 정밀해진다.

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
(semantic 없음 - voxel-level 3D Semantics Head에서만 제공)
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
  F_0.6_temporal (dense, 100 x 34 x 10 x C)  <- 이것만 사용 (self-contained)
  -> z-flatten: 높이축을 채널로 펼침 -> 100 x 34 x (10*C)
     (ZPool로 누르지 않는다: 바닥 높이를 예측하는 head가
      입력에서 높이 정보를 먼저 버리면 안 되기 때문 - L3)
  -> 1x1 conv로 (10*C) -> C_bev 압축 -> 100 x 34 x C_bev

output (dense BEV, 0.6m 격자 또는 0.3m로 upsample -> 수평 선명도 향상):
  z_surface        : 바닥 높이 (연속 회귀라 0.3m bin에 안 묶임 -> sub-voxel vertical precision 가능; 실제 정밀도는 GT/depth 품질에 의존)
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
    적용 조건: kept 바닥 voxel이 있는 (x,y)에서만 consistency loss를 건다.
      원거리/저텍스처로 바닥이 프루닝된 곳은 비교 대상이 없으므로
      Surface head 단독(z_surface regression)으로만 학습/추론한다.
```

v1에서도 Tesla처럼 별도 head로 두되, 모듈은 가볍게(BEV conv 몇 layer) 시작한다. 3D Semantics에는 의존하지 않는다(Geometry only).

---

## 12. Runtime Masking and Multi-rate Scheduling

Active mask는 별도 prediction head가 아니라, Sub-Voxel Shape Head와 Queryable Interface를 어디에 얼마나 비싸게 실행할지 정하는 runtime priority map이다.

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
  (heavy local query 대상은 occupied 표면 voxel만 센다.
   near-surface free shell voxel(L2)은 여기서 제외 -> K 예산은 표면 기준 유지)

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
  단일 stage @ 0.6m 3D (Tesla/PanoOcc-style, z 유지), align+concat+3D residual conv
  1.2m 3D는 Orin 예산 초과 시 fallback로만
  full 3D temporal memory 금지
  0.6m/0.3m full global temporal attention 금지
  0.3m temporal 금지

Feature:
  1.2m -> 0.6m는 dense (temporal/coarse occupancy가 여기).
  coarse dense occupancy/visibility는 0.6m에서 dense.
  0.6m -> 0.3m만 sparse deconv + 프루닝.
  30cm dense volume feature는 만들지 않는다.
  C는 48 or 64로 시작한다.

Prune (2단 게이트):
  게이트① @0.6m parent (연산량): occupied/boundary parent만 확장.
  게이트② @0.3m child (선명도/anti-dilation): 빈 child 제거, keep ~0.5.
  far-field는 keep 더 높여 원거리 물체 보호 (프루닝 비가역, recall 우선).

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
  Tesla/PanoOcc-style 3D temporal @ 0.6m (align+concat+3D conv, z 유지)
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
8. Tesla/PanoOcc-style single 3D temporal @ 0.6m (z 유지: 3D align + concat + 3D residual conv, NOT attention; streaming memory 선택)
9. Per-voxel 3D flow (vx/vy/vz) -> F_0.6_temporal
10. Coarse dense occupancy / visibility head @ 0.6m (free/unknown, 프루닝 guide)
11. Sparse deconv + 2단 게이트 prune: 0.6m parent gate -> 0.3m child prune
12. Produce F_0.3 sparse surface feature (kept voxels)
13. Occupancy Head (sparse fine 표면 갈래)
14. Occupancy Flow Head (0.3m sparse 출력, motion은 step 8/9에서)
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
0.6m 3D에서 Tesla/PanoOcc-style 시간 정렬(align+concat+3D conv) + per-voxel flow + coarse dense free/unknown을 만들고,
sparse deconv + 프루닝으로 occupied 표면(+ 인접 free 1겹)을 0.3m까지 올린 뒤,
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

### Temporal at 0.6m 3D, Before Final Deconvolution (Tesla/PanoOcc-style)

Tesla 그림은 temporal alignment가 최종 deconvolution 이후 고해상도 feature가 아니라, 그 이전 coarse latent에 붙는 흐름으로 읽힌다. 본 문서는 그 coarse latent를 **0.6m 단일 stage**로 둔다.

```text
0.3m에서 temporal:
  voxel 272,000개 -> Orin 위협, 금지

1.2m에서 temporal:
  Tesla coarsest 원칙엔 맞지만 PanoOcc(~2m coarse)보다도 거칢
  flow motion이 1.2m cell에 평균화됨 -> Orin 예산 초과 시 fallback로만

0.6m 3D에서 temporal (채택):
  z 유지(34,000 voxel, PanoOcc 40k 이내) -> 높이별 motion / vz 가능
  Z=10층 유지(붕괴 없음), 여전히 최종 deconv 이전 latent
  align + concat + 3D conv (attention 아님) -> voxel 수에 선형 -> Orin 가능
```

Tesla/PanoOcc 근거: coarse 3D feature(z 유지)를 trajectory로 정렬(align)해 concat하고 residual 3D conv로 융합해 occupancy/flow를 낸다(attention 아님). Tesla 그림(Spatial Frame Alignment -> Spatiotemporal Features stack)과 PanoOcc temporal encoder(align+fuse)가 같은 계열이다. ViewFormer의 z-squeeze + temporal attention은 높이 motion을 잃고 Tesla와 달라 폐기한다(streaming memory만 학습 효율용으로 선택). 단, raw deconv feature는 날것일 수 있으므로 temporal 직전 0.6m pre-temporal refinement를 둔다(BEVDet4D 교훈).

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

이 분리는 Tesla 데모(표면에만 voxel 존재 = surface-centric)와 맞고, dense 0.3m 비용을 피하면서 4 volume head를 그대로 유지한다. 정밀도가 필요한 표면을 30cm로 풀되, **표면 바로 앞 빈 공간 1겹(near-surface free shell)도 함께 30cm로 살린다**(L2, 기본 ON) — 2m 로봇의 좁은 통로 통과 판단을 위해. 그 외 넓은 빈 공간만 0.6m coarse로 둔다.

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
  Tesla/PanoOcc-style single 3D temporal @ 0.6m 추가 (z 유지)
  ego-motion 3D warp(align) / concat / 3D residual conv (streaming memory 선택)

Stage 3:
  Occupancy Flow Head 추가 (0.6m 3D per-voxel flow vx/vy/vz -> 0.3m 매핑)
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

Temporal module (Tesla/PanoOcc-style, single 3D stage @ 0.6m, z 유지):
  3D voxel memory queue (N=3~4, 과거3+현재1)
  ego-motion 3D alignment(warp) at 0.6m
  concat current + aligned history -> 3D residual conv fuse (attention 금지)
  per-voxel 3D occupancy flow (vx/vy/vz)
  (선택) ViewFormer streaming memory: 학습 효율용
  1.2m 3D fallback: Orin 20Hz 예산 초과 시에만

Coarse dense occupancy:
  occupied / free / unknown / visibility @ 0.6m
  (free/unknown 담당 + 프루닝 guide)

Sparse decoder:
  sparse deconv + 2단 게이트: 0.6m parent gate(연산) + 0.3m child prune(anti-dilation), keep ~0.5

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
  kept voxel 근처: xyz + sampled F_0.3_sparse + shape code + local coordinate
                   -> Queryable MLP -> continuous occupancy / uncertainty (occupancy 전용)
                   (semantic 질의는 없음; voxel-level 3D Semantics Head에서만 제공)
  pruned 영역:     부모 0.6m coarse occupancy로 보수적 판정 (MLP 생략, Section 10)
                   high-conf free->free / unknown->unknown / occupied·uncertain->occupied

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
   - 이 문서에서는 **streaming memory(학습 효율)와 occupancy flow supervision(FlowOcc3D 방식)** 만 차용한다. z-squeeze + temporal attention은 Tesla(align+concat, z 유지)와 달라 폐기하고, temporal 본체는 PanoOcc식 3D(아래)를 따른다

3. [PanoOcc](https://github.com/Robertwyq/PanoOcc) ([paper](https://arxiv.org/abs/2306.10013)) — 이 문서의 **temporal + sparse decoder 양쪽 핵심 레퍼런스**
   - (temporal) temporal encoder: coarse 3D voxel을 ego-pose로 align -> concat -> residual 3D conv로 fuse (z 유지, 과거3+현재1=4 frame, coarse 50x50x16=40,000 voxel). Tesla 그림(Spatial Frame Alignment->Spatiotemporal stack)과 동일 계열
   - (sparse) coarse-to-fine **sparse deconvolution + occupancy sparsification(프루닝)**, 3단계 keep ratio 0.2/0.5/0.5로 점유 후보만 남겨 dense 고해상 3D conv 회피
   - 이 문서 반영: **0.6m 3D 단일 temporal stage(z 유지, per-voxel flow)** + 1.2m->0.6m->0.3m sparse decoder + 프루닝 (keep ratio는 원거리 보호 위해 더 보수적으로)

4. BEVFormer / BEVDet4D
   - BEV feature ego-motion alignment와 temporal fusion 위치에 대한 교훈 참고
   - 이 문서에서는 pre-temporal refinement와 deconv 전 temporal alignment에 반영

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
2. PanoOcc (temporal align+concat+3D conv + sparse decoder; 본 문서 temporal/sparse 핵심)
3. BEVDet4D / BEVFormer temporal fusion sections
4. REO (vanilla attention 종류)
5. ViewFormer-Occ (streaming memory + flow supervision만 차용)
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
  -> 0.6m per-voxel 3D flow GT (vx/vy/vz, FlowOcc3D 방식), 0.3m voxel로 매핑

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
  Section 7 temporal 3D voxel memory의 ego-motion warp 입력

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
