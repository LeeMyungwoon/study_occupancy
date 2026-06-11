# Tesla-style Queryable Occupancy Network Architecture

이 문서는 Tesla Occupancy Network 공개 발표에서 보이는 방향성을 참고하되, 실제 구현을 위해 REO, ViewFormer, PanoOcc, SparseOcc, GaussianFormer 계열 아이디어를 섞은 아키텍처 초안이다.

이 구조는 여러 논문/레포의 장점을 다음처럼 가져온 하이브리드 설계다.

| Source | 가져오는 핵심 아이디어 | 이 문서에서 쓰는 위치 |
|---|---|---|
| REO | calibration-free / calibration-light vanilla cross-attention, query-based fine prediction | canonical-view spatial lifting |
| ViewFormer | streaming temporal BEV memory, ego-motion alignment, temporal attention | 1.6m 3D feature 직후 temporal module |
| PanoOcc | coarse-to-fine 3D deconvolution, unified occupancy representation | 1.6m -> 0.8m -> 0.4m dense decoder |
| SparseOcc / sparse conv 계열 | 중요한 영역만 sparse하게 refine | active mask / quota Top-K 기반 refinement 영역 선택 |
| GaussianFormer / GaussianFormer-2 | object-centric 3D Gaussian, deformable image re-query, probabilistic Gaussian superposition | budget-K Gaussian refinement branch, probabilistic queryable evaluation |
| RoadBEV / FastRSR | BEV에서 road surface elevation을 직접 예측 | road surface geometry head |

특히 spatial lifting은 **vanilla cross-attention**을 기본으로 한다. 즉 projection-first deformable attention을 기본 구조로 두지 않고, 1.6m 3D voxel query가 multi-camera image tokens를 `Q/K/V` attention으로 읽는 방식이다.

예외는 budget-K Gaussian refinement branch 하나다. 이 branch는 선택된 소수의 refinement Gaussian이 자기 위치를 canonical virtual camera에 재투영해 image feature를 다시 읽는 deformable re-query를 사용한다. projection 대상이 하드웨어와 무관한 virtual camera(설계 상수)이므로, raw calibration에 의존하지 않는다는 calibration-light 원칙은 유지된다.

Reference visual style: [Screenshot from 2026-06-05 22-38-02.png](<Screenshot from 2026-06-05 22-38-02.png>)

## 0. Target Space and Whole Architecture

이 아키텍처가 고려하는 3D 공간은 ego vehicle / robot 기준의 전방 중심 local occupancy volume이다.

```text
X range:
  rear 10m ~ front 50m
  total 60m

Y range:
  left 10m ~ right 10m
  total 20m

Z range:
  -1m ~ +4m
  total 5m
  (z축은 차체가 아니라 중력에 정렬, Section 20)

Final voxel size:
  0.2m x 0.2m x 0.2m

Final dense occupancy output:
  300 x 100 x 25
  = 750,000 voxels
```

가장 중요한 설계 구분:

```text
20cm dense occupancy output:
  만든다.

20cm full dense feature volume:
  만들지 않는다.

Budget-K refinement Gaussians:
  active mask가 고른 중요한 voxel에서만 초기화한다.
  deformable image re-query로 파라미터를 refine하고,
  probabilistic queryable evaluation의 연속 local 표현으로 사용한다.

Road surface outputs:
  20cm BEV grid에서 Road Surface Geometry Head로 만든다.
  volume occupancy의 바닥 slice로 대체하지 않는다.
  v1 core에서는 z_surface / valid / uncertainty에 집중한다.
```

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
| Spatial Attention at 1.6m     |
| REO-style vanilla attention   |
| Q: 1.6m 3D voxel queries      |
| K,V: image tokens             |
+---------------+---------------+
                |
                v
+-------------------------------+
| 1.6m Coarse 3D Feature        |
| 38 x 13 x 4 x C               |
+---------------+---------------+
                |
                v
+-------------------------------+       +--------------------------+
| ViewFormer-style Temporal     |<----->| Streaming BEV Memory     |
| z-squeeze to BEV              |       | ego-motion aligned       |
| temporal attention            |       | N history frames         |
| inject back to 3D feature     |       +--------------------------+
+---------------+---------------+
                |
                v
+-------------------------------+
| Dense Deconvolutions          |
| 1.6m -> 0.8m                  |
| 0.8m -> 0.4m                  |
+---------------+---------------+
                |
                v
+-------------------------------+
| 0.4m Dense 3D Feature         |
| valid: 150 x 50 x 13 x C      |
+-------+-----------+-----------+
        |           |
        |           +------------------------------+
        |                                          |
        v                                          v
+-----------------------+              +--------------------------+
| Surface Outputs       |              | Structured Sub-voxel Head|
| road surface geometry |              | 1x1x1 conv on 0.4m feat  |
| z_surface / valid     |              | 2x2x2 logits per cell    |
| uncertainty           |              | reshape / crop           |
| derived slope / step  |              |                          |
+-----------------------+              +------------+-------------+
                                                    |
                                                    v
                                       +--------------------------+
                                       | Volume Outputs           |
                                       | 300 x 100 x 25           |
                                       | occupancy / free         |
                                       | unknown                  |
                                       | dynamic / flow           |
                                       +------------+-------------+
                                                    |
                                                    v
                                       +--------------------------+
                                       | Active Mask Selection    |
                                       | surface / boundary       |
                                       | uncertainty / dynamic    |
                                       | planner path / visibility|
                                       +------------+-------------+
                                                    |
                                                    v
                                       +--------------------------+
                                       | Budget-K Gaussian Refine |
                                       | init from selected voxels|
                                       | deformable image re-query|
                                       | 1~2 rounds / K hard cap  |
                                       +------------+-------------+
                                                    |
                                                    v
                                       +--------------------------+
                                       | Queryable Outputs        |
                                       | x,y,z -> probabilistic   |
                                       |   Gaussian superposition |
                                       | continuous occupancy     |
                                       | adaptive surface overlay |
                                       +--------------------------+
```

축약하면:

```text
Canonical-view REO-style spatial attention
+ ViewFormer-style temporal BEV memory
+ PanoOcc-style 2-stage deconv to 0.4m
+ RoadBEV/FastRSR-style road surface geometry
+ flow occupancy output
+ structured sub-voxel 20cm dense occupancy head
+ dynamic-resolution refinement
+ GaussianFormer-style budget-K refinement Gaussians
+ GaussianFormer-2-style probabilistic queryable evaluation
```

핵심 목표는 다음과 같다.

- 최종 출력은 20cm 정사각 voxel 기반 dense occupancy grid로 만든다.
- 하지만 20cm 해상도의 dense feature volume은 만들지 않는다.
- spatial attention은 coarse grid에서 수행해 비용을 낮춘다.
- temporal context는 ViewFormer처럼 BEV memory에서 streaming으로 처리한다.
- PanoOcc처럼 coarse-to-fine deconvolution을 사용하되, 0.4m feature까지만 dense하게 키운다.
- 0.2m final occupancy는 structured sub-voxel channel head로 예측한다.
- 중요한 영역은 budget-K refinement Gaussian으로 표현하고, deformable image re-query로 이미지에서 새 정보를 받아 refine한다.
- 임의 좌표 query는 GaussianFormer-2-style probabilistic superposition으로 평가한다.
- road surface output은 volume occupancy와 별도 head로 예측한다.
- road surface geometry는 `z_ground / z_surface`, valid, uncertainty를 핵심 출력으로 둔다.
- slope, normal, step, roughness는 처음에는 `z_surface`에서 유도하고, 별도 head로 만들지 않는다.
- semantic surface, traversability, volume semantic은 v1 core head에서 제외하고 나중에 필요한 경우 확장한다.

---

## 1. Prediction Range and Output Grid

예측 영역:

```text
X axis: rear 10m ~ front 50m = 60m
Y axis: left 10m ~ right 10m = 20m
Z axis: -1m ~ +4m = 5m
```

Z 하한이 0이 아니라 -1m인 이유:

```text
내리막 경사 / 경사로 하행 / 연석 아래 / 도로 침하처럼
ego 기준면보다 낮은 표면을 표현해야 한다.
차체 pitch만으로도 원거리 바닥이 z=0 아래로 내려간다.
grid의 z축은 차체가 아니라 VIO의 중력 추정에 정렬한다 (Section 20).
총 높이는 5m로 동일하므로 voxel 수와 비용은 변하지 않는다.
```

최종 voxel 크기:

```text
voxel_size = 0.2m
```

최종 dense occupancy output 크기:

```text
X: 60m / 0.2m = 300
Y: 20m / 0.2m = 100
Z:  5m / 0.2m = 25

final_output = 300 x 100 x 25 = 750,000 voxels
```

중요한 구분:

```text
20cm dense output은 만든다.
20cm dense feature volume은 만들지 않는다.
```

즉 최종적으로 `300 x 100 x 25` occupancy probability는 출력하지만, 내부에서 `300 x 100 x 25 x C` 형태의 큰 3D feature tensor를 오래 유지하지 않는다.

---

## 2. Overall Architecture

추천 전체 구조:

```text
multi-camera images
-> canonical-view preprocessing
-> image backbone + neck
-> canonical ray positional embedding
-> calibration-light vanilla spatial attention at 1.6m
-> 1.6m coarse 3D feature

-> ViewFormer-style streaming temporal module
-> temporally enhanced 1.6m 3D feature

-> dense deconv 1
-> 0.8m 3D feature

-> dense deconv 2
-> 0.4m 3D feature

-> road surface geometry head
-> 300 x 100 z_surface / valid / uncertainty output

-> structured sub-voxel channel head
-> 300 x 100 x 25 dense occupancy output

-> dynamic / flow head
-> dynamic probability / occupancy flow output

-> active mask selection
-> budget-K Gaussian refinement branch
-> selected voxels initialize refinement Gaussians
-> deformable image re-query refines Gaussian parameters
-> probabilistic queryable evaluation inside selected regions
```

해상도별 grid 크기:

```text
0.2m: 300 x 100 x 25 = 750,000 voxels
0.4m: 150 x  50 x 13 =  97,500 voxels
0.8m:  75 x  25 x  7 =  13,125 voxels
1.6m:  38 x  13 x  4 =   1,976 voxels
```

주의: 이 크기는 실제 예측 영역을 덮기 위한 logical grid 크기다. stride-2 deconv를 깔끔하게 쓰려면 내부 tensor는 padding된 크기로 만들고, 마지막에 valid region만 crop/mask하는 편이 구현이 쉽다.

예:

```text
1.6m internal: 38 x 13 x 4
0.8m internal: 76 x 26 x 8
0.4m internal: 152 x 52 x 16

valid 0.4m region:
150 x 50 x 13
```

최종 20cm query도 valid region 안의 `300 x 100 x 25` voxel center에 대해서만 수행한다.

`1.6m`에서 spatial attention을 수행하면 `0.8m`에서 attention을 수행하는 것보다 query 수가 약 6배 이상 줄어든다. 대신 deconvolution을 두 번 사용해 `0.4m` feature까지 복원한다.

---

## 2.1 Core Heads and Branches

이 문서의 v1 core는 너무 많은 head를 두지 않는다. 핵심은 `동적 해상도 복셀 + flow occupancy + 바닥 surface`다.

따라서 core head / branch는 다음 6개로 본다.

| Head / Branch | 출력 크기 | 역할 |
|---|---:|---|
| Volume Occupancy Head | `300 x 100 x 25` | 각 20cm voxel의 occupied / free / unknown 확률 |
| Dynamic / Flow Head | `300 x 100 x 25` 또는 active voxel | dynamic probability, occupancy flow, velocity |
| Road Surface Geometry Head | `300 x 100` | BEV cell마다 바닥/표면의 `z_surface`, valid, uncertainty |
| Active Mask Head | `0.4m cell` 또는 `20cm voxel` | 어떤 영역을 더 자세히 볼지 선택 |
| Budget-K Gaussian Refinement Branch | selected `K` Gaussians | 선택된 voxel에서 Gaussian 초기화, deformable image re-query로 파라미터 refine |
| Probabilistic Queryable Evaluation | query points in selected regions | Gaussian superposition으로 임의 좌표의 continuous occupancy / boundary / normal 평가 |

구조적으로는 다음처럼 묶는다.

```text
Core outputs:
  1. Volume Occupancy
  2. Occupancy Flow
  3. Road / Ground Surface Geometry

Dynamic-resolution refinement:
  4. Active Mask
  5. Budget-K Gaussian Refinement
  6. Probabilistic Queryable Evaluation

Runtime rules inside dynamic-resolution refinement:
  fixed K quota scheduler
  distance / planner-aware LOD budget
  packed batched Gaussian evaluation
```

이 중 `Active Mask + Budget-K Gaussian Refinement + Probabilistic Queryable Evaluation`은 따로따로 독립된 최종 prediction task라기보다, 동적 해상도 복셀을 만들기 위한 하나의 refinement pipeline이다.

```text
20cm base occupancy
-> active mask로 refine score 계산
-> quota-based Top-K scheduler로 K개 voxel 선택
-> 선택된 voxel center에서 refinement Gaussian 초기화
-> deformable image re-query로 Gaussian 파라미터 refine
-> 모든 query point는 packed batch로 superposition을 한 번에 평가
```

실시간성을 위해 다음 세 가지 runtime rule을 core에 포함한다.

```text
1. fixed K budget:
   매 frame refine할 voxel 수를 제한한다.

2. quota-based selection:
   boundary / dynamic / planner / uncertainty / near-field가 한쪽에 밀리지 않게 나눠 뽑는다.

3. packed batched query:
   Gaussian마다 평가를 따로 호출하지 않고, 모든 query point를 Q x 3 batch로 펴서 superposition을 한 번에 평가한다.
```

v1에서 core head로 두지 않는 것:

```text
Volume Semantic Head:
  나중에 semantic occupancy가 필요할 때 추가

Road Surface Semantics / Traversability:
  나중에 road, curb, ramp, stair, terrain, cost가 필요할 때 추가

Surface Normal / Slope / Step Head:
  처음에는 z_surface에서 계산
  필요할 때만 auxiliary head로 분리
```

---

## 3. Canonical-view REO-style Input

REO의 calibration-free vanilla attention 아이디어를 그대로 쓰되, 실제 하드웨어 변경 유연성을 위해 입력을 canonical view로 정렬한다.

정확한 이름은 순수 `calibration-free`보다 다음에 가깝다.

```text
calibration-light canonical-view REO-style attention
```

입력 전처리:

```text
raw camera image
-> undistortion
-> rectification
-> virtual camera layout으로 정렬
-> fixed crop / resize
-> canonical ray grid 생성
```

목적:

- 카메라 하드웨어가 바뀌어도 downstream network는 같은 virtual camera layout을 보게 한다.
- 실제 intrinsic/extrinsic 변화는 preprocessing adapter에서 최대한 흡수한다.
- attention 모듈은 매번 다른 raw camera geometry를 직접 학습하지 않아도 된다.

주의:

depth 없이 이미지를 완전히 다른 physical viewpoint로 재투영할 수는 없다. 여기서 말하는 canonical view는 완전한 3D reprojection이 아니라, 렌즈/화각/방향을 정규화한 ray-aligned image representation에 가깝다.

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

vanilla cross-attention은 projection 힌트 없이 3D 대응을 데이터로만 학습하므로, 수렴이 느리고 데이터 요구량이 크다. 이를 완화하기 위해 학습 시에만 image feature 위에 작은 per-pixel depth head를 둔다 (BEVDepth 계열의 교훈).

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

효과: depth를 맞추려면 image feature가 "이 픽셀이 얼마나 먼 표면인지"를 인코딩해야 하고, 그 feature는 cross-attention의 3D 대응 학습을 직접 돕는다. 추론 비용이 0이므로 v1 학습에 기본 포함한다.

---

## 5. Coarse Spatial Attention at 1.6m

spatial attention은 20cm나 40cm가 아니라 `1.6m` coarse grid에서 수행한다.

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
coarse_query_grid = 38 x 13 x 4
num_queries ~= 1,976
```

기본 형태:

```text
Q: 1.6m 3D voxel query positional embedding
K: multi-camera image tokens + canonical ray positional embedding
V: multi-camera image features

coarse_3d_feature = VanillaCrossAttention(Q, K, V)
```

이 방식은 REO의 vanilla cross-attention 아이디어를 3D coarse query에 적용한 형태다. REO 원 논문은 BEV query를 중심으로 설명하지만, 여기서는 Tesla-style occupancy를 위해 low-resolution 3D voxel query를 사용한다.

장점:

- `0.8m` attention보다 query 수가 훨씬 적다.
- calibration projection에 과하게 의존하지 않는다.
- canonical ray positional embedding 덕분에 완전 무기하 attention보다 수렴이 안정적이다.

위험:

- 너무 coarse한 query는 얇은 물체나 작은 장애물 정보를 놓칠 수 있다.
- 그래서 `3.2m` attention까지 낮추는 것은 초기 버전에서는 피한다.

권장 시작점:

```text
1.6m attention
-> ViewFormer-style temporal module
-> 0.8m deconv
-> 0.4m deconv
-> 20cm query
```

---

## 6. ViewFormer-style Streaming Temporal Module

temporal module은 ViewFormer의 핵심 아이디어를 접목한다.

핵심은 다음과 같다.

```text
3D voxel feature를 그대로 과거 프레임 memory에 저장하지 않는다.
z축을 squeeze해서 BEV memory로 저장한다.
ego-motion으로 과거 BEV feature를 현재 좌표계에 align한다.
aligned memory와 현재 BEV feature 사이에 temporal attention을 수행한다.
temporal BEV context를 다시 현재 3D feature에 주입한다.
```

현재 구조에서는 temporal module을 `1.6m coarse 3D feature` 직후에 둔다.

```text
current 1.6m 3D feature
-> z-squeeze / height pooling
-> current 1.6m BEV feature
-> ego-motion aligned BEV memory
-> streaming temporal attention
-> temporal BEV context
-> restore / inject into 1.6m 3D feature
-> temporally enhanced 1.6m 3D feature
```

권장 tensor 흐름:

```text
F3D_t:
  38 x 13 x 4 x C

B_t = ZPool(F3D_t):
  38 x 13 x Cb

B08_t = Upsample2x(B_t):
  75 x 25 x Cb
  저장 / warp 해상도

Memory:
  [B08_{t-1}, B08_{t-2}, B08_{t-3}, ...]
  keyframe 간격 ~0.2s (연속 frame이 아니라 시간 간격 기반)

Aligned memory:
  Warp(B08_{t-k}, pose_{t-k -> t})
  0.8m grid에서 수행

B_temporal:
  TemporalAttention(query=B08_t, key/value=aligned_memory)

F3D_temporal:
  F3D_t + Inject3D(AvgPool2x(B_temporal), z_embedding)
```

`ZPool`은 단순 mean/max pooling으로 시작할 수 있지만, 최종적으로는 learned height pooling이 더 좋다.

```text
ZPool options:
  mean pooling over z
  max pooling over z
  learned weighted pooling over z
  attention pooling over z
```

`Inject3D`는 temporal BEV context를 z축으로 broadcast한 뒤, z positional embedding과 함께 현재 3D feature에 residual/gating 방식으로 더한다.

```text
Inject3D(B_temporal, z_embedding):
  broadcast B_temporal to X x Y x Z
  concatenate or add z_embedding
  1x1x1 conv / MLP
  residual or FiLM-style gating
```

추천 memory 길이:

```text
N_history = 3 or 4 keyframes
keyframe 간격 ~0.2s

20 FPS 연속 4 frame은 0.15~0.2s 창이라
occlusion / 가려진 물체 추론에 너무 짧다.
시간 간격 기반으로 띄엄띄엄 저장해 0.6~0.8s 창을 확보한다.
```

memory는 0.4m feature도, 1.6m BEV도 아닌 **0.8m BEV feature**로 저장하고 warp한다.

이유:

- 3D feature memory보다 훨씬 작다. (0.8m BEV 4 keyframes, C=64 fp16 기준 약 1MB)
- ego-motion alignment를 BEV plane에서 간단하게 처리할 수 있다.
- ViewFormer처럼 temporal interaction 비용을 낮출 수 있다.
- 1.6m로 저장하면 warp가 망가진다: 10m/s 주행 시 frame당 이동 0.5m는 1.6m cell의 1/3이라, sub-cell 이동이 보간으로 매 frame 번져 기억이 흐려진다. 0.8m는 같은 이동을 절반의 상대 오차로 정렬한다.

주의:

- BEV로 z축을 squeeze하면 height 정보 일부가 손실될 수 있다.
- 따라서 temporal context는 3D feature를 대체하지 않고, 현재 1.6m 3D feature에 보조 context로 주입한다.
- 높이가 중요한 물체나 경사로 표현은 현재 frame의 3D feature와 z embedding에 맡긴다.

첫 프레임 또는 memory가 부족한 경우:

```text
if memory is empty:
  B_temporal = B08_t
else:
  B_temporal = TemporalAttention(B08_t, aligned_memory)
```

학습 시에는 temporal robustness를 위해 다음 augmentation을 사용한다.

```text
history frame dropout
pose noise
time gap jitter
memory detach / truncated BPTT
```

추가 head:

```text
dynamic probability
occupancy flow / velocity
```

temporal module을 넣으면 final occupancy뿐 아니라 dynamic object, motion consistency, occupancy flow를 같이 학습하기 쉬워진다.

---

## 7. Coarse-to-fine Dense Deconvolution

PanoOcc-style coarse-to-fine decoder를 사용하되, dense deconv는 `0.4m`까지만 수행한다.

```text
1.6m coarse 3D feature
-> deconv block 1
-> 0.8m 3D feature

-> deconv block 2
-> 0.4m 3D feature
```

구현상 deconv 출력 크기는 stride-2에 맞춘 padded internal grid로 잡고, 예측 범위를 벗어나는 padding voxel은 valid mask로 제외한다.

```text
deconv internal:
38 x 13 x 4
-> 76 x 26 x 8
-> 152 x 52 x 16

valid feature:
150 x 50 x 13
```

비추천:

```text
1.6m
-> 0.8m
-> 0.4m
-> 0.2m dense feature
```

`0.2m` dense feature volume은 너무 크다.

```text
300 x 100 x 25 x C
```

예를 들어 fp16, 64 channels라면 feature tensor 하나만 약 96MB가 된다. 여기에 deconv 중간 activation, normalization, residual, training gradient가 붙으면 메모리와 latency가 크게 증가한다.

따라서 dense feature는 여기서 멈춘다.

```text
0.4m dense feature = 150 x 50 x 13 x C
```

---

## 8. Road Surface Geometry Output

Tesla 발표 그림의 `Road Surface Geometry`에 해당하는 출력은 3D volume occupancy와 별도 head로 둔다.

중요한 전제:

```text
road surface output은 3D occupancy의 특정 z slice가 아니다.
0.4m dense 3D feature에서 별도 surface BEV feature를 만들고,
그 feature에서 20cm BEV surface geometry를 예측한다.
```

권장 feature 흐름:

```text
0.4m dense 3D feature
  150 x 50 x 13 x C

-> learned z-pooling / surface query pooling
-> surface BEV feature
  150 x 50 x Cs

-> BEV upsample or 20cm surface query
-> 20cm surface BEV feature
  300 x 100 x Cs

-> road surface geometry head
```

`z-pooling`은 단순 mean/max pooling으로 시작할 수 있지만, 최종적으로는 learned pooling 또는 surface query pooling이 더 좋다. 표면은 보통 3D volume 전체가 아니라 특정 높이 근처에 있으므로, z축 전체를 동일하게 압축하면 바닥 단서가 흐려질 수 있다.

geometry head는 BEV cell마다 주행/이동 가능한 주요 표면의 높이와 형상을 예측한다.

v1 core 출력:

```text
z_ground / z_surface:
  각 (x, y) 위치의 대표 바닥/표면 높이

valid / unknown:
  해당 위치에 신뢰할 수 있는 표면이 있는지

uncertainty:
  카메라 가림, 원거리, texture 부족, multi-layer 구조에 대한 불확실성
```

derived output:

```text
normal / slope:
  표면 법선과 기울기
  z_surface의 spatial gradient에서 유도

step height / height discontinuity:
  curb, speed bump, pothole, stair처럼 높이 변화가 큰 위치
  z_surface gradient peak에서 후보 생성

roughness:
  표면 요철 정도
  local height variance에서 유도
```

따라서 초기 구현에서 실제 head가 직접 예측하는 것은 다음 세 개다.

```text
z_surface(x, y)
valid_or_unknown(x, y)
surface_uncertainty(x, y)
```

normal, slope, roughness는 먼저 `z_surface`에서 미분해 만든다.

```text
dz/dx, dz/dy -> normal / slope
local height variance -> roughness
height gradient peak -> step / curb 후보
```

이후 성능이 부족하면 normal, slope, step을 별도 auxiliary head로 분리할 수 있다. 하지만 v1 core에서는 별도 head로 두지 않는다.

### Optional Surface Semantics

surface semantics와 traversability는 지금 core head가 아니다. Road Surface Geometry Head가 안정된 뒤, 필요하면 다음 class/cost를 optional extension으로 붙인다.

```text
surface class:
  road / lane marking / curb / sidewalk / ramp / stair / terrain / unknown

traversability / cost:
  platform-conditioned moving feasibility
```

중요한 점은 `surface class`와 `traversability`를 분리하는 것이다. 예를 들어 `stair`는 같은 class라도 차량에는 이동 불가, 족형 로봇에는 조건부 이동 가능일 수 있다.

### Stair / Ramp / Curb Handling

계단 같은 구조도 표현 가능하다. 다만 v1에서는 `stair semantic`을 직접 분류하기보다, surface geometry와 volume occupancy, adaptive query가 계단 구조를 드러내도록 한다.

v1에서 보는 방식:

```text
surface geometry:
  tread 후보의 z_surface
  step / edge 후보는 z_surface gradient로 유도

volume occupancy:
  riser / vertical face / obstacle boundary 표현

budget-K Gaussian refinement:
  선택된 step / boundary voxel을 refinement Gaussian으로 더 세밀하게 표현
```

나중에 semantics/traversability를 추가하면 다음처럼 확장할 수 있다.

```text
semantic class:
  stair

surface role:
  tread / riser / boundary

traversability:
  platform-conditioned probability or cost
```

플랫폼별 해석:

```text
passenger vehicle:
  stair -> non-traversable

wheeled robot:
  stair -> mostly non-traversable
  ramp -> traversable 가능

legged robot:
  stair tread -> conditionally traversable
  stair riser -> obstacle / boundary
  cost -> high
```

즉 "계단도 이동 가능하다"는 절대 semantic이 아니라 플랫폼 조건부 affordance다.

height-map 형태의 surface geometry는 기본적으로 한 `(x, y)` 위치에 하나의 대표 surface 높이를 예측한다. 이 방식은 도로, 경사로, 연석, 계단의 tread에는 잘 맞지만, overpass나 다층 구조처럼 한 `(x, y)`에 여러 surface가 있는 장면에는 한계가 있다.

초기 버전의 처리:

```text
one primary surface per (x, y)
multi-layer / ambiguous region -> unknown or high uncertainty
vertical riser / wall-like face -> volume occupancy + budget-K Gaussian refinement에서 처리
```

나중에 필요하면 multi-surface head로 확장한다.

```text
z_surface_1, z_surface_2, ...
surface_valid_1, surface_valid_2, ...
```

하지만 첫 구현에서는 하나의 대표 `z_surface`와 uncertainty를 두는 편이 낫다.

### Consistency with Volume Occupancy

surface output과 volume output은 서로 독립적으로 아무렇게나 예측되면 안 된다. 다음 consistency가 필요하다.

```text
surface z 주변:
  occupied / boundary probability가 높아야 함

surface 위쪽:
  free 또는 unknown이어야 함

curb / stair / step 위치:
  height discontinuity와 occupancy boundary가 같이 나타나야 함

drivable surface:
  volume occupancy에서 obstacle과 충돌하면 안 됨
```

권장 loss:

```text
L_surface_height:
  Huber or L1 on z_surface with valid mask

L_surface_valid:
  BCE or focal loss

L_surface_semantic:
  CE or focal loss
  optional extension에서만 사용

L_surface_normal:
  1 - dot(normal_pred, normal_gt)
  or derived-normal consistency from z_surface

L_surface_smooth:
  edge-aware smoothness
  semantic boundary와 step에서는 smoothness 약화

L_occ_surface_consistency:
  surface height와 occupancy boundary 정합
```

이 surface branch의 목적은 planner/control이 바로 쓸 수 있는 `2.5D ground/surface geometry`를 만드는 것이다. 3D occupancy는 전체 장애물과 빈 공간을 말해주고, road surface geometry head는 "바닥/표면이 어디에 있고 얼마나 믿을 수 있는지"를 더 직접적으로 알려준다.

---

## 9. 20cm Dense Output Head

최종 20cm occupancy는 20cm dense feature volume을 만들어서 예측하지 않는다. 하지만 20 FPS v1에서는 모든 20cm voxel을 `GridSample + MLP`로 하나씩 query하는 방식도 피한다.

20 FPS v1의 기본 dense output은 structured sub-voxel channel head로 만든다.

```text
0.4m dense feature:
  150 x 50 x 13 x C

-> small 1x1x1 conv / MLP head
-> each 0.4m cell predicts 2 x 2 x 2 sub-voxel logits
-> reshape / crop
-> 300 x 100 x 25 dense occupancy
```

이 방식은 20cm dense feature를 만들지 않으면서도, 모든 20cm voxel logit을 빠르게 만든다.

```text
0.4m feature tensor:
  유지한다.

20cm dense feature tensor:
  만들지 않는다.

20cm dense occupancy logits:
  structured channel output으로 만든다.
```

임의 좌표 query와 selected voxel 내부의 정밀화는 dense 경로가 아니라 budget-K Gaussian refinement branch가 담당한다 (Section 10, 11).

```text
selected x, y, z points
-> 주변 refinement Gaussian 탐색
-> GaussianFormer-2-style probabilistic superposition 평가
-> occupancy / unknown / local surface-boundary probability
```

이 구조는 Tesla demo의 queryable occupancy 성격과 잘 맞는다.

핵심:

```text
0.4m까지는 dense feature
0.2m는 dense output logits
0.2m dense feature는 만들지 않음
all-voxel dense output은 structured head
selected fine refinement는 budget-K Gaussian branch
```

---

## 10. Budget-K Gaussian Refinement Branch

선택된 영역의 fine refinement는 GaussianFormer / GaussianFormer-2 아이디어를 budget-bounded 형태로 가져온 branch로 수행한다. 이전 초안의 `sparse local feature anchor + exposed-face adaptive point query` 조합은 이 branch로 대체한다.

대체 이유:

```text
1. 정보 병목 해소:
   기존 refinement는 0.4m feature를 다시 sampling하는 구조라
   1.6m lifting 이후 이미지에서 새 정보가 들어오지 않았다.
   Gaussian의 deformable image re-query는 선택 영역만
   이미지 feature를 다시 읽는 budget-bounded local re-query다.

2. latency 구조 단순화:
   probe -> boundary search의 순차 query 루프가
   single feed-forward refinement로 바뀐다.

3. analytic surface 표현:
   covariance에서 surface normal과 local 두께가 직접 나온다.
   anisotropic scale이 20cm grid보다 얇은 구조를 표현할 수 있다.

4. queryable 표현 일치:
   probabilistic superposition은 임의 좌표 x에서
   p(occupied | x)를 closed-form으로 평가한다.
   Tesla demo의 queryable occupancy 인터페이스와 같다.
```

GaussianFormer 원본을 그대로 쓰지 않는다. 원본은 desktop GPU에서도 수백 ms급이다 (GaussianFormer-2 기준 RTX 4090에서 6,400 Gaussians에 약 313ms, 그중 이미지 기반 초기화 모듈만 약 142ms). 따라서 다음과 같이 비용 요소를 제거한 형태로만 가져온다.

```text
가져오지 않는 것:
  이미지에서 Gaussian을 직접 생성하는 초기화 모듈
  전체 장면의 Gaussian 표현 (수만 개 단위)
  full-scene Gaussian-to-voxel splatting CUDA op

가져오는 것:
  Gaussian 파라미터화: mean / scale / rotation / opacity / feature
  deformable image re-query 기반 파라미터 refinement
  GaussianFormer-2-style probabilistic superposition 평가
```

비용 제거가 가능한 이유는 base 경로가 이미 있기 때문이다. base dense occupancy는 structured head가 만들고, active mask + quota Top-K가 "어디에 Gaussian이 필요한지"를 이미 알려주므로, Gaussian은 이미지 기반 초기화 모듈 없이 selected voxel에서 바로 시작한다.

### Gaussian Initialization

```text
input:
  quota-based Top-K가 선택한 voxel index
  K_total hard cap (2048 권장)

per-Gaussian parameters:
  mean:     voxel center + learnable offset
  scale:    0.2m 근방 isotropic 초기화
  rotation: identity quaternion
  opacity:  base occupancy logit에서 초기화
  feature:  0.4m feature trilinear sample
```

### Deformable Image Re-query

각 Gaussian이 자기 위치를 canonical virtual camera에 재투영해 image feature를 다시 읽고 파라미터를 갱신한다. 이 단계가 이 branch의 존재 이유다. 1.6m lifting 이후 처음으로 이미지의 새 정보가 선택 영역에 유입된다.

```text
Gaussian mean (+ scale 방향 offset points)
-> canonical virtual camera로 projection
-> P3 feature level에서 deformable sampling
-> sampled image feature + Gaussian feature
-> small MLP
-> 파라미터 잔차 갱신:
   mean offset / scale / rotation / opacity / feature
```

권장 budget:

```text
rounds:
  1 round로 시작, 최대 2 rounds

sampling points:
  Gaussian당 4~8 points

feature level:
  P3 중심, P2 추가는 v1.5 이후
```

주의:

```text
projection 대상은 canonical virtual camera다.
virtual camera는 하드웨어와 무관한 설계 상수이므로,
raw calibration에 의존하지 않는다는
calibration-light 원칙은 유지된다.
```

이 re-query는 이전 초안에서 v1.5로 미뤘던 `P2/P3 local image re-query`를 budget-bounded 형태로 기본 경로에 편입한 것이다. Orin latency 측정 결과 부담이 크면 re-query round를 0으로 두는 fallback(0.4m feature 기반 refine만)으로 내릴 수 있다.

### Query Budget

refinement 영역마다 queryable evaluation point 수 `M_i`는 다르게 둔다.

```text
M_i = f(
  uncertainty_score,
  boundary_score,
  planner_relevance,
  dynamic_score,
  visibility_score
)
```

거리/플래너 LOD도 `M_i`에 반영한다.

```text
near + planner path + dynamic:
  larger M_i

far + static + low uncertainty:
  smaller M_i

certain free / occupied interior:
  M_i = 0
```

### Derived Properties

```text
surface normal:
  covariance의 최소 고유값 방향에서 analytic하게 유도
  road surface geometry head의 derived normal과 cross-check 가능

thin object:
  anisotropic scale이 20cm grid보다 얇은 구조를 직접 표현

boundary:
  기존 probe -> sign crossing -> refine의 순차 boundary search 불필요
  Gaussian 파라미터 자체가 boundary 위치/방향을 표현

dynamic object:
  flow head와 결합해 Gaussian velocity로 확장 가능 (v2)
```

기존 exposed-face mask는 핵심 메커니즘에서 제외한다. 필요하면 Section 11의 query point 배치 prior로만 사용한다.

### Supervision

```text
voxel GT만 있는 경우:
  superposition 평가 결과를 20cm voxel GT로 supervise
  sub-voxel 신호는 없음 (GT 해상도의 한계)

연속 GT가 있는 경우:
  aggregated LiDAR / mesh / simulation 또는
  Section 19 pipeline의 연속 SDF에서 만든
  continuous point GT로 직접 supervise

옵션:
  Gaussian rendering 기반 self-supervision
  (GaussTR / GaussianOcc 계열)
  voxel GT 없이 이미지에서 sub-voxel 신호를 얻는 경로
```

---

## 11. Probabilistic Queryable Evaluation

이 branch는 optional debugging idea가 아니라, 이 문서의 목표 구조에서는 기본 경로에 포함한다.

먼저 기본 20cm dense occupancy는 0.4m feature에서 structured sub-voxel channel head로 만든다 (Section 9).

```text
0.4m dense feature
-> structured sub-voxel channel head
-> 300 x 100 x 25 dense occupancy
```

이 결과는 전체 map의 base output이다. 하지만 이 출력은 fixed 20cm grid의 결과다. Tesla-style로 임의 좌표나 voxel 내부를 더 깊게 query하려면 선택된 영역의 연속 표현이 필요하고, 그 연속 표현이 Section 10의 refinement Gaussian이다.

임의의 query point는 주변 refinement Gaussian의 probabilistic superposition으로 평가한다 (GaussianFormer-2 방식).

```text
query point x
-> selected voxel grid 기반 local neighbor lookup
-> 주변 refinement Gaussian 탐색
-> 각 Gaussian의 occupancy 확률 기여를 superposition
-> p(occupied | x), local boundary, surface normal
```

packed execution:

```text
selected region count:
  K (refinement Gaussian budget과 동일)

query count per region:
  M_i (distance / planner-aware LOD budget으로 결정)

total query count:
  Q = sum_i M_i

packed tensors:
  p_query_packed: Q x 3
  query_offsets:  K + 1
  neighbor_index: query당 상위 N개 Gaussian index

실행 규칙:
  모든 query point를 Q x 3 batch로 펴서
  superposition을 한 번의 batched 연산으로 평가
  region별 loop 금지
```

평가가 MLP가 아니라 analytic mixture이므로, 기존 Queryable MLP 방식 대비 추가 학습 파라미터 없이 query 수를 늘릴 수 있다. 필요하면 superposition 출력 위에 아주 작은 MLP를 둬서 semantic / dynamic 보정만 한다.

역할 분담:

```text
0.4m dense feature:
  전체 공간의 coarse context

20cm dense occupancy:
  전체 map의 기본 occupied/free/unknown 결과 (structured head)

refinement Gaussians:
  선택된 어려운 영역의 연속 local 표현

probabilistic evaluation:
  임의 좌표의 continuous occupancy / boundary / normal
```

이 branch가 특히 필요한 경우:

```text
얇은 물체
boundary / surface-shell
dynamic object 주변
occlusion / visibility frontier
planner path 주변
한쪽 면만 노출된 voxel 내부 carving
```

최종 표현:

```text
base voxel:
  fixed 20cm grid cell

surface-shell overlay:
  Gaussian superposition에서 유도한
  adaptive partial cuboid / local surface patch
```

전체 grid는 여전히 20cm fixed voxel grid다. 가변적인 것은 selected 영역의 연속 local 표현이다.

---

## 12. Active Mask Design

active mask는 최종 occupancy를 계산할 때 쓰는 마스크가 아니다. 전체 20cm occupancy는 기본 경로에서 이미 계산한다.

```text
0.4m dense feature
-> structured sub-voxel channel head
-> dense occupancy
```

active mask의 역할은 다음 두 가지다.

```text
1. budget-K refinement Gaussian을 초기화할 voxel 선택
2. probabilistic queryable evaluation을 집중할 surface-shell / dynamic / uncertain 영역 선택
```

즉 active mask는 "이 voxel이 occupied인가?"를 결정하는 마스크가 아니라, "이 영역을 더 깊게 볼 것인가?"를 결정하는 refinement selection mask다.

정확히는 binary mask만 출력하는 것이 아니라, 각 voxel 또는 cell의 refinement priority score를 만든다.

```text
S_refine =
    boundary_score
  + uncertainty_score
  + dynamic_score
  + planner_relevance
  + surface_score
  + visibility_frontier_score
  + near_field_score
```

이 score는 바로 dense refinement를 실행한다는 뜻이 아니다. 다음 quota-based Top-K scheduler가 고정된 개수의 voxel만 선택한다.

단순히 occupied probability가 높은 voxel만 남기면 위험하다.

나쁜 pruning:

```text
keep = sigmoid(occ_logit) > threshold
```

이렇게 하면 camera-only 환경에서 가려진 물체, 얇은 물체, 멀리 있는 물체를 너무 빨리 버릴 수 있다.

더 안전한 active mask:

```text
keep =
    occupied 가능성 높은 voxel
  + uncertainty 높은 voxel
  + boundary 근처 voxel
  + ego path / planned trajectory 주변 voxel
  + dynamic object 주변 voxel
  + 일부 random negative voxel
```

특히 `uncertain voxel`을 keep하는 것이 중요하다.

```text
free != unknown
```

자율주행 occupancy에서는 비어 있음과 모름을 구분해야 한다.

### Quota-based Top-K Refinement Scheduler

실시간성을 위해 매 frame refinement 대상 수를 고정한다.

```text
S_refine
-> category별 candidate 분리
-> quota-based Top-K
-> selected voxel index K개
```

단순 Top-K만 쓰면 score가 큰 한 종류의 영역에 예산이 몰릴 수 있다. 예를 들어 큰 벽의 boundary가 대부분을 차지하면, 작은 동적 물체나 planner path 주변 장애물이 선택되지 않을 수 있다.

따라서 category별 quota를 둔다.

```text
K_total = fixed runtime budget

K_boundary
K_near
K_planner
K_dynamic
K_uncertain
K_visibility_frontier
K_random_probe
```

20 FPS v1 권장 예시:

```text
K_total = 2048

K_boundary            = 768
K_near                = 384
K_planner             = 384
K_dynamic             = 256
K_uncertain           = 192
K_random_probe        = 64
```

`K_total`은 latency와 품질을 직접 조절하는 knob이다.

```text
K = 1024:
  빠름, refinement 적음

K = 2048:
  20 FPS v1 권장 시작점

K = 4096:
  품질은 좋지만 20 FPS hard target에서는 부담 증가
```

### Distance / Planner-aware LOD Budget

여기서 LOD는 octree처럼 voxel grid를 재귀적으로 쪼개는 뜻이 아니다. 기본 20cm grid는 유지하고, adaptive query budget만 거리와 중요도에 따라 다르게 주는 뜻이다.

권장 정책:

```text
near field, 0~10m:
  높은 refinement budget
  planner path / dynamic / boundary 우선

mid field, 10~25m:
  중간 refinement budget
  boundary / dynamic / uncertainty 중심

far field, 25~50m:
  낮은 refinement budget
  큰 obstacle boundary와 high-uncertainty만 선택
```

planner가 현재 평가 중인 trajectory corridor는 거리와 무관하게 quota를 별도로 둔다.

```text
planned trajectory 주변:
  near / mid / far와 별도로 planner quota로 보장

확실한 free space:
  refinement 제외

확실한 occupied interior:
  refinement 제외

unknown / visibility frontier:
  일부 quota 유지
```

추천 output heads:

```text
occupancy probability
free probability
unknown / visibility probability
dynamic probability
flow / velocity
surface-related priority score
```

---

## 13. 20 FPS Runtime Target and Multi-rate Scheduling

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

20 FPS를 목표로 하면 모든 고비용 출력을 모든 frame에서 같은 강도로 실행하지 않는다. 대신 perception의 기본 출력은 매 frame 유지하고, 무거운 refinement는 작은 budget 또는 낮은 주기로 실행한다.

핵심 개념:

```text
매 frame 반드시 나와야 하는 것:
  base occupancy
  road surface geometry
  active mask
  small refinement

매 frame 전부 무겁게 하지 않아도 되는 것:
  large adaptive refinement
  dense flow refinement
  optional semantic / traversability extension
```

권장 multi-rate schedule:

```text
20 FPS, every frame:
  image backbone
  1.6m spatial attention
  temporal BEV memory update
  dense deconv to 0.4m
  20cm base occupancy
  road surface geometry
  active mask
  small K refinement

10 FPS, every 2 frames:
  heavier adaptive refinement
  larger Q_total query budget
  flow refinement for wider active regions

planner-triggered, as needed:
  planned trajectory corridor refinement
  near-field collision check refinement
  dynamic object boundary refinement
```

즉 시스템은 매 frame 20 FPS로 기본 occupancy를 내보낸다. 하지만 고비용 query refinement는 모든 영역에서 매 frame full budget으로 돌리지 않고, 중요 영역 또는 저주기 branch로 나눠서 실행한다.

20 FPS v1 default budget:

```text
K_total:
  1024 ~ 2048 refinement Gaussians

re-query budget:
  Gaussian당 4~8 deformable sampling points, 1 round

Q_total:
  8k ~ 32k queryable evaluation points

recommended start:
  K_total = 2048
  re-query points = 8, 1 round
  average M_i = 8
  Q_total ~= 16k
```

20 FPS implementation guardrails:

```text
Base dense occupancy:
  structured sub-voxel channel head 사용
  모든 20cm voxel에 generic GridSample + MLP를 호출하지 않음

Gaussian refinement:
  active mask가 선택한 K개 voxel에서만 초기화
  이미지 기반 Gaussian 초기화 모듈 금지
  deformable re-query는 1~2 round, P3 중심으로 제한
  full-scene Gaussian-to-voxel splatting 금지
  K_total / Q_total hard cap 유지
  packed batched evaluation 필수

Dense features:
  0.4m까지만 dense feature 유지
  0.2m dense feature volume 금지

Temporal:
  0.8m BEV memory만 저장 (시간 간격 keyframe)
  full 3D temporal memory 금지

Precision / deployment:
  TensorRT FP16을 기본 목표로 둠
  backbone / MLP INT8은 v1.5 이후 선택적으로 검토

Runtime profiling:
  PyTorch eager mode 성능으로 20 FPS를 기대하지 않음
  p50 / p95 latency와 peak GPU memory를 매 stage별 측정
```

20 FPS v1에서 제외하는 것:

```text
dense 0.2m feature volume
full 3D temporal memory
unbounded dense P2/P3 local image re-query
full-scene Gaussian splatting / 이미지 기반 Gaussian 초기화 모듈
all-camera high-resolution local attention
all-voxel generic GridSample + MLP dense query
large K_total such as 4096 by default
```

local image re-query는 무제한 dense 형태로는 제외한다. 대신 budget-K Gaussian branch의 deformable re-query가 `K hard cap / 1~2 round / P3 중심` 제한 안에서 같은 역할을 한다. Orin 측정 결과 이 비용도 부담이면 re-query round를 0으로 내리는 fallback(0.4m feature 기반 refine만)을 쓴다.

---

## 14. Final Inference Path

최종 inference 구조:

```text
1. Raw multi-camera images
2. Canonical-view preprocessing
3. Image backbone + neck
4. Canonical ray positional embedding
5. Vanilla spatial attention at 1.6m
6. ViewFormer-style temporal BEV memory alignment
7. Temporal context injection into 1.6m 3D feature
8. Dense deconv: 1.6m -> 0.8m
9. Dense deconv: 0.8m -> 0.4m
10. Produce road surface geometry: z_surface / valid / uncertainty
11. Produce 300 x 100 x 25 dense occupancy with structured sub-voxel channel head
12. Produce dynamic probability / occupancy flow
13. Build active mask from occupancy / uncertainty / boundary / flow / surface / planner scores
14. Quota-based Top-K scheduler with fixed K budget
15. Distance / planner-aware LOD query budget assignment
16. Initialize budget-K refinement Gaussians at selected voxels
17. Deformable image re-query refines Gaussian parameters
18. Packed batched probabilistic queryable evaluation
```

최종 구조를 한 줄로 정리하면:

```text
Canonical-view REO-style attention
+ ViewFormer-style streaming temporal BEV memory
+ PanoOcc-style 2-stage coarse-to-fine deconv
+ RoadBEV/FastRSR-style road surface geometry head
+ structured sub-voxel 20cm dense occupancy head
+ dynamic / flow head
+ quota-based Top-K refinement scheduler
+ distance / planner-aware LOD budget
+ GaussianFormer-style budget-K refinement Gaussians
+ deformable image re-query on canonical virtual cameras
+ GaussianFormer-2-style packed probabilistic queryable evaluation
```

---

## 15. Why This Architecture

`0.8m attention + 1 deconv`보다 `1.6m attention + 2 deconv`가 Jetson AGX Orin 추론에서 더 유리할 가능성이 있다.

이유:

```text
0.8m query count ~= 13,125
1.6m query count ~=  1,976
```

attention query 수가 약 6배 이상 줄어든다.

추가되는 deconv는 다음 단계다.

```text
1.6m -> 0.8m
```

이 단계의 output voxel 수는 약 13k라서 상대적으로 작다. 반면 둘 다 공통으로 필요한 무거운 deconv는 다음 단계다.

```text
0.8m -> 0.4m
```

따라서 전체적으로는 다음 trade-off가 된다.

```text
attention 비용: 크게 감소
deconv 비용: 작게 증가
```

Jetson AGX Orin처럼 embedded GPU에서 attention, sampling, softmax, memory movement가 병목이 될 수 있기 때문에 이 trade-off가 유리할 수 있다.

temporal module도 같은 이유로 3D memory가 아니라 BEV memory를 사용한다. 단 저장/warp 해상도는 1.6m이 아니라 0.8m이다. 1.6m cell은 frame당 ego 이동(~0.5m)의 sub-cell warp가 보간으로 번져 기억이 흐려지기 때문이다.

```text
1.6m 3D feature:
38 x 13 x 4 x C

0.8m BEV memory:
75 x 25 x Cb
(4 keyframes, C=64 fp16 기준 약 1MB)
```

여러 프레임을 저장할 때 z축이 빠지는 것만으로 memory와 temporal attention 비용이 크게 줄어든다. 이후 temporal context는 현재 3D feature에 다시 주입하므로, 과거 정보는 싸게 쓰고 현재 frame의 height 구조는 유지할 수 있다.

refinement branch가 sparse feature anchor + Queryable MLP가 아니라 budget-K Gaussian인 이유도 같은 비용 논리다.

```text
sparse anchor + Queryable MLP:
  0.4m feature를 다시 보간하는 구조
  1.6m lifting 이후 이미지의 새 정보가 유입되지 않음
  boundary 정밀도가 0.4m feature의 정보량에 갇힘

budget-K Gaussian refinement:
  같은 K budget으로 deformable image re-query 1 round 추가
  선택 영역만 이미지에서 새 정보를 받아옴
  covariance가 normal / thin structure를 analytic하게 표현
  임의 좌표 평가가 추가 파라미터 없는 closed-form superposition
```

GaussianFormer 계열을 전면 채택하지 않는 이유도 명확하다. 원본 구조는 desktop GPU에서도 수백 ms급이고, 커스텀 splatting CUDA op과 이미지 기반 초기화 모듈은 TensorRT/Jetson 배포에 불리하다. 이 문서는 그 중 비용 대비 효과가 확실한 세 가지(파라미터화, deformable re-query, superposition 평가)만 budget-bounded로 가져온다.

---

## 16. Implementation Notes

초기 구현 권장 순서:

```text
Stage 1:
  canonical-view preprocessing
  + 1.6m spatial attention
  + 2-stage dense deconv to 0.4m
  + 20cm structured occupancy head
  + basic road surface geometry head
  + auxiliary depth head (training-only)

Stage 2:
  ViewFormer-style 0.8m temporal BEV memory 추가 (시간 간격 keyframe)
  ego-motion alignment / BEV warp 추가
  dynamic / flow head 추가

Stage 3:
  unknown / uncertainty head 추가
  active mask 생성
  quota-based Top-K refinement scheduler 추가
  distance / planner-aware LOD budget 추가

Stage 4:
  budget-K Gaussian refinement branch 추가 (re-query 없이)
  selected voxel에서 Gaussian 초기화
  packed batched probabilistic evaluation 적용

Stage 5:
  deformable image re-query 추가 (1 round, P3)
  Gaussian 파라미터 잔차 갱신 학습

Stage 6:
  optional surface semantics / traversability 확장
```

구현 순서 참고: Stage 4~5는 packed Queryable MLP 기반 refinement를 같은 인터페이스(selected voxel -> 파라미터/컨텍스트 -> 임의 점 평가)의 중간 baseline으로 먼저 만들어도 된다. plan.md의 Phase 08이 이 baseline에 해당하며, Gaussian branch는 인터페이스를 유지한 채 구현체만 교체한다.

v1에서 명시적으로 제외:

```text
unbounded dense P2/P3 local image re-query
full-scene Gaussian splatting / 이미지 기반 Gaussian 초기화 모듈
flow-aware dynamic feature correction
dense 0.2m deconvolution
20cm full dense feature volume
all-voxel generic GridSample + MLP dense query
```

local image re-query는 budget-K Gaussian branch의 deformable re-query(1~2 round, P3 중심, K hard cap)로만 허용한다. 무제한 dense 형태는 v2 이후에도 기본 경로에 넣지 않는다.

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

이 augmentation은 canonical-view preprocessing에 overfit되는 것을 줄이고, 하드웨어 변경 유연성을 높이기 위해 필요하다.

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
  1.6m 3D voxel queries
  vanilla cross-attention with image tokens

Temporal module:
  z-squeeze 1.6m 3D feature to BEV
  upsample to 0.8m BEV for storage / warp
  streaming BEV memory queue (time-spaced keyframes, ~0.2s)
  ego-motion alignment at 0.8m
  temporal attention
  inject temporal BEV context back into 1.6m 3D feature

Dense decoder:
  deconv 1: 1.6m -> 0.8m
  deconv 2: 0.8m -> 0.4m

Surface output:
  learned z-pooling / surface query pooling from 0.4m feature
  road surface geometry: z_surface / valid / uncertainty
  derived geometry: slope / normal / step from z_surface

Volume occupancy output:
  structured sub-voxel channel head from 0.4m feature
  each 0.4m cell predicts 2 x 2 x 2 occupancy logits
  reshape / crop to 300 x 100 x 25
  occupancy / free / unknown output

Dynamic / flow output:
  dynamic probability
  occupancy flow / velocity
  apply densely or on active voxels depending on compute budget

Runtime refinement scheduler:
  active mask produces S_refine
  quota-based Top-K selects fixed K voxels per frame
  quotas: boundary / near / planner / dynamic / uncertain / visibility frontier / random probe
  distance and planner-aware LOD assigns query budget M_i

Gaussian refinement branch:
  initialize K Gaussians at active-mask-selected voxels
  mean / scale / rotation / opacity / feature parameters
  deformable image re-query on canonical virtual cameras
  1~2 rounds, P3-level, K_total hard cap
  residual parameter update

Probabilistic queryable evaluation:
  pack query points into Q x 3 tensors
  GaussianFormer-2-style probabilistic superposition
  continuous occupancy / boundary / surface normal
  render adaptive partial cuboid / local surface overlay

V1 exclusions:
  no unbounded dense P2/P3 local image re-query
  no full-scene Gaussian splatting
  no image-based Gaussian initialization module
  no dense 0.2m feature volume
  no full 3D temporal memory
```

가장 중요한 설계 원칙:

```text
20cm는 결과 해상도다.
20cm는 내부 dense feature 해상도가 아니다.
```

---

## 18. Recommended Papers and Repositories

이 문서의 아키텍처를 구현하려면 논문을 모두 같은 깊이로 볼 필요는 없다. 목적별로 보는 순서를 나눈다.

### Core Architecture

1. [REO](https://github.com/ICEORY/REO)
   - calibration-free / calibration-light vanilla attention과 query-based prediction 참고
   - 이 문서에서는 1.6m 3D voxel query가 image tokens를 읽는 spatial lifting에 반영

2. [ViewFormer-Occ](https://github.com/viewformerocc/viewformer-occ)
   - streaming temporal BEV memory, ego-motion alignment, temporal attention, flow occupancy 계열 참고
   - 이 문서에서는 1.6m feature 직후 temporal module과 Dynamic / Flow Head에 반영

3. [PanoOcc](https://github.com/Robertwyq/PanoOcc)
   - coarse-to-fine 3D deconvolution과 unified occupancy representation 참고
   - 이 문서에서는 1.6m -> 0.8m -> 0.4m dense decoder에 반영

4. SparseOcc / sparse convolution 계열
   - 중요한 영역만 sparse하게 refine하는 방식 참고
   - 이 문서에서는 Active Mask Head와 quota Top-K refinement 영역 선택에 반영

5. [GaussianFormer](https://github.com/huang-yh/GaussianFormer) / [GaussianFormer-2](https://arxiv.org/abs/2412.04384)
   - object-centric 3D Gaussian 표현, deformable image re-query, probabilistic Gaussian superposition 참고
   - 이 문서에서는 Budget-K Gaussian Refinement Branch와 Probabilistic Queryable Evaluation에 반영
   - 주의: 이미지 기반 초기화 모듈, full-scene splatting, 수만 개 단위 Gaussian은 가져오지 않는다 (Section 10)

### Road Surface Geometry

1. [RoadBEV](https://github.com/ztsrxh/RoadBEV)
   - BEV에서 road surface elevation을 직접 예측하는 기본 참고자료
   - `z_surface(x, y)`와 road surface geometry head를 이해하기 위해 먼저 볼 것

2. [FastRSR](https://arxiv.org/abs/2504.09535)
   - RoadBEV 계열의 효율 개선 방향 참고
   - Jetson AGX Orin 같은 embedded inference를 생각하면 RoadBEV 다음으로 볼 것

3. [BEV-GS](https://arxiv.org/abs/2504.13207)
   - road surface를 Gaussian/grid representation으로 표현하는 옵션
   - 첫 구현 필수는 아니고, 표면 rendering이나 texture reconstruction까지 관심이 생기면 볼 것

### Optional Extensions

1. [MapTR](https://github.com/hustvl/MapTR)
   - lane, divider, boundary 같은 online vectorized HD map element 예측 참고
   - surface semantics를 확장할 때 선택적으로 참고

2. [HDMapNet](https://arxiv.org/abs/2107.06307)
   - BEV semantic map learning의 기본 참고자료
   - road / lane / curb 같은 raster BEV semantics를 추가할 때 선택적으로 참고

3. [Cam2BEV](https://github.com/ika-rwth-aachen/Cam2BEV)
   - multi-camera image에서 BEV semantic segmentation을 만드는 고전적 참고자료
   - surface semantics를 단순 raster BEV segmentation으로 시작할 때 선택적으로 참고

4. [H3O](https://arxiv.org/abs/2503.04059)
   - depth, semantic segmentation, surface normal을 3D occupancy 학습에 보조 supervision으로 쓰는 방향 참고
   - surface normal / depth / semantic auxiliary loss가 필요할 때 선택적으로 참고

5. [UnScenes3D](https://www.nature.com/articles/s41597-025-05532-5)
   - 3D semantic occupancy와 road surface elevation reconstruction을 함께 정의한 데이터/태스크 참고
   - volume output과 surface geometry output을 함께 학습/평가하는 관점에서 참고

6. [GaussianWorld](https://arxiv.org/abs/2412.10373)
   - Gaussian을 ego-motion으로 이월하며 갱신하는 streaming Gaussian world model
   - refinement Gaussian의 temporal 이월(v2)을 검토할 때 참고

7. [GaussTR](https://arxiv.org/abs/2412.13193)
   - Gaussian rendering 기반 self-supervision과 foundation model alignment
   - sub-voxel supervision이 없을 때 rendering 기반 보조 supervision을 붙일 때 참고

### Minimal Reading Order

최소로만 본다면 다음 순서를 추천한다.

```text
1. RoadBEV
2. FastRSR
3. ViewFormer-Occ
4. REO
5. PanoOcc
6. SparseOcc / sparse convolution examples
7. GaussianFormer -> GaussianFormer-2
```

surface output을 이해하려는 목적이면 `RoadBEV -> FastRSR`를 먼저 본다. flow occupancy와 temporal memory는 `ViewFormer-Occ`, 20cm query 구조는 `REO`, coarse-to-fine decoder는 `PanoOcc`, refinement 영역 선택은 sparse convolution / SparseOcc 계열을 본다. budget-K Gaussian refinement와 probabilistic queryable evaluation은 `GaussianFormer -> GaussianFormer-2` 순서로 본다.

---

## 19. GT Data Pipeline (Camera-only Auto-labeling)

이 문서의 모든 head는 supervision이 필요하다. LiDAR가 없는 camera-only 환경에서는 DUSt3R 계열 3D foundation model을 기반으로 한 offline auto-labeling으로 GT를 만든다. offline이므로 추론 latency 제약과 무관하게 무거운 모델을 쓸 수 있다.

### Why Not Vanilla DUSt3R

DUSt3R 아이디어 자체는 출발점이지만, vanilla DUSt3R를 직접 쓰지 않는 이유는 네 가지다.

```text
1. scale 모호성:
   DUSt3R pointmap은 up-to-scale이다.
   20cm metric voxel GT에는 metric scale이 필수다.

2. pair 단위 추론 + 비싼 global alignment:
   DUSt3R는 이미지 쌍 단위로 추론하고,
   multi-view 통합에 최적화 기반 global alignment가 필요하다.
   수백~수천 frame의 주행 시퀀스에 비실용적이다.

3. 정적 장면 가정:
   동적 물체는 재구성 기하가 깨진다 (ghosting).
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
  (images만: pointmap 상대오차 ~0.16,
   images + intrinsics + pose + depth: ~0.01)
  multi-camera rig + odometry 보유 상황에 정확히 부합

대안: MASt3R / MASt3R-SfM, VGGT

동적 분리 보조: MonST3R 또는 2D segmentation + tracking
```

### Two-stage Structure

DUSt3R 계열과 NeRF/3DGS는 경쟁 관계가 아니라 순차 역할 분담이다.

```text
Stage A: 기하 + 포즈 (MapAnything / MASt3R)
  metric pointmap, camera pose, dense 초기 기하
  NeRF/3DGS는 포즈 없이 시작하지 못하므로 반드시 선행

Stage B: 정제 + 연속 표면 (3DGS 권장, NeRF 가능)
  Stage A 출력을 초기화로 photometric 최적화
  표면 정제 후 mesh / SDF 추출
```

Stage B의 목적은 화질이 아니다. Stage A 점군을 voxelize하면 20cm 이산 GT까지만 나온다. Stage B가 만드는 연속 mesh/SDF가 Section 10 budget-K Gaussian refinement branch를 학습시킬 **sub-voxel continuous GT**다. 즉 Stage B는 refinement branch가 실데이터에서 학습 가능해지는 조건이다.

NeRF 대신 3DGS를 권장하는 이유: 학습이 빠르고, Stage A의 점군 출력이 그대로 3DGS 초기화가 되며, mesh 추출 도구가 성숙해 있다. 동적 장면까지 neural rendering으로 다루려면 EmerNeRF-style static/dynamic decomposition을 참고한다.

### Pipeline

```text
1. 수집:      multi-camera 시퀀스 + Basalt VIO pose (Section 20) + 시각 동기화

2. 기하:      MapAnything (images + known intrinsics + pose prior)
              -> metric pointmaps, chunk 단위 처리

3. 동적 분리: SAM2 + tracking -> dynamic mask
              -> 정적 aggregation에서 동적 물체 제외

4. 정적:      static 점군 multi-frame aggregation -> TSDF fusion
              -> mesh / SDF
              -> 20cm voxelize -> volume occupancy GT
              -> ground segmentation -> z_surface / valid GT
              -> 연속 SDF 보존 -> sub-voxel GT

5. 가시성:    per-frame camera ray casting -> free / unknown 분리
              (free != unknown 라벨의 유일한 출처)

6. 동적:      per-frame instance volume + track association
              -> dynamic occupancy + occupancy flow GT

7. 옵션:      3DGS refinement -> 표면 정밀화 + 연속 GT 강화

8. 검증:      같은 파이프라인을 nuScenes에 적용
              -> Occ3D LiDAR GT와 정량 비교
              -> 거리별 오차 프로파일 확보 후 자체 데이터에 적용
```

### Head별 GT 매핑

```text
Volume Occupancy Head:
  step 4 voxelize + step 5 visibility

Road Surface Geometry Head:
  step 4 ground segmentation (z_surface / valid / uncertainty)

Dynamic / Flow Head:
  step 6 tracking

Active Mask / quota Top-K:
  base GT에서 유도 (boundary / uncertainty score)

Budget-K Gaussian Refinement Branch:
  step 4 / 7의 연속 SDF (sub-voxel supervision)
```

### Limitations

```text
원거리 정확도:
  camera 기반 depth 오차는 거리에 따라 급증한다.
  20cm GT 신뢰 범위는 현실적으로 ~20-25m.
  far-field는 GT uncertainty를 낮춰 loss weight를 줄이거나
  unknown으로 라벨한다.
  이 부분은 LiDAR GT 대비 구조적 열세이며 회피 불가다.

검증 없는 적용 금지:
  자체 데이터에는 비교 기준이 없으므로,
  step 8의 nuScenes 검증으로 오차 수준을 먼저 정량화한다.

품질 보증:
  가능하면 GT 수집용으로만 LiDAR 1대를 일부 시퀀스에 장착해
  spot-check한다. 추론은 여전히 camera-only다.
```

### References

1. [DUSt3R](https://github.com/naver/dust3r) / [MASt3R](https://github.com/naver/mast3r)
   - pointmap 기반 unconstrained 3D 재구성의 출발점 (Naver Labs)

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
  Section 6 temporal BEV memory의 ego-motion warp 입력

중력 방향:
  voxel grid의 z축을 차체가 아니라 중력에 정렬
  차체 pitch/roll에도 grid가 흔들리지 않음
  Section 1의 z range [-1m, +4m]는 이 중력 정렬 기준이다
```

### B. Landmark -> Sparse Depth Supervision (v1, 학습)

Section 4의 auxiliary depth supervision에 들어가는 온라인 GT 소스다.

```text
landmark를 이미지에 재투영
-> 해당 픽셀의 metric depth GT
-> depth head를 sparse하게 supervise
-> landmark covariance로 loss 가중
```

주의: landmark는 코너/텍스처 영역에만 찍히므로 분포가 편향된다 (흰 벽, 균일한 노면에는 없음). Section 19의 dense GT를 대체하지 않고 보완한다. 운영 중 공짜로 계속 쌓이는 것이 장점이다.

### C. Landmark 입력 주입 (v1.5)

supervision을 넘어 추론 시 입력 힌트로도 쓸 수 있다. MapAnything이 known geometry 입력으로 정확도가 급상승하는 것과 같은 원리로, vanilla attention의 기하 힌트 부족을 보완한다.

```text
방법 1: voxel grid prior
  landmark 3D 위치를 1.6m voxel grid에 splat
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
-> dynamic head의 weak pseudo-label
-> active mask dynamic quota의 후보 영역
```

노이즈가 많으므로(조명 변화, occlusion도 기각됨) 단독 라벨이 아니라 약한 보조 신호로만 쓴다.

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
