# Tesla-style Queryable Occupancy Network Architecture

이 문서는 Tesla AI Day 2022의 Occupancy Network 공개 구조를 참고하되, 실제 구현을 위해 REO, ViewFormer, PanoOcc, BEVFormer/BEVDet4D 계열 아이디어를 섞은 아키텍처 초안이다.

현재 설계의 핵심은 다음이다.

```text
최종 해상도:
  30cm voxel (occupied/surface-band 후보 + near-surface free)

최종 feature (2-해상도 hybrid):
  free/unknown 빈 공간: coarse dense @ 0.6m
  occupied/surface 후보: sparse deconv + 프루닝으로 30cm까지
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
| PanoOcc | (1) coarse 3D voxel **temporal: align + concat + residual 3D conv** (z 유지, coarse ~40k voxel, 과거3+현재1) (2) coarse-to-fine **sparse deconvolution + occupancy 프루닝** | (1) 0.6m 3D 단일 temporal stage (z 유지, coarse motion seed, Tesla 그림 align+concat과 동일 계열) (2) 1.2m->0.6m->0.3m sparse decoder (점유 후보만 keep) |
| SparseOcc / sparse execution 계열 | 중요한 위치만 더 비싼 연산을 실행하는 runtime control | sparse decoder의 프루닝 + Sub-Voxel Shape Head의 masked / face-aware execution |
| RoadBEV / FastRSR | BEV에서 road surface elevation을 직접 예측하는 관점 | 별도 Surface Outputs head(dense BEV)에서 z_surface와 geometry-derived risk 직접 회귀 (Section 11) |

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
  left 10.2m ~ right 10.2m
  total 20.4m
  (20m가 아니라 20.4m로 잡은 이유: 20/0.3 = 66.67이라 정수 cell이 안 됨.
   ±10.2m = 20.4m = 68 cells로 딱 떨어지게 해 격자=타깃을 만든다)

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

세 축 모두 30cm로 딱 떨어지도록 range를 잡았다(X 60m=200, Y 20.4m=68, Z 6m=20). 따라서 **격자=타깃이라 padding도 valid mask도 필요 없다.** (이전엔 Y를 20m로 잡아 20.4m padding+mask가 필요했으나, 20/0.3=66.67의 비정수 문제로 Y를 ±10.2m=20.4m로 변경했다.)

> 외부 metric 기준 주의: model의 valid ROI는 이제 Y ±10.2m다. planner/평가가 정확히 **±10m**를 기대하면, 0.3m cell과 ±10m 경계가 딱 맞지 않으므로 단순 whole-row crop으로는 정확히 ±10m가 나오지 않는다. export/eval은 (1) metric mask로 ±10m 밖을 제외하거나, (2) 필요하면 resampling으로 정확한 20m 출력을 만들고, (3) whole-cell만 허용되는 소비자는 full 68 cells(±10.2m) 또는 center-inside 66 cells(약 19.8m)를 명시적으로 선택한다. 학습 grid는 ±10.2m를 유지한다.

```text
Internal 30cm grid (= valid ROI, 전 축 격자=타깃):
  X: 200 cells = 60.0m   (rear 10m ~ front 50m)
  Y:  68 cells = 20.4m   (left 10.2m ~ right 10.2m)
  Z:  20 cells =  6.0m   (-2m ~ +4m)

0.3m 표현:
  이 200 x 68 x 20 grid 위에서 sparse하게 유지한다.
  dense하게 채우지 않고, sparse deconv + 프루닝으로 candidate voxel만 활성화한다.
  활성 index 집합(prelim_kept) =
    child_score_keep_candidates ⊎ prelim_free_shell
    (gate② cheap score로 정한 O_occ_fine 실행 대상. 아직 최종 surface/boundary/free label이 아니다.)
  O_occ_fine 이후에 kept는 pred_occupied_surface / pred_boundary / pred_visibility_frontier
    / pred_free_shell / pred_free_kept / pred_uncertain_kept로 완전 분할된다 (Section 9).
  (padding/valid mask 없음 -> loss/metric/parent-child mapping이 모든 cell에서 명확)
```

가장 중요한 설계 구분 (2-해상도 hybrid):

```text
Coarse dense occupancy / visibility:
  0.6m에서 dense하게 만든다 (temporal stage와 같은 해상도).
  occupied / free / unknown / visibility + mixed_surface_risk를 담당한다.
  빈 공간은 크고 매끄러우므로 30cm 정밀도가 필요 없다.

Sparse fine 30cm output:
  PanoOcc식 sparse deconv + occupancy 프루닝으로 candidate voxel(prelim_kept)을 남긴다.
  prelim_kept는 최종 label 집합이 아니라 O_occ_fine을 실행할 sparse index set이다
    (= child_score_keep_candidates ⊎ prelim_free_shell).
  O_occ_fine 이후 pred_* label로 분할되며, 무거운 head(Flow / Sub-Voxel Shape / Semantics)는
  그중 pred_heavy_mask(occupied|boundary)
  에서만 실행하고, free_shell/free_kept는 occupancy 해상도(좁은 통로 판단)에 쓴다. (Section 9)

별도 30cm dense 3D feature volume:
  만들지 않는다. 프루닝으로 sparse하게 유지한다.

Sub-Voxel Shape Information:
  4개 volume head 중 하나로 예측한다.
  pred_heavy_mask voxel 중에서도 비싼 local query는
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

이 분리의 동기는 Tesla 데모 영상의 **렌더링**이다. Tesla occupancy 데모는 빈 공간이 아니라 **표면 voxel만** 그린다.

다만 "Tesla 내부 표현이 sparse"라는 뜻은 아니다. 공개 그림/특허 기준 Tesla는 **dense voxel grid의 occupancy를 평가하고 deconvolution으로 voxel을 만들며, 기본 voxel(~30cm)에서 표면/주행면 근처는 더 작은 voxel로 refine**하는 쪽에 가깝다. 즉 Tesla는 "dense 평가 + 표면 근처 finer refine + 표면 중심 렌더링"이다.

> **Tesla public vs 본 문서(Jetson 적응)**: 본 문서의 2-해상도 sparse hybrid(free/unknown=0.6m coarse dense, occupied/surface-band 후보=0.3m sparse deconv+prune)는 Tesla와 **동일한 구현이 아니라**, Tesla-style 출력 구조(4 Volume Head / Surface / Queryable, 표면 중심 표현)를 **Orin 비용에 맞춰 sparse runtime으로 변형**한 것이다. 이하 문서에서 "Tesla처럼"은 출력 구조/철학을 따른다는 뜻이고, sparse prune·near-surface free shell·2단 게이트는 우리 Jetson 적응이다.

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
| coarse motion seed (vx,vy,vz) |       +--------------------------+
+---------------+---------------+
                |
                v
+-------------------------------+       +--------------------------+
| Coarse Dense Occupancy @0.6m  |       | Sparse Deconv (2단 게이트)|
| occ/free/unknown/vis+risk     |       | ①0.6m parent gate(연산)  |
| (빈 공간 free/unknown 담당)   |       | ②0.3m child prune(선명도) |
| (프루닝 게이트 기준도 담당)   |       +------------+-------------+
+---------------+---------------+                    |
                |                                    v
                |                       +--------------------------+
                |                       | Sparse 0.3m Candidate    |
                |                       | prelim_kept voxels      |
                |                       +------------+-------------+
                |                                    |
                v                                    v
+--------------------------------------------------------------+
| Tesla-style 4 Volume Heads (= Tesla "Volume Outputs")        |
| 1. Occupancy Head (coarse dense free/unknown +               |
|                    sparse fine occ/surface-band)             |
| 2. Occupancy Flow Head        (N_heavy, vx/vy/vz,           |
|                                0.6m motion seed readout)   |
| 3. Sub-Voxel Shape Info Head  (N_heavy + exposed-face mask) |
| 4. 3D Semantics Head          (N_heavy)                     |
+------+----------------------------------------+--------------+
       |                                        |
       v                                        v
+---------------------------+        +-------------------------------+
| Surface Outputs (별도 head)|       | Queryable Output Interface    |
| = Tesla Road Surface       |       | occ/boundary -> MLP query     |
|   Geometry + risk       |       | free_shell -> child-level      |
| dense BEV: z_surface,   |       |   observed-free일 때만 free    |
| slope, step, cost       |       | uncertain/pruned -> consumer mode |
+---------------------------+        +-------------------------------+
```

축약하면:

```text
Canonical-view vanilla attention (REO 방식) on 3D voxel query (Tesla 타깃)
+ dense deconv 1.2m -> 0.6m
+ 0.6m pre-temporal refinement (BEVDet4D 교훈)
+ Tesla/PanoOcc-style single 3D temporal @ 0.6m (align+concat+3D conv, coarse motion seed)
+ coarse dense occupancy/visibility @ 0.6m (free/unknown 담당)
+ PanoOcc-style sparse deconv + 프루닝 0.6m -> 0.3m
+ Tesla-style 4 Volume Heads (Occupancy fine=[N_kept], heavy heads=[N_heavy])
+ Surface Outputs head (Tesla처럼 별도 브랜치)
+ face-aware Sub-Voxel Shape Information
+ queryable occupancy interface
```

핵심 목표는 다음과 같다.

- occupied/surface-band 후보와 near-surface free shell은 30cm sparse로, free/unknown 빈 공간은 coarse dense로 만든다 (2-해상도 hybrid).
- 30cm dense volume feature는 만들지 않는다. sparse deconv + 프루닝으로 0.3m candidate만 남긴다.
- volume output은 Tesla 그림처럼 4개 head로 나눈다. Flow / Sub-Voxel Shape / Semantics는 pred_heavy_mask(occupied|boundary)에서만 실행하고, Occupancy fine branch만 `[N_kept]` 전체를 본다.
- Occupancy Head만 두 갈래로 나뉜다: free/unknown은 coarse dense, 30cm occupied/surface-band evidence는 sparse fine.
- Surface(road geometry)는 Tesla 그림처럼 **별도 출력 브랜치(Surface Outputs head)** 로 둔다. volume head readout이 아니다 (Section 11).
- Sub-Voxel Shape Information은 별도 예측 경로가 아니라 head다.
- "보이는 면만 refine"은 Sub-Voxel Shape Head 내부의 exposed-face mask와 masked query execution으로 처리한다.
- temporal은 **단일 stage @ 0.6m 3D** (Tesla/PanoOcc-style, z-squeeze 없이 Z 유지)로 한다. deconv 1.2m -> 0.6m 직후, deconv 0.6m -> 0.3m 이전의 coarse latent에서 처리한다. 융합은 **trajectory align + concat + 3D residual conv** (attention 아님).
- Flow는 0.3m sparse로 출력하되, motion 정보는 0.6m 3D temporal의 coarse motion seed에서 온다 (vx/vy/vz seed -> pred_heavy voxel readout/refine). Z 유지라 높이별 motion(vz) seed까지 잡는다.
- 1.2m 별도 temporal stage는 두지 않는다. 단 **1.2m 3D는 Orin 실측이 20Hz 예산 초과 시의 fallback**으로만 둔다.
- z-squeeze BEV temporal, temporal attention(ViewFormer식), full global temporal attention(0.6m/0.3m), 0.3m temporal, all-voxel heavy local query는 v1에서 금지한다 (ViewFormer streaming memory는 학습 효율용으로만 선택 차용).

---

## 1. Prediction Range and Output Grid

예측 영역:

```text
X axis: rear 10m ~ front 50m = 60m target = 60.0m grid (격자=타깃)
Y axis: left 10.2m ~ right 10.2m = 20.4m target = 20.4m grid (격자=타깃)
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

**전 축 격자=타깃이라 grid ROI valid mask(`roi_valid_mask`)는 없다.** X 60m=200, Y 20.4m=68, Z 6m=20 모두 30cm로 정확히 나누어떨어진다. 따라서 200×68×20 grid 전체가 곧 valid ROI다.

> 용어 주의: "valid mask"가 두 의미로 쓰이지 않게 분리한다. **`roi_valid_mask`(없음)** = grid 경계/padding용(여기선 불필요). **`warp_valid_mask`(있음)** = temporal warp에서 범위 밖 sample 표시(Section 1.1). **`surface_observed_mask`(있음)** = surface/GT에서 관측된 cell만 supervise(Section 11, 19). 세 mask는 서로 다른 것이다.

```text
model output 좌표계:
  200 x 68 x 20 grid = valid ROI (X·Y·Z 전부 타깃과 정확히 일치, padding 없음)
  (30cm 후보는 이 중 점유/표면/near-free/uncertain 후보만 sparse 활성,
   free/unknown은 coarse dense에서 관리)

loss / metric / planner:
  grid ROI 관점에서는 모든 cell이 valid (padding용 ignore 불필요).
  단 supervision 관점의 confidence/visibility/unknown ignore mask는 별도로 사용한다.
```

30cm feature 비용 감각 (dense였다면):

```text
만약 F_0.3를 dense로 만들면, C=64, fp16:
  200 x 68 x 20 x 64 x 2 bytes
  ~= 34.8 MB
  (272,000 voxel 전부에 3D conv -> Orin 20Hz 위협)
```

본 문서는 dense를 만들지 않는다. sparse deconv + 프루닝으로 0.3m candidate voxel만 남긴다:

```text
주행 장면의 occupied 비율은 대략 수~십수 %.
2단 게이트(0.6m parent + 0.3m child, keep ~0.5) 적용 시
  0.3m 단계 활성 voxel은 대략 수만 voxel 수준 (272k의 일부).
  + near-surface free shell(L2, 기본 ON)이 occupied/surface 후보 대비 ~0.5~1배를 더해
    kept 총량이 대략 1.5~2배가 된다 (Section 8.2 N3 참조).
-> 그래도 272k dense 대비 메모리와 3D conv 연산은 크게 감소.
```

free / unknown 빈 공간은 coarse dense @ 0.6m(100 x 34 x 10 = 34,000 voxel)에서만 dense로 다룬다. 0.6m dense는 voxel 수가 0.3m(272k) 대비 8배 작아(34k) 훨씬 가볍지만 "부담 없음"은 아니며 **profile 대상**이다(Section 13). 여기서 temporal(Section 7)과 coarse occupancy를 함께 처리한다. dense 비용이 가장 위험한 곳은 0.3m(272k)이며, 그 단계만 sparse로 만든다.

Jetson AGX Orin 20Hz를 목표로 하면 C는 처음에 48 또는 64로 시작하고, head는 1x1x1 (sparse) conv / light MLP 중심으로 둔다.

---

## 1.1 Coordinate Conventions (구현 전 고정)

grid_sample / temporal warp / ray casting / parent-child mapping은 좌표 규약이 조금만 어긋나도 바로 에러가 난다. 구현 전에 아래를 **단일 진실**로 고정한다.

```text
좌표계 (frames):
  world       : 고정 inertial frame
  ego/body    : 로봇 base_link (IMU). pose = T_world_ego.
  camera_k    : 각 카메라. T_ego_cam_k (extrinsic), intrinsic K_k.
  grid frame  : 현재 ego 위치 + (ego yaw, gravity-up) 정렬된 local frame.
                (gravity로 roll/pitch만 보정, yaw·translation은 ego를 따라감.
                 "gravity frame"이 아니라 "ego-yaw + gravity-up 정렬 local grid frame"이다.)

grid 표현을 3가지로 분리해서 쓴다 (혼동 방지):
  grid_metric : grid frame(ego-yaw+gravity-up)에서의 미터 좌표, **ego origin 기준** (min corner 아님). min corner는 index 변환(grid_index) 때만 뺀다.
  grid_index  : voxel index (i,j,k) 실수/정수
  grid_norm   : grid_sample용 [-1,1] normalized 좌표
  변환:
    p_ego         = T_ego_world · p_world                 # world -> ego (T_ego_world = T_world_ego^-1)
    p_grid_metric = R_grid_ego · p_ego                    # ego -> grid 방향(roll/pitch만 보정), ego origin 기준
                                                          # (p_ego는 이미 ego frame이므로 t_ego를 또 빼지 않는다)
    grid_index    = (p_grid_metric - origin) / s          # origin = min corner. 연속 index: cell i 중심 -> i+0.5
    grid_norm     = 2·grid_index / N - 1                  # align_corners=False (여기서 +0.5를 다시 더하지 않는다)
                                                          # 검산: cell i 중심 grid_index=i+0.5 -> norm=2(i+0.5)/N-1 (정상)

  PyTorch 5D grid_sample 축 매핑 (혼동 방지):
    feature tensor = [B, C, D, H, W] = [B, C, Z, Y, X]   (D<-Z, H<-Y, W<-X)
    grid 의 마지막 차원 = (gx, gy, gz) 순서이고 각각 W,H,D에 대응
      -> gx = grid_norm[X], gy = grid_norm[Y], gz = grid_norm[Z]
    즉 index (i,j,k)=(X,Y,Z)를 grid의 (gx,gy,gz)=(X,Y,Z)로 넣되,
       tensor는 (Z,Y,X) 순으로 저장한다. (i<->W, j<->H, k<->D)

origin & axes:
  origin = grid frame의 min corner = (X_min, Y_min, Z_min) = (-10, -10.2, -2) m
  축: +X 전방, +Y 좌, +Z 위(중력 반대). 오른손 좌표계.
  voxel index (i,j,k), i in [0,200), j in [0,68), k in [0,20).

cell center (half-open):
  각 cell은 left-closed / right-open: cell i = [X_min + i·s, X_min + (i+1)·s)
  cell center = X_min + (i + 0.5)·s          (s = 0.3m)
  점 -> index = floor((x - X_min) / s)
    -> 오른쪽 경계점 X_min + (i+1)·s는 다음 cell(i+1)에 속한다.
    -> x == X_max는 half-open 규약상 out-of-range로 ignore한다 (clamp하지 않음).
  grid_sample은 align_corners=False와 위 grid_norm 식을 일관되게 쓴다.

multi-resolution (parent-child):
  1.2m parent (i) -> 0.6m (2i, 2i+1) -> 0.3m (4i..4i+3)  각 축 독립.
  origin은 세 해상도가 동일(같은 min corner). child index = parent index * 2 + offset.
  s_1.2=1.2, s_0.6=0.6, s_0.3=0.3. 전 축 격자=타깃이라 ratio가 정확히 2 (padding 없음).

temporal warp:
  먼저 각 frame의 grid frame을 world에 대해 정의한다:
    회전 방향 정의: R_grid_ego = (ego -> grid, roll/pitch만 gravity로 보정).
                   R_ego_grid = (grid -> ego) = R_grid_ego^-1.   # 이후로는 R_ego_grid만 쓴다
    grid frame의 world pose (homogeneous, 동일 기호 그대로):
      T_world_gridcur  = T_world_egocur  · [R_ego_grid_cur  | 0; 0 1]
      T_world_gridprev = T_world_egoprev · [R_ego_grid_prev | 0; 0 1]
    (grid origin은 ego origin과 같은 위치, 회전만 ego-yaw+gravity-up; min corner는 index 단계에서만 뺀다.)
  과거 feature를 현재 grid로 가져오는 변환 (방향 명시):
    T_gridprev_gridcur = T_world_gridprev^-1 · T_world_gridcur
    p_prev_gridmetric  = T_gridprev_gridcur · p_cur_gridmetric
      (구현: p_cur_gridmetric에 homogeneous 1을 붙여 [x,y,z,1]^T로 4x4 변환 후 앞 3성분 사용)
  즉 "현재 grid의 점"을 "과거 grid 좌표"로 보낸 뒤 거기서 sample한다.
  (inverse 방향을 헷갈리면 과거 feature가 반대로 밀린다. 단위 테스트: 순수 +X 전진이면
   과거 feature는 현재 grid에서 -X쪽으로 나타나야 한다.)
  grid_sample(mode=bilinear, align_corners=False), 범위 밖은 zero-pad + warp_valid_mask.
  (참고: 5D input에서 mode="bilinear"는 PyTorch 명명상 그렇게 부를 뿐 실제로는 3D trilinear로 동작한다.)
```

이 규약은 Section 7(temporal warp), Section 10(queryable sparse-aware sampling / center-snap), Section 19(ray casting/voxelize), Section 20(VIO pose)에서 그대로 참조한다.

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
   -> coarse motion seed (vx,vy,vz)
-> temporally enhanced 0.6m 3D feature

-> coarse dense occupancy / visibility head @ 0.6m (free/unknown + 프루닝 게이트 기준)

-> [sparse fine 가지] 2단 게이트로 0.3m sparse 생성:
   게이트① @0.6m (연산량, recall-first): high-confidence free parent만 버림 (occupied/mixed-surface-risk/unknown/uncertain은 keep)
                          -> 남은 parent만 sparse 업샘플 (빈 영역 0.3m는 생성조차 안 함)
   게이트② @0.3m (선명도+recall): high-conf free child만 drop (uncertain/unknown/thin/far/dynamic keep), 표면+인접 free 1겹 유지 (anti-dilation)
-> 30cm sparse fine candidate index set
   (= prelim_kept: child_score_keep_candidates ⊎ prelim_free_shell; 아직 최종 pred label 아님)

-> Occupancy Head fine branch ([N_kept], occupied logit + surface-band auxiliary)
-> pred_heavy_mask 확정 후:
   Occupancy Flow / Sub-Voxel Shape / 3D Semantics ([N_heavy])

-> Surface Outputs head (Tesla처럼 별도 브랜치, dense BEV)
-> queryable output interface
```

해상도별 역할:

```text
1.2m:
  image-to-3D spatial lifting (vanilla attention)
  coarse scene / occlusion / long-term context

0.6m (dense):
  single Tesla/PanoOcc-style 3D temporal (z 유지) + coarse motion seed
  coarse dense occupancy / free / unknown / visibility
  occupancy 프루닝 guide
  mid-level geometry refinement

0.3m (sparse):
  sparse fine candidate feature (= prelim_kept; O_occ_fine 이전 index set)
  O_occ_fine 이후 pred_* 6개 label로 분할
  Occupancy fine branch는 [N_kept], heavy heads는 pred_heavy_mask의 [N_heavy]
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

따라서 image cross-attention은 1.2m에서 수행하고, 30cm fine candidate feature는 sparse deconvolution + 프루닝으로 복원한다.

---

## 2.1 Core Volume Heads and Supporting Runtime Modules

이 문서의 core volume heads는 정확히 네 개다(= Tesla "Volume Outputs"). Surface는 별도 브랜치(Section 11), Queryable은 인터페이스(Section 10)다.

| Head | 출력 위치 | 역할 |
|---|---|---|
| Occupancy Head | coarse 갈래: dense `100 x 34 x 10` @0.6m / fine 갈래: `[N_kept] x 2` @0.3m | **coarse가 free/unknown/visibility 담당**, fine은 **occupied logit + surface-band auxiliary**(free=낮은 p_occ) |
| Occupancy Flow Head | 내부 coarse branch: `O_motion_0.6` / 최종 output: `[N_heavy]` | dynamic probability, occupancy flow, velocity (vx/vy/vz). 0.6m seed를 pred_heavy voxel로 readout/refine |
| Sub-Voxel Shape Information Head | `[N_heavy]` + exposed-face masked local query | 30cm voxel 내부의 exposed face, surface offset, normal, shape code, uncertainty |
| 3D Semantics Head | `[N_heavy]` 실행, loss는 valid semantic mask | voxel semantic class |

별도 출력 브랜치 (Tesla 그림 기준):

| 출력 | 위치 | 역할 |
|---|---|---|
| Surface Outputs head | dense BEV (Section 11) | z_surface / slope / step / uncertainty / traversability_cost / drop_risk |
| Queryable Outputs | MLP + fallback 인터페이스 (Section 10) | `surface_shell_prob` + planner용 `collision_state` (semantic은 3D Semantics Head) |

아래 항목들은 head가 아니라 runtime/control mechanism이다.

```text
Exposed-face mask:
  Sub-Voxel Shape Head가 어느 face를 자세히 볼지 정한다.

Active / priority mask:
  planner path, uncertainty, dynamic, visibility frontier 등으로
  expensive local query budget을 배분한다.

Queryable MLP (occupancy 전용, 4-way 분기 - Section 10):
  (A) occupied/boundary 근처 -> Sub-Voxel Shape code + F_0.3_sparse로 MLP 정밀 평가
  (B) kept-fine-free(pred_free_shell|pred_free_kept) -> observed-free 증거가 있을 때만 free 즉답, 아니면 coarse fallback
  (C) pred_visibility_frontier / pred_uncertain_kept -> parent 0.6m coarse/visibility로 보수적 판정
  (D) pruned 영역            -> parent 0.6m coarse occupancy로 보수적 판정
                                unknown/occluded/occupied/uncertain은 consumer mode
  (semantic은 질의하지 않음; voxel-level 3D Semantics Head에서만)
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

### Auxiliary Depth / Free-space Evidence Head (v1 inference에도 필요)

vanilla cross-attention은 projection 힌트 없이 3D 대응을 데이터로만 학습하므로, 수렴이 느리고 데이터 요구량이 커질 수 있다. 이를 완화하기 위해 image feature 위에 작은 depth/free-space evidence head를 둔다.

중요: 이 head는 단순 auxiliary loss가 아니다. Section 10의 `observed_free_evidence_0.3(child)`를 v1에서 만들려면 추론 시에도 저해상도 depth/range confidence가 필요하다. 이 head를 완전히 제거하면 `pred_free_shell`은 planner에 free 즉답을 할 수 없고, 항상 coarse fallback/unknown으로 내려가야 한다.

```text
학습 시:
  image feature -> tiny depth/free-space evidence head
  -> per-pixel or low-res depth / range confidence / ray free-space confidence
  -> L_depth + L_free_space_evidence를 전체 loss에 추가

depth GT 소스:
  dense:     Section 19 pipeline의 metric pointmap depth
  sparse:    Section 20 Basalt landmark 재투영 depth
             (landmark covariance로 가중)
  synthetic: 시뮬레이션 depth

추론 시:
  v1 기본: tiny low-res depth/free-space evidence head 유지
    -> Section 10의 observed_free_evidence_0.3(child)에 사용
    -> confidence가 낮으면 free 증거로 쓰지 않음
  budget 때문에 이 head를 끄는 설정:
    -> near-surface free shell은 0.3m occupancy 후보로는 keep 가능하지만,
       planner free 즉답은 금지하고 coarse fallback/unknown으로 처리
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

### 5.1 Attention Token Budget (Orin 비용 통제, v1 필수)

vanilla cross-attention을 "진짜 global"로 두면 비용이 폭발한다. Q=4,250, K=Σ(camera) H_f·W_f인데, 6캠 × (예: 32×88) = 약 17k token이면 score 행렬이 4,250×17k ≈ 7.2e7. 이건 그대로 두면 Orin에서 위험하다. 따라서 **token budget을 v1부터 명시**한다.

```text
필수:
  - image token downsampling: backbone feature를 attention 입력 해상도로 줄인다
    (예: stride 16 -> H/16·W/16, 또는 추가 1회 pooling). K token 수를 목표로 cap.
  - K token cap (예: 전체 <= 8k~16k). 초과 시 view/region pooling으로 축소.
  - camera / view gating: 각 1.2m query가 모든 카메라를 보지 않고, FOV로 본 가능성이
    있는 카메라 token만 attend (geometry-light gating; full projection 아님).
  - FlashAttention / memory-efficient attention 커널 사용 (score 행렬 materialize 회피).
  - head 수 / channel을 작게 시작 (C=48~64).

profile 실패 시 hard fallback:
  - K cap을 4k~8k로 더 낮춤
  - view gating을 더 강하게 해 query별 camera/token 수 제한
  - 1.2m spatial lifting을 multi-rate로 낮춤(예: 10Hz + temporal hold)
  - 그래도 안 되면 local windowed / deformable attention을 v1로 당겨 사용
    (vanilla full attention 철학은 유지하되 Orin budget을 우선)

선택 (v1.5, budget 여유나 품질 개선 시):
  - local windowed cross-attention, deformable sampling을 더 정교화.
```

> 이 token budget(특히 downsampling·K cap·gating·FlashAttention)은 1.2m attention이 20Hz 안에 드는지의 핵심 변수다. 실제 수치는 Section 13의 Orin 프로파일로 검증한다.
>
> ⚠️ **20Hz 최대 리스크는 sparse head가 아니라 여기(1.2m attention)일 가능성이 높다.** Q=4,250 × K=8k~16k는 sparse decoder보다 큰 단일 블록이다. Section 13 프로파일에서 **이 stage를 1순위로 측정**하고, 초과 시 K cap 축소 / view gating 강화 / token downsampling 추가 / 필요 시 windowed·deformable attention을 v1로 당겨 단계적 축소한다.

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
-> Occupancy fine [N_kept] + heavy heads [N_heavy] (+ Surface Outputs head)
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

출처를 정확히 분리하면:

```text
Tesla 공개 그림/특허가 지지하는 것 (= 우리가 따르는 골격):
  t, t-1, t-2, t-3의 3D representation을 trajectory로 정렬(align) + concat(fuse)하고
  deconvolution으로 voxel을 만든다. 즉 "3D align + concat + deconv".

PanoOcc에서 가져온 구현 선택 (Tesla 확정 아님):
  fuse를 구체적으로 "concat 후 residual 3D conv"로 한다.
  (과거 3 + 현재 1 = 4 frame, coarse 50×50×16 = 40,000 voxel.)
```

즉 **"align + concat"은 Tesla 지지**, **"residual 3D conv로 fuse"는 PanoOcc 구현 선택**이다. 본 문서는 이 조합을 따른다. ViewFormer의 z-squeeze + temporal attention은 Tesla에서 벗어난 쪽이라 v1에서 폐기한다 (streaming memory만 학습 효율용으로 선택 차용).

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
  PanoOcc가 40k(단 A100 기준)에서 동작 -> 우리 34k는 그 envelope 안이라 Orin 후보. Z 유지로 vz까지 예측. (실제 Orin FPS는 Section 13 프로파일로 검증.)

-> v1 main temporal은 0.6m 3D 단일 stage (align + concat + 3D residual conv).
   1.2m 3D는 Orin 실측이 20Hz 예산을 넘을 때의 fallback로만 둔다 (품질↓·비용↓,
   단계만 1.2m로 내리면 되어 구조 변경 없이 후퇴 가능).
```

비용의 핵심은 해상도가 아니라 **operator**다.

```text
0.6m 3D = 34,000 voxel

full self-attention:  34,000^2 -> 금지 (ViewFormer가 무겁던 진짜 이유)
align + concat + depthwise/separable 3D conv:  voxel 수에 선형 -> Orin 후보(실측 필요)
history N=3 메모리:  100 x 34 x 10 x 64 x 2B ≈ 4.4MB/frame -> 무시 가능

ViewFormer가 3090에서 4 FPS인 것은 0.8m라서가 아니라 temporal attention 때문이다.
우리는 PanoOcc식 concat + 3D conv로 한다.
```

### 7.2 흐름 (3D align -> concat -> residual 3D conv -> coarse motion seed)

z-squeeze 없이 0.6m 3D feature를 그대로 trajectory 정렬해 concat한 뒤 residual 3D conv로 융합한다. 이 단계에서 나오는 주 산물은 `F_0.6_temporal`이다. 여기에 붙는 `O_motion_0.6`은 독립적인 5번째 head가 아니라 **Occupancy Flow Head 내부의 coarse branch**다. 즉 "motion 정보의 원천"은 0.6m temporal이고, "최종 flow output"은 `pred_heavy_mask` 위의 Occupancy Flow Head readout이다.

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

Coarse motion seed (Z 유지, Occupancy Flow Head 내부 coarse branch):
  O_motion_0.6 = Conv1x1x1(F_0.6_temporal)
  -> dynamic_seed, vx_seed, vy_seed, vz_seed, motion_uncertainty_seed per 0.6m voxel
  -> Section 9.2 Occupancy Flow Head가 이 seed를 0.3m pred_heavy voxel에
     sample/broadcast하고, 필요하면 F_0.3_sparse로 residual 보정한다.
  -> loss는 Section 9.2/19의 Flow Head loss 안에서 L_flow_coarse_seed로 관리한다.
```

### 7.3 temporal fusion operator (v1 / v1.5)

```text
권장 v1:
  ego-motion 3D warp
  concatenate current + aligned history (채널 stack)
  depthwise / separable 3D conv
  gated residual fusion
  3D history queue는 runtime temporal state로 사용
  (선택) ViewFormer streaming memory: 학습 시 과거 feature를 재계산하지 않고
         캐시 -> 학습 효율↑, 추론 latency 변화 없음
         (runtime 3D history queue와 다른 개념이다)

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
0.6m 3D:  100 x 34 x 10 = 34,000 voxel   <- 본 문서 temporal 위치 (PanoOcc 40k 이내, A100 기준이므로 Orin은 검증 필요)
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
  occupied / free / unknown / visibility + mixed_surface_risk logits (dense, 100 x 34 x 10)
  -> free vs unknown 구분의 dense source
  -> 동시에 0.3m sparse 프루닝을 guide

갈래 2) Sparse fine 가지: 2단 게이트로 0.3m sparse 생성
  게이트① @0.6m (parent gate, recall-first):
    coarse occupancy로 occupied / mixed-surface-risk / unknown / 저신뢰(uncertain) parent를 keep하고,
    **high-confidence free parent만 버린다** (Section 8.2.1 false-negative 방어와 일치).
    -> keep된 parent만 0.3m로 sparse 업샘플
    -> high-conf free 영역의 0.3m children만 "애초에 생성하지 않음" (272k 전체 materialize 회피)
    (주의: "free/unknown 버림"이 아니다. unknown/uncertain은 얇은 물체 보호를 위해 keep.)
  게이트② @0.3m (child prune):
    (실행 순서, 순환 방지)
    1) cheap child score head: 확장된 child마다 preliminary keep/free/surface/shell-tag score 계산
       (이건 게이트②용 가벼운 score이지, 최종 occupancy가 아니다)
    2) 이 score로 prune (recall-first): **high-confidence free child만 drop**.
       uncertain/unknown/thin/far/dynamic child는 keep/quota로 보호 (Section 8.2.1).
       + 표면 인접 free child 1겹을 free shell 후보로 추가 keep (near-surface 1-shell, 기본 ON)
       -> child_score_keep_candidates 먼저 확정
       -> prelim_free_shell_raw 계산 (= shell-tag score + near-surface evidence로 고른 후보)
       -> prelim_free_shell = prelim_free_shell_raw - child_score_keep_candidates
          (집합 겹침 방지용 extra index set. child_score_keep_candidates가 우선권을 가진다.)
       -> kept(=prelim_kept) index set 확정 =
          child_score_keep_candidates ⊎ prelim_free_shell
       -> near_surface_free_shell_tag = prelim_free_shell_raw ∩ kept
          (최종 pred_free_shell semantic은 이 tag로 정한다.
           prelim_free_shell은 index 추가분일 뿐 semantic source가 아니다.)
          (occupied/boundary/visibility_frontier/free_shell/free_kept/uncertain_kept 최종 label은
           3)의 O_occ_fine 이후 pred_*로 정해진다)
       (keep ~0.5는 soft budget 목표일 뿐, mandatory recall keep이 우선)
    3) 이후 O_occ_fine(갈래 B)이 [N_kept] 위에 occupied logit + surface-band auxiliary를 낸다
       (free_shell도 여기서는 낮은 p_occ일 때 fine_free_candidate가 될 뿐이다.
        planner에 free로 즉답하려면 Section 10의 observed-free 증거까지 필요하다.)
    -> F_0.3_sparse = prelim_kept
       (= child_score_keep_candidates ⊎ prelim_free_shell;
          prelim_free_shell은 겹침 제거된 near-surface free extra index)
```

주의: `prelim_free_shell`은 sparse index 중복 제거를 위한 "추가 keep 집합"이고,
`near_surface_free_shell_tag`는 최종 `pred_free_shell`을 정하는 semantic tag다.
따라서 near-surface free child가 이미 `child_score_keep_candidates`로 살아남았더라도
`near_surface_free_shell_tag`가 있으면 O_occ_fine 이후 `pred_free_shell`이 될 수 있다.
planner에 high-confidence free로 즉답하려면 여전히 child-level observed-free 증거가 필요하다(Section 10).
즉 free_shell 생성과 최종 occupancy는 순환이 아니라 score(빠름) -> prune -> final logit -> observed-free 검증 순서다.

주의: dense 0.6m feature(temporal/free-unknown용)는 그대로 둔다. 게이트는 그 dense를 자르는 게 아니라, **0.3m로 내려보내는 sparse fine 가지에만** 적용된다. 즉 "0.6m은 dense 유지" 원칙과 충돌하지 않는다.

0.6m을 dense로 두는 이유:

```text
0.6m dense = 100 x 34 x 10 = 34,000 voxel.
3D conv가 가볍고, 3D temporal(z 유지)과 coarse dense occupancy를
여기서 한 번에 처리할 수 있다 (PanoOcc coarse 40k voxel 이내).
반면 0.3m dense = 272,000 voxel -> 이 단계만 sparse로 피한다.
```

Volume output block은 `F_0.3_sparse`를 sparse source로 쓴다. 단 실행 대상은 다르다. Occupancy fine branch는 `[N_kept]` 전체에 붙고, Flow / Sub-Voxel Shape / Semantics는 O_occ_fine 이후 확정된 `pred_heavy_mask`의 `[N_heavy]`에서만 돈다. coarse dense feature를 별도 branch로 직접 쓰는 것은 Occupancy Head(free/unknown @0.6m)뿐이다.

### 8.2 2단 게이트의 두 가지 목적 (연산량 + 선명도)

프루닝은 단순 비용 절감이 아니다. 두 게이트가 **서로 다른 목적**을 가진다.

```text
게이트① @0.6m (parent gate)  -> 목적: 연산량 (단, recall-first)
  coarse occupancy로 high-confidence free parent만 버려
  그 free 영역의 0.3m children을 생성조차 안 한다.
  occupied / mixed-surface-risk / unknown / uncertain parent는 keep (얇은 물체 보호).
  기준: 0.6m coarse occupied 확률 + uncertainty + surface-risk prior
        (Section 8.2.1 recall-first/uncertainty keep).
  주의: 이 시점에는 O_occ_fine 전이라 pred_boundary가 아직 없다.
        parent gate의 mixed-surface-risk는 O_occ_coarse의 mixed_surface_risk_logit을 기본으로 쓰고,
        coarse p_occ gradient, depth/ray surface proximity, 2D edge/semantic prior,
        temporal dynamic cue 같은 cheap signal을 보조로 더한 recall 보호용 prior다.
        최종 pred_boundary는 Section 9에서 O_occ_fine 이후에만 정한다.
  high-confidence free parent =
    p_coarse_free 높음 AND p_coarse_occ 낮음 AND p_coarse_unknown 낮음 AND visibility 높음.
    visibility가 낮거나 unknown이 높으면 free로 drop하지 않고 keep/unknown 처리.

게이트② @0.3m (child prune)  -> 목적: 선명도 (anti-dilation) + recall 방어
  확장된 child 중 빈 child를 제거해 표면을 또렷하게 유지.
  단, 표면 인접 1겹 free child는 함께 keep한다 (near-surface 1-shell, 기본 ON).
  high-confidence free child =
    cheap free score 높음 AND cheap keep/surface score 낮음
    AND mandatory keep target/ quota에 걸리지 않음.
    이것은 drop 결정용 preliminary score이며, planner free 판정이 아니다.
  keep ratio ~0.5는 **soft budget target**일 뿐이고, **mandatory recall keep이 우선**한다.
    (어려운 장면 / 원거리·동적·얇은 물체 quota가 많으면 0.5를 넘는 게 정상.)
  false-negative 방어 (parent gate와 동일 원칙, Section 8.2.1):
    high-confidence free child만 drop. uncertain/unknown child는 keep.
    thin / far / dynamic child는 quota로 최소 keep 보장.
    -> parent를 살려도 child에서 얇은 물체를 자르면 똑같이 복구 불가이므로,
       gate②도 gate①과 같은 recall-first/quota 규칙을 적용한다.

  runtime worst-case guard (recall-first가 budget을 넘길 때, Orin 안전):
    복잡/occlusion 큰 장면은 keep이 0.5를 쉽게 초과한다. soft budget만으론 부족하므로:
      - class별 quota (occupied/thin/dynamic 우선, 나머지 후순위)
      - distance별 cap (원거리는 절대 cap, 근거리·planner corridor는 보호)
      - emergency fallback: N_kept hard cap 초과 시, 안전 우선순위(근거리>planner>dynamic>thin)
        낮은 것부터 0.3m fine 실행에서 제외하되, 제외된 voxel을 free로 간주하지 않는다.
        제외된 후보는 parent 0.6m coarse의 unknown/occupied-risk fallback으로 degrade한다.
        mandatory occupied/surface-band keep만으로도 cap을 넘으면 해당 frame은
        0.3m fine coverage를 일부 포기하고 conservative output(unknown/high-cost)을 내며 로그를 남긴다.
    -> "recall은 지키되, 터지면 품질을 떨어뜨리지 안전을 떨어뜨리지 않는다."
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
근거: Tesla 내부는 dense voxel grid를 평가한다(벽 앞 빈 공간도 voxel 단위로 평가됨).
      우리는 비용상 그 빈 공간을 0.6m coarse로만 다루는데, 표면 child만 남기면
      표면 바로 앞 free가 0.6m로만 해석되어, 2m 로봇이 좁은 통로를 "막힘"으로
      과소평가할 수 있다. 한정된 sparse 예산을 Tesla의 dense 평가에 가장 가깝게
      쓰는 길은 "장애물 바로 앞 빈 공간"을 0.3m로 살리는 것이다 (항법상 가장 중요).
      (이건 Tesla 동일 구현이 아니라, dense의 이점을 sparse에서 흉내 내는 Jetson 적응이다.)

동작: 게이트②에서 occupied/surface 후보 child + 그 표면에 인접한 free child 1겹을 함께 keep.
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

### 8.2.1 Pruning False Negative 방어 (v1 필수)

**프루닝 false negative가 이 설계의 가장 위험한 실패 모드다.** 게이트①이 0.6m parent를 high-confidence free로 버리거나, 게이트②가 0.3m child를 high-confidence free로 drop하면 그 위치는 뒤에서 복구하기 어렵다(프루닝은 비가역). 특히 coarse/cheap score가 얇은 물체(난간/막대/유리/전선)나 surface-band를 free로 오판하면 0.3m fine/head supervision 자체가 사라진다. 충돌 안전에 직결되므로, 보호장치를 v1.5로 미루지 않고 **v1부터 전부** 넣는다.

```text
v1 필수 (모두):
  1. recall-first gate:
     게이트①/② 임계값을 precision이 아니라 recall 기준으로 잡는다
     (occupied를 놓치는 것 << free를 더 keep하는 것).
  2. uncertainty keep:
     coarse/fine cheap score가 애매한(p~0.5) / unknown / 저신뢰 parent-child는 무조건 keep.
  3. category quota keep:
     아래 카테고리에 최소 keep 보장 (단순 top-k가 얇은/원거리/저신뢰를 먼저 자르는 것 방지)
       near-field occupied
       observed surface-band / boundary-band
       near-surface free shell candidate
       far-field low-confidence
       dynamic
       planner corridor
       thin-object / high-gradient
       random exploration
  4. GT-guided warmup:
     학습 초기엔 GT occupied와 observed surface-band parent/child를 강제 keep해
     gate가 recall을 먼저 배우게 한다
     (이후 점진적으로 자기 예측 기반으로 전환).
  5. soft / straight-through top-k:
     hard top-k는 gradient가 안 흐른다. soft top-k 또는 straight-through estimator로
     keep 결정에 gradient를 흘려 gate가 학습되게 한다.
  6. gate② mandatory keep target:
     positive keep = GT occupied
                   | observed surface-band
                   | near-surface observed free shell
                   | thin/far/dynamic/planner/unknown quota bucket
     negative drop = high-confidence observed free
                     AND not near-surface
                     AND not quota-protected
     이 target이 없으면 O_occ_fine과 L_surface_band가 실행되기 전에
     surface/boundary 후보가 잘려서 뒤 loss로 복구할 수 없다.
  7. gate② shell-tag supervision:
     near-surface observed free shell은 keep target이기도 하지만,
     `pred_free_shell`과 `pred_free_kept`를 가르는 semantic tag target이기도 하다.
     따라서 keep/drop loss와 별도로 L_gate2_shell_tag를 둔다.
     이 loss가 없으면 voxel은 살아남아도 shell tag가 사라져,
     표면 옆 free가 내부 free처럼 coarse_free_conf를 요구받을 수 있다.
```

> v1.5 이후엔 quota 비율 튜닝, per-class recall 목표, active learning 기반 hard-negative 보강 등으로 정교화한다. 단 위 1~7은 v1 출시 기준이다.

### 8.3 이전 버전과의 차이

```text
이전:
  dense feature를 0.3m까지 만들고, 4 head를 F_0.3 dense 위에 붙임

현재:
  0.6m까지 dense, 0.3m는 sparse deconv + 프루닝.
  free/unknown은 0.6m coarse dense, occupied/surface-band 후보는 0.3m sparse fine.
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

Volume output block은 `F_0.3_sparse`를 sparse source로 쓴다. 단 coarse dense feature를 별도 branch로 직접 쓰는 것은 Occupancy Head(free/unknown @0.6m)뿐이다. Flow / Sub-Voxel Shape / Semantics는 O_occ_fine 이후 확정된 `pred_heavy_mask`의 `[N_heavy]`에서만 실행된다.

**용어 고정 (구현 mask 일관성):**

순환 의존을 피하려고 **2단계(prelim → pred)** 로 분리한다. prelim은 gate② cheap score 산물(O_occ_fine 이전), pred는 O_occ_fine 이후의 최종 label이다.

먼저 임계값과 auxiliary score를 정의한다. 여기서 중요한 점은 `boundary`를 단순히 "p가 애매한 모든 voxel"로 두지 않는 것이다. p가 애매한 voxel에는 실제 표면 전이대도 있지만, occlusion/unknown/저신뢰 공간도 섞인다. 동시에 `p_occ`만 보고 free를 확정하면 안 된다. `p_surf`가 강하게 surface-band를 가리키는 voxel은 `p_occ`가 낮게 나와도 곧바로 free로 보내지 않고, 관측 free 증거가 없으면 boundary/uncertain 쪽으로 보호한다.

```text
threshold:
  p_occ   = sigmoid(O_occ_fine_occ)      # occupied 확률
  p_surf  = sigmoid(O_occ_fine_surface)  # near-surface / boundary-band auxiliary 확률

  tau_occ  : occupied 판정 (recall 우선; 낮게 잡아 occupied 놓침을 줄인다)
  tau_free : free 판정     (precision 우선; 낮게 잡아 free 오답을 줄인다).
             반드시 tau_free + eps <= tau_occ로 둔다(eps > 0).
             tau_free == tau_occ는 raw_occ/raw_free가 같은 경계값에서 겹칠 수 있으므로 invalid config다.
  tau_surf : boundary/surface-band 판정 (SDF band recall 우선)

  raw_occ_candidate  = (p_occ >= tau_occ)
  raw_free_candidate = (p_occ <= tau_free)
  raw_uncertain      = (tau_free < p_occ) AND (p_occ < tau_occ)
  surface_band_candidate = (p_surf >= tau_surf)

  invariant:
    raw_occ_candidate / raw_free_candidate / raw_uncertain은 서로 겹치지 않고,
    p_occ에 대해 complete partition이 되어야 한다.
    구현 시 assert(tau_free + eps <= tau_occ)를 둔다.

  # free false-positive 방어:
  # p_surf가 높은데 p_occ가 낮은 voxel은 얇은 표면/경계가 free로 잘못 빠진 경우일 수 있다.
  # 단 ray/depth가 해당 child를 실제로 지나간 high-conf observed-free이면 free를 허용한다.
  observed_free_child_evidence =
      observed_free_evidence_0.3(child)
      # 구현 시 pred label이 아니라 Section 10의 child-level ray/depth/free-space evidence를 사용한다.
      # 0.6m parent free는 mixed parent에서 너무 거칠 수 있으므로 surface_free_conflict 해제 조건으로 쓰지 않는다.

  surface_free_conflict =
      raw_free_candidate AND surface_band_candidate AND NOT observed_free_child_evidence

  fine_occ_candidate  = raw_occ_candidate
  fine_free_candidate = raw_free_candidate AND NOT surface_free_conflict
  fine_uncertain      = raw_uncertain OR surface_free_conflict

calibration 원칙 (구현 시 검증):
  occupied recall 우선 (놓치면 충돌 위험), free precision 우선 (잘못 free면 위험),
  boundary는 "uncertain 전체"가 아니라 near-surface evidence가 있는 transition band로 제한.
  p_surf가 높은 free conflict는 free로 확정하지 않고 boundary/uncertain 판정으로 넘긴다.
```

```text
[1단계: gate② cheap child score - O_occ_fine 이전. 여기서 sparse 텐서 index가 정해진다]
child_score_keep_candidates =
    cheap keep/surface score와 mandatory recall keep으로 살아남은 child
    # occupied/uncertain/thin/far/dynamic/unknown quota 포함

prelim_free_shell_raw =
    표면 인접 free child 1겹 후보 (cheap score 기반, near-surface 1-shell)
    # sparse index 추가 후보이면서, 최종 pred_free_shell semantic tag의 source다.

prelim_free_shell =
    prelim_free_shell_raw - child_score_keep_candidates
    # 집합 우선순위: child_score_keep_candidates가 먼저다.
    # 이것은 "추가로 살리는 near-surface free index"일 뿐 semantic label source가 아니다.
    # child_score_keep_candidates 안에 이미 들어간 near-surface free child도
    # 아래 near_surface_free_shell_tag로 shell semantic을 유지한다.

kept (= prelim_kept) = child_score_keep_candidates ⊎ prelim_free_shell
                       # = F_0.3_sparse 텐서에 실제로 할당되는 voxel 집합 ([N_kept])
fine_occ_mask     = kept    # O_occ_fine(갈래 B)이 logit을 내는 대상 (kept 전체)

near_surface_free_shell_tag =
    prelim_free_shell_raw AND kept
    # 최종 pred_free_shell을 정하는 semantic tag.
    # sparse index 중복 제거 때문에 prelim_free_shell에서 빠졌더라도,
    # prelim_free_shell_raw에 있었고 kept에 남아 있으면 shell로 취급한다.

[2단계: O_occ_fine 실행 후 - 그 logit/aux score로 최종 label을 붙인다.
 kept를 6개로 완전 분할(disjoint·complete)]

pred_occupied_surface = fine_occ_candidate
  # 이름은 surface-centric runtime을 따른 것이지만, 의미는 "p_occ가 높은 occupied candidate"다.
  # solid interior 후보가 섞일 수 있으며, 실제 노출 표면/face는 Sub-Voxel Shape의 exposed-face mask에서 가른다.

pred_free  = fine_free_candidate 를 near-surface shell semantic으로 둘로 나눈다:
  pred_free_shell      = fine_free_candidate AND near_surface_free_shell_tag
                         # 표면 옆 free. child_score_keep_candidates에서 온 voxel이어도
                         # near_surface_free_shell_tag가 있으면 shell로 본다.
  pred_free_kept       = fine_free_candidate AND NOT near_surface_free_shell_tag
                         # 내부/일반 free. Section 10에서 coarse_free_conf까지 요구한다.

fine_uncertain 영역은 다시 셋으로 나눈다:
  physical_surface_evidence =
      surface_band_candidate
      OR face_sign_change_surface_evidence
      OR observed_surface_proximity

  visibility_frontier_evidence =
      unknown_frontier_evidence
      AND NOT physical_surface_evidence

  pred_boundary        = fine_uncertain AND physical_surface_evidence
                         # 실제 surface/occupied-free 전이대. Shape/Queryable 정밀 경로 후보.
                         # surface_free_conflict는 surface_band_candidate를 포함하므로
                         # 관측 free 증거가 없는 "p_occ free + p_surf high" voxel을 여기서 구제한다.

  pred_visibility_frontier =
      fine_uncertain AND visibility_frontier_evidence
      # occlusion/unknown 경계. priority/query budget에는 중요하지만,
      # 실제 표면이라고 보지 않으므로 Flow/Shape/Semantics heavy head는 실행하지 않는다.

  pred_uncertain_kept  = fine_uncertain AND NOT physical_surface_evidence
                                      AND NOT visibility_frontier_evidence
                         # 그냥 p가 애매한 voxel. unknown/occlusion/저신뢰가 섞일 수 있으므로
                         # heavy head를 실행하지 않고 coarse fallback으로 보수 처리.

evidence 계산 규칙 (순환 방지):
  - O_occ_fine의 p_occ/p_surf, 0.6m coarse state, 관측 ray/depth로 계산한다.
  - pred_boundary라는 최종 label을 다시 입력으로 쓰지 않는다.

face_sign_change_surface_evidence는 "voxel label을 바꾸는 규칙"이 아니라 face-level signal이다.
  - pred_boundary 후보를 고르는 보조 evidence / exposed-face priority에만 사용한다.
  - occupied/free voxel을 boundary로 재분류하지 않는다.
  - kept neighbor는 base state(p_occ >= tau_occ, p_occ <= tau_free, uncertain)로 본다.
  - pruned neighbor는 0.6m coarse occupied/free/unknown fallback state로 본다.
  - occupied 쪽과 observed-free 쪽이 같은 face 주변에 있을 때만 physical surface transition evidence로 세운다.
  - unknown/occluded 쪽과 맞닿은 경우는 physical boundary가 아니라 unknown_frontier_evidence로 보낸다.
  - 학습 GT 기준(surface-band): SDF |d| < 0.5·voxel band, observed surface 근처만 신뢰.

unknown_frontier_evidence:
  - kept/pruned neighbor 또는 0.6m coarse fallback이 unknown/occluded이고,
    해당 voxel이 fine_uncertain인 경우에만 세운다.
  - active/priority mask, planner 추가 query, conservative fallback을 위한 신호다.
  - Shape/Flow/Semantics supervision 대상 surface로 쓰지 않는다.

observed_surface_proximity는 다음 중 하나다:
  - 추론: depth/ray consistency상 voxel center 또는 face가 surface hit 근처(<= 0.5 voxel)
  - 학습: SDF/mesh surface band + visibility가 있는 관측 표면
  - 단순히 camera frustum 안에 있다는 사실만으로는 surface proximity가 아니다.

pred_heavy_mask       = pred_occupied_surface | pred_boundary
                        # Flow/Shape/Semantics 실행 대상.
                        # pred_visibility_frontier / pred_uncertain_kept는 제외.

검증: prelim_kept = child_score_keep_candidates ⊎ prelim_free_shell
      kept = pred_occupied_surface ⊎ pred_boundary ⊎ pred_visibility_frontier
             ⊎ pred_free_shell ⊎ pred_free_kept ⊎ pred_uncertain_kept
      (빠짐없이, 겹침없이)
```

주의(free_shell/frontier 제외 조건): boundary/heavy에서 제외되는 것은 **pred_free_shell / pred_free_kept**(즉 fine이 free로 본 voxel), **pred_visibility_frontier**(occlusion/unknown 경계), **pred_uncertain_kept**(surface evidence 없는 애매한 voxel)이다. `near_surface_free_shell_tag`가 있더라도 O_occ_fine 결과가 occupied이면 `pred_occupied_surface`, uncertain이면서 physical surface evidence가 있으면 `pred_boundary`, unknown-frontier evidence만 있으면 `pred_visibility_frontier`, evidence가 없으면 `pred_uncertain_kept`가 된다. 반대로 child_score_keep_candidates에서 온 voxel이라도 fine이 free이고 `near_surface_free_shell_tag`가 있으면 `pred_free_shell`이다.

즉 **index 집합(kept)** 은 cheap score로 먼저 정하고, **pred_* 6개 label** 과 **pred_heavy_mask** 는 O_occ_fine 이후에 정한다(순환 없음). 무거운 head(Flow/Shape/Semantics)는 pred_heavy_mask에서만 동작하고, fine이 free로 본 voxel(pred_free_shell / pred_free_kept), visibility frontier(pred_visibility_frontier), surface evidence 없는 uncertain voxel(pred_uncertain_kept)은 제외된다. 문서 전체에서 이 명명으로 통일한다.

**실행/출력 텐서 규약 (구현 모순 방지):**

```text
fine occupancy (갈래 B): [N_kept] x 2
  1) occupied logit: occupied/free/uncertain을 나누는 주 logit.
     "free"는 별도 free logit이 아니라 낮은 p_occ(=fine_free_candidate)로 해석된다.
     (free label = low p_occ + observed free 판정)
  2) surface-band auxiliary logit: boundary/near-surface evidence용.
     별도 volume head가 아니라 Occupancy Head fine 갈래의 보조 채널이다.
Flow / Sub-Voxel Shape / 3D Semantics: [N_heavy] 위에서 실행/출력
  -> 결과가 kept index 공간이 필요하면 scatter([N_heavy] -> [N_kept], free_shell은 빈 값/0).
즉 "[N_kept] 출력"이라고 쓰면 안 되는 head가 있다(heavy head는 [N_heavy]).
```

**실행 mask ≠ loss/supervision mask (head별로 분리):**

```text
실행 mask (forward 도는 곳): 셋 다 pred_heavy_mask (= pred_occupied_surface | pred_boundary)
loss/supervision mask (학습 신호 주는 곳)는 head마다 다르다:
  Sub-Voxel Shape : pred_heavy_mask 전체 (occupied + boundary 모두 의미 있음 - 표면/전이대 형상)
  Occupancy Flow  : valid_flow_mask = confident occupied/dynamic GT 중심
                    (pred_boundary 전체엔 약하게/ignore)
  3D Semantics    : valid_semantic_mask = confident occupied/surface GT 중심
                    (pred_boundary 전체에 주면 noisy -> 약하게/ignore)
-> boundary는 surface evidence가 있는 전이대라 shape엔 유용하지만 flow/semantic엔 noisy하므로 supervision을 제한한다.

deployment에서 Orin 예산이 부족하면 실행 mask도 head별로 더 좁힐 수 있다:
  shape_exec_mask = pred_heavy_mask
  flow_exec_mask  = pred_occupied_surface | dynamic_boundary_subset
  sem_exec_mask   = pred_occupied_surface | confident_surface_boundary_subset
단 v1 학습/디버깅 기준은 pred_heavy_mask 공통 실행으로 두고, loss mask만 먼저 분리한다.
```

**실행 순서 (head는 병렬이 아니라 이 순서다):**

```text
1. gate② cheap child score          -> prelim_kept 확정
                                        (= child_score_keep_candidates ⊎ prelim_free_shell)
2. O_occ_fine (갈래 B) on prelim_kept -> occupied logit + surface-band auxiliary (free_shell 포함)
3. pred label 확정 (2의 logit 기준, 6-way):
   pred_occupied_surface / pred_boundary / pred_visibility_frontier
   / pred_free_shell / pred_free_kept / pred_uncertain_kept
   pred_heavy_mask = pred_occupied_surface | pred_boundary
     (pred_visibility_frontier / pred_free_shell / pred_free_kept / pred_uncertain_kept 제외)
4. Flow / Sub-Voxel Shape / 3D Semantics on pred_heavy_mask ([N_heavy])
즉 heavy head는 O_occ_fine 이후에 mask가 정해지므로 4개 head가 완전 병렬은 아니다.
(아래 다이어그램은 source 의존을 보여줄 뿐, 실행 순서는 위와 같다.)
```

```text
Coarse dense (0.6m)
└── Occupancy Head (free / unknown / visibility 갈래)

F_0.3_sparse (prelim_kept candidate voxels)
├── Occupancy Head fine branch ([N_kept], occupied logit + surface-band auxiliary)
└── O_occ_fine 이후 pred_heavy_mask 확정
    ├── Occupancy Flow Head ([N_heavy], 0.6m motion seed readout/refine)
    ├── Sub-Voxel Shape Information Head ([N_heavy])
    └── 3D Semantics Head ([N_heavy])

(Surface Outputs head는 별도 브랜치 - Section 11)
```

설계 원리: Flow / Sub-Voxel Shape / Semantics 세 head는 **surface/occupied 후보에서만 의미**가 있으므로 sparse가 자연스럽다(명확한 빈 공간엔 flow도 표면도 semantic도 없다). Occupancy Head만 "빈 공간의 free/unknown"이라는 추가 책임이 있어 coarse dense 갈래를 함께 갖는다. Sub-Voxel Shape Information Head는 pred_heavy_mask 중에서도 exposed face가 있는 voxel만 골라 masked execution을 사용한다.

### 9.1 Occupancy Head (2-갈래)

Occupancy Head는 두 갈래로 나뉜다. free/unknown은 빈 공간의 저주파 속성이라 coarse dense에서, 30cm 후보 voxel의 occupied 확률과 surface-band evidence는 sparse fine에서 예측한다.

```text
갈래 A: Coarse Dense Occupancy / Visibility (@ 0.6m)
  input:  F_0.6_temporal (dense, 100 x 34 x 10 x C)
  output: occupied / free / unknown / visibility logits
          + mixed_surface_risk auxiliary logit
  역할:   빈 공간 free vs unknown 구분 (프루닝 / queryable의 기준)
  또한 이 occupancy score와 mixed_surface_risk가 0.6m->0.3m sparse deconv 프루닝을 guide한다.

갈래 B: Sparse Fine Occupancy + Surface Band (@ 0.3m)
  input:  F_0.3_sparse (kept voxels)
  output: 살아남은 voxel의 occupied logit + surface-band auxiliary logit
  역할:   30cm occupied/free/uncertain 분리 + boundary/surface-band evidence
```

권장 시작:

```text
# 갈래 A (dense @ 0.6m)
O_occ_coarse = Conv1x1x1(F_0.6_temporal)
  -> 100 x 34 x 10 x 5
     channels = occupied_logit, observed_free_logit, unknown_logit,
                visibility_logit, mixed_surface_risk_logit
     occupied/free/unknown은 softmax class logits
     visibility는 독립 sigmoid logit (관측 가능성/신뢰도)
     mixed_surface_risk는 독립 sigmoid auxiliary logit
       - parent 안에 observed surface-band / thin object / high-gradient geometry가 섞인 위험도
       - softmax occupied class가 free 쪽으로 흔들려도 parent gate가 child 생성을 보존하게 하는 recall prior
       - Occupancy Head 내부 보조 채널이지 별도 Volume Head가 아니다.

# 갈래 B (sparse @ 0.3m)
O_occ_fine = SparseConv1x1x1(F_0.3_sparse)
  -> [N_kept] x 2   (occupied logit, surface-band auxiliary logit)
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
v1에서는 `observed_free_logit`과 `unknown_logit`을 명시적으로 둔다. `occ + visibility`만으로
free/unknown을 유도하는 압축 표현은 calibration이 안정된 뒤의 v1.5 옵션이다. 특히
Queryable Interface의 `coarse_free_conf`, planner의 conservative fallback, 프루닝된 voxel의
free/unknown 분기는 observed-free와 unknown이 분리되어 있어야 모순 없이 동작한다.

### 9.2 Occupancy Flow Head

Occupancy Flow Head는 dynamic probability와 flow / velocity를 예측한다. 이 head는 두 branch를 갖는다. (1) `F_0.6_temporal`에서 `O_motion_0.6` coarse seed를 만들고, (2) 그 seed를 0.3m `pred_heavy_mask(= pred_occupied_surface | pred_boundary)` voxel 위로 readout/refine해 최종 `[N_heavy]` flow output을 낸다. `O_motion_0.6`은 독립 head가 아니라 Flow Head 내부 coarse branch다(pred_visibility_frontier/free_shell/free_kept/uncertain_kept는 flow 없음). Z를 유지하므로 vx, vy에 더해 **vz(수직 속도)** seed까지 유지된다.

```text
motion 정보 (정보의 출처):
  Section 7의 0.6m 3D temporal -> O_motion_0.6 coarse motion seed
  (dynamic_seed, vx_seed, vy_seed, vz_seed, uncertainty_seed) per 0.6m voxel
  (Z=10, 높이별 motion seed 유지)

Flow Head 내부 branch:
  branch A: O_motion_0.6 coarse seed
            dense @0.6m, Flow Head 소속, L_flow_coarse_seed로 supervise 가능
  branch B: O_flow final output
            sparse [N_heavy] @0.3m, pred_heavy_mask 위에서 seed + F_0.3_sparse를 readout/refine

출력 ([N_heavy], pred_heavy_mask = pred_occupied_surface | pred_boundary):
  0.3m occupied/surface-band voxel별로 flow 벡터를 붙인다
  (pred_visibility_frontier / free_shell / free_kept / uncertain_kept 제외).
  0.6m voxel motion seed를 그 안의 0.3m pred_heavy voxel로 sample/broadcast하고,
  F_0.3_sparse의 fine geometry와 concat해 O_flow를 readout한다.
  v1은 seed를 거의 그대로 쓰고, v1.5에서 small residual refinement를 추가할 수 있다.
```

왜 이렇게 나누나 (Tesla/PanoOcc 근거):

```text
Tesla/PanoOcc는 coarse 3D에서 motion을 추정하고 fine voxel에 펼친다.
flow의 motion granularity는 temporal을 한 해상도(0.6m)가 상한이다.
0.3m로 출력해도 기본 motion은 0.6m seed를 펼친 것이며, v1에서는 0.3m에서 새 motion을 추정하지 않는다.
강체(차량/보행자/카트)는 한 voxel column이 같이 움직이므로 0.6m로 충분.
3D 유지 덕에 높이별 다른 motion(예: 사람 다리/몸통, 포크)도 vz로 잡힌다.

-> 출력 격자 = 0.3m (다른 head와 일치),
   motion 해상도 = 0.6m 3D (Tesla/PanoOcc 수준, Orin 후보 - 프로파일 검증 대상).
   0.3m temporal은 하지 않는다.
```

권장 출력:

```text
O_flow:
  [N_heavy] x 5   (pred_heavy_mask = pred_occupied_surface | pred_boundary 위에서만 실행)
  -> 필요 시 kept index 공간으로 scatter (frontier / fine-free / uncertain_kept voxel은 flow 없음)

channels:
  dynamic_logit
  vx
  vy
  vz
  flow_uncertainty
```

flow는 실제 occupied/dynamic surface에서만 의미가 있으므로 프루닝된 빈 공간, pred_visibility_frontier, uncertain_kept에는 두지 않는다. 실행은 pred_heavy_mask에서 하더라도 supervise는 더 좁은 valid_flow_mask로 제한한다.

```text
occupied dynamic voxel:
  L_flow_fine_readout 적용

static occupied voxel:
  near-zero flow regularization

free / unknown / visibility_frontier / uncertain_kept voxel:
  flow loss 약하게 또는 ignore

coarse 0.6m dynamic voxel:
  L_flow_coarse_seed 적용 가능
  단 unknown/occluded/저신뢰 parent는 confidence mask로 ignore
```

motion 해상도 한계 (명시):

```text
v1: motion 해상도 상한 = 0.6m (3D, vx/vy/vz seed).
    0.3m flow output은 0.6m coarse motion seed를 pred_heavy voxel에 매핑/readout한 것이다.
    0.3m에서 새로운 motion 정보가 생기지 않는다.

v1.5 (필요 시): F_0.3_sparse 기반 small residual flow head를 추가해
    0.3m 수준 motion을 보정한다 (full 0.3m temporal은 여전히 금지).
```

### 9.3 Sub-Voxel Shape Information Head

Sub-Voxel Shape Information Head는 30cm voxel 내부의 local shape를 표현한다.

Occupancy Head가 말하는 것:

```text
이 30cm voxel은 occupied / free / unknown / surface-band / uncertain인가?
```

Sub-Voxel Shape Information Head가 말하는 것:

```text
occupied 또는 surface-band boundary voxel 내부에서
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
  pred_heavy voxel의 6-neighbor 중
  free 또는 unknown neighbor와 맞닿은 face

camera visibility:
  query priority와 confidence를 조정하는 보조 신호
```

기본 exposed-face 계산 (6-way pred label 기준):

이 루프는 **pred_heavy_mask(= pred_occupied_surface | pred_boundary)** voxel 위에서만 돈다. neighbor는 6-way pred label(pred_occupied_surface / pred_boundary / pred_visibility_frontier / pred_free_shell / pred_free_kept / pred_uncertain_kept) 또는 pruned(coarse fallback)로 판정한다. 핵심은 "neighbor가 occupied solid인가"이지 단순 "kept인가"가 아니다 — pred_free_shell/pred_free_kept(=fine이 free로 본 kept)와 맞닿는 면은 **진짜 노출면**이고, pred_visibility_frontier/pred_uncertain_kept와 맞닿는 면은 unknown/occlusion 후보로 다룬다.

```text
for each v in pred_heavy_mask:           # pred_occupied_surface | pred_boundary
  for face in [+x, -x, +y, -y, +z, -z]:
    n = neighbor voxel adjacent to face
    label(n) =
      pred_occupied_surface              -> not exposed (occupied 표면끼리 맞닿음)
      pred_boundary                      -> transition candidate
                                            (boundary끼리는 기본 not exposed,
                                             occupied->boundary face는 face_sign_change/visibility가
                                             있으면 exposed-candidate)
      pred_visibility_frontier           -> exposed-to-unknown candidate
      pred_free_shell / pred_free_kept   -> exposed candidate (fine이 free로 본 kept)
      pred_uncertain_kept                -> exposed-to-unknown candidate
      pruned                             -> exposed candidate (부모 0.6m coarse로 판정)
```

중요: **프루닝됨 = "fine 30cm 표면 증거 없음"이지 곧 free가 아니다.** 프루닝된 voxel에는 빈 공간뿐 아니라 **물체 내부(표면 뒤 꽉 찬 부분)** 도 섞인다. exposed candidate의 실제 free/unknown/uncertain 여부는 아래로 확정한다(neighbor가 kept면 그 pred label, pruned면 0.6m coarse).

```text
n = pred_free_shell / pred_free_kept (high-conf observed free) -> exposed-to-free
n = pred_free_shell / pred_free_kept (저신뢰/미관측)            -> exposed-candidate (uncertain)
n = pred_visibility_frontier                                  -> exposed-to-unknown / priority-only
n = pred_uncertain_kept                                        -> exposed-to-unknown candidate
n = pred_boundary + face_sign_change/visibility outward        -> exposed-candidate
n = 프루닝 + 부모 coarse free      -> exposed-to-free (진짜 노출면)
n = 프루닝 + 부모 coarse unknown   -> exposed-to-unknown (occlusion 경계)
n = 프루닝 + 부모 coarse occupied  -> 내부 경계일 수 있음, 노출면 아님 (보수적)
```

이 head는 **pred_heavy_mask(= pred_occupied_surface | pred_boundary)** 위에서만 동작한다(fine-free / uncertain_kept voxel 제외). 모든 voxel을 dense하게 도는 비용이 처음부터 없다.

권장 출력:

```text
O_shape:  [N_heavy] x channels   (pred_heavy_mask 위; 필요 시 kept index로 scatter)
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
  [N_heavy] x N_class   (pred_heavy_mask = pred_occupied_surface | pred_boundary 위에서만 실행)
  -> 필요 시 kept index 공간으로 scatter
     (pred_visibility_frontier / pred_free_shell / pred_free_kept / pred_uncertain_kept voxel은 semantic 없음)
```

semantic은 실제 occupied/surface voxel에서만 의미가 있으므로 pred_heavy_mask 위에서 실행하되, loss는 더 좁은 `valid_semantic_mask`에만 준다. `pred_boundary` 전체에 강한 semantic label을 주면 noisy해질 수 있으므로, confident occupied/surface GT 또는 관측된 surface-near voxel 중심으로 supervise한다. free / unknown / pred_visibility_frontier / pred_uncertain_kept 영역에 semantic label을 강제로 주지 않는다.

---

## 10. Queryable Occupancy Interface (occupancy 전용)

Queryable output은 별도 volume head가 아니다. sparse fine candidate feature(F_0.3_sparse=prelim_kept) / shape code와 coarse dense occupancy를 사용해 임의 좌표를 평가하는 interface다. v1에서는 두 의미를 분리한다. (1) `surface_shell_prob`: 표면 shell 근처의 연속 occupancy 확률, (2) `collision_state`: planner가 쓰는 free/unknown/occupied-risk 보수 상태다. `surface_shell_prob`를 곧바로 collision probability로 해석하면 안 된다.

**Tesla와의 의도적 차이 (occupancy 전용):** Tesla AI Day 그림의 Queryable Outputs는 MLP가 **두 개** — ① Continuous Occupancy Probability, ② Continuous 3D Semantics — 로 명시적으로 분리되어 있다. 본 시스템은 **임의 좌표의 연속 semantic 질의가 필요 없어** Queryable MLP를 surface-shell occupancy용 **하나만** 둔다. semantic은 voxel-level **3D Semantics Head(Section 9.4)** 에서만 제공한다(연속 좌표 질의는 지원하지 않음). planner collision 판정은 MLP 단독이 아니라 `collision_state` fallback 규칙을 함께 쓴다. 필요해지면 v1.5에서 Tesla처럼 semantic MLP를 추가할 수 있다.

```text
query point x, y, z
-> 30cm voxel index 찾기
-> 이 voxel의 분류는? (4-way runtime 분기; 내부 pred label은 6-way)

	  (A) pred_heavy_mask (= pred_occupied_surface | pred_boundary):
	        정밀 경로. local coord u + sample F_0.3_sparse + shape_code
	        -> small Queryable MLP -> piecewise-continuous surface_shell_prob / uncertainty (semantic 없음)
	        -> collision_state는 surface_shell_prob + coarse/visibility + solid/occlusion fallback으로 별도 결정

  (B) kept이지만 fine이 free로 본 칸 (pred_free_shell | pred_free_kept):
        fine-level에서 free로 "예측"된 칸 (확정 아님 - 추론 시 게이트/모델 산물이라 false-free 가능).
        - high_conf_observed_free(아래 정의) 이면 -> free로 즉답 (MLP 생략).
        - 그 외(저신뢰/미관측) -> 부모 0.6m coarse로 fallback (unknown/uncertain 판정, 분기 C/D 규칙).
        (원하면 인접 occupied 표면까지 거리 기반의 아주 싼 occupancy interpolation만.)

  (C) kept이지만 pred_visibility_frontier 또는 pred_uncertain_kept:
        visibility frontier 또는 surface evidence가 없는 uncertain kept. 정밀 MLP 생략.
        부모 0.6m coarse + visibility로 unknown/uncertain/occupied-risk를 보수 판정.

	  (D) pruned (kept 아님):
	        부모 0.6m coarse occupancy를 읽어 보수적 판정 (MLP 생략):
          high-confidence observed free -> free
          unknown / occluded / occupied / uncertain -> consumer mode에 따라 (아래)
```

**high_conf_observed_free 정의** (planner가 추측 free를 관측 free로 오인하지 않게). pred_free_kept(내부 free)와 pred_free_shell(표면 옆 free)은 요구 증거가 다르다. `pred_free_shell` 여부는 sparse index source(`prelim_free_shell`에서 왔는지)가 아니라 `near_surface_free_shell_tag`로 정한다. free_shell의 0.6m parent는 표면을 포함해 coarse occupied/uncertain이기 쉬우므로, coarse_free를 AND로 요구하면 free_shell이 항상 막혀 좁은 통로 목적이 무력화된다. 대신 free_shell에는 **child-level observed-free evidence**를 요구한다.

```text
공통: fine_free_candidate (p_occ <= tau_free)
      AND NOT unknown/occluded frontier

pred_free_kept (내부 free):
      + observed_free_evidence_0.6(parent)
      + coarse_free_conf (부모 0.6m도 free에 high-confidence 동의)
      # 내부 free는 coarse도 free여야 안전하다.

pred_free_shell (표면 옆 free):
      + observed_free_evidence_0.3(child)
      + NOT solid_interior_veto
      # 이유: free_shell의 parent는 표면 포함이라 coarse occupied/uncertain일 수 있다.
      #       parent가 occupied라는 이유만으로 막으면 free_shell이 항상 막힌다.
      #       대신 해당 child 자체를 ray/depth가 지나갔다는 증거가 있어야 한다.
      #       veto는 "parent occupied"가 아니라 "해당 child가 표면 뒤 solid interior/occluded 쪽"이라는
      #       증거일 때만 건다.

solid_interior_veto 예:
  - coarse가 high-conf occupied이고, 해당 child를 통과하는 observed-free ray가 없음
  - depth hit/surface 뒤쪽 child라 occluded interior로 보임
  - visibility head가 unknown/occluded로 강하게 봄
  - dynamic/static instance volume 내부로 추정됨

-> 위 조건을 만족하면 planner에 free 즉답, 아니면 unknown/uncertain fallback (보수적).
```

추론 시 observed-free 신호 출처 (Section 19는 offline GT용이므로 추론용을 따로 명시):
```text
- observed_free_evidence_0.6(parent):
    coarse visibility/free-space head (0.6m, 매 frame 추론)의 observed-free 신호.
    pred_free_kept / pruned fallback처럼 0.6m 단위로 답해도 되는 곳에 사용한다.

- observed_free_evidence_0.3(child):
    v1 기본: Section 4의 tiny depth/free-space evidence head로 만든
    low-res depth/range confidence를 현재 camera ray에 back-project해,
    해당 0.3m child voxel을 ray가 통과했거나 surface hit 앞쪽 free space로 검증한 경우.
    v1.5 보강: VIO landmark ray(Section 20.E), stereo/temporal depth, planner corridor ray cache.
    pred_free_shell에는 이 child-level 증거가 필수다.
    이 head가 꺼져 있거나 confidence가 낮으면 pred_free_shell도 free 즉답 금지.

- camera frustum evidence는 충분조건이 아니다.
  FOV 안에 들어온 것은 "볼 수 있었을 가능성"일 뿐 free 증거가 아니므로,
  visibility confidence를 낮추거나 ray/depth 검증 후보를 고르는 약한 prior로만 쓴다.
-> observed-free 증거가 없으면 NOT observed로 보수적 처리.
```

즉 **정밀 MLP는 (A) pred_heavy_mask(occupied/boundary)에서만** 돈다. (B) kept-but-fine-free(pred_free_shell | pred_free_kept)는 **high-conf observed free일 때만 free 즉답, 그 외엔 coarse fallback**(MLP 생략)이고, (C) pred_visibility_frontier/pred_uncertain_kept와 (D) pruned는 coarse/visibility 기반 보수 판정이다. (kept를 6개 pred_* label로 완전 분할했으므로 모든 query가 정확히 한 분기에 속한다.)

프루닝된 칸이나 pred_visibility_frontier/pred_uncertain_kept가 occupied/uncertain/unknown일 때의 반환은 **소비 모드(consumer mode)** 로 일원화한다(같은 내부 상태를 모드별로 다르게 표현).

```text
planner collision mode:
  collision_state == observed_free만 통과 가능.
  unknown / occluded / occupied-risk / uncertain은 conservative occupied 또는 high-cost로 취급.
  surface_shell_prob가 낮아도 solid interior/occluded/pruned occupied-risk면 통과 가능으로 보지 않는다.

map / query probability mode:
  unknown은 unknown으로 보존하고, occupied-risk는 높은 점유 확률/불확실성으로 반환.
```

핵심: **프루닝됨, pred_visibility_frontier, pred_uncertain_kept는 곧 free가 아니다.** 여기에는 빈 공간, occlusion, 물체 내부, 단순 모델 저신뢰가 섞여 있으므로, coarse/visibility가 **확실히 observed free**라고 동의할 때만 free로 답한다. occupied/uncertain이면 위 모드 규칙을 따른다(절대 free 아님). 이렇게 하면 차량 내부 등 solid interior를 free로 오답하는 충돌 위험을 막는다. 정밀 MLP는 분기 (A) kept occupied/boundary에서만 돈다.

입력 (분기 A에서만):

```text
F_query (sparse-aware, dense trilinear 아님):
  F_0.3_sparse는 좌표가 듬성듬성이라 일반 trilinear는 성립하지 않는다
  (주변 8 corner가 모두 kept라는 보장이 없음). 따라서 규칙을 고정한다:

  v1 (기본): query를 분기 (A) voxel의 중심으로 스냅 -> 그 kept voxel feature 사용.
             내부 위치 정밀화는 shape_code/face_surface_offset(Sub-Voxel Shape)로.
  대안:      sparse hash로 8-neighbor 조회, 존재하는 corner만 가중평균,
             없는 corner는 zero + 그 corner는 unknown 처리 (가중치 0).
  -> 어느 쪽이든 "dense trilinear"는 쓰지 않는다. (분기 B/C는 애초에 MLP 미실행)

  연속성 주의: v1 center-snap은 voxel 내부에선 local coord로 연속이지만 voxel 경계에서
    feature가 튈 수 있다 -> 출력은 "continuous"가 아니라 **piecewise-continuous**다.
    v1.5 개선 옵션(택1): boundary consistency loss / sparse 8-neighbor interpolation /
    SDF consistency. 매끈한 연속성이 필요하면 이 중 하나를 도입한다.

shape_context:
  shape_code
  exposed_face_mask
  face_surface_offset
  local_normal
  shape_uncertainty

local coordinate:
  u in [-0.5, 0.5]^3   (분기 A voxel 내부 상대좌표)
```

출력:

```text
surface_shell_prob(x)         # MLP branch A에서만 정밀. surface shell 근처 확률.
surface_shell_uncertainty(x)
collision_state(x)            # observed_free | surface_hit | occupied_risk | unknown | occluded | uncertain
collision_cost(x)             # planner용 scalar cost(optional)
(semantic 없음 - voxel-level 3D Semantics Head에서만 제공)
optional signed distance / boundary probability
```

주의: v1 Queryable MLP는 분기 (A) pred_heavy_mask에서만 실행되므로 `p_unknown`을 직접 예측하지 않는다.
unknown/occluded 여부는 분기 (C)/(D)의 coarse occupancy + visibility fallback에서 결정한다.
MLP의 uncertainty는 local shape/surface-shell confidence일 뿐, visibility unknown class가 아니다.

또한 v1의 Queryable MLP가 예측하는 occupancy는 Section 19의 GT convention과 맞춰 **surface-shell 중심 점유**다. camera-only로 닫힌 증거가 없는 물체 내부를 연속 MLP가 solid occupied로 채우지 않는다. solid interior / occluded interior / pruned occupied-risk는 `collision_state`에서 보수적으로 처리한다.

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

## 11. Surface Outputs Head (Tesla-style 별도 브랜치, Geometry 중심)

Tesla AI Day 2022 그림에서 Surface Outputs는 Volume Outputs와 **나란한 별도 출력 브랜치**다(Queryable Outputs와도 분리). 본 문서도 이를 따라, road surface geometry를 4개 volume head의 readout 후처리가 아니라 **독립 head**로 둔다.

```text
Tesla 그림의 출력 그룹:
  Volume Outputs   = 4 volume heads (Occupancy/Flow/Sub-Voxel Shape/Semantics)
  Surface Outputs  = 별도 브랜치  <- 이 Section
  Queryable Outputs = MLP 인터페이스 (Section 10)
```

**Geometry 중심 (Road Surface Semantics는 두지 않는다).** Tesla 그림의 Surface Outputs는 실제로 두 갈래 — `Road Surface Geometry`(높이/형상)와 `Road Surface Semantics`(차선/주행영역 같은 노면 평면 raster) — 다. 그러나 본 시스템은 **실내 및 실외(도로 아님)** 대상이라 차선/도로표시 같은 노면 semantic이 없다. 따라서 차선/주행구역 같은 Road Surface Semantics 갈래는 두지 않는다. 다만 planner가 바로 쓸 수 있도록 `traversability_cost` / `drop_risk` 같은 **geometry-derived risk**는 v1 Surface output에 포함한다. 이것은 semantic class가 아니라 z_surface, slope, step, uncertainty, ground absence에서 계산/학습되는 기하 기반 비용이다.

### 11.1 입력과 출력

Surface는 바닥 높이장(z_surface)으로, BEV column마다 하나의 매끄러운 값이다. 따라서 sparse 표면 voxel이 아니라 **0.6m dense feature만 입력으로 받는 self-contained dense BEV head**로 둔다(RoadBEV식 BEV elevation regression). sparse head(3D Semantics, Sub-Voxel Shape) 출력을 입력으로 끌어오지 않는다 — sparse->dense 역류와 실행 순서 의존을 피하기 위함이다.

```text
input:
  F_0.6_temporal (dense, 100 x 34 x 10 x C)  <- 이것만 사용 (self-contained)
  -> z-flatten: 높이축을 채널로 펼침 -> 100 x 34 x (10*C)
     (ZPool로 누르지 않는다: 바닥 높이를 예측하는 head가
      입력에서 높이 정보를 먼저 버리면 안 되기 때문 - L3)
  -> 1x1 conv로 (10*C) -> C_bev 압축 -> 100 x 34 x C_bev

output (dense BEV, 기본 0.6m 격자; 필요 시 0.3m로 upsample 가능):
  z_surface        : 바닥 높이 (연속 회귀라 0.3m bin에 안 묶임 -> sub-voxel vertical precision 가능; 실제 정밀도는 GT/depth 품질에 의존)
  valid            : 유효/관측 여부
  uncertainty      : 높이 불확실도
  slope / normal   : 경사
  step_height      : 연석 / 계단 높이
  traversability_cost : geometry-derived 통과 비용 (semantic class 아님)
  drop_risk / ground_absence : 구멍, 급격한 하강, no-ground 위험
```

주의: 0.6m feature만 입력으로 쓰는 Surface head를 0.3m BEV로 upsample해도 새로운 수평 detail이 생기지는 않는다. 0.3m output은 downstream grid 정렬과 sparse ground consistency를 위한 표현상 세분화이며, 실제 sharp edge authority는 0.3m pred_heavy voxel + Sub-Voxel Shape offset이다.

**범위 한정 (one-z-per-BEV의 한계와 분담):** z_surface는 BEV column당 **하나의 값**이라, 한 column에 "테이블 밑 + 바닥", 복층/mezzanine, overhang, 낮은 천장, 구멍+턱이 동시에 있으면 표현이 부족하다. 따라서 Surface head는 **"주 주행/보행 가능 surface"(주로 ground)만** 담당하도록 한정한다.

```text
역할 분담:
  Surface head (BEV, 1-z)     : 주 ground surface 높이/경사/단차 (traversability 기준면)
  3D Occupancy volume         : 머리 위 overhang / 천장 / 복층 / 테이블 밑 공간은
                                여기서 표현된다 (full 3D라 column 안 여러 층 표현 가능).
  -> "한 column에 여러 surface"는 surface head가 아니라 3D occupancy가 담당한다.

옵션 (v1.5, 실내 복잡 구조 시):
  - multi-surface head: column당 상위 K개 surface 높이 (바닥 + 머리 위 장애 하단 등)
  - traversability semantic: per-BEV "주행/보행 가능 / 불가" 라벨을 semantic class로 추가
    (v1의 geometry-derived traversability_cost보다 강한 supervision이 필요할 때)
```

### 11.2 다른 head와의 관계 (입력 의존이 아니라 분업 + consistency loss)

Surface head는 **자립**한다. 다른 head 출력을 입력으로 받지 않는다. 대신 세부 바닥 기하를 두 경로로 나눠 담당하고, 둘을 consistency loss로 묶는다.

```text
Surface Outputs Head (dense, 0.6m feature):
  매끄러운 z_surface 전역 높이장. 경사 / 완만한 ramp / 관측 안 된 곳 보간.
  장점: 어디든 값이 있음. 한계: 날카로운 모서리는 다소 뭉개짐.

sparse 0.3m ground pred_heavy voxel + Sub-Voxel Shape offset:
  연석 lip / 단차 edge 같은 날카로운 sub-30cm 기하.
  바닥/도로 surface는 occupied surface 또는 boundary라 pred_heavy_mask에 들어간다.
  offset/normal이 30cm voxel 내부의 실제 표면 위치까지 잡음.

결합:
  consistency loss로 두 경로 z를 일치시킴 (입력 의존 아님).
  planner는 "Surface head 전역 높이장 + 날카로운 곳은 sparse offset"을 융합해 사용.
```

핵심: z_surface 수직 정밀도는 회귀라 0.3m 격자에 갇히지 않는다. 단 가장 날카로운 연석/단차의 위치 정밀도는 sparse sub-voxel offset이 authority다(Surface head는 매끄러운 base).

### 11.3 학습

```text
GT:
  Section 19의 ground/road segmentation + 연속 표면(SDF/mesh)에서
  per-BEV-cell z_surface / slope / step / valid / traversability_cost / drop_risk 라벨 생성

loss:
  z_surface:
    L_surface_z = regression loss (surface_observed_mask: 관측된 cell만)
  valid / observed:
    L_surface_valid = BCE(valid, observed_ground_or_surface_valid)
    valid가 낮은 cell은 z/slope/step/cost loss를 약하게 주거나 ignore.
  uncertainty:
    heteroscedastic 또는 ensemble. z/slope/step/cost loss weight에 사용.
  slope / normal:
    L_surface_slope_or_normal = observed ground/road surface에서만 supervise.
  step_height:
    L_step_height = curb/step/drop edge 주변에 가중. 일반 평면에서는 near-zero regularization.
  traversability_cost:
    L_traversability_cost = geometry-derived cost regression 또는 ordinal CE.
    입력 GT는 slope, step_height, roughness, ground_absence, uncertainty, robot clearance rule에서 만든다.
    semantic class가 아니라 planner cost다.
  drop_risk / ground_absence:
    L_drop_risk = BCE 또는 focal BCE.
    구멍/급격한 하강/no-ground 라벨이 있는 observed cell에 강하게 주고,
    미관측/occluded cell은 unknown/ignore로 둔다.
  consistency loss:
    L_surface_sparse_consistency = Surface head z_surface <-> sparse ground pred_heavy voxel의
      sub-voxel surface 높이가 같은 (x,y)에서 일치하도록 (soft coupling)
    해상도 규칙:
      v1 기본은 Surface output을 0.3m BEV로 upsample한 뒤 0.3m sparse ground voxel과 비교.
      0.6m BEV 그대로 쓸 경우, 같은 0.6m parent 안의 0.3m ground child 4개는
      bilinear/nearest로 sample한 동일 parent surface와 비교하고 loss weight를 1/child_count로 나눈다.
      한 0.6m cell 안에 서로 다른 높이의 ground child가 공존하면(step/curb),
      Surface head는 매끄러운 base만 맞추고 sharp edge는 Sub-Voxel Shape 쪽 loss를 더 신뢰한다.
    적용 조건: observed ground/road surface이고, 그 위치의 sparse voxel이
      pred_heavy_mask 또는 GT_heavy(teacher-forcing 단계)에 들어간 경우만 건다.
      pred_visibility_frontier / pred_free_shell / pred_free_kept / pred_uncertain_kept는 consistency 대상이 아니다.
      원거리/저텍스처로 바닥 surface가 프루닝되었거나 uncertain으로 남은 곳은 비교 대상이 없으므로
      Surface head 단독(z_surface regression)으로만 학습/추론한다.
```

v1에서도 Tesla처럼 별도 head로 두되, 모듈은 가볍게(BEV conv 몇 layer) 시작한다. 3D Semantics에는 의존하지 않는다(Geometry 중심).

---

## 12. Runtime Masking and Multi-rate Scheduling

Active mask는 별도 prediction head가 아니라, Sub-Voxel Shape Head와 Queryable Interface를 어디에 얼마나 비싸게 실행할지 정하는 runtime priority map이다. 단 `pred_visibility_frontier`는 heavy head 실행 대상이 아니므로, shape용 priority와 planner/query용 priority를 분리한다.

priority score:

```text
S_shape (Sub-Voxel Shape expensive local query용):
  정의역: pred_heavy_mask 내부 voxel만
  S_shape =
    exposed_face_score
  + occupancy_uncertainty
  + shape_uncertainty
  + dynamic_score
  + planner_relevance
  + exposed_to_unknown_neighbor_score

  # pred_visibility_frontier voxel 자체는 shape 실행 대상이 아니다.
  # 다만 pred_heavy voxel의 neighbor가 visibility frontier이면
  # "unknown 쪽 노출면"으로 priority를 올릴 수 있다.

S_query (Queryable / planner high-detail readout용):
  정의역: pred_heavy_mask + pred_visibility_frontier + pred_uncertain_kept + pruned fallback 영역
  S_query =
    planner_relevance
  + occupancy_uncertainty
  + dynamic_score
  + visibility_frontier_score
  + near_field_collision_score
```

사용처:

```text
1. Sub-Voxel Shape Head의 expensive local query 위치 선택
   -> pred_heavy_mask 안에서만 S_shape top-k
2. Queryable MLP의 point budget 배분
   -> MLP 실행은 여전히 분기 (A) pred_heavy_mask에서만
   -> frontier/uncertain/pruned는 coarse fallback 또는 추가 관측 우선순위로 사용
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
  Occupancy fine head (occupied logit + surface-band auxiliary)
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
  (heavy local query 대상은 pred_heavy_mask voxel만 센다.
   pred_visibility_frontier / near-surface free shell / free_kept / uncertain_kept는 여기서 제외 -> K 예산은 surface 기준 유지)

Q_total:
  16k ~ 64k local query points

start:
  K = 4096
  average M_i = 4~8
```

---

## 13. 20 FPS Runtime Target

Jetson AGX Orin 기준 v1 **목표**는 20 FPS다. 단 이는 **검증 대상 목표**이며, PyTorch eager에서 미달이어도 정상이다 - Section 5.1 token budget과 아래 프로파일로 실제 도달성을 측정한 뒤, 미달 stage는 token/channel/K·Q/해상도 축소나 multi-rate로 내린다. (PanoOcc 등 레퍼런스 FPS는 A100 기준이라 Orin 보장 근거가 아님.)

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
  0.3m full-res / global temporal memory 금지 (0.6m coarse 3D voxel memory는 허용/핵심)
  0.6m/0.3m full global temporal attention 금지
  0.3m temporal 금지

Feature:
  1.2m -> 0.6m는 dense (temporal/coarse occupancy가 여기).
  coarse dense occupancy/visibility는 0.6m에서 dense.
  0.6m -> 0.3m만 sparse deconv + 프루닝.
  30cm dense volume feature는 만들지 않는다.
  C는 48 or 64로 시작한다.

Prune (2단 게이트):
  게이트① @0.6m parent (연산량, recall-first): high-conf free parent만 버림 (occupied/mixed-surface-risk/unknown/uncertain keep).
  게이트② @0.3m child (선명도/anti-dilation + recall): high-conf free child만 drop
    (uncertain/unknown/thin/far/dynamic keep/quota). keep ~0.5는 soft budget, recall keep 우선.
  False-negative 방어 (v1 필수, Section 8.2.1): recall-first gate + uncertainty keep
    + category quota + GT-guided warmup + soft/straight-through top-k.
    (coarse가 얇은 물체를 free로 오판하면 0.3m에서 복구 불가 -> recall 우선.)

Volume heads:
  1x1x1 (sparse) conv / light MLP 중심
  head 내부 large attention 금지

Sub-voxel shape:
  pred_heavy_mask([N_heavy]) voxel 위에서만 동작
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
  Occupancy fine [N_kept] + heavy heads [N_heavy] + Surface Outputs head
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
8. Tesla/PanoOcc-style single 3D temporal @ 0.6m
   (z 유지: 3D history queue + ego-motion align + concat + 3D residual conv, NOT attention)
9. Coarse 0.6m motion seed (vx/vy/vz seed) from F_0.6_temporal
10. Coarse dense occupancy / visibility head @ 0.6m (free/unknown, 프루닝 guide)
11. Sparse deconv + 2단 게이트 prune: 0.6m parent gate -> 0.3m child prune
12. Produce F_0.3_sparse fine candidate feature (= prelim_kept)
13. Occupancy Head fine branch ([N_kept], occupied logit + surface-band auxiliary)
14. pred_heavy_mask 확정
15. Occupancy Flow Head ([N_heavy], step 9 motion seed를 0.3m pred_heavy voxel로 readout/refine)
16. Sub-Voxel Shape Information Head ([N_heavy], exposed-face masked)
17. 3D Semantics Head ([N_heavy])
18. Surface Outputs head (별도 브랜치, dense BEV)
19. Build exposed-face / priority mask
20. Packed queryable MLP for selected local points (pruned/frontier 영역은 coarse로 보수 판정)
21. Planner readout
```

최종 구조를 한 줄로 정리하면:

```text
1.2m에서 이미지를 3D로 들어올리고,
0.6m 3D에서 Tesla/PanoOcc-style 시간 정렬(align+concat+3D conv) + coarse motion seed + coarse dense free/unknown을 만들고,
sparse deconv + 프루닝으로 occupied/surface-band 후보(+ 인접 free 1겹)를 0.3m까지 올린 뒤,
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
  z 유지(34,000 voxel, PanoOcc 40k 이내 - A100 기준) -> 높이별 motion / vz 가능
  Z=10층 유지(붕괴 없음), 여전히 최종 deconv 이전 latent
  align + concat + 3D conv (attention 아님) -> voxel 수에 선형 -> Orin 후보(검증 대상)
```

근거(출처 분리): **align + concat은 Tesla 공개 그림/특허 지지**(Spatial Frame Alignment -> Spatiotemporal Features stack -> deconv). **fuse를 "residual 3D conv"로 구체화한 것은 PanoOcc 구현 선택**(Tesla 확정 아님). 둘 다 attention이 아니다. ViewFormer의 z-squeeze + temporal attention은 높이 motion을 잃고 Tesla와 달라 폐기한다(streaming memory만 학습 효율용으로 선택). 단, raw deconv feature는 날것일 수 있으므로 temporal 직전 0.6m pre-temporal refinement를 둔다(BEVDet4D 교훈).

### 30cm Sparse Surface Feature + Coarse Dense Occupancy

final feature를 전부 coarse하게 유지하면 Tesla식 4 volume head와 거리가 생기고, 반대로 0.3m를 완전 dense로 만들면 272,000 voxel 3D deconv가 Orin 20Hz를 위협한다. 그래서 둘로 나눈다.

```text
free / unknown (빈 공간, 저주파):
  coarse dense @ 0.6m

occupied/surface-band 후보 + near-surface free shell (고주파 / 좁은 통로):
  sparse deconv + prune @ 0.3m
  -> F_0.3_sparse (= prelim_kept)
  -> Occupancy fine branch는 [N_kept]
  -> Flow/Shape/Semantics는 pred_heavy_mask 확정 후 [N_heavy]
```

이 분리는 Tesla 데모 **렌더**(표면 voxel만 보임)와 시각적으로 맞고, dense 0.3m 비용을 피하면서 4 volume head를 그대로 유지한다(Tesla 내부 dense 평가의 sparse 적응판). 정밀도가 필요한 표면을 30cm로 풀되, **표면 바로 앞 빈 공간 1겹(near-surface free shell)도 함께 30cm로 살린다**(L2, 기본 ON) — 2m 로봇의 좁은 통로 통과 판단을 위해. 그 외 넓은 빈 공간만 0.6m coarse로 둔다.

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
  + Occupancy Head fine branch ([N_kept], occupied logit + surface-band auxiliary)
  + 3D Semantics Head의 최소 class
  + tiny depth/free-space evidence head
    (v1 inference에도 유지. 끄면 pred_free_shell free 즉답 금지)

Stage 2:
  0.6m pre-temporal refinement 추가
  Tesla/PanoOcc-style single 3D temporal @ 0.6m 추가 (z 유지)
  runtime 3D history queue + ego-motion 3D warp(align) / concat / 3D residual conv
  ViewFormer식 streaming cache는 학습 효율용 선택 사항(temporal history와 다른 개념)

Stage 3:
  Occupancy Flow Head 추가
  (0.6m coarse motion seed vx/vy/vz -> 0.3m pred_heavy voxel readout/refine)
  dynamic probability / flow supervision 추가

Stage 4:
  Sub-Voxel Shape Information Head 추가
  exposed_face_mask / local offset / normal / uncertainty 학습

Stage 5:
  packed Queryable MLP 추가
  exposed-face / planner-aware query budget 추가

Stage 6:
  Surface Outputs head 추가 (z_surface / slope / step / uncertainty / traversability_cost / drop_risk)
```

v1에서 명시적으로 제외:

```text
0.6m or 0.3m image cross-attention
0.3m full global temporal attention
0.3m full-res / global temporal memory (0.6m coarse 3D voxel memory는 허용)
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
  coarse motion seed (vx/vy/vz seed; final flow output은 Section 9.2 head)
  (선택) ViewFormer streaming memory: 학습 효율용
  1.2m 3D fallback: Orin 20Hz 예산 초과 시에만

Coarse dense occupancy:
  occupied / free / unknown / visibility + mixed_surface_risk @ 0.6m
  (free/unknown 담당 + 프루닝 guide)

Sparse decoder:
  sparse deconv + 2단 게이트: 0.6m parent gate(연산) + 0.3m child prune(anti-dilation)
    둘 다 recall-first: high-conf free만 drop, uncertain/unknown/thin/far/dynamic keep·quota.
    keep ~0.5는 soft budget target일 뿐, mandatory recall keep이 우선.

Final feature:
  F_0.3_sparse = prelim_kept
                 (child_score_keep_candidates ⊎ prelim_free_shell; cheap score로 disjoint 확정)
  pred_heavy_mask = pred_occupied_surface | pred_boundary
    (O_occ_fine 이후; Flow/Shape/Semantics는 여기서만,
     pred_visibility_frontier / pred_free_shell / pred_free_kept / pred_uncertain_kept 제외)

Volume heads (= Tesla Volume Outputs):
  Occupancy Head (coarse dense free/unknown + sparse fine [N_kept] occupied/surface-band)
  Occupancy Flow Head ([N_heavy], 0.6m motion seed를 0.3m pred_heavy voxel로 readout/refine)
  Sub-Voxel Shape Information Head ([N_heavy], exposed-face masked)
  3D Semantics Head ([N_heavy])

Surface Outputs head (= Tesla Surface Outputs, 별도 브랜치):
  dense BEV: z_surface / slope / step / uncertainty / traversability_cost / drop_risk

Sub-voxel shape:
  exposed face mask
  face surface offset
  local surface normal
  shape uncertainty
  optional thinness
  shape code for queryable MLP

Queryable output (surface-shell occupancy + collision_state, 4-way 분기):
  (A) occupied/boundary: xyz + sampled F_0.3_sparse(sparse-aware) + shape code + local coord
                         -> Queryable MLP -> surface_shell_prob / uncertainty
                         -> collision_state는 coarse/visibility/solid fallback과 결합
  (B) kept-fine-free(pred_free_shell|pred_free_kept): high-conf observed면 free 즉답, 아니면 coarse fallback (MLP 생략)
  (C) pred_visibility_frontier / pred_uncertain_kept: 부모 0.6m coarse/visibility로 보수적 판정 (MLP 생략)
  (D) pruned:            부모 0.6m coarse로 보수적 판정 (MLP 생략, Section 10)
                         observed free만 free / unknown·occluded·occupied·uncertain은 consumer mode
  (semantic 질의는 없음; voxel-level 3D Semantics Head에서만 제공)

Runtime controls:
  exposed-face mask
  planner-aware priority mask
  packed query execution
```

가장 중요한 설계 원칙:

```text
30cm는 occupied/surface-band 후보와 near-surface free의 결과 해상도다. 단 내부 표현은 dense가 아니라 sparse다.
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
   - 이 문서 반영: **0.6m 3D 단일 temporal stage(z 유지, coarse motion seed)** + 1.2m->0.6m->0.3m sparse decoder + 프루닝 (keep ratio는 원거리 보호 위해 더 보수적으로)

4. BEVFormer / BEVDet4D
   - BEV feature ego-motion alignment와 temporal fusion 위치에 대한 교훈 참고
   - 이 문서에서는 pre-temporal refinement와 deconv 전 temporal alignment에 반영

5. SparseOcc / sparse execution examples
   - 모든 위치에 같은 비용을 쓰지 않고 중요한 위치만 더 비싼 연산을 쓰는 runtime idea 참고
   - 이 문서에서는 (1) sparse decoder의 프루닝 (2) Sub-Voxel Shape Head의 exposed-face / priority masked execution 두 곳에 반영

### Surface Geometry and Semantics

1. [RoadBEV](https://github.com/ztsrxh/RoadBEV)
   - BEV에서 road surface elevation을 직접 예측하는 기본 참고자료
   - 이 문서에서는 별도 Surface Outputs head가 0.6m dense feature에서 z_surface와 geometry-derived risk를 직접 회귀하는 방식에 반영 (Section 11)

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
              -> ground / road segmentation -> Surface Outputs head GT
                 (z_surface / slope / step / valid / traversability_cost / drop_risk 등)
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

### Occupancy GT Convention (surface shell vs solid fill)

camera-only reconstruction에서 가장 위험한 애매함은 "occupied"가 표면 shell인지, 물체 내부까지 채운 solid volume인지다. v1에서는 다음 규칙으로 고정한다.

```text
0.3m fine occupancy GT:
  occupied positive:
    observed surface shell / mesh-SDF surface band / 동적 instance의 관측된 occupied volume
    (기본은 surface-centric. 관측되지 않은 물체 내부를 자동 solid-fill하지 않는다.)

  observed free negative:
    camera ray가 실제 통과한 free space.

  unknown / ignore:
    occlusion 뒤, 물체 내부로 추정되지만 관측 ray가 없는 영역,
    TSDF confidence가 낮은 영역, 동적 분리가 불확실한 영역.

solid interior:
  watertight mesh, CAD/sim GT, LiDAR/멀티뷰로 충분히 닫힌 instance 등
  강한 근거가 있을 때만 occupied/solid interior로 쓴다.
  camera-only TSDF가 닫히지 않은 내부는 occupied positive가 아니라 unknown/ignore가 기본이다.

surface-band:
  observed SDF |d| < 0.5 voxel 또는 mesh surface 근처.
  Shape/Queryable supervise용이며, occlusion frontier와 분리한다.
```

이 규칙 때문에 `solid_interior_veto`는 "parent가 occupied라서 무조건 veto"가 아니라, 해당 child가 관측 표면 뒤 solid/occluded 쪽이라는 별도 증거가 있을 때만 켠다.

### Head별 GT 매핑

```text
Occupancy Head (coarse dense 갈래 @ 0.6m):
  step 4 voxelize + step 5 visibility
  -> 0.6m로 다운샘플한 free / unknown / occupied GT
  -> occupied/free/unknown은 3-class CE(or focal CE), visibility는 별도 BCE,
     mixed_surface_risk는 별도 BCE(or focal BCE)
     (free와 unknown은 softmax class에서 배타적, visibility는 관측 신뢰도 보조)

  0.6m mixed-cell 다운샘플 규칙 (parent 안의 0.3m child 8개):
    occupied:
      child 중 하나라도 occupied GT 또는 observed surface-band가 있으면 occupied/mixed-keep.
      parent gate가 얇은 물체를 free로 삭제하지 않게 occupied가 free보다 우선한다.

    mixed_surface_risk auxiliary:
      child 중 occupied/surface-band/thin/high-gradient/dynamic 후보와 observed free가 섞이면 positive.
      parent가 softmax상 occupied로 supervise되더라도 별도 risk logit으로 남긴다.
      이 logit은 parent gate의 mixed-surface-risk keep에 직접 쓰인다.
      모든 child가 clean observed free이면 negative.
      unknown/occluded가 섞인 경우는 risk positive 또는 ignore로 두되, free drop negative로 쓰지 않는다.

    free:
      8개 child가 모두 observed free이고,
      occupied/surface-band/unknown/occluded child가 하나도 없을 때만 free.

    unknown:
      occupied/surface-band는 없지만 unknown/occluded child가 하나라도 있으면 unknown.

    visibility:
      parent 안 child visibility의 max/mean 둘 다 기록할 수 있지만,
      high-confidence free 판단에는 "모든 relevant child가 observed free"라는 보수 조건을 쓴다.

    loss:
      L_occ_coarse_cls = occupied/free/unknown 3-class CE(or focal CE).
      L_occ_coarse_visibility = visibility BCE.
      mixed parent(occupied child + free child가 섞임)는 occupied class로 supervise하되,
      calibration용 auxiliary로 mixed/free ratio를 로그하거나 낮은 weight를 둘 수 있다.
      L_coarse_mixed_surface_risk는 parent gate recall 보호용이므로 FN penalty를 크게 둔다.

Occupancy Head (sparse fine 갈래 @ 0.3m) + 프루닝:
  step 4 voxelize의 occupied GT
  -> 0.6m / 0.3m scale에서 multi-scale occupancy supervision
  -> 프루닝 keep/drop 학습 (occupied GT만이 아니라 surface-band/free-shell recall도 keep 타깃)
  주의: GT에서 occupied인 voxel이 프루닝되면 영구 손실 -> recall 우선 keep ratio
  prelim_kept 전체 supervision (필수, free_shell만이 아님):
    fine occupancy는 N_kept = prelim_kept(= child_score_keep_candidates ⊎ prelim_free_shell) 전체에
    occupied logit + surface-band auxiliary logit을 내므로, free_shell뿐 아니라
    child_score_keep_candidates 중 free/uncertain인 voxel도 올바른 label 또는 ignore가 필요하다.
    occupied logit 규칙 (prelim_kept 전체에 적용):
      -> occupied GT voxel              = occupied (positive)
      -> observed free voxel            = free (negative)   # free_shell이든 child_keep이든 동일
      -> unknown / occluded / low-confidence voxel = ignore
    surface-band auxiliary 규칙:
      -> observed SDF |d| < 0.5·voxel 또는 mesh surface 근처 = surface-band positive
      -> 충분히 관측된 free/solid interior                 = surface-band negative
      -> unknown / occluded / 저신뢰 SDF                    = ignore
    near-surface observed free shell GT:
      -> observed free voxel 중 observed surface-band 또는 occupied GT의 6-neighbor 1겹
      -> 해당 child를 ray가 실제 통과했거나 surface hit 앞쪽 free space로 검증한 경우만 positive
      -> 단순 frustum 내부 / 미관측 / occluded는 shell positive 금지
      -> 이 target은 `prelim_free_shell_raw` / `near_surface_free_shell_tag`를 supervise한다.
         해당 child가 child_score_keep_candidates로도 살아남았으면 sparse index는 중복 추가하지 않지만,
         shell tag는 유지해야 하며 fine이 free일 때 `pred_free_shell`로 분류한다.
    (이게 없으면 fine head가 occupied 표면만 배우고 kept 내부 free/frontier/uncertain
     (=pred_free_shell/pred_free_kept/pred_visibility_frontier/pred_uncertain_kept)를 애매하게 둔다.)
  false-free 교정 (preliminary score 오분류):
    gate② preliminary score가 어떤 voxel을 free_shell로 분류했지만 GT가 occupied면,
    -> O_occ_fine은 그 voxel에 occupied(positive) label을 받는다 (free_shell이라고 free 강제 금지).
    -> 또한 학습 시 pred_heavy_mask를 "예측 mask" 대신 "GT heavy mask(GT occupied | observed surface-band)"로
       teacher-forcing해 Flow/Shape/Semantics를 보조 supervise한다 (mask 오류가
       heavy head 학습을 막지 않도록). 추론 시엔 예측 mask 사용.
  pred_heavy_mask GT->predicted 전환 스케줄 (train/infer mismatch 방지):
    warmup        : 100% GT heavy mask (heavy head가 깨끗한 입력으로 먼저 학습)
    mid (mixed)   : GT mask와 predicted mask를 확률 p로 섞고 p를 점진적으로 predicted 쪽으로
                    (scheduled sampling; 동시에 mask 예측 자체도 학습됨)
    late          : 100% predicted mask (추론과 동일 조건으로 마무리)
    -> 추론은 항상 predicted mask. 이 스케줄이 없으면 train/infer mismatch로 heavy head가 무너짐.
  mask / gate loss 설계 (mask가 틀리면 heavy head가 supervision을 못 받으므로):
    L_gate2_keep  : gate② cheap child score를 keep/drop target으로 직접 supervise (BCE 또는 focal),
                    recall-weighted (keep해야 할 voxel을 drop하는 FN에 큰 penalty).
                    positive keep target =
                      GT occupied
                      | observed surface-band
                      | near-surface observed free shell
                      | thin/far/dynamic/planner/unknown quota bucket
                    negative drop target =
                      high-confidence observed free
                    AND not near-surface
                    AND not quota-protected
                    unknown/occluded/저신뢰 voxel은 negative로 강제하지 말고 keep 또는 ignore.
    L_gate2_shell_tag :
                    gate② cheap shell-tag score / prelim_free_shell_raw / near_surface_free_shell_tag를
                    near-surface observed free shell GT로 supervise (BCE 또는 focal BCE).
                    positive shell tag =
                      observed free voxel
                      AND observed surface-band 또는 occupied GT의 6-neighbor 1겹
                      AND child-level ray/depth가 실제 observed-free로 검증
                    negative shell tag =
                      충분히 관측된 일반 free
                      AND not near-surface
                      AND not quota-protected
                    unknown/occluded/미관측/저신뢰 voxel은 shell negative로 강제하지 말고 ignore.
                    이 loss는 L_gate2_keep과 다르다. L_gate2_keep은 "살릴지"를 배우고,
                    L_gate2_shell_tag는 살아남은 free voxel이 `pred_free_shell`인지
                    `pred_free_kept`인지 가르는 semantic tag를 배운다.
    L_occ_fine       : O_occ_fine_occ를 prelim_kept 전체의 occupied/free/ignore label로 supervise.
                       observed free는 negative, unknown/occluded/저신뢰는 ignore.
    L_heavy_recall    : (pred_occupied_surface | pred_boundary)의 recall loss.
                        GT occupied 또는 observed surface-band를 heavy mask에서 놓치면 큰 penalty.
                        surface-band/partial voxel은 pred_boundary로 가도 recall 성공으로 본다.
    L_surface_band    : O_occ_fine_surface를 GT transition band(SDF |d|<0.5·voxel)로 supervise.
                        pred_boundary는 이 surface-band score와 p_occ uncertain 구간의 조합으로 만든다.
    L_heavy_mask      : 초기엔 heavy head 입력 mask = (GT_heavy ∪ pred_heavy)로 넓게 줘서
                        mask 예측이 미숙해도 heavy head가 supervision을 받게 한 뒤,
                        스케줄에 따라 pred_heavy로 좁힌다 (위 scheduled sampling과 함께).
    원칙: occupied/surface-band recall을 우선 안정화하고, precision은 그 다음에 조인다.

Occupancy Flow Head:
  step 6 tracking
  -> 0.6m coarse motion/flow seed GT (vx/vy/vz, FlowOcc3D 방식)
  -> 0.3m pred_heavy voxel flow GT는 해당 parent seed를 매핑하고,
     필요하면 instance/track 기반 residual GT로 보강
  loss:
    L_flow_coarse_seed  : O_motion_0.6 dense coarse branch supervision
    L_flow_fine_readout : O_flow [N_heavy] final output supervision
    L_dynamic           : dynamic_logit / dynamic_seed classification
  주의(신뢰도): camera-only로 vz까지 안정 추정은 어렵다. flow GT는 confidence mask 필수,
    track이 확실한 dynamic voxel에서만 supervise(observed-only). 저신뢰는 ignore.
    가능하면 sim GT 또는 LiDAR spot-check로 보정한다.

Sub-Voxel Shape Information Head:
  step 4 / 7의 연속 SDF와 boundary samples
  exposed face / local offset / normal / uncertainty supervision
  loss:
    L_shape_exposed_face : exposed_face_logits 6-way BCE/focal BCE.
    L_shape_offset       : exposed face의 face_surface_offset regression.
    L_shape_normal       : observed surface normal cosine/L1 loss.
    L_shape_uncertainty  : heteroscedastic shape/offset/normal uncertainty calibration.
  주의(신뢰도): camera-only SDF/normal은 관측 면에서만 신뢰 가능. observed surface 근처에서만
    supervise(occlusion 뒤/미관측은 ignore), per-sample confidence로 가중.
    sub-voxel offset/normal은 sim GT 또는 LiDAR spot-check로 검증 권장.

Queryable Occupancy Interface / MLP:
  별도 Volume Head는 아니지만 learnable MLP이므로 GT/loss가 필요하다.
  Section 10의 MLP는 `collision_state` 전체를 직접 학습하는 것이 아니라,
  분기 (A) pred_heavy_mask 내부의 `surface_shell_prob`와 local uncertainty를 학습한다.
  `collision_state`는 이 확률 + coarse occupancy/visibility + observed-free/solid fallback 규칙으로 만든다.

  학습 입력 mask:
    warmup        : GT_heavy = GT occupied | observed surface-band 중심으로 query sample
    mid (mixed)   : GT_heavy와 pred_heavy_mask를 섞어 train/infer mismatch 완화
    late          : pred_heavy_mask 중심으로 sample
    원칙: MLP는 추론처럼 branch (A)에서만 의미가 있다. 다만 초기에 pred mask가 틀려
          MLP 학습이 막히지 않도록 heavy head와 같은 scheduled sampling을 쓴다.

  query sample 생성:
    positive surface-shell samples:
      observed mesh/SDF surface 또는 dynamic observed surface 주변에서
      |SDF| <= tau_shell 인 query point.
      기본 tau_shell은 0.3m voxel의 sub-voxel band 안에서 잡고,
      GT confidence / distance / visibility로 loss weight를 조절한다.

    negative observed-free local samples:
      같은 GT_heavy 또는 pred_heavy voxel 내부/인접 face에서,
      camera ray가 통과했다고 확인된 observed-free query point.
      특히 exposed face 앞쪽 free side를 충분히 sampling해
      `surface_shell_prob`가 표면 밖 free 공간까지 퍼지지 않게 한다.

    ignore samples:
      occlusion 뒤, 미관측 interior, low-confidence SDF, dynamic 분리 불확실 영역.
      단순 frustum 내부이거나 모델이 free라고 추측한 점은 negative로 쓰지 않는다.

    solid interior:
      v1 Queryable MLP는 surface-shell 중심이므로 camera-only로 닫힌 증거가 없는
      물체 내부를 positive로 채우지 않는다. watertight mesh / CAD / sim / LiDAR 등
      강한 solid GT가 있으면 collision_state의 occupied-risk fallback rule 평가/검증에는 쓰되,
      surface_shell_prob는 surface band 중심으로 유지한다.

  loss:
    L_query_surface_shell:
      weighted BCE 또는 focal BCE(surface_shell_prob, surface_shell_label).
      positive/negative query 비율을 균형화하고, observed/confident sample에 더 큰 weight를 준다.

    L_query_uncertainty:
      heteroscedastic BCE/NLL 또는 calibration loss.
      SDF confidence가 낮거나 ray consistency가 약한 sample은 uncertainty target을 높인다.

    optional L_query_sdf:
      MLP가 optional signed distance를 출력할 때만 사용.
      observed surface 근처 narrow band에서 truncated SDF regression으로 둔다.
      이 loss를 켜도 planner collision은 여전히 `collision_state` fallback 규칙을 따른다.

    L_query_collision_consistency (약한 규칙 loss 또는 unit test):
      high-conf observed-free query가 surface_hit/occupied-risk로 반환되지 않는지,
      high surface_shell_prob query가 observed_free로 반환되지 않는지 검사한다.
      미관측/occluded query를 free negative로 강제하지 않는다.

Auxiliary Depth / Free-space Evidence Head (Section 4):
  core Volume Head는 아니지만 v1 inference에서 `observed_free_evidence_0.3(child)`를 만들기 위한
  inference-critical auxiliary head다. 따라서 Section 19 loss checklist에 포함한다.

  GT source:
    dense metric depth:
      Section 19 Stage A의 metric pointmap / mesh render / synthetic depth.
    sparse metric depth:
      Section 20 Basalt landmark reprojection depth + covariance weight.
    free-space evidence:
      calibrated camera ray casting으로 depth hit 앞쪽 구간을 observed free로 라벨.
      occlusion 뒤와 hit 뒤쪽 interior는 free negative가 아니라 unknown/ignore.

  output:
    low-res depth or range
    depth/range confidence
    per-ray free-space confidence

  loss:
    L_depth:
      scale-aware L1/Huber 또는 inverse-depth loss, confidence로 가중.
    L_depth_conf:
      heteroscedastic depth NLL 또는 confidence calibration.
    L_free_space_evidence:
      ray가 통과한 구간의 free-space BCE/focal BCE.
      depth hit 뒤쪽 / occluded / 미관측 구간은 negative로 쓰지 않고 ignore.

  inference connection:
    이 head의 high-confidence ray/depth 결과만 Section 10의
    `observed_free_evidence_0.3(child)`와 `observed_free_evidence_0.6(parent)`에 들어간다.
    confidence가 낮으면 observed-free가 아니라 unknown으로 보수 처리한다.

3D Semantics Head:
  static / dynamic segmentation
  road / vehicle / pedestrian / curb / wall 등 semantic labels
  loss:
    L_semantic = valid_semantic_mask 위에서만 CE/focal CE.
    pred_boundary 전체에 강한 semantic label을 주지 않고, confident observed surface 중심으로 supervise.

Surface Outputs head:
  ground / road segmentation + 연속 표면(SDF/mesh)
  -> per-BEV-cell z_surface / slope / step / valid / uncertainty / traversability_cost / drop_risk
  loss:
    Section 11.3의 L_surface_valid / L_surface_z / L_surface_slope_or_normal /
    L_step_height / L_traversability_cost / L_drop_risk /
    L_surface_sparse_consistency를 사용한다.

전체 loss checklist (누락 방지):
  L_occ_coarse_cls + L_occ_coarse_visibility + L_coarse_mixed_surface_risk
  + L_gate2_keep + L_gate2_shell_tag + L_occ_fine + L_surface_band + L_heavy_recall + L_heavy_mask
  + L_flow_coarse_seed + L_flow_fine_readout + L_dynamic
  + L_shape_exposed_face + L_shape_offset + L_shape_normal + L_shape_uncertainty
  + L_query_surface_shell + L_query_uncertainty
    (+ optional L_query_sdf, optional/rule-test L_query_collision_consistency)
  + L_depth + L_depth_conf + L_free_space_evidence
  + L_semantic
  + L_surface_valid + L_surface_z + L_surface_slope_or_normal
  + L_step_height + L_traversability_cost + L_drop_risk
  + L_surface_sparse_consistency
```

### Limitations

```text
근본 전제 (강조):
  camera-only reconstruction으로 30cm occupancy + sub-voxel offset/normal +
  dynamic flow(vx/vy/vz)를 안정적으로 만드는 것은 그 자체가 별도 연구 과제다.
  "MapAnything이 metric scale을 준다"는 출발점일 뿐, head별 GT 품질은 보장되지 않는다.
  -> 모든 head GT에 confidence mask + observed-only supervision을 기본으로 깐다.
     (미관측/occluded/저신뢰는 supervise하지 않고 ignore.)

원거리 정확도:
  camera depth 오차는 거리에 따라 급증 -> 30cm GT 신뢰 범위도 원거리에서 급락.
  far-field는 GT uncertainty를 반영해 loss weight를 줄이거나 unknown으로 라벨.

신뢰도 등급 (head별로 다름):
  높음:  coarse occupancy(0.6m), 주 ground surface 높이 (관측 기반)
  중간:  0.3m occupied 표면 경계
  낮음:  sub-voxel offset/normal, dynamic flow vz  -> sim GT / LiDAR spot-check 권장

검증 없는 적용 금지:
  자체 데이터엔 비교 기준이 없으므로 step 8 nuScenes(Occ3D LiDAR GT) 검증으로
  거리별/head별 오차를 먼저 정량화한 뒤 적용한다.

품질 보증:
  가능하면 GT 수집용으로만 LiDAR 1대를 일부 시퀀스에 장착해 spot-check한다
  (특히 flow/sub-voxel). 추론은 여전히 camera-only다.
  sim(합성) GT는 sub-voxel/flow의 신뢰 가능한 보강 소스로 적극 활용한다.
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

Section 4의 tiny depth/free-space evidence head를 sparse하게 supervise하는 온라인 GT 소스다. 이 supervision 자체는 학습용이지만, head는 v1 inference에서도 observed-free evidence를 만들기 위해 유지된다.

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
