# 내가 생각하는 테슬라 Occupancy Network 아키텍처 - Ver.4

> ⚠️ **이 문서는 Ver.4 사고 기록(아카이브)입니다. 최신 canonical 설계가 아닙니다.**
>
> 최신 정본은 [occupancy_network_architecture.md](occupancy_network_architecture.md) 다. 두 문서의 수치가 다르면 **occupancy_network_architecture.md가 우선**한다.
>
> 주요 차이 (이 문서 = 옛 예시 / canonical = 현재):
> - 최종 voxel: **0.2m → 0.3m**
> - 해상도 단계: **1.6/0.8/0.4m → 1.2/0.6/0.3m**
> - Y range: **±10m → ±10.2m (20.4m, 격자=타깃)**, Z: **[0, +5] → [-2, +4] (중력 정렬)**
> - 최종 표현: structured sub-voxel head → **sparse deconv + 2단 게이트 prune + near-surface free shell**
> - temporal: ViewFormer z-squeeze BEV → **Tesla/PanoOcc식 0.6m 3D (align+concat+3D conv)**
> - Surface: ZPool → **z-flatten**, Queryable: **occupancy 전용**(semantic은 3D Semantics Head)
>
> 아래 본문의 수치/구조는 사고 과정 기록으로만 참고하고, 구현은 canonical 문서와 그 6개월 plan.md를 따른다.

```yaml
# A. Tesla public main architecture
#    공개 발표 그림에 직접 대응되는 기본 흐름

1. Multi-camera image input + rectify
↓
2. RegNet + BiFPN image featurizers
↓
3. Spatial Attention
   - 3D positional query
   - image-space K/V from multi-camera features
↓
4. Low-res Spatial Features
↓
5. Temporal Alignment
   - trajectory used to align past features to current coordinate frame
   - temporal context / spatial frame alignment
↓
6. Spatiotemporal Features
↓
7. Deconvolutions
↓
8. Surface Outputs
   - road / ground surface geometry
   - road / ground surface semantics
↓
9. Volume Outputs
   - occupancy
   - occupancy flow
   - sub-voxel shape information
   - 3D semantics
↓
10. Queryable Outputs
   - x,y,z → MLP → continuous occupancy probability
   - x,y,z → MLP → continuous 3D semantics

# B. Tesla-like implementation interpretation
#    공개 그림을 구현으로 옮길 때 가능한 해석

- low-res feature는 2D BEV만이 아니라 3D-aware slab / volume으로 유지하는 설계로 둘 수 있음
- temporal alignment 연산은 BEV/top-down ego-motion warp로 해석할 수 있음
- height slice별 같은 x-y-yaw transform을 적용하는 것은 단순 구현 가정
- SubVoxelShape_base를 Queryable MLP의 shape prior / conditioning feature로 쓰는 것은 optional 구현 해석

# C. Robot-specific runtime query / visualization strategy
#    Tesla public block이 아니라 실시간 로봇 구현 정책

- surface-shell refine score calculation
- quota-based Top-K surface voxel selection
- adaptive in-voxel query generation
- exposed face / edge / corner 기반 query budget 조절
- Queryable MLP 기반 boundary search / carving
- sparse adaptive partial cuboid overlay
- planner-friendly local occupancy / distance field conversion

# D. Optional extensions / separate task branches
#    Tesla strict occupancy main path 밖의 확장 또는 별도 task branch

- low-res flow-aware dynamic feature correction
- visible near-field / dynamic image re-query enhancement
- DuOcc-style QueryAgg injection
- object / agent sparse branch for trajectory / shape / pose
```

이 문서는 위 네 층을 분리해서 읽는다.

```text
A:
Tesla AI Day / CVPR 공개 그림에 직접 보이는 구조

B:
공개 그림을 구현으로 옮길 때의 해석

C:
로봇 실시간성과 Tesla 데모식 시각화를 위한 runtime 최적화

D:
strict occupancy main path 밖의 optional extension / separate task branch
```

---

Temporal buffer는 `T`개 frame의 video feature queue로 둔다.
로봇 초기 구현에서는 2초 정도를 예시값으로 사용할 수 있지만, Tesla 공개 그림 자체가 2초를 고정값으로 명시한 것은 아니다.

### 기본 공간 / 해상도 설정

아래 공간 범위와 `voxel_size = 0.2m`는 이 문서의 로봇 구현 target 예시이다.
Tesla 공개 발표는 voxel size가 application / memory / compute trade-off에 따라 configurable하다고 설명하며, 20cm를 고정 public spec으로 명시하지 않는다.

```text
voxel_size = 0.2m

coordinate:
x: ego 기준 전후 방향, +x = front
y: ego 기준 좌우 방향
z: height

x_range = [-10m, +50m]   # rear 10m, front 50m
y_range = [-10m, +10m]   # left 10m, right 10m
z_range = [0m, +5m]
x_length = 60m
y_length = 20m
z_height = 5m

N_x = x_length / voxel_size = 300
N_y = y_length / voxel_size = 100
N_z = z_height / voxel_size = 25

low_res_shape = N_x_low × N_y_low × N_z_low
예: 60 × 20 × 5
```

초기 구현에서는 실제 target resolution과 toy/debug resolution을 분리한다.

```text
target:
20cm base voxel grid
(robot implementation example, not Tesla fixed public spec)
x_range = [-10m, +50m], y_range = [-10m, +10m], z_range = [0m, +5m]
N_x = 300, N_y = 100, N_z = 25

toy/debug:
32 × 32 × 4 또는 64 × 64 × 8 low-res grid부터 시작
```

Level of Detail = LOD

| **거리** | **복셀 표현** |
| --- | --- |
| 0~10m | 높은수준 |
| 10~25m | 중간수준 |
| 25m~50m | 낮은수준 |

여기서 LOD는 Octree처럼 부모-자식 voxel을 재귀적으로 분할한다는 뜻이 아니다.
이 문서의 로봇 구현 예시에서는 기본 20cm base grid를 유지하되, surface-shell / near-field / dynamic / planner-relevant 영역에만 sub-voxel point query를 더 많이 사용하는 의미이다.

## 1. Multi-camera image

### **역할**

- 여러 대의 카메라에서 들어오는 원본 이미지
- Tesla 차량 공개 설명은 8 camera 입력을 기준으로 한다
- 아래 목록은 로봇 또는 단순화된 surround-camera 예시이다:
    
    ```
    front
    front-left
    front-right
    rear
    rear-left
    rear-right
    ```
    

---

### **왜 필요한가?**

- Occupancy Network는 3D 공간을 예측해야 한다
- 단일 카메라만 쓰면 한쪽 방향만 보이므로 주변 전체 공간을 만들기 어렵다
- multi-camera를 쓰면 각 카메라가 서로 다른 시야를 담당한다
    
    ```
    camera image 1 → 전방 정보
    camera image 2 → 좌측 정보
    camera image 3 → 우측 정보
    ...
    ```
    

---

### **여기서 중요한 정보**

- 이 단계에서 이미 단순 이미지뿐 아니라 다음 정보가 같이 필요하다
    
    ```
    camera intrinsic
    camera extrinsic
    timestamp
    image resolution
    camera index
    ```
    
- 특히 intrinsic/extrinsic이 없으면 나중에 3D query를 image feature에 연결하기 어렵다

---

### Rectify / camera calibration alignment

- Tesla AI Day 그림에는 image input 뒤에 `Rectify` 단계가 명시되어 있다
- 이 단계는 각 카메라 이미지를 네트워크가 안정적으로 쓰기 좋게 보정하는 전처리이다
    
    ```
    raw camera image
    +
    intrinsic / distortion parameters
    +
    extrinsic calibration
    ↓
    rectified image / calibrated camera ray
    ```
    
- 3D spatial query가 image feature를 attend하려면 각 image token이 어떤 camera ray / 3D direction에 해당하는지 알아야 한다
- 따라서 rectification은 단순 이미지 보정이 아니라, 이후 spatial attention에서 3D query와 image feature를 연결하기 위한 geometry alignment 단계이다

---

## 2. RegNet + BiFPN image featurizers

### **역할**

- 원본 이미지를 image featurizer에 넣어서 feature map을 뽑는다
    
    ```
    image
    ↓
    RegNet
    ↓
    BiFPN
    ↓
    image feature
    ```
    
- Tesla AI Day 요약 그림 기준 기본값은 다음이다
    
    ```
    RegNet + BiFPN
    ```
    
- 다른 backbone은 대체 구현으로 둘 수 있다
    
    ```
    ResNet
    EfficientNet
    ConvNeXt
    Swin Transformer
    ```
    
- FPN은 여러 해상도의 feature를 만드는 구조이다
    
    ```
    P2: high-resolution, low-level feature
    P3
    P4
    P5: low-resolution, high-level semantic feature
    ```
    

---

### **왜 필요한가?**

- 원본 이미지를 바로 attention에 넣으면 너무 크다
- 예를 들어 1280×720 이미지를 그대로 token화하면 계산량이 폭발한다. 그래서 backbone을 통해 압축된 feature를 만든다

---

### **FPN의 의미**

- FPN은 작은 물체와 큰 문맥을 같이 다루기 위한 구조이다
    
    ```
    P2/P3 → 경계, 작은 물체, 디테일
    P4/P5 → 큰 물체, 전역 문맥, 장면 구조
    ```
    
- 이 문서의 단순 구현에서는 계산량을 줄이기 위해 P4/P5 같은 low-res feature 중심으로 3D feature를 만들 수 있다
- Tesla 공개 발표에서 직접 확인되는 것은 RegNet + BiFPN backbone과 3D positional query / image feature attention이며, P-level 선택 방식은 공개된 고정 spec으로 보지 않는다
- P2/P3는 메인 refinement 필수 입력이 아니라, 근거리 / 동적 / 얇은 구조물에 대한 optional image re-query branch에서 선택적으로 사용할 수 있다

---

## 3. Low-res image feature + 3D PE

### **역할**

- 낮은 해상도의 image feature에 **3D positional encoding**을 추가한다
    
    ```
    low-res image feature
    +
    3D positional encoding
    =
    3D-aware image feature
    ```
    

---

### 왜 low-res feature를 쓰나?

- P4/P5 같은 low-res feature는 해상도는 낮지만 semantic 정보가 강하다
    
    ```
    벽이 있는지
    차량이 있는지
    도로/바닥 구조가 어떤지
    장면 전체 흐름이 어떤지
    ```
    
- 같은 큰 구조를 파악하는 데 좋다

---

### 3D PE는 왜 넣나?

- 일반 image feature는 기본적으로 2D 정보다
    
    ```
    feature at (u, v)
    ```
    
- 하지만 occupancy는 3D 공간을 예측한다
    
    ```
    voxel at (x, y, z)
    ```
    
- 그래서 image feature에 다음 정보를 주입한다
    
    ```
    이 feature는 어느 카메라에서 왔는가?
    이 픽셀은 3D 공간의 어느 ray 방향을 보는가?
    대략 어떤 depth 후보와 연결될 수 있는가?
    ```
    
- 이게 PETR-style 요소입니다.

---

### 핵심 직관

```
그냥 image feature:
"나는 이미지의 이 위치 feature다"

3D PE가 들어간 image feature:
"나는 이 카메라에서 이 방향의 3D 공간을 보는 feature다"
```

---

## 4. Low-res fixed 3D spatial query

### 역할

- 낮은 해상도의 3D 공간 query를 만든다
- 예를 들어 ego 기준 3D 공간을 작게 나눈다
    
    ```
    32 × 32 × 4
    ```
    
- 그러면 query 수는
    
    ```
    4096개
    ```
    
- 각 query는 하나의 3D 공간 위치를 대표한다
    
    ```
    query 1 = voxel (x1, y1, z1)
    query 2 = voxel (x2, y2, z2)
    query 3 = voxel (x3, y3, z3)
    ...
    ```
    

---

### fixed라는 의미

- 여기서 fixed는 매 프레임마다 query 위치가 바뀌는 게 아니라, **ego vehicle 또는 robot 기준으로 고정된 3D grid를 사용한다**는 뜻이다
    
    ```
    query index 0 = 항상 같은 3D 위치
    query index 1 = 항상 같은 3D 위치
    ...
    ```
    

---

### 왜 low-res인가?

- 처음부터 고해상도 voxel query를 만들면 너무 비싸다
- 예를 들어:
    
    ```
    target base grid:
    N_x × N_y × N_z = 300 × 100 × 25
    ```
    
- 이 정도면 attention 비용이 매우 크다
- 그래서 처음에는 낮은 해상도로 전체 구조를 본다

---

## **5. Global cross-attention + optional local attention**

### 역할

- low-res 3D spatial query가 multi-camera image feature를 읽는다
- 구조는 다음과 같다
    
    ```
    Q = low-res 3D spatial query
    K = 3D PE가 들어간 image feature
    V = 3D PE가 들어간 image feature
    ```
    
- attention은 이렇게 동작한다
    
    ```
    3D query:
    "나는 이 3D 위치다. 내가 봐야 할 이미지 feature는 어디인가?"
    
    image feature:
    "나는 이 카메라의 이 위치에서 나온 feature다."
    ```
    

---

### global이라는 의미

- coarse stage에서는 query가 전체 image feature 또는 넓은 범위의 multi-camera feature를 볼 수 있다
- 즉, 특정 projection 주변만 보는 것이 아니라, 비교적 넓게 봅니다
- optional image re-query branch를 켠 경우에만 projection 주변의 local attention을 제한적으로 사용할 수 있다

---

### 왜 필요한가?

- 이 단계가 실질적인 **2D → 3D view transformation이다**
    
    ```
    multi-camera 2D feature
    ↓
    3D spatial feature
    ```
    

---

## 6. Low-res 3D-aware spatial feature volume

### 역할

- global cross-attention 결과로 낮은 해상도의 3D feature volume이 만들어진다
- 예를 들면:
    
    ```
    32 × 32 × 4 × C
    ```
    
- 여기서 각 voxel feature는 다음 정보를 담는다
    
    
    ```
    이 공간이 비어 있는지
    점유되어 있는지
    동적 물체가 있는지
    주변 구조가 어떤지
    semantic clue가 있는지
    ```
    

---

### 중요한 점

- 이 단계의 출력은 아직 최종 occupancy map이 아니다
- 정확히는:
    
    ```
    3D occupancy를 예측하기 위한 latent feature volume
    ```
    
- 즉, 아직 feature 이다

---

### Temporal alignment 관점에서의 의미

- 이 단계의 출력은 단순한 3D voxel 확률 map이 아니라, **low-resolution 3D-aware spatial latent feature**이다
- 즉, feature 자체는 height / occupancy / semantic / dynamic clue를 포함할 수 있다
    
    ```
    F_t ∈ R^(N_x_low × N_y_low × N_z_low × C)
    ```
    
- 하지만 이후 temporal alignment를 할 때 꼭 완전한 3D volume warp만 해야 하는 것은 아니다
- Tesla 발표자료의 temporal alignment 그림은 top-down trajectory 기반 정렬로 표현되어 있으므로, 가능한 구현 해석은 다음과 같다
    
    ```
    feature representation:
    low-res 3D-aware spatial feature
    
    alignment operation:
    BEV / top-down ego-motion warp
    ```
    
- 즉, 이 feature는 3D occupancy를 예측하기 위한 feature이지만, 이 문서의 구현 해석에서는 시간 정렬을 계산량을 줄이기 위해 low-resolution BEV-like grid 기준으로 수행한다
- 메인 구현에서는 `N_z_low`를 유지한 3D-aware latent를 기본 표현으로 둔다
- 즉, temporal alignment는 top-down으로 해석하되, deconv 입력 feature는 다시 3D slab / volume 형태로 두는 설계이다
    
    ```
    main feature:
    F_t ∈ R^(N_x_low × N_y_low × N_z_low × C)
    
    alignment:
    x-y-yaw BEV warp를 각 height slice에 동일 적용
    
    deconv input:
    N_z_low가 유지된 3D-aware spatiotemporal feature
    ```

---

## 7. Tesla-like implementation interpretation: low-res BEV-like temporal alignment + aligned feature stack

### 역할

- 이 섹션은 Tesla 공개 그림에 직접 적힌 표현과, 이 문서의 구현 해석을 분리해서 읽어야 한다
    
    ```
    Tesla public:
    trajectory used to align past features to current coordinate frame
    Temporal Context
    Spatial Frame Alignment
    Spatiotemporal Features
    
    implementation interpretation:
    low-res 3D-aware spatial feature를 유지하고,
    alignment 연산은 BEV/top-down x-y-yaw warp로 수행한다고 가정한다
    ```
    
- 현재 frame의 low-res spatial feature와 이전 frame들의 low-res spatial feature를 현재 ego 좌표계로 정렬한다.
- 이때 정렬은 디컨볼루션 이후 high-res voxel에서 수행하지 않는다.
- 이 문서의 기본 구현에서는 **3D deconv 이전의 low-resolution spatial feature 단계**에서 수행하는 것으로 둔다.
    
    ```
    F_t
    F_{t-1}
    F_{t-2}
    F_{t-3}
    ...
    ```
    
- 여기서 각 feature는 3D occupancy를 예측하기 위한 low-res spatial latent이다.
    
    ```
    F_t ∈ R^(N_x_low × N_y_low × N_z_low × C)
    ```
    
- 하지만 temporal alignment 연산 자체는 완전한 3D volume warp라기보다, **BEV / top-down ego-motion warp**로 해석할 수 있다.
    
    ```
    low-res 3D-aware spatial feature
    ↓
    BEV-like ego-motion alignment
    ↓
    aligned low-res spatial feature
    ```
    

---

### 왜 BEV-like alignment인가?

- Tesla AI Day temporal alignment 그림은 과거 feature들을 trajectory를 이용해 현재 coordinate frame에 맞추는 구조로 표현되어 있다.
- 이 그림은 3D volume 전체를 직접 돌리는 표현이라기보다, top-down spatial frame을 정렬하는 표현에 가깝게 보인다.
- 따라서 이 문서의 구현 해석은 다음이다.
    
    ```
    feature는 3D-aware spatial latent
    alignment는 BEV/top-down coordinate에서 수행
    ```
    
- 즉, 최종 출력은 3D occupancy이지만, 이 문서에서는 temporal memory alignment를 low-res BEV-like spatial feature에서 수행하는 것으로 둔다.
- 첨부한 Tesla AI Day 요약 그림도 이 해석과 양립한다.
- 그림의 `Spatial Features`와 `Spatiotemporal Features`는 완전한 2D BEV map이라기보다 얇은 3D slab처럼 표현되어 있고, `Spatial Frame Alignment`만 trajectory 기반 top-down 정렬로 표현되어 있다. 다만 공개 자료만으로 내부 tensor shape이 확정되는 것은 아니다.
- 따라서 이 문서의 기본 해석은 다음이다.
    
    ```
    feature representation:
    low-res 3D slab / volume
    
    temporal alignment:
    BEV/top-down ego-motion warp
    
    deconv:
    3D-aware spatiotemporal feature를 고해상도 voxel feature로 복원
    ```

---

### 왜 low-res에서 수행하나?

- temporal fusion을 high-res voxel에서 수행하면 계산량이 너무 크다.
    
    ```
    low-res:
    32 × 32 × 4
    
    target base:
    N_x × N_y × N_z = 300 × 100 × 25
    ```
    
- 따라서 이 문서의 기본 구현에서는 temporal alignment와 fusion을 deconv 전에 수행하는 것이 좋다.
    
    ```
    Spatial Attention
    ↓
    Low-res spatial feature
    ↓
    Temporal alignment / fusion
    ↓
    Spatiotemporal feature
    ↓
    3D deconv
    ↓
    High-res occupancy feature
    ```
    
- 이렇게 하면 시간 정보가 반영된 feature를 이후 deconv가 고해상도로 복원할 수 있다.

---

### Ego-motion alignment

- 과거 feature는 과거 ego frame 기준으로 만들어졌다.
- 현재 frame과 좌표계가 다르기 때문에 그대로 합치면 위치가 어긋난다.
    
    ```
    F_{t-1}
    F_{t-2}
    F_{t-3}
    ```
    
- VIO / SLAM / wheel odometry / IMU integration에서 얻은 ego-motion을 이용해 과거 feature를 현재 ego frame으로 warp한다.
    
    ```
    F'_{t-1} = Warp_BEV(F_{t-1}, T_{t ← t-1})
    F'_{t-2} = Warp_BEV(F_{t-2}, T_{t ← t-2})
    F'_{t-3} = Warp_BEV(F_{t-3}, T_{t ← t-3})
    ```
    
- 여기서 `Warp_BEV`는 이 문서의 구현 가정에서 사용하는 x-y-yaw 중심의 top-down grid warp이다.
- feature가 `N_x_low × N_y_low × N_z_low × C` 형태라면, 단순 버전에서는 각 height slice에 같은 BEV transform을 적용한다.
- 즉, 이 구현 해석에서는 `z`축을 메인 표현에서 channel로 완전히 접지 않는다.
- temporal attention 내부에서 효율을 위해 임시로 z-squeeze / BEV query를 만들 수는 있지만, 이 문서의 기본 설계에서는 3D deconv 전에는 `N_z_low`가 있는 3D-aware feature로 복원한다.
    
    ```
    low-res 3D-aware feature
    ↓
    height slice별 BEV warp
    ↓
    aligned low-res 3D-aware feature
    ```
    

---

### Aligned feature stack

- alignment가 끝나면 현재 feature와 과거 feature들이 모두 현재 ego 좌표계에 놓인다.
    
    ```
    F_t
    F'_{t-1}
    F'_{t-2}
    F'_{t-3}
    ...
    ```
    
- 이 feature들을 시간축으로 stack한다.
    
    ```
    F_stack =
    Stack(
        F_t,
        F'_{t-1},
        F'_{t-2},
        F'_{t-3}
    )
    ```
    
- 이 stack은 단순히 channel concat만 의미하지 않는다.
- 시간 dimension을 가진 aligned feature memory로 보는 것이 더 정확하다.
    
    ```
    F_stack ∈ R^(T × N_x_low × N_y_low × N_z_low × C)
    ```
- ViewFormer처럼 temporal attention 내부에서 계산량을 줄이기 위해 z축을 squeeze할 수는 있다.
- 하지만 이 경우에도 squeeze된 BEV query는 내부 연산용 표현이며, 최종 temporal output은 다시 `N_z_low`를 가진 3D-aware latent로 복원한다.
    
    ```
    optional internal:
    voxel feature
    ↓
    z-squeeze BEV query
    ↓
    temporal attention
    ↓
    z-unsqueeze / restore
    ↓
    3D-aware temporal feature
    ```
    

---

### Temporal context fusion / temporal attention

- aligned feature stack을 그대로 평균내거나 단순 concat-conv로 합치지 않는다.
- 현재 feature를 query로 두고, 정렬된 과거 feature들을 key/value로 사용한다.
    
    ```
    Q = F_t
    K = [F_t, F'_{t-1}, F'_{t-2}, F'_{t-3}]
    V = [F_t, F'_{t-1}, F'_{t-2}, F'_{t-3}]
    ```
    
- 공개 그림의 큰 표현은 `Temporal Context` / `Spatial Frame Alignment`이고, 이 문서에서는 이를 구현할 fusion block으로 temporal attention을 사용한다고 해석한다.
- temporal attention은 현재 위치에서 어떤 과거 feature를 더 신뢰할지 선택한다.
    
    ```
    현재 frame이 명확함
    → 현재 feature에 높은 weight
    
    현재 frame이 가려짐
    → 과거 feature에 높은 weight
    
    과거 feature가 ego-motion 오차로 흐림
    → 낮은 weight
    
    동적 객체 주변에서 temporal inconsistency가 큼
    → 최근 frame 또는 flow-corrected feature에 높은 weight
    ```
    
- 결과적으로 하나의 spatiotemporal feature가 만들어진다.
    
    ```
    F_st = TemporalAttention(
        Q = F_t,
        K = F_stack,
        V = F_stack
    )
    ```
    

---

### 출력

- 이 단계의 출력은 temporal 정보를 포함한 low-res spatiotemporal feature이다.
    
    ```
    F_st ∈ R^(N_x_low × N_y_low × N_z_low × C)
    ```
- 내부적으로 BEV query를 사용했더라도, 이 단계의 외부 출력은 `N_z_low`를 가진 3D-aware feature로 둔다.
    
- 이 feature에는 다음 정보가 들어 있다.
    
    ```
    현재 frame의 관측 정보
    과거 frame에서 정렬된 구조 정보
    occlusion 보완 정보
    temporal consistency 정보
    동적 객체 motion clue
    occupancy flow 예측을 위한 시간 단서
    ```
    

---

### 이 단계의 핵심

- 이 단계는 단순히 과거 feature를 누적하는 단계가 아니다.
- 정확한 의미는 다음이다.
    
    ```
    1. 과거 low-res spatial feature를 저장한다.
    2. VIO / odometry trajectory로 현재 ego frame에 BEV-like alignment한다.
    3. 정렬된 feature들을 aligned feature stack으로 만든다.
    4. temporal attention으로 필요한 과거 정보만 선택적으로 fusion한다.
    5. low-res spatiotemporal feature F_st를 만든다.
    ```
    
- 따라서 이 구조는 다음 표현이 가장 정확하다.
    
    ```
    Ego-motion alignment
    +
    Aligned feature stack
    +
    Temporal attention
    ```
    

---

### 주의할 점

- ego-motion alignment는 기본적으로 static world에 강하다.
- 벽, 바닥, 기둥, 정지 물체는 잘 정렬된다.
- 하지만 사람, 차량, 로봇, 카트 같은 dynamic object는 ego-motion만으로 완전히 정렬되지 않는다.
    
    ```
    static object:
    ego-motion alignment로 정렬 가능
    
    dynamic object:
    객체 자체의 motion 때문에 추가 보정 필요
    ```
    
- strict Tesla-style main path에서는 이 feature를 바로 3D deconv로 보낸다.
- 동적 객체 misalignment를 더 줄이고 싶을 때만 optional low-res flow-aware correction을 켠다.

---

#### 이 아키텍처에서 temporal module의 핵심

- Low-res에서 수행한다.
- BEV-like ego-motion alignment를 사용한다.
- Aligned feature stack을 만든다.
- Temporal attention으로 fusion한다.
- 그 이후에 deconv로 high-resolution occupancy feature를 복원한다.

---

#### 구조 핵심 순서

- Spatial attention
    - low-res 3D-aware spatial feature
    - BEV-like temporal alignment
    - aligned feature stack
    - temporal attention
    - 3D deconv
    - occupancy / flow / refinement

---

## D1. Optional research extension: low-res flow-aware dynamic feature correction

이 branch는 Tesla AI Day 요약 그림의 strict main path에는 표시되지 않는다.
기본 Tesla-style path는 다음처럼 간다.

```text
F_st
↓
3D deconv
↓
Volume Outputs에서 Occupancy Flow 예측
```

따라서 이 branch는 동적 객체가 많은 로봇 환경에서 실험할 수 있는 optional 보강 구조로 둔다.

### 역할

- temporal attention으로 만들어진 `F_st`에서 occupancy와 flow의 초기 단서를 예측한다.
- 이 flow 정보를 이용해 ego-motion alignment만으로는 맞지 않는 동적 객체 feature를 추가 보정한다.
    
    ```
    F_st
    ↓
    preliminary occupancy / flow / dynamic score head
    ↓
    O_pre, Flow_pre, Dyn_pre
    ↓
    flow-aware dynamic correction
    ↓
    F_flow
    ```
    
- 이 branch는 최종 occupancy를 바로 만드는 단계가 아니다.
- 목적은 **동적 객체 주변의 low-res spatiotemporal feature를 한 번 더 정렬하고 보강하는 것**이다.

---

### 왜 필요한가?

- ego-motion alignment는 static world를 기준으로 한다.
- 따라서 정적 물체는 잘 맞지만, 움직이는 객체는 여전히 어긋난다.
    
    ```
    벽 / 바닥 / 기둥:
    ego-motion alignment로 정렬 가능
    
    사람 / 차량 / 로봇 / 카트:
    ego-motion alignment 이후에도 위치 오차가 남음
    ```
    
- 예를 들어 사람이 왼쪽에서 오른쪽으로 움직이면, 과거 feature는 ego-motion으로 정렬해도 과거 사람 위치에 남을 수 있다.
    
    ```
    t-2: 사람 위치 A
    t-1: 사람 위치 B
    t  : 사람 위치 C
    ```
    
- 이 경우 필요한 것은 객체 자체의 motion을 반영하는 추가 보정이다.

---

### Occupancy flow의 의미

- occupancy flow는 occupied region이 어느 방향으로 움직이는지 나타내는 vector이다.
    
    ```
    Flow_pre[x, y, z] = 해당 occupied feature의 motion vector
    ```
    
- 초기 구현의 기본값은 BEV 2D flow이다.
    
    ```
    Flow_pre ∈ R^(N_x_low × N_y_low × N_z_low × 2)
    flow = (dx, dy)
    ```
    
- height slice별로 같은 BEV flow를 공유하는 단순 버전은 다음처럼 둘 수 있다.
    
    ```
    Flow_pre_bev ∈ R^(N_x_low × N_y_low × 2)
    ```
    
- 3D motion까지 명시적으로 다루는 확장 버전은 다음처럼 둔다.
    
    ```
    Flow_pre_3d ∈ R^(N_x_low × N_y_low × N_z_low × 3)
    ```
    

---

### Dynamic correction 방식

- 먼저 temporal attention 이후 feature에서 dynamic score와 flow를 예측한다.
    
    ```
    Dyn_pre = DynamicHead(F_st)
    Flow_pre = FlowHead(F_st)
    ```
    
- dynamic score가 높은 영역만 선택한다.
    
    ```
    dynamic mask =
    Dyn_pre > threshold
    또는
    top-K dynamic region
    ```
    
- 선택된 동적 영역에 대해 flow 기반 feature warp를 수행한다.
    
    ```
    F_flow = FlowWarp(F_st, Flow_pre, Dyn_pre)
    ```
    
- 직관적으로는 다음과 같다.
    
    ```
    ego-motion alignment:
    내 차량/로봇 움직임 보정
    
    flow-aware correction:
    객체 자체 움직임 보정
    ```
    

---

### Feature update 방식

- 전체 feature를 무조건 flow로 이동시키면 static world가 망가질 수 있다.
- 따라서 dynamic score를 gate로 사용한다.
    
    ```
    F_flow =
    (1 - Dyn_gate) * F_st
    +
    Dyn_gate * FlowWarp(F_st, Flow_pre)
    ```
    
- 여기서 `Dyn_gate`는 dynamic score를 sigmoid로 변환한 값이다.
    
    ```
    Dyn_gate = sigmoid(Dyn_pre)
    ```
    
- 이렇게 하면 정적 영역은 기존 temporal feature를 유지하고, 동적 영역만 flow 기반 보정을 받는다.

---

### Temporal attention과의 관계

- temporal attention은 어떤 과거 feature를 쓸지 고르는 단계이다.
- flow-aware correction은 동적 객체의 위치 어긋남을 추가로 보정하는 단계이다.
    
    ```
    Temporal attention:
    어떤 시간의 feature를 믿을지 선택
    
    Flow-aware correction:
    움직이는 객체 feature를 현재 위치에 더 가깝게 이동
    ```
    
- 따라서 둘은 역할이 다르며, 동적 객체가 많은 로봇 환경에서는 optional로 함께 실험할 수 있다.

---

### 출력

- 이 단계의 출력은 dynamic correction이 반영된 low-res spatiotemporal feature이다.
    
    ```
    F_flow ∈ R^(N_x_low × N_y_low × N_z_low × C)
    ```
- 이 출력도 3D deconv로 들어가야 하므로 `N_z_low`를 유지한 3D-aware latent로 둔다.
- flow 자체는 BEV 2D vector를 height slice별로 공유할 수 있지만, feature tensor shape은 3D slab 형태를 유지한다.
    
- 이 optional branch를 켠 configuration에서는 이후 deconv 입력으로 기존 `F_st` 대신 `F_flow`를 사용한다.
    
    ```
    F_st
    ↓
    Occupancy-flow-aware dynamic correction
    ↓
    F_flow
    ↓
    3D deconv
    ```
    

---

### 이 단계의 핵심

- 이 구조는 다음 세 가지를 결합한다.
    
    ```
    Ego-motion alignment
    +
    Temporal attention
    +
    Occupancy-flow-aware dynamic correction
    ```
    
- 정적 배경은 ego-motion으로 맞춘다.
- 어떤 과거 정보를 쓸지는 temporal attention으로 고른다.
- 동적 객체는 occupancy flow로 한 번 더 보정한다.
- 따라서 이 방식은 단순 temporal fusion보다 동적 객체에 강할 수 있지만, Tesla AI Day 요약 그림 기준으로는 main path가 아니라 optional extension이다.

## D2. Optional research extension: visible near-field / dynamic image re-query enhancement

이 branch는 기본 Tesla-style main path에 넣지 않는다.
기본 구조는 `F_st → 3D deconv → heads → queryable MLP`로 간다.

---

#### 역할

- temporal fusion과 occupancy-flow-aware correction 이후에도 불확실한 일부 voxel만 현재 frame의 high-resolution image feature로 다시 확인한다
- 대상은 전체 voxel이 아니라 visible near-field / dynamic / thin obstacle 후보로 제한한다
- 이 branch는 실시간 기본 모드가 아니라 ablation, 정밀 모드, 또는 로봇 근거리 안전성 보강 옵션이다
    
    ```
    F_occ_low
    ↓
    visible near-field / dynamic / uncertainty 후보 선택
    ↓
    3D→2D projection
    ↓
    P2/P3 local image feature re-query
    ↓
    ΔF_img
    ↓
    F_occ_img = F_occ_low + ΔF_img
    ```
    

---

#### 언제 켜는가?

- 실내+실외 로봇에서 다음 상황이 자주 문제될 때 켠다
    
    ```
    의자 다리
    사람 발 / 카트 바퀴
    낮은 턱
    케이블
    얇은 기둥
    유리문 / 난간 경계
    현재 frame에서 새로 나타난 근거리 장애물
    ```
    
- 하지만 기본 inference path에서는 끄는 것이 더 단순하고 Tesla 공개 아키텍처와도 더 가깝다

---

#### 후보 선택 기준

- 모든 voxel에 적용하지 않는다
- `dynamic_score`, `pre_uncertainty`, `pre_boundary_score`는 `F_occ_low`에서 나온 low-res preliminary head 또는 temporal inconsistency에서 얻는다
- 다음 점수가 높은 상위 `K_img`개만 선택한다
    
    ```
    image_requery_score =
    λ1 * near_field_score
    + λ2 * dynamic_score
    + λ3 * pre_uncertainty
    + λ4 * pre_boundary_score
    + λ5 * visibility_score
    ```
    
- 초기 구현에서는 다음 정도로 제한한다
    
    ```
    K_img = 128 ~ 512
    ```
    

---

#### P2/P3 local re-query

- 선택된 voxel center 또는 sub-voxel 후보 point를 image plane으로 projection한다
- projection 주변의 P2/P3 local window만 참조한다
    
    ```
    P2/P3 feature
    ↓
    3×3 local window
    또는 deformable sampling 4~8 points
    ```
    
- 이 branch는 high-res image detail을 feature에 더하는 역할만 한다
- occupancy logit을 직접 덮어쓰지 않는다

---

#### main path와의 관계

- 기본 deconv 입력은 `F_st`이다
- optional flow-aware correction을 켠 경우에는 `F_flow`를 사용한다
- optional image re-query까지 켠 경우에는 deconv 입력을 `F_occ_img`로 바꾼다
    
    ```
    기본:
    F_deconv_in = F_st
    
    optional flow-aware correction:
    F_deconv_in = F_flow
    
    optional image re-query:
    F_deconv_in = F_occ_img
    ```
    
- 따라서 이 branch는 메인 단계 번호에 넣지 않는다
- Tesla AI Day / CVPR 공개 설명에 가까운 기본 흐름은 temporal alignment와 temporal attention으로 spatiotemporal feature를 만든 뒤 바로 deconv하는 구조이다

---

## 8. 3D deconv to decoded high-res voxel feature volume

### 역할

- 이 단계의 입력은 temporal alignment와 temporal attention이 반영된 low-res spatiotemporal feature이다.
- strict Tesla-style main path의 기본 입력은 `F_st`이다.
- optional branch를 켠 경우에만 `F_flow` 또는 `F_occ_img`를 대신 사용할 수 있다.
    
    ```
    F_deconv_in = F_st
    
    optional:
    F_deconv_in = F_flow
    또는
    F_deconv_in = F_occ_img
    
    F_deconv_in
    ↓
    3D deconv
    ↓
    F_dec
    ```
    
- 즉, 단순 coarse feature를 바로 upsampling하는 것이 아니라, 다음 정보가 이미 반영된 feature를 고해상도로 복원한다.
    
    ```
    multi-view image evidence
    ego-motion aligned temporal memory
    temporal attention
    optional occupancy-flow-aware dynamic correction
    optional visible near-field image re-query
    ```
    
- temporal / dynamic correction이 반영된 low-res spatial feature volume을 더 높은 해상도로 키운다.
- toy/debug 예를 들면:
    
    ```
    32 × 32 × 4
    ↓
    64 × 64 × 8
    ↓
    128 × 128 × 16
    ```
    
- toy/debug에서는 이런 식으로 올릴 수 있다
- target base grid에서는 최종 decoded feature 크기를 다음처럼 둔다
    
    ```
    N_x × N_y × N_z
    =
    300 × 100 × 25
    ```
    
- 여기서 사용할 수 있는 방법은 두 가지이다
    
    ```
    ConvTranspose3D
    또는
    Trilinear upsampling + Conv3D
    ```
    

---

### 왜 필요한가?

- coarse volume만으로는 해상도가 너무 낮다
- 로봇이 실제로 주행하려면 작은 장애물, 경계, 통로, 벽면 등을 더 촘촘하게 알아야 한다
- 그래서 전체 3D feature volume을 high-res 쪽으로 확장한다

---

### 한계

- 디컨볼루션은 정보를 새로 관측하는 게 아니다
    
    ```
    low-res feature를 기반으로 high-res feature를 추정
    ```
    
- 따라서 low-res 단계에서 완전히 사라진 작은 물체는 deconv만으로 정확히 복원하기 어렵다

---

### 이 단계의 핵심 출력

- 이 단계의 가장 중요한 출력은 최종 occupancy가 아니라 **decoded high-res voxel feature volume**이다
    
    ```
     F_dec ∈ R^(N_x × N_y × N_z × C)
    ```
    
- 예를 들면:
    
    ```
    F_dec = 300 × 100 × 25 × C
    ```
    
- 여기서 각 voxel은 단순한 occupied/free 값이 아니라 latent feature를 가진다
    
    ```
    F_dec[x, y, z] = C차원 feature vector
    ```
    
- 이 feature는 이후 세 방향으로 사용된다
    
    ```
    1. surface output heads
    
    2. volume output heads
    
    3. queryable MLP point decoder
    ```
    
- 따라서 이 단계에서 feature를 바로 occupancy probability로 압축해서 버리면 안 된다
- 최종 logits 이전의 rich feature volume을 유지해야 한다

---

## **9. Tesla-style surface / volume output heads**

### 역할

- `F_dec`를 여러 개의 head에 넣어 Tesla AI Day 그림의 Surface Outputs와 Volume Outputs를 만든다
- 여기서의 high-resolution은 이 문서의 로봇 target 예시에서 전체 공간을 20cm base voxel grid로 표현하는 base resolution을 의미한다
- Tesla 공개 설명에서는 voxel size가 configurable하므로, 20cm는 Tesla 고정값이 아니라 이 문서의 구현 예시이다
- 이 단계의 출력은 queryable MLP와 planner 표현의 기준이 되는 dense base output이다
    
    ```
    F_dec
    ↓
    surface / volume heads
    ↓
    Surface Outputs
    +
    Volume Outputs
    ```
    

---

### 출력 종류

- Tesla 그림 기준 출력은 두 갈래로 둔다
    
    ```
    Surface Outputs:
    RoadSurfaceGeom_base
    RoadSurfaceSem_base
    
    Volume Outputs:
    O_base
    Flow_base
    SubVoxelShape_base  # 문서 내 구현명
    Sem3D_base
    ```
    
- 예를 들면:
    
    ```
    O_base              ∈ R^(N_x × N_y × N_z)
    Flow_base           ∈ R^(N_x × N_y × N_z × 2 or 3)
    SubVoxelShape_base  ∈ R^(N_x × N_y × N_z × C_shape)  # optional implementation head
    Sem3D_base          ∈ R^(N_x × N_y × N_z × C_sem)
    
    RoadSurfaceGeom_base ∈ surface parameter map
    RoadSurfaceSem_base  ∈ surface semantic map
    ```
    
- `Unc_base`와 `Bnd_base`는 Tesla 그림의 주 output이라기보다, query scheduling과 refinement 후보 선택을 위한 auxiliary head로 둘 수 있다
- `SubVoxelShape_base` 역시 Tesla 공개 그림의 `sub-voxel shape information`을 구현으로 옮기는 한 가지 방식이다. 이것이 직접 local mask인지, MLP conditioning인지, 별도 feature head인지는 공개 자료만으로 확정하지 않는다
    
    ```
    Aux outputs:
    Unc_base ∈ R^(N_x × N_y × N_z)
    Bnd_base ∈ R^(N_x × N_y × N_z)
    ```
    

---

### Surface Outputs는 surface-shell refinement와 다르다

- Tesla 그림의 `Surface Outputs`는 뒤쪽의 surface-shell query scheduling과 같은 개념이 아니다
- `Surface Outputs`는 `F_dec`에서 바로 나오는 별도 prediction head이다
    
    ```
    F_dec
    ↓
    RoadSurfaceGeometryHead
    ↓
    RoadSurfaceGeom_base
    
    F_dec
    ↓
    RoadSurfaceSemanticHead
    ↓
    RoadSurfaceSem_base
    ```
    
- 반면 `surface-shell refinement`는 `O_base`, `Flow_base`, `Unc_base`, `Bnd_base` 등을 보고 queryable MLP를 어디에 호출할지 정하는 runtime scheduling 단계이다
    
    ```
    Surface Outputs:
    road / ground surface 자체를 직접 예측하는 head
    
    surface-shell refinement:
    occupancy boundary 주변을 더 정밀하게 query하는 실행 최적화
    ```
    
- 따라서 이름은 비슷하지만 역할은 다르다

---

### Road / ground surface geometry head

- Tesla 차량 기준으로는 road surface geometry를 예측한다
- 실내+실외 로봇 기준으로는 road보다 넓게 ground / floor surface geometry로 일반화할 수 있다
- 바닥이나 도로에 voxel을 만들지 않는다는 뜻이 아니다
- `O_base`의 voxel grid는 바닥 / 도로 / 벽 / 장애물 / 빈 공간을 모두 포함한다
- surface geometry head는 그중 robot이 실제로 접촉하거나 주행할 수 있는 ground / floor surface를 별도로 더 직접적으로 예측하는 보완 head이다
- 예측 대상은 다음처럼 둘 수 있다
    
    ```
    surface height / elevation
    surface normal
    slope
    curb / step / ramp boundary
    local traversability geometry
    ```
    
- 출력 형태는 구현에 따라 dense 2.5D map 또는 surface parameter map으로 둘 수 있다
    
    ```
    RoadSurfaceGeom_base
    또는
    GroundSurfaceGeom_base
    ```
    
- 이 head는 3D volume occupancy와 별도로, 주행/보행 가능한 표면의 형태를 안정적으로 주는 역할을 한다
- 즉 occupancy volume을 대체하는 것이 아니라, planner/control이 쓰기 쉬운 2.5D surface prior를 추가로 제공한다

---

### Voxel grid와 surface head의 관계

- voxel / occupancy head는 3D 공간 전체를 표현한다
    
    ```
    바닥 / 도로
    벽
    장애물
    사람 / 카트 / 로봇
    free space
    unknown space
    ```
    
- surface head는 그 3D 공간 안에서 이동 가능한 접지면을 직접 뽑는다
    
    ```
    x, y 위치의 ground height z
    surface normal
    slope
    traversability
    surface semantic
    ```
    
- Tesla가 말한 "voxel grid가 surface와 align된다"는 것은 두 출력이 독립 모델로 따로 예측되는 것이 아니라, 같은 `F_dec` feature에서 나오기 때문에 서로 일관되도록 학습된다는 뜻으로 해석할 수 있다
    
    ```
    shared F_dec
    ├─ occupancy / volume heads
    └─ surface geometry / semantic heads
    ```
    
- 따라서 바닥 근처의 occupancy boundary와 surface geometry는 서로 맞아야 한다
    
    ```
    O_base:
    ground surface 근처는 occupied / boundary
    그 위쪽은 free
    
    GroundSurfaceGeom_base:
    같은 위치의 ground height / normal / slope
    ```
    
- planner 관점에서는 두 출력의 역할이 다르다
    
    ```
    occupancy:
    충돌하면 안 되는 3D 공간
    
    surface:
    어디를 어떻게 지나갈 수 있는지
    ```

---

### Road / ground surface semantic head

- surface semantic은 volume semantic과 다르게, 바닥/도로 표면 위의 의미를 직접 예측한다
- 차량 기준 예시는 다음과 같다
    
    ```
    road
    lane area
    shoulder
    curb
    sidewalk
    crosswalk
    driveway
    ```
    
- 실내+실외 로봇 기준 예시는 다음과 같다
    
    ```
    floor
    carpet
    tile
    grass
    gravel
    ramp
    stair
    curb
    forbidden area
    ```
    
- 출력은 planner가 바로 사용할 수 있는 traversability / costmap의 강한 prior가 된다
    
    ```
    RoadSurfaceSem_base
    또는
    GroundSurfaceSem_base
    ```
    
- 즉 surface semantic은 `Sem3D_base`의 단순 축소판이 아니라, planner가 접지면과 이동 가능 영역을 해석하기 쉽게 만든 별도 head로 보는 것이 좋다

---

### CVPR'22 dynamic obstacle 해석

- `O_base`는 static obstacle 전용 output이 아니다
- 현재 scene에서 occupied인 모든 공간을 하나의 occupancy field로 예측한다
    
    ```
    wall
    tree
    stopped truck
    moving cart
    pedestrian
    mannequin / statue
    unknown obstacle
    ↓
    all occupied space
    ```
    
- 즉 pure occupancy는 "왜 occupied인가?"보다 "공간이 차 있는가?"를 먼저 예측한다
- semantic class는 control strategy를 돕는 보조 정보로 둘 수 있지만, occupancy 자체는 static / dynamic ontology gap에 빠지지 않는 것이 중요하다
- 이렇게 하면 moving object network와 static obstacle network 사이에서 object가 빠지는 문제를 줄일 수 있다

---

### 왜 occupancy flow가 필요한가?

- instantaneous occupancy만 있으면 움직이는 object도 현재 위치의 stationary obstacle처럼 해석될 수 있다
- 예를 들어 앞차를 따라갈 때, 현재 occupied voxel이 ego가 도달할 시점에는 이미 앞으로 이동해 있을 수 있다
- 따라서 occupancy와 함께 `Flow_base`를 예측한다
    
    ```
    O_base:
    현재 어떤 공간이 occupied인가?
    
    Flow_base:
    occupied region이 시간에 따라 어떻게 이동하는가?
    ```
    
- `Flow_base`는 object detection head의 object velocity가 아니라 voxel / occupancy region 단위의 motion field이다
- 입력으로는 여러 time step의 occupancy feature를 사용하고, ego-motion으로 현재 coordinate frame에 align한 뒤 deconv / upsample 계열에서 occupancy와 flow를 함께 예측한다

---

### 중요한 점

- 이 문서의 로봇 구현 예시에서 `O_base`는 20cm base voxel 단위의 기본 occupancy이다
- 이 결과만으로는 planning/control에 충분히 정밀하지 않을 수 있다
- 로봇 runtime 구현에서는 이 output에서 끝내지 않고, surface-shell 후보 voxel만 골라 Queryable MLP를 추가 호출한다
    
    ```
    20cm base voxel:
    전체 scene을 빠르게 표현
    
    runtime query:
    표면 / 경계 / 경로 주변 / 동적 객체 주변만 세밀하게 표현
    ```
    

---

### Robot runtime에서는 왜 occupancy logits 뒤가 아니라 F_dec에서 query하는가?

- refinement MLP는 최종 occupancy probability만 사용하는 것이 아니라, `F_dec`의 latent feature를 사용해야 한다
- occupancy probability는 이미 정보가 많이 압축된 값이다
    
    ```
    나쁜 구조:
    O_base → MLP refinement
    
    좋은 구조:
    F_dec + local query coordinate → MLP refinement
    ```
    
- 따라서 `O_base`는 refine할 위치를 고르는 데 사용하고, 실제 fine prediction은 `F_dec`에서 가져온 feature를 사용한다

---

### 이 단계의 의미

- 이 단계는 다음 역할을 동시에 한다
    
    ```
    1. road surface geometry / semantics 생성
    2. 전체 공간에 대한 dense occupancy 생성
    3. occupancy flow 생성
    4. sub-voxel shape clue 생성
    5. 3D semantics 생성
    6. 이후 query scheduling을 위한 auxiliary score 생성
    ```
    
- 대부분의 공간은 `O_base`를 그대로 사용한다
- 중요한 surface-shell 영역만 C 레이어의 runtime query scheduling으로 Queryable MLP를 추가 호출한다

---

### Flow_pre와 Flow_base의 차이

- `Flow_pre`는 optional low-res flow-aware correction branch를 켰을 때만 사용하는 중간 flow이다.
    
    ```
    F_st
    ↓
    Flow_pre
    ↓
    dynamic feature correction
    ```
    
- `Flow_base`는 deconv 이후 high-res voxel feature에서 예측하는 최종 base occupancy flow이다.
    
    ```
    F_dec
    ↓
    Flow_base
    ```
    
- 즉 두 flow의 역할은 다르다.
    
    ```
    Flow_pre:
    low-res temporal feature 보정용
    
    Flow_base:
    최종 occupancy flow 출력용
    ```
    
- 최종적으로는 `Flow_base`와 이후 queryable refinement에서 얻은 `Flow_query`를 함께 사용할 수 있다.

---

## C1. Robot-specific runtime query scheduling: surface-shell refine score calculation

### 역할

- 이 단계는 Tesla 공개 architecture block이 아니라, `Queryable Outputs`를 실시간 로봇에서 어디에 호출할지 정하는 runtime scheduling 정책이다
- `O_base`, `Unc_base`, `Bnd_base`, `Flow_base`, planner path 정보를 이용해 **sub-voxel query를 날릴 voxel**을 고른다
- 핵심은 모든 voxel을 refine하지 않는 것이다
- refine 대상은 대부분 다음 영역이다
    
    ```
    occupied / free 경계
    물체 표면
    planner path 주변 장애물 경계
    동적 객체 외곽
    uncertainty가 높은 얇은 구조물 후보
    visibility / occlusion frontier
    ```
    

---

### 왜 surface-shell 중심인가?

- 완전히 빈 공간은 굳이 query할 필요가 없다
    
    ```
    p_occ ≈ 0
    주변도 free
    → refine 필요 낮음
    ```
    
- 완전히 물체 내부인 공간도 굳이 query할 필요가 적다
    
    ```
    p_occ ≈ 1
    주변도 occupied
    → planner 입장에서는 그냥 막힌 공간
    ```
    
- 가장 중요한 곳은 free와 occupied가 만나는 경계이다
    
    ```
    free | occupied
    → 실제 obstacle surface 가능성 높음
    → sub-voxel refinement 필요
    ```
    

---

### 기준 1: Boundary / surface candidate

- 주변 voxel과 occupancy 차이가 큰 곳은 surface 후보이다
    
    ```
    boundary_score ≈ |∇O_base|
    ```
    
- 직관적으로는 다음과 같다
    
    ```
    주변 voxel과 occupancy가 비슷함
    → 내부 또는 빈 공간일 가능성 큼
    → refine 우선순위 낮음
    
    주변 voxel과 occupancy가 크게 다름
    → 경계일 가능성 큼
    → refine 우선순위 높음
    ```
    

---

### 기준 2: Uncertainty

- occupancy 확률이 애매한 곳이다
    
    ```
    p_occ ≈ 0.5
    ```
    
- 단순한 uncertainty는 다음처럼 둘 수 있다
    
    ```
    uncertainty = 1 - |2p - 1|
    ```
    
- 의미는 다음과 같다
    
    ```
    p = 0.5 → uncertainty 높음
    p = 0.0 또는 1.0 → uncertainty 낮음
    ```
    

---

### 기준 3: Surface shell mask

- 좁은 의미의 surface-shell 후보는 다음 조건으로 만들 수 있다
    
    ```
    1. p_occ가 threshold 이상인 voxel
    2. 이웃 voxel 중 free 또는 low occupancy voxel이 존재
    ```
    
- 즉, occupied voxel 중에서도 바깥과 맞닿은 voxel만 surface-shell로 본다
    
    ```
    완전 내부 occupied voxel:
    주변도 occupied
    → surface-shell 아님
    
    경계 occupied voxel:
    주변에 free 존재
    → surface-shell 후보
    ```
    
- 다만 최종 refine 후보는 좁은 surface-shell만 쓰지 않는다
- uncertainty / visibility frontier / planner relevance / dynamic boundary는 아직 `p_occ` threshold를 넘지 않아도 refine 후보가 될 수 있다
    
    ```
    refine_candidate =
    surface_shell
    OR uncertainty
    OR visibility_frontier
    OR planner_relevant
    OR dynamic_boundary
    ```
    

---

### 기준 4: Near-field

- ego vehicle 또는 robot에 가까운 영역은 더 중요하다
    
    ```
    0m ~ 5m   → 높은 가중치
    5m ~ 10m  → 중간 가중치
    10m 이상 → 낮은 가중치
    ```
    
- 이 문서의 20cm 예시 기준으로는 근거리에서 voxel 하나의 오차도 collision checking에 직접 영향을 줄 수 있다

---

### 기준 5: Planning relevance

- local planner의 후보 경로 주변은 높은 우선순위를 갖는다
    
    ```
    path corridor 주변
    swept volume 주변
    collision margin 주변
    ```
    
- 모든 장애물 표면이 같은 중요도를 갖는 것은 아니다
- 실제로 중요한 것은 ego trajectory와 충돌할 가능성이 있는 표면이다

---

### 기준 6: Dynamic / flow importance

- `Flow_base`가 크거나 temporal inconsistency가 큰 영역은 refinement 우선순위가 높다
    
    ```
    dynamic_score 높음
    flow magnitude 큼
    이전 frame과 현재 frame occupancy 차이 큼
    ```
    
- 사람, 차량, 로봇, 카트, moving obstacle 주변은 surface 위치와 flow가 중요하다

---

### 최종 refine score

- 초기 구현에서는 다음처럼 단순하게 시작할 수 있다
    
    ```
    S_refine =
    λ1 * boundary_score
    + λ2 * uncertainty
    + λ3 * near_field_score
    ```
    
- 확장 버전은 다음과 같다
    
    ```
    S_refine =
    λ1 * boundary_score
    + λ2 * uncertainty
    + λ3 * surface_shell_score
    + λ4 * near_field_score
    + λ5 * planning_relevance
    + λ6 * dynamic_score
    + λ7 * visibility_frontier_score
    ```
    

---

### refine에서 제외할 영역

- 다음 영역은 기본적으로 refinement 대상에서 제외한다
    
    ```
    확실한 free space
    확실한 occupied interior
    ego path와 무관한 먼 static wall 내부
    semantic 변화가 없는 넓은 평면 내부
    ```
    
- 단, 얇은 구조물이나 occlusion boundary는 base occupancy가 확실하지 않을 수 있으므로 uncertainty 기준으로 일부 남긴다

---

### 출력

- 이 단계의 출력은 모든 base voxel에 대한 refine score이다
    
    ```
    S_refine ∈ R^(N_x × N_y × N_z)
    ```
    
- 이 score는 다음 단계에서 Top-K surface voxel selection에 사용된다

---

## C2. Robot-specific runtime query scheduling: quota-based Top-K surface voxel selection

### 역할

- 이 단계도 Tesla 공개 그림의 독립 block이 아니라, 제한된 query budget을 어디에 쓸지 정하는 runtime optimization이다
- `S_refine`이 높은 voxel 중 실제로 sub-voxel query를 수행할 voxel을 선택한다
- 이 단계는 Octree처럼 recursive하게 voxel을 쪼개는 단계가 아니다
- GPU에서 계산량을 일정하게 유지하기 위해 매 frame 선택 voxel 수를 제한한다
    
    ```
    S_refine
    ↓
    top-K / quota-based top-K
    ↓
    selected surface voxel indices
    ```
    

---

### 왜 단순 Top-K만 쓰면 부족한가?

- 단순 Top-K는 score가 큰 한 종류의 영역에 선택이 몰릴 수 있다
- 예를 들어 큰 벽의 boundary가 score를 많이 차지하면, 작은 동적 객체가 선택되지 않을 수 있다
- 그래서 category별 quota를 두는 것이 안전하다
    
    ```
    boundary quota
    near-field quota
    planner corridor quota
    dynamic quota
    uncertainty quota
    visibility frontier quota
    ```
    

---

### 추천 quota 예시

- 전체 K가 4096이라고 하면 다음처럼 나눌 수 있다
    
    ```
    K_total = 4096
    
    K_boundary   = 1536
    K_near       = 768
    K_planner    = 768
    K_dynamic    = 512
    K_uncertain  = 384
    K_random     = 128
    ```
    
- `K_random`은 coarse prediction이 놓친 얇은 물체나 예외 상황을 탐색하기 위한 probe query 용도이다
- 처음 구현에서는 random을 생략해도 되지만, 얇은 구조물 detection을 고려하면 소량 유지하는 것이 좋다

---

### K의 의미

- K는 연산량과 품질을 직접 조절하는 하이퍼파라미터다
    
    ```
    K = 2048  → 빠름, refinement 적음
    K = 4096  → 균형
    K = 8192  → 더 정밀하지만 무거움
    K = 16384 → 고품질, embedded에서는 부담 큼
    ```
    

---

### 선택 대상

- 선택 대상은 모든 voxel이 아니라 surface-shell 후보 voxel이다
    
    ```
    후보:
    boundary voxel
    uncertainty voxel
    path 주변 voxel
    dynamic object 외곽 voxel
    visibility frontier voxel
    
    제외:
    확실한 free voxel
    확실한 occupied interior voxel
    ```
    

---

### 출력

- 선택된 voxel index
    
    ```
    selected_idx ∈ R^K
    ```
    
- 선택된 voxel의 feature
    
    ```
    F_selected = gather(F_dec, selected_idx)
    F_selected ∈ R^(K × C)
    ```
    
- 선택된 voxel의 coarse output도 함께 가져올 수 있다
    
    ```
    O_selected
    Sem_selected
    Flow_selected
    Unc_selected
    ```
    

---

## C3. Robot-specific runtime query generation: adaptive in-voxel query generation

### 역할

- 이 단계는 Tesla의 `Queryable Outputs`를 실제로 호출하기 위한 로봇용 adaptive query 생성 정책이다
- 선택된 surface voxel 내부에 arbitrary `x, y, z` point query를 생성한다
- 이 query는 base voxel을 같은 비율의 작은 voxel로 고정 분할하기 위한 것이 아니다
- Queryable MLP가 base voxel 내부의 continuous occupancy field를 물어보고, 표면 boundary가 어디 있는지 찾기 위한 local query이다
    
    ```
    selected surface voxel
    ↓
    adaptive local point query
    ↓
    Queryable MLP
    ↓
    in-voxel occupancy boundary
    ```
    
- 이 단계는 global Octree가 아니다
- 전체 grid는 base voxel grid로 유지하고, 선택된 surface voxel 내부에서만 local adaptive query를 수행한다

---

### Query는 자유롭고, rendering은 선택한다

- Queryable MLP는 base voxel 내부의 임의 local coordinate를 받을 수 있다
- 따라서 query 위치는 같은 비율의 `2×2×2`, `3×3×3` 격자에 묶이지 않는다
- 다만 시각화는 두 가지 방식 중 하나로 선택할 수 있다
    
    ```
    option A:
    adaptive cuboid rendering
    - Tesla 데모처럼 작은 axis-aligned cuboid 조합으로 표시
    - cuboid 위치와 크기는 고정 비율일 필요 없음
    
    option B:
    local surface rendering
    - query 결과의 iso-surface를 작은 local patch로 표시
    - 더 부드럽지만 Tesla 데모의 blocky voxel 느낌은 줄어듦
    ```
    
- 이 문서의 기본 시각화 목표는 option A이다
- 즉 query는 자유롭게 찍고, 최종 표시는 Tesla 데모처럼 axis-aligned partial cuboid로 한다

---

### Adaptive boundary search

- 먼저 selected surface voxel 내부에 소수의 initial probe point를 찍는다
    
    ```
    cube corners
    face centers
    voxel center
    edge centers
    optional random / stratified samples
    ```
    
- 이 initial probe는 최종 shape을 같은 비율로 자르기 위한 grid가 아니다
- 목적은 occupied / free가 바뀌는 boundary 후보를 찾는 것이다
    
    ```
    p_occ(q_a) > threshold
    p_occ(q_b) < threshold
    ↓
    q_a와 q_b 사이에 surface boundary 가능성
    ```
    
- boundary 후보가 있는 구간에는 추가 query를 더 찍는다
    
    ```
    bisection
    secant-style refinement
    gradient-guided refinement
    local marching-cubes-like refinement
    ```
    
- 이렇게 하면 base voxel 내부의 표면 위치가 균일 sub-voxel 격자에 고정되지 않고, Queryable MLP의 continuous occupancy field에 따라 더 자유롭게 결정된다

---

### Exposed face / edge / corner 기반 query 방향

- 모든 surface voxel을 같은 방식으로 3D 전체 query하지 않는다
- 먼저 이웃 occupancy를 보고 어떤 면이 free / unknown 공간과 맞닿아 있는지 계산한다
    
    ```
    exposed_face_mask = {
        +x: neighbor(+x) is free or unknown,
        -x: neighbor(-x) is free or unknown,
        +y: neighbor(+y) is free or unknown,
        -y: neighbor(-y) is free or unknown,
        +z: neighbor(+z) is free or unknown,
        -z: neighbor(-z) is free or unknown
    }
    ```
    
- 노출된 면이 하나뿐이면 해당 face normal 방향 위주로 query한다
    
    ```
    top face only:
    exposed +z
    → z 방향으로만 boundary refinement 집중
    
    side face only:
    exposed +x 또는 -x / +y 또는 -y
    → 해당 side normal 방향으로 boundary refinement 집중
    ```
    
- 두 면이 동시에 노출되면 edge voxel로 보고 두 normal 방향을 함께 query한다
    
    ```
    exposed +x and +z
    → side + top edge
    → x 방향 refinement + z 방향 refinement
    → edge 주변 추가 query
    ```
    
- 세 면 이상이 노출되면 corner / thin-structure 후보로 보고 query budget을 더 많이 준다
    
    ```
    exposed +x, +y, +z
    → corner voxel
    → 3D adaptive query 증가
    ```
    
- 이렇게 하면 윗면만 보이는 voxel은 높이 방향으로만 더 세밀하게 깎고, 옆면만 보이는 voxel은 해당 옆면 방향으로만 깎을 수 있다
- 모서리 / 꼭짓점 / 얇은 구조물은 여러 방향에서 boundary가 생길 수 있으므로 더 많은 adaptive query를 사용한다

---

### Query budget 예시

- query budget은 exposed face 개수와 planner relevance에 따라 달리 둔다
    
    ```
    one exposed face:
    M_i 낮음
    normal direction line search 중심
    
    two exposed faces:
    M_i 중간
    two-direction boundary search + edge probe
    
    three or more exposed faces:
    M_i 높음
    corner / thin object용 3D adaptive query
    ```
    
- camera visibility는 가중치로 사용할 수 있지만, exposed face 판단을 camera visibility에만 의존하면 안 된다
- 기본 exposed face는 occupancy neighbor / occupancy gradient로 계산하고, camera visibility는 query priority를 조정하는 보조 신호로 쓴다
    
    ```
    exposed face:
    occupancy neighbor / gradient 기반
    
    query priority:
    camera visibility
    planner path
    dynamic score
    uncertainty
    ```
    
- 최종 query 수는 voxel마다 다르다
    
    ```
    M_i = f(
        number_of_exposed_faces,
        boundary_score,
        uncertainty,
        planner_relevance,
        dynamic_score,
        visibility_score
    )
    ```

---

### Uniform query는 fallback / debug 용도

- `2×2×2`, `3×3×3`, `4×4×4` 같은 균일 query는 구현 검증이나 fallback으로는 유용하다
    
    ```
    2 × 2 × 2 → quick debug
    3 × 3 × 3 → stable visualization fallback
    4 × 4 × 4 → denser fallback
    ```
    
- 하지만 이 문서의 목표 구현에서는 균일 subdivision이 최종 surface shape을 결정하지 않는다
- 최종 surface voxel carving은 adaptive query와 boundary search가 결정한다

---

### query 위치

- 각 query는 selected voxel 내부의 local coordinate를 가진다
- local coordinate는 다음 연속 범위 안의 임의 위치이다
    
    ```
    q_local ∈ [-0.5, 0.5]^3
    ```
    
- 예시:
    
    ```
    initial probe:
    q_local = (-0.5, -0.5, -0.5)
    q_local = ( 0.0,  0.0,  0.0)
    q_local = ( 0.5,  0.5,  0.5)
    
    adaptive refinement:
    q_local = (-0.17, 0.23, -0.04)
    q_local = (-0.11, 0.19, -0.02)
    ```
    
- 이 local coordinate는 MLP가 selected voxel 내부에서 query point가 어디에 있는지 알게 해준다

---

### world coordinate 변환

- local query는 실제 world / ego 좌표 query point로 변환된다
    
    ```
    p_query = voxel_center + q_local * voxel_size
    ```
    
- 예를 들어 voxel_size가 0.2m이면:
    
    ```
    p_query = voxel_center + q_local * 0.2m
    ```
    

---

### surface-shell query의 의미

- 모든 selected voxel 내부를 무조건 full grid로 보는 것이 아니라, selected voxel 자체가 이미 surface-shell 후보이다
- 따라서 전체 3D 공간 기준으로 보면 query는 대부분 surface 주변에 집중된다
- 각 selected voxel마다 필요한 query 수 `M_i`는 달라질 수 있다
    
    ```
    전체 공간:
    coarse base voxel로 유지
    
    surface-shell:
    adaptive query로 boundary 위치 탐색
    ```
    

---

### 출력

- local query set
    
    ```
    q_local_packed ∈ R^(Q × 3)
    query_offsets ∈ R^(K + 1)
    ```
    
- ego/world coordinate query set
    
    ```
    p_query_packed ∈ R^(Q × 3)
    ```
    
- selected feature repeat
    
    ```
    F_query_context ∈ R^(Q × C)
    ```
    
- 여기서 `Q = Σ_i M_i`이고, `M_i`는 selected voxel마다 다를 수 있다


---

### Tesla 데모 표현과의 관계

- 이 구조를 사용하면 최종 시각화에서 다음 표현이 가능하다
    
    ```
    완전히 찬 base voxel
    일부만 찬 voxel
    계단식 경계
    작은 직육면체 block으로 튀어나온 표면
    ```
    
- 하지만 각 block의 면은 여전히 축 정렬되어 있다
- 즉, 기본 시각화는 사다리꼴/평행사변형 mesh가 아니라 cuboid 기반 partial voxel rendering이다
- 중요한 점은 cuboid가 반드시 같은 비율의 균일 sub-voxel일 필요는 없다는 것이다

---

### 목표 시각화: base voxel 내부 partial occupancy

- 구현 목표는 Tesla CVPR 데모처럼 fixed-size voxel grid 위에서 표면 voxel만 내부를 더 자유롭게 채우는 것이다
- 이 문서의 로봇 구현 예시에서는 base voxel 한 변을 20cm로 둔다
- 여기서 "자유롭게"는 base voxel의 외곽 면을 기울어진 polyhedron으로 변형한다는 뜻이 아니다
- base voxel 내부의 임의 query point에서 continuous occupancy를 물어보고, 그 결과로 occupied region의 partial shape을 만든다는 뜻이다
    
    ```
    base voxel:
    fixed-size cube
    이 문서 예시: 20cm
    
    surface voxel:
    base cube 내부의 adaptive partial occupancy shape
    
    visualization:
    occupied region만 작은 cuboid 조합 또는 local surface로 표시
    ```
    
- 따라서 전체 공간 grid는 여전히 fixed-size base grid이고, 표면 주변만 effective resolution이 더 높아진다
    
    ```
    non-surface region:
    base cube 그대로 렌더링
    
    surface-shell region:
    base cube 내부 임의 위치를 adaptive하게 query
    occupancy boundary를 찾음
    occupied partial region만 렌더링
    ```
    
- 이 방식은 Tesla 데모 이미지처럼 벽/장애물 표면이 계단식 작은 cuboid 조합으로 보이는 표현과 양립한다
- 같은 비율의 균일 shrink / expand가 아니라, Queryable MLP가 찾은 local boundary에 따라 voxel 내부 shape이 달라지는 구조이다

---

### SubVoxelShape와 Queryable MLP의 역할 분담

- `SubVoxelShape_base`는 각 base voxel 내부의 local shape clue를 주는 optional volume output으로 둘 수 있다
- `Queryable MLP`는 실제 `x, y, z` query point에서 continuous occupancy를 예측한다
- 이 문서의 기본 구현은 Queryable MLP가 최종 sub-block occupancy를 query point별로 예측하는 방식으로 둔다
- `SubVoxelShape_base`를 직접 local mask 확정 head가 아니라 Queryable MLP가 표면 voxel을 더 잘 깎도록 돕는 shape prior / conditioning feature로 사용하는 것은 optional 구현 해석이다
- Tesla 공개 자료만으로 `sub-voxel shape information`의 정확한 내부 역할은 확정하지 않는다
- 시각화 pipeline은 다음처럼 고정한다
    
    ```
    F_dec
    ↓
    O_base / SubVoxelShape_base
    ↓
    surface-shell voxel 선택
    ↓
    voxel 내부 adaptive point query
    ↓
    Queryable MLP
    ↓
    continuous occupancy probability
    ↓
    boundary search / carving
    ↓
    adaptive partial cuboid or local surface rendering
    ```
    
- 이 문서의 구현에서는 표면 voxel을 "깎는" 최종 판단을 `SubVoxelShape_base`가 아니라 Queryable MLP가 담당하도록 둔다
- 이 구조는 Tesla 그림의 `x, y, z → MLP → continuous occupancy probability` Queryable Outputs와 잘 맞는 구현 선택이다

---

### Adaptive partial rendering rule

- 각 selected surface voxel 내부에서 Queryable MLP를 여러 위치에 호출한다
- MLP output이 threshold를 가로지르는 위치를 찾아 occupancy boundary를 추정한다
    
    ```
    p_occ = QueryableMLP(F_query, q_local, PE(p_query))
    
    p_occ > threshold:
    occupied side
    
    p_occ < threshold:
    free side
    
    sign change:
    local boundary candidate
    ```
    
- boundary를 찾은 뒤에는 두 가지 rendering이 가능하다
    
    ```
    Tesla-demo-like default:
    adaptive axis-aligned cuboid rendering
    - cuboid 위치와 크기가 voxel마다 달라질 수 있음
    - 같은 S×S×S 비율로 고정하지 않음
    
    optional smooth view:
    local iso-surface rendering
    - 더 부드러운 surface patch
    - blocky voxel demo 느낌은 줄어듦
    ```
    
- 기본 구현은 Tesla 데모 느낌을 위해 adaptive axis-aligned cuboid rendering을 사용한다
    
    ```
    for each selected surface voxel:
        exposed_face_mask = compute_exposed_faces(O_base)
        initial_query(exposed_face_mask)
        boundary_candidates = find_sign_changes()
        refined_boundary = adaptive_refine(boundary_candidates, exposed_face_mask)
        cuboids = fit_axis_aligned_partial_cuboids(refined_boundary)
        render(cuboids)
    ```
    
- 이때 cuboid는 전역 dense grid로 승격하지 않는다
- `FineOverlay` 안에 선택된 voxel의 local adaptive shape으로만 sparse하게 저장한다
    
    ```
    FineOverlay[selected_voxel_idx] = {
        cuboid_1: center, size, p_occ, semantic,
        cuboid_2: center, size, p_occ, semantic,
        ...
    }
    ```

---

## 13. Tesla public queryable output decoder: Queryable MLP point decoder

### 역할

- `x, y, z` 임의 query point에 대해 Tesla-style 기본 출력으로 occupancy / 3D semantics를 예측한다
- 로봇용 확장에서는 flow / distance field / uncertainty도 함께 예측할 수 있다
- Tesla 그림의 `Queryable Outputs`에 해당한다
- 원리적으로는 arbitrary location query가 가능하지만, 실시간 시스템에서는 앞 단계의 surface-shell scheduling으로 호출할 point 수를 제한한다
- 이 단계는 fixed-size base voxel grid보다 더 세밀한 sub-voxel 정보를 얻기 위한 MLP decoder이다
    
    ```
    F_dec
    +
    p_query
    +
    q_local
    ↓
    MLP
    ↓
    continuous occupancy / 3D semantics
    optional: flow / distance / uncertainty
    ```
    

---

### 입력 feature

- 단순히 selected voxel center feature만 쓰는 방법도 가능하다
    
    ```
    F_selected = gather(F_dec, selected_idx)
    ```
    
- 더 좋은 방식은 query point 위치에서 `F_dec`를 trilinear interpolation하는 것이다
    
    ```
    F_query = trilinear_sample(F_dec, p_query)
    ```
    
- 이렇게 하면 query point가 voxel 내부 어디에 있는지에 따라 더 부드러운 feature를 사용할 수 있다

---

### MLP 입력

- MLP 입력은 다음처럼 구성한다
    
    ```
    MLP input =
    [
        F_query,
        q_local,
        PE(p_query),
        O_selected,
        Flow_selected,
        Unc_selected
    ]
    ```
    
- 최소 구현은 다음만 사용해도 된다
    
    ```
    MLP input = [F_query, q_local]
    ```
    
- 확장 구현에서는 absolute positional encoding과 coarse logits를 추가한다
    
    ```
    MLP input = [F_query, q_local, PE(p_query), coarse_logits]
    ```
    

---

### 출력

- Tesla-style 기본 출력은 query point의 continuous occupancy / 3D semantics이다
    
    ```
    O_query_packed     ∈ R^(Q × 1)
    Sem3D_query_packed ∈ R^(Q × C_sem)
    ```
    
- 로봇용 확장 출력은 다음과 같다
    
    ```
    Flow_query_packed ∈ R^(Q × 2 or 3)
    Dist_query_packed ∈ R^(Q × 1)
    Unc_query_packed  ∈ R^(Q × 1)
    ```
    
- 여기서 `Q = Σ_i M_i`이고, `M_i`는 selected surface voxel마다 adaptive하게 달라질 수 있다
    

---

### distance field를 같이 예측하는 이유

- occupancy는 단순히 막힘/비어있음을 말한다
    
    ```
    occupied / free
    ```
    
- distance field는 가장 가까운 장애물 표면까지의 거리를 표현한다
    
    ```
    distance to nearest surface
    ```
    
- planner 입장에서는 distance field가 유용하다
    
    ```
    trajectory가 장애물에서 얼마나 떨어져 있는가?
    collision margin이 충분한가?
    DWA / MPC / trajectory optimization에서 cost로 쓸 수 있는가?
    ```
    

---

### 왜 batched 방식인가?

- voxel마다 if문으로 처리하면 GPU 효율이 낮다
- 선택된 K개 voxel에서 나온 adaptive query들을 하나의 packed batch로 펴서 처리한다
    
    ```
    Q queries
    ↓
    MLP
    ↓
    Q predictions
    ```
    
- 예를 들어:
    
    ```
    K = 4096
    평균 M_i = 20
    Q ≈ 81,920 queries
    ```
    
- 이 방식은 recursive Octree branching보다 GPU 친화적이다

---

### 이 단계의 의미

- 이 구조는 완전한 implicit scene representation이 아니다
- 전체 공간은 dense base voxel로 유지하고, 중요한 surface-shell만 implicit point query로 세밀화한다
    
    ```
    Dense base voxel grid
    +
    Sparse implicit surface-shell refinement
    ```
    

---

## C4. Robot-specific visualization / planner overlay: sparse surface-shell fine occupancy overlay

### 역할

- 이 단계는 Tesla 공개 architecture의 별도 neural block이라기보다, Queryable MLP 결과를 로봇 map과 Tesla-demo-like visualization에 반영하는 표현 방식이다
- implicit point-query decoder가 예측한 fine 결과를 최종 map 표현에 반영한다
- 전체 공간을 dense fine voxel grid로 만들지 않는다
- 대신 dense base occupancy 위에 sparse fine occupancy를 overlay한다
    
    ```
    Dense base occupancy
    +
    Sparse surface-shell fine overlay
    ```
    

---

### 왜 full dense fine map을 만들지 않나?

- 모든 20cm voxel을 10cm sub-voxel로 바꾸면 전체 voxel 수가 8배 증가하고, 약 6.7cm sub-voxel로 바꾸면 27배 증가한다
- 예를 들어:
    
    ```
    N_x × N_y × N_z
    ↓
    2N_x × 2N_y × 2N_z  # 10cm
    또는
    3N_x × 3N_y × 3N_z  # 약 6.7cm
    ```
    
- 이는 실시간 시스템에서 부담이 크다
- 따라서 필요한 surface-shell 영역만 sparse하게 유지한다

---

### overlay의 의미

- base map은 전체 공간을 표현한다
    
    ```
    O_base[x, y, z]
    ```
    
- fine overlay는 선택된 voxel 내부의 adaptive partial shape만 표현한다
    
    ```
    FineOverlay[selected_voxel_idx] =
    local adaptive cuboid / surface shape
    ```
    
- 최종적으로는 다음처럼 저장한다
    
    ```
    FineOverlay = {
        selected_idx_1: [adaptive_cuboid_1, adaptive_cuboid_2, ...],
        selected_idx_2: [adaptive_cuboid_1, adaptive_cuboid_2, ...],
        ...
    }
    ```
    

---

### adaptive partial voxel rendering

- 시각화할 때는 Queryable MLP가 찾은 local boundary를 작은 axis-aligned cuboid 조합으로 렌더링할 수 있다
- cuboid의 위치와 크기는 voxel마다 달라질 수 있으며, 같은 비율의 `2×2×2` 또는 `3×3×3` grid에 고정하지 않는다
    
    ```
    Queryable MLP
    ↓
    adaptive boundary search
    ↓
    fit local occupied cuboids
    ↓
    render adaptive partial voxel
    ```
    
- 이렇게 하면 다음 표현이 가능하다
    
    ```
    완전히 찬 voxel
    일부만 찬 voxel
    계단식 경계
    작은 cuboid로 표현되는 표면 detail
    ```
    
- 단, 이 표현은 mesh surface가 아니라 block-based partial occupancy이다

---

### planner용 변환

- planner에는 raw point query 결과를 그대로 넘기기보다 다음 형태로 변환하는 것이 좋다
    
    ```
    local high-res occupancy grid
    distance field
    costmap
    swept-volume collision map
    ```
    
- 즉, 시각화용 partial voxel과 planner용 map은 분리해서 생각할 수 있다
    
    ```
    visualization:
    sparse fine block rendering
    
    planning:
    local high-res occupancy / distance field
    ```
    

---

### 출력

- 이 단계의 출력은 다음과 같다
    
    ```
    DenseBaseMap
    SparseFineOverlay
    LocalHighResOccupancy
    DistanceField
    ```
    
- 전체 map을 완전한 dense fine voxel로 바꾸지 않고, 필요한 영역만 sparse하게 유지한다

---

## D3. Optional research extension: DuOcc-style QueryAgg injection

### 역할

- 이 branch는 Tesla strict main path에는 없는 optional extension이다
- 목적은 dynamic object 주변 occupancy feature를 object / instance query로 보강하는 것이다
    
    ```
    dynamic object / instance query
    ↓
    image / voxel feature aggregation
    ↓
    dynamic object 주변 voxel feature 보강
    ↓
    occupancy / semantics 품질 향상
    ```
    
- Tesla public main path는 dynamic object를 occupancy target과 occupancy flow output에 포함하는 방식에 더 가깝다
- DuOcc-style QueryAgg는 object query를 다시 occupancy feature에 주입하는 연구적 확장으로 분리한다

---

## D4. Optional separate Tesla task branch: object / agent sparse prediction branch

### 역할

- 이 branch는 occupancy main path와 별도로 동작하는 object / agent 중심 예측 구조이다
- Tesla AI Day의 `Object Prediction & Properties` 그림에 해당한다
- dense spatial feature에서 중요한 object / agent 후보만 sparse하게 뽑고, agent별 video module과 head를 적용한다
    
    ```
    image features
    ↓
    transformer / spatial feature
    ↓
    video module
    ↓
    detection head
    ↓
    convert to sparse
    ↓
    sparse agent features
    ↓
    per-agent video module + head
    ↓
    future trajectory / shape mesh / pedestrian pose / agent properties
    ```
    

---

### occupancy flow와의 차이

- occupancy flow는 voxel / occupancy cell 단위의 motion field이다
    
    ```
    어느 공간 cell이 어느 방향으로 움직이는가?
    ```
    
- object / agent branch는 object identity와 agent-level 속성을 다룬다
    
    ```
    어떤 agent가 있고, 앞으로 어디로 갈 것인가?
    shape / pose / intent는 무엇인가?
    ```
    
- 따라서 둘은 비슷한 목적을 갖지만 표현 단위가 다르다
    
    ```
    occupancy flow:
    dense or semi-dense voxel-level motion
    
    object / agent branch:
    sparse instance-level prediction
    ```
    

---

### DuOcc QueryAgg와의 관계

- DuOcc의 Query-guided Feature Aggregation은 dynamic object의 instance-level query feature를 image feature에서 추출하고, 해당 voxel region에 주입해 occupancy 품질을 높이는 구조이다
- 따라서 이 optional branch와 개념적으로 닮아 있다
- 특히 object / agent query가 dynamic occupancy 주변 feature를 보강한다는 관점에서는 네가 말한 것처럼 매우 가까운 아이디어이다
- 차이는 다음과 같다
    
    ```
    DuOcc QueryAgg:
    instance query를 occupancy voxel feature 보강에 사용
    
    Tesla object / agent branch:
    sparse agent를 뽑고 future trajectory / shape mesh / pose 같은 agent property를 예측
    ```
    
- 로봇용 구현에서는 두 아이디어를 분리해서 쓰는 것이 좋다
    
    ```
    main occupancy:
    dense occupancy / flow / semantics
    
    optional QueryAgg-style injection:
    dynamic object 주변 voxel feature 보강
    
    optional object / agent branch:
    agent trajectory / pose / shape prediction
    ```
    
- 정리하면 DuOcc QueryAgg는 object query를 occupancy 품질 향상에 직접 주입하는 방식이고, Tesla의 object / agent branch는 sparse agent feature로 미래 궤적, shape, pose 같은 agent property를 예측하는 별도 branch에 가깝다
    

---

## 15. Final Tesla-style occupancy representation + planner maps

### 역할

- 최종 occupancy 표현을 만든다
- 최종 출력은 Tesla 그림처럼 surface / volume / queryable output을 함께 갖는 hybrid structure이다
    
    ```
    1. Surface Outputs
       - road surface geometry
       - road surface semantics
    
    2. Volume Outputs
       - occupancy
       - occupancy flow
       - sub-voxel shape information
       - 3D semantics
    
    3. Queryable Outputs
       - continuous occupancy probability
       - continuous 3D semantics
    
    4. Robot/planner-friendly derived maps
       - sparse fine overlay
       - local high-res occupancy
       - distance field / costmap
    ```
    

---

### Dense base occupancy

- 이 문서의 로봇 구현 예시에서는 전체 공간을 20cm base voxel grid로 표현한다
    
    ```
    O_base ∈ R^(N_x × N_y × N_z)
    ```
    
- `O_base`는 static / dynamic을 먼저 나누지 않고, 현재 scene에서 차 있는 모든 공간을 포함한다
- 따라서 unknown obstacle이나 ontology가 애매한 물체도 우선 occupancy로 잡는 것이 목표이다
- 이 map은 전체 scene understanding에 사용된다
    
    ```
    free / occupied / unknown
    semantic class
    occupancy flow
    ```
    

---

### Sparse adaptive surface-shell fine overlay

- surface-shell 후보 영역만 adaptive query와 partial cuboid로 더 정밀하게 표현한다
    
    ```
    FineOverlay = {
        selected_voxel_idx:
        adaptive partial shape / cuboid list
    }
    ```
    
- 이 overlay는 다음 영역에 집중된다
    
    ```
    occupied/free 경계
    물체 표면
    planner path 주변 장애물
    동적 객체 외곽
    uncertainty 높은 얇은 구조물 후보
    occlusion frontier
    ```
    

---

### local high-res occupancy

- planner가 필요한 근거리 영역은 sparse fine overlay를 local grid로 변환할 수 있다
- 예를 들어 ego 주변 10m 또는 local planner corridor 주변만 10cm grid로 변환한다
    
    ```
    sparse fine overlay
    ↓
    local high-res occupancy grid
    ```
    
- 이 local high-res grid는 DWA, MPC, trajectory rollout, collision checking에 사용할 수 있다

---

### distance field

- local high-res occupancy에서 distance field를 계산할 수 있다
    
    ```
    occupancy
    ↓
    distance transform
    ↓
    distance field
    ```
    
- distance field는 각 위치가 가장 가까운 장애물 표면에서 얼마나 떨어져 있는지를 나타낸다
    
    ```
    distance > safety_margin → 안전
    distance < safety_margin → 위험
    ```
    

---

### occupancy flow

- 동적 객체 또는 움직이는 occupancy에 대해 flow를 유지한다
    
    ```
    Flow_base
    +
    Flow_query
    ```
    
- base flow는 전체 공간 motion을 제공하고, query flow는 surface-shell / dynamic object 주변 motion을 보강한다

---

### Tesla-style 해석

- 이 구조는 다음 공개 설명과 잘 맞는다
    
    ```
    fixed-size voxel grid
    +
    per-voxel feature map
    +
    MLP with 3D spatial point queries
    +
    arbitrary location prediction
    ```
    
- 즉, voxel 자체를 Octree처럼 recursive하게 분할하는 것이 아니다
- MLP는 원리적으로 arbitrary `x, y, z` query가 가능하다
- 실시간 로봇에서는 계산량을 제한하기 위해 surface-shell / planner / dynamic 후보를 중심으로 query call을 scheduling한다

---

### 최종 파이프라인 요약

```yaml
A. Tesla public outputs from F_dec

F_dec
├─ surface outputs
│  - road / ground surface geometry
│  - road / ground surface semantics
├─ volume outputs
│  - occupancy
│  - occupancy flow
│  - sub-voxel shape information
│  - 3D semantics
└─ queryable outputs
   - x,y,z → MLP → continuous occupancy probability
   - x,y,z → MLP → continuous 3D semantics

C. Robot runtime query / visualization strategy

surface-shell refine score calculation
↓
quota-based Top-K surface voxel selection
↓
adaptive in-voxel query generation
↓
queryable MLP point decoder 호출
↓
adaptive boundary search / carving
↓
sparse adaptive partial cuboid overlay
↓
local high-res occupancy / distance field conversion
↓
final planner-friendly occupancy representation
```

---

### 핵심 수식

```
F_query_packed = trilinear_sample(F_dec, p_query_packed)

O_query_packed, Sem3D_query_packed, Flow_query_packed, Dist_query_packed
=
MLP(F_query_packed, q_local_packed, PE(p_query_packed), coarse_logits_packed)
```

- 최종 map은 단순한 dense tensor 하나가 아니라 다음 조합이다

```
FinalRepresentation = {
    SurfaceOutputs,
    VolumeOutputs,
    QueryableOutputs,
    SparseFineOverlay,
    PlannerMaps
}
```
