# Phase 07 - Temporal Memory and Flow

기간: D081-D095  
목표: Tesla/PanoOcc-style 0.6m 3D temporal(z 유지, align+concat+3D residual conv, NOT attention)과 Occupancy Flow Head 내부의 `O_motion_0.6` coarse motion seed를 구현한다. 최종 flow는 Phase 08에서 O_occ_fine 이후 확정되는 `pred_heavy_mask` 위의 `[N_heavy]` readout/refine으로 낸다.

이 phase의 목적은 단순히 프레임을 여러 장 넣는 것이 아니다.  
핵심은 현재 프레임의 0.6m 3D feature(z 유지)를 만들고, 과거 프레임의 3D voxel memory를 현재 ego pose 기준으로 정렬(warp)한 뒤, concat + 3D residual conv로 융합하는 것이다.

v1에서 temporal은 **z-squeeze BEV가 아니라 3D(z 유지)** 로 한다. (ViewFormer의 z-squeeze + temporal attention은 architecture에서 폐기됨)

```text
이유:
1. Tesla 그림(Spatial Frame Alignment -> Spatiotemporal stack)과 PanoOcc temporal encoder가
   모두 3D를 align + concat 한다 (attention 아님).
2. z를 유지해야 높이별 motion / vz(수직 속도)를 살릴 수 있다.
3. 융합이 attention이 아니라 concat + 3D conv라 비용은 voxel 수에 선형 -> Orin 가능
   (0.6m 3D = 34,000 voxel, PanoOcc coarse 40k 이내).
4. Orin 예산 초과 시에는 1.2m 3D fallback (해상도만 한 단계 내림, 3D 유지는 동일).
```

v1에서 이 phase가 직접 만드는 것은 Flow Head의 0.6m coarse branch다.

```text
O_motion_0.6:
  dynamic_seed_logit: B x 1 x X x Y x Z
  coarse_flow_seed:   B x 3 x X x Y x Z  # vx, vy, vz

final flow:
  Phase 08의 pred_heavy_mask 위에서 seed + F_0.3_sparse를 readout/refine
  output: [N_heavy] x 5 또는 필요한 channel(dynamic + vx/vy/vz + confidence)
```

## 이 phase가 끝나면 할 수 있어야 하는 것

```text
1. 0.6m 3D feature(z 유지)를 z-squeeze 없이 다룰 수 있다.
2. ego pose delta로 과거 3D voxel feature를 현재 frame에 맞게 3D warp할 수 있다.
3. 최근 N개(과거3+현재1=4)의 3D voxel memory를 queue로 관리할 수 있다.
4. concat + 3D residual conv로 temporal 융합할 수 있다 (attention 아님).
5. moving cube toy sequence에서 O_motion_0.6 coarse seed target을 만들 수 있다.
6. L_flow_coarse_seed와 [N_heavy] readout interface를 분리할 수 있다.
```

## D081 - PanoOcc / Tesla temporal 발췌 읽기

읽을 것:

```text
PanoOcc temporal encoder (temporal align + temporal fuse, 3D voxel)
ego-motion 3D alignment 부분
Tesla AI Day Temporal Alignment 그림 (Spatial Frame Alignment -> Spatiotemporal Features stack)
(ViewFormer는 streaming memory / flow supervision 방식만 참고)
```

기록 파일:

```text
phases/phase_07_temporal_flow/notes.md
```

기록할 질문:

```text
1. 왜 BEV z-squeeze가 아니라 3D(z 유지)로 정렬+concat 하는가? (높이별 motion / vz)
2. ego-motion alignment는 무엇을 정렬하는가? (과거 3D feature -> 현재 좌표계)
3. 융합이 attention이 아니라 concat + 3D residual conv인 이유는?
4. temporal memory가 occupancy에 도움 되는 경우와 방해되는 경우는?
5. moving object는 memory에 어떻게 남고, 이것이 flow head와 어떻게 연결되는가?
```

완료 기준:

```text
notes.md에 "current frame coordinate로 과거 3D feature를 warp한다"는 설명이 있다.
temporal memory가 z-squeeze BEV가 아니라 0.6m 3D voxel feature memory임을 명확히 적는다.
```

## D081a - 0.6m Pre-temporal Feature Refinement (architecture Section 6)

만들 파일:

```text
phases/phase_07_temporal_flow/src/pre_temporal.py
phases/phase_07_temporal_flow/tests/test_pre_temporal.py
```

구현:

```python
class PreTemporalRefine3D(nn.Module):
    # dense deconv(1.2m->0.6m) 직후, temporal 직전에 현재 frame 내부만 한 번 정리
    # light 3D conv block: depthwise 3x3x3 + pointwise 1x1x1 + residual/gating
    # input/output: B x C x 100 x 34 x 10 (shape 유지, z 유지)
```

왜 필요한가 (BEVDet4D 교훈):

```text
1.2m attention 직후 feature는 너무 coarse해서 temporal cue를 바로 쓰면
velocity error가 오른다(+11.9%). temporal 전에 작은 encoder로 한 번 정리한다.
큰 모듈/추가 camera attention이 아니라 작은 3D conv block이다.
위치: deconv(1.2m->0.6m) 직후, D082~D088 temporal 직전.
이후 TemporalEnhancer(D088)는 F_0.6_refined를 입력으로 받는다.
```

검증:

```text
input/output shape 동일 (B x C x 100 x 34 x 10, z 유지).
residual이라 초기엔 거의 identity에 가깝다.
backward 통과, NaN 없음.
```

## D082 - 3D voxel memory (z 유지, no z-squeeze)

만들 파일:

```text
phases/phase_07_temporal_flow/src/temporal_memory.py
phases/phase_07_temporal_flow/tests/test_temporal_memory.py
```

구현:

```python
def stack_history_3d(current, aligned_history):
    """
    current:         B x C x X x Y x Z
    aligned_history: list[B x C x X x Y x Z]
    return:          B x (C*(1+N)) x X x Y x Z  (채널 방향 concat 준비)
    z를 유지한 채 다룬다. ZPool/z-squeeze 없음.
    """
```

검증:

```text
B=2, C=64, X=100, Y=34, Z=10 입력이 z를 유지한 채 concat된다.
어디서도 Z dimension을 BEV로 누르지 않는다.
backward가 통과한다.
```

이해:

```text
surface head는 z-flatten(채널로 펼치기)으로 BEV를 만들지만,
temporal은 아예 3D를 유지한다 (z 유지 -> 높이별 motion / vz).
```

## D083 - 3D ego-motion warp

수정 파일:

```text
phases/phase_07_temporal_flow/src/temporal_memory.py
```

구현:

```python
def make_voxel_warp_grid(shape_xyz, pose_delta, voxel_size, origin):
    """
    현재 voxel grid 위치가 과거 3D feature의 어느 위치를 sample해야 하는지 만든다 (3D).
    """

def warp_3d_feature(feat_3d, pose_delta, voxel_size=0.6, origin=None):
    """
    grid_sample 3D 기반 voxel warp.
    과거 3D feature를 현재 ego coordinate로 정렬한다 (PanoOcc식).
    """
```

구현 순서:

```text
1. 3D voxel index grid를 만든다.
2. index를 metric coordinate로 바꾼다.
3. current coordinate를 previous coordinate로 inverse transform한다 (회전 포함).
4. previous coordinate를 grid_sample(3D)의 normalized coordinate로 바꾼다.
5. F.grid_sample(mode='bilinear', 5D tensor)로 sample한다.
```

검증:

```text
identity pose면 output이 input과 같다.
x 방향 translation pose면 feature 위치가 예측한 방향으로 이동한다.
yaw rotation 0도/90도 toy case를 따로 테스트한다.
grid_sample align_corners 설정을 notes.md에 기록한다.
```

실패 체크:

```text
warp 방향이 반대로 되기 쉽다.
3D 중앙에 single hot voxel을 두고 이동 방향을 눈으로 확인한다.
```

## D084 - 3D memory queue

수정 파일:

```text
phases/phase_07_temporal_flow/src/temporal_memory.py
```

구현:

```python
class VoxelMemoryQueue:
    def __init__(self, max_history=3):   # 과거3 + 현재1 = 4 frame (PanoOcc와 동일)
        ...

    def push(self, feature_3d, pose, timestamp=None):
        ...

    def get_aligned(self, current_pose):
        ...

    def clear(self):
        ...
```

저장할 것:

```text
feature:   B x C x X x Y x Z   (0.6m 3D, z 유지)
pose:      ego/global pose
timestamp: optional
```

검증:

```text
max_history=3이면 최근 3개만 유지한다 (현재 포함 4 frame).
empty queue에서는 빈 list 또는 None을 반환한다.
current_pose와 동일 pose면 aligned feature가 거의 원본과 같다.
```

설계 주의:

```text
architecture 기준 memory는 0.6m 3D voxel(z 유지)로 저장/warp하고,
keyframe은 ~0.2s 시간 간격으로 띄엄띄엄 저장한다 (0.6~0.8s 창).
queue 인터페이스에 (저장 해상도 0.6m, keyframe 간격, 1.2m fallback) 설정을 열어둔다.
학습 때 batch sequence와 inference 때 streaming queue는 다르게 동작할 수 있다.
```

## D085 - temporal fusion (concat + 3D residual conv)

수정 파일:

```text
phases/phase_07_temporal_flow/src/temporal_memory.py
```

구현:

```python
class TemporalFusion3D(nn.Module):
    """
    current_3d:  B x C x X x Y x Z
    memory_3d:   list[B x C x X x Y x Z]  (이미 ego-motion warp로 정렬됨)
    output:      B x C x X x Y x Z

    X = concat(current_3d, aligned memory_3d)   # 채널 stack
    out = current_3d + Residual3DConv(X)        # attention 아님
    """
```

검증:

```text
memory가 없으면 current_3d와 같은 shape의 output이 나온다.
memory 1개/3개 모두 shape가 같다.
attention을 쓰지 않는다 (concat + 3D conv만).
```

이유:

```text
Tesla/PanoOcc식 align + concat + 3D residual conv가 baseline이자 v1 채택안이다.
temporal attention / z-squeeze는 쓰지 않는다.
```

## D086 - streaming memory option (학습 효율)

수정 파일:

```text
phases/phase_07_temporal_flow/src/temporal_memory.py
```

구현:

```text
TemporalFusion3D에 ViewFormer streaming memory option 추가
(학습 시 과거 feature를 재계산하지 않고 캐시 -> 학습 효율, 추론 latency 변화 없음)
```

검증:

```text
streaming on/off 결과가 동일하다 (정확도 동치, 속도만 차이).
output shape가 유지된다.
```

주의:

```text
streaming memory는 ViewFormer에서 "학습 효율"용으로만 차용한다.
temporal 본체(align+concat+3D conv)는 PanoOcc식이다.
```

## D087 - 1.2m fallback 스위치 + 융합 마무리

수정 파일:

```text
phases/phase_07_temporal_flow/src/temporal_memory.py
```

구현:

```text
융합 해상도 스위치: 기본 0.6m 3D, Orin 예산 초과 시 1.2m 3D fallback.
(3D 유지는 동일, 해상도만 한 단계 내림.)
```

검증:

```text
0.6m / 1.2m 두 해상도에서 fusion forward가 동작한다.
fallback이 켜져도 출력 인터페이스(shape 규약)는 동일하다.
```

이해:

```text
1.2m fallback은 "z-squeeze로 후퇴"가 아니라 "3D를 한 단계 거친 해상도로" 가는 것이다.
높이 정보(vz)는 fallback에서도 유지된다.
```

## D088 - temporal model integration

만들 파일:

```text
phases/phase_07_temporal_flow/src/model_temporal.py
phases/phase_07_temporal_flow/tests/test_model_temporal.py
```

구현:

```text
0.6m 3D feature (z 유지)
-> 3D voxel memory align (ego-motion warp)
-> concat + 3D residual conv (TemporalFusion3D)
-> enhanced 0.6m 3D feature
-> deconv decoder로 전달
```

권장 class:

```python
class TemporalEnhancer(nn.Module):
    def forward(self, feat_06m, memory_3d=None, pose=None):
        return enhanced_feat_06m, current_feat_06m   # 다음 frame memory용
```

검증:

```text
memory 없음/1개/3개 모두 forward 성공.
enhanced_feat_06m shape가 feat_06m과 같다 (z 유지).
```

기록:

```text
notes.md에 "temporal은 dense deconv(1.2m->0.6m) 이후,
0.6m->0.3m sparse/final deconv 이전의 0.6m 3D feature에 들어간다"라고 적는다.
0.3m 이후에 temporal을 넣지 않는 이유(272k voxel)도 적는다.
```

## D089 - moving cube sequence dataset

만들 파일:

```text
phases/phase_07_temporal_flow/src/sequence_dataset.py
phases/phase_07_temporal_flow/tests/test_sequence_dataset.py
```

구현:

```text
moving cube sequence
ego pose delta
dynamic mask
O_motion_0.6 coarse flow seed target (vx, vy, vz)
[N_heavy] fine flow readout target placeholder
occlusion/unknown mask optional
```

데이터 구성:

```text
scene A: static cube, ego stationary
scene B: moving cube, ego stationary
scene C: static cube, ego moving
scene D: moving cube, ego moving
scene E: 높이 방향으로 움직이는 cube (vz != 0 검증용)
```

검증:

```text
cube가 t에서 t+1로 이동한 방향이 flow target과 일치한다 (vx,vy,vz).
static cube는 dynamic mask가 False다.
높이 방향 이동 scene에서 vz가 0이 아니다 (z 유지의 이점 확인).
ego moving only scene에서 static object flow를 어떻게 정의할지 notes.md에 적는다.
```

중요:

```text
flow target은 coordinate convention이 매우 중요하다.
v1에서는 ego/current frame 기준 flow로 통일한다.
```

## D090 - Occupancy Flow Head coarse seed + readout skeleton

만들 파일:

```text
phases/phase_07_temporal_flow/src/flow_head.py
phases/phase_07_temporal_flow/tests/test_flow_head.py
```

구현:

```python
class OccupancyFlowHead(nn.Module):
    """
    branch A:
      input:  F_0.6_temporal, B x C x X x Y x Z
      output: O_motion_0.6 dense coarse seed
              dynamic_seed_logit + coarse vx, vy, vz

    branch B skeleton:
      Phase 08 이후 pred_heavy_mask voxel에
      O_motion_0.6 seed + F_0.3_sparse를 gather해
      final [N_heavy] dynamic_logit + vx, vy, vz를 readout/refine한다.
    """
```

검증:

```text
O_motion_0.6 dynamic seed shape 확인
O_motion_0.6 coarse flow seed shape 확인 (3 channel: vx,vy,vz)
toy pred_heavy_mask gather/readout interface shape 확인
Flow Head가 독립 5번째 head가 아니라 4 volume head 중 Flow Head의 내부 coarse branch임을 notes에 기록
```

주의:

```text
coarse seed는 0.6m dense에서 supervise할 수 있다.
최종 flow는 pred_heavy_mask 위에서 실행하되, supervise는 valid_flow_mask에만 준다.
pred_visibility_frontier / pred_free_shell / pred_free_kept / pred_uncertain_kept에는 final flow를 두지 않는다.
```

## D091 - flow losses

만들 파일:

```text
phases/phase_07_temporal_flow/src/flow_losses.py
phases/phase_07_temporal_flow/tests/test_flow_losses.py
```

구현:

```python
def dynamic_bce_loss(dynamic_logit, dynamic_gt, valid_mask=None):
    ...

def flow_coarse_seed_loss(o_motion_06, flow_gt_06, valid_flow_seed_mask):
    ...

def flow_fine_readout_loss(flow_pred_heavy, flow_gt_heavy, valid_flow_mask):
    ...

def flow_endpoint_error(flow_pred, flow_gt, mask):
    ...
```

mask 설계:

```text
valid_flow_seed_mask = confident coarse dynamic/occupied GT
valid_flow_mask = confident occupied/dynamic GT on pred_heavy_mask
```

검증:

```text
L_flow_coarse_seed는 0.6m dense seed에 적용한다.
L_flow_fine_readout은 [N_heavy] final readout에만 적용한다.
flow loss는 valid_flow_mask에서만 계산한다.
mask가 전부 False여도 NaN이 나지 않는다.
perfect flow prediction은 EPE가 0에 가깝다.
vz 채널도 loss/EPE에 포함된다.
```

기록:

```text
notes.md에 "flow를 free space에서 학습하면 왜 이상한가"를 적는다.
```

## D092 - flow one-batch overfit

만들 파일:

```text
phases/phase_07_temporal_flow/scripts/train_flow_one_batch.py
```

작업:

```text
1. moving cube toy sequence를 고정한다 (vz != 0 scene 포함).
2. TemporalEnhancer + OccupancyFlowHead branch A를 붙인다.
3. L_flow_coarse_seed와 dynamic BCE를 먼저 학습한다.
4. toy pred_heavy_mask gather/readout skeleton의 shape를 확인한다.
5. 300 iteration 내에서 overfit되는지 확인한다.
```

출력 로그:

```text
iter
loss_dynamic
loss_flow_coarse_seed
loss_flow_fine_readout(toy readout)
dynamic_acc
flow_epe (vx,vy,vz)
```

완료 기준:

```text
moving cube의 O_motion_0.6 seed 방향(vx,vy,vz)을 one-batch에서 맞춘다.
높이 방향 이동 cube에서 vz가 살아난다.
dynamic mask가 cube 주변에서만 높아진다.
```

실패 체크:

```text
flow 방향이 반대면 pose/flow coordinate convention을 다시 본다.
dynamic은 맞는데 flow가 안 맞으면 mask와 target scale을 확인한다.
```

## D093 - flow visualization

만들 파일:

```text
phases/phase_07_temporal_flow/scripts/visualize_flow.py
```

출력:

```text
dynamic_mask_pred.png
dynamic_mask_gt.png
flow_bev_arrows_pred.png   # vx,vy를 BEV arrow로
flow_bev_arrows_gt.png
flow_vz_slice.png          # vz를 height slice로 (z 유지 확인)
flow_error_map.png
```

구현 기준:

```text
1. vx,vy는 BEV arrow로, vz는 별도 slice로 시각화한다.
2. arrow는 너무 조밀하게 그리지 않는다.
3. pred/gt arrow scale을 동일하게 둔다.
4. dynamic mask 위에 arrow를 overlay한다.
```

검증:

```text
toy moving cube에서 arrow 방향을 눈으로 확인할 수 있다.
높이 방향 이동 scene에서 vz slice가 0이 아니다.
static cube에는 arrow가 거의 없다.
```

## D094 - temporal ablation

실험:

```text
A0 no temporal
A1 3D concat + residual conv (채택)
A2 + streaming memory option
A3 1.2m fallback 해상도
```

기록할 항목:

```text
occupancy IoU
dynamic accuracy
flow endpoint error (특히 vz가 살아나는지)
latency
peak memory
학습 안정성
```

기록 파일:

```text
phases/phase_07_temporal_flow/notes.md
```

판단 기준:

```text
3D concat+conv가 no temporal 대비 flow EPE를 개선하는지 본다.
streaming memory가 학습 속도를 줄이는지 (정확도는 동치) 확인한다.
0.6m 3D가 예산을 넘으면 1.2m fallback의 품질/비용 trade-off를 본다.
```

## D095 - Phase 07 report

검증:

```bash
pytest phases/phase_07_temporal_flow/tests -q
python phases/phase_07_temporal_flow/scripts/train_flow_one_batch.py
python phases/phase_07_temporal_flow/scripts/visualize_flow.py
```

기록 파일:

```text
phases/phase_07_temporal_flow/notes.md
```

기록할 내용:

```text
1. 3D voxel memory shape (z 유지)
2. memory queue max_history (과거3+현재1=4)
3. ego-motion 3D warp convention
4. temporal fusion 방식 (concat + 3D residual conv, attention 아님)
5. O_motion_0.6 coarse seed와 [N_heavy] fine readout target coordinate convention
6. valid_flow_mask와 dynamic mask
7. final_occnet_v1로 가져갈 파일 목록
```

최종 완료 기준:

```text
3D warp identity/translation/rotation test가 통과한다.
TemporalEnhancer forward가 memory 없음/있음 모두 통과한다.
moving cube one-batch flow overfit이 된다 (vz 포함).
flow visualization으로 방향을 확인할 수 있다.
```
