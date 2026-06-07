# Phase 07 - Temporal Memory and Flow

기간: D081-D095  
목표: ViewFormer-style 1.6m BEV temporal memory와 `Dynamic / Flow Head`를 구현한다.

이 phase의 목적은 단순히 프레임을 여러 장 넣는 것이 아니다.  
핵심은 현재 프레임의 1.6m 3D feature를 만들고, 과거 프레임에서 만들어둔 BEV memory를 현재 ego pose 기준으로 정렬한 뒤, 다시 현재 3D feature에 context로 주입하는 것이다.

v1에서 temporal memory는 full 3D memory가 아니라 BEV memory로 제한한다.

```text
이유:
1. 3D full memory는 메모리와 latency가 너무 커지기 쉽다.
2. 1.6m BEV memory는 ego-motion alignment가 쉽다.
3. 주행 환경에서는 시간적으로 유지되는 구조물이 BEV에서 강하게 표현된다.
4. 20 FPS 목표에서는 temporal을 가볍게 유지해야 한다.
```

v1에서 flow는 다음 두 가지를 예측한다.

```text
dynamic_logit: B x 1 x X x Y x Z
flow:          B x 3 x X x Y x Z  # dx, dy, dz
```

## 이 phase가 끝나면 할 수 있어야 하는 것

```text
1. 1.6m 3D feature를 BEV memory로 압축할 수 있다.
2. ego pose delta로 BEV memory를 현재 frame에 맞게 warp할 수 있다.
3. 최근 N개의 BEV memory를 queue로 관리할 수 있다.
4. temporal BEV context를 현재 3D feature에 다시 inject할 수 있다.
5. moving cube toy sequence에서 dynamic mask와 flow target을 만들 수 있다.
6. flow loss가 occupied/dynamic 영역에서만 계산되도록 만들 수 있다.
```

## D081 - ViewFormer 발췌 읽기

읽을 것:

```text
ViewFormer temporal memory overview
ego-motion alignment 부분
temporal fusion 부분
occupancy / flow 관련 부분
```

기록 파일:

```text
phases/phase_07_temporal_flow/notes.md
```

기록할 질문:

```text
1. 왜 3D feature memory가 아니라 BEV memory를 쓰는가?
2. ego-motion alignment는 무엇을 정렬하는가?
3. 과거 feature를 현재 frame으로 가져올 때 좌표계 기준은 무엇인가?
4. temporal memory가 occupancy에 도움 되는 경우와 방해되는 경우는 무엇인가?
5. moving object는 memory에 어떻게 남을 수 있고, 이것이 flow head와 어떻게 연결되는가?
```

완료 기준:

```text
notes.md에 "current frame coordinate로 과거 BEV를 warp한다"는 설명이 있다.
temporal memory가 camera feature memory가 아니라 1.6m BEV feature memory임을 명확히 적는다.
```

## D082 - ZPool

만들 파일:

```text
phases/phase_07_temporal_flow/src/temporal_memory.py
phases/phase_07_temporal_flow/tests/test_temporal_memory.py
```

구현:

```python
class ZPool(nn.Module):
    """
    입력:  B x C x X x Y x Z
    출력:  B x Cb x X x Y

    1.6m 3D feature를 temporal memory용 BEV feature로 압축한다.
    """
```

최소 구현:

```text
1. mean pooling mode
2. max pooling mode
3. learned pooling mode
```

검증:

```text
B=2, C=64, X=38, Y=13, Z=4 입력
출력은 B=2, Cb, X=38, Y=13
backward가 통과한다.
height 정보 일부가 손실된다는 점을 notes.md에 기록한다.
```

이해:

```text
ZPool은 surface head의 LearnedZPooling과 비슷하지만 목적이 다르다.
surface pooling은 ground geometry 예측용이고,
temporal ZPool은 과거 frame memory 저장용이다.
```

## D083 - BEV ego-motion warp

수정 파일:

```text
phases/phase_07_temporal_flow/src/temporal_memory.py
```

구현:

```python
def make_bev_warp_grid(shape, pose_delta, voxel_size, origin):
    """
    현재 BEV grid 위치가 과거 BEV feature의 어느 위치를 sample해야 하는지 만든다.
    """

def warp_bev_feature(bev, pose_delta, voxel_size=1.6, origin=None):
    """
    grid_sample 기반 2D BEV warp.
    과거 BEV feature를 현재 ego coordinate로 정렬한다.
    """
```

구현 순서:

```text
1. BEV index grid를 만든다.
2. index를 metric coordinate로 바꾼다.
3. current coordinate를 previous coordinate로 inverse transform한다.
4. previous coordinate를 grid_sample의 normalized coordinate로 바꾼다.
5. F.grid_sample로 sample한다.
```

검증:

```text
identity pose면 output이 input과 같다.
x 방향 translation pose면 feature 위치가 예측한 방향으로 이동한다.
rotation 0도, 90도 toy case를 따로 테스트한다.
grid_sample align_corners 설정을 notes.md에 기록한다.
```

실패 체크:

```text
warp 방향이 반대로 되기 쉽다.
테스트용으로 BEV 중앙에 single hot point를 두고 이동 방향을 눈으로 확인한다.
```

## D084 - memory queue

수정 파일:

```text
phases/phase_07_temporal_flow/src/temporal_memory.py
```

구현:

```python
class BEVMemoryQueue:
    def __init__(self, max_history=3):
        ...

    def push(self, feature, pose, timestamp=None):
        ...

    def get_aligned(self, current_pose):
        ...

    def clear(self):
        ...
```

저장할 것:

```text
feature:   B x Cb x X x Y
pose:      ego/global pose
timestamp: optional
```

검증:

```text
N_history=3이면 최근 3개만 유지한다.
empty queue에서는 빈 list 또는 None을 반환한다.
current_pose와 동일 pose면 aligned feature가 거의 원본과 같다.
```

주의:

```text
학습 때 batch 안의 sequence와 inference 때 streaming queue는 다르게 동작할 수 있다.
처음에는 toy streaming 방식으로 구현하고, D088에서 model integration을 확인한다.
```

## D085 - simple temporal fusion

수정 파일:

```text
phases/phase_07_temporal_flow/src/temporal_memory.py
```

구현:

```python
class TemporalFusion(nn.Module):
    """
    current_bev: B x C x X x Y
    memory_bevs: list[B x C x X x Y]
    output: B x C x X x Y
    """
```

처음 구현할 mode:

```text
mean:
  current + aligned memories를 평균낸다.

gated:
  concat(current, memory_mean) -> gate -> current * gate + memory_mean * (1-gate)
```

검증:

```text
memory가 없으면 current_bev와 같은 shape의 output이 나온다.
mean mode와 gated mode 모두 shape가 같다.
gate 값이 0~1 범위에 있다.
```

이유:

```text
처음부터 temporal attention으로 가면 문제 원인을 찾기 어렵다.
mean/gated fusion이 baseline이다.
```

## D086 - temporal attention option

수정 파일:

```text
phases/phase_07_temporal_flow/src/temporal_memory.py
```

구현:

```text
TemporalFusion에 attention mode 추가
```

권장 방식:

```text
1. current + N memory를 time dimension으로 stack한다.
2. 각 BEV cell마다 시간축 attention을 수행한다.
3. attention은 spatial attention이 아니라 temporal attention으로 제한한다.
4. 즉 X*Y 전체를 token으로 섞는 큰 attention은 하지 않는다.
```

검증:

```text
output shape가 simple fusion과 같다.
N_history가 달라도 동작한다.
attention weight의 time dimension 합이 1이다.
```

주의:

```text
v1 runtime 목표에서는 temporal attention이 필수는 아니다.
D094 ablation에서 mean/gated보다 이득이 없으면 final v1에서 끌 수 있다.
```

## D087 - InjectBEVTo3D

수정 파일:

```text
phases/phase_07_temporal_flow/src/temporal_memory.py
```

구현:

```python
class InjectBEVTo3D(nn.Module):
    """
    bev_context: B x Cb x X x Y
    feat_3d:     B x C x X x Y x Z
    output:      B x C x X x Y x Z
    """
```

구현 순서:

```text
1. BEV context를 Z축으로 broadcast한다.
2. z embedding을 추가해서 모든 height slice가 완전히 같아지지 않게 한다.
3. 1x1x1 conv로 channel을 맞춘다.
4. residual add 또는 concat+conv 방식으로 3D feature에 주입한다.
```

검증:

```text
input 3D feature와 output 3D feature shape가 같다.
temporal context가 0이면 output이 input과 크게 다르지 않다.
```

이해:

```text
BEV memory는 높이 정보를 압축했기 때문에 다시 3D로 넣을 때 z embedding이 중요하다.
```

## D088 - temporal model integration

만들 파일:

```text
phases/phase_07_temporal_flow/src/model_temporal.py
phases/phase_07_temporal_flow/tests/test_model_temporal.py
```

구현:

```text
1.6m 3D feature
-> ZPool
-> BEV memory alignment
-> temporal fusion
-> InjectBEVTo3D
-> enhanced 1.6m feature
-> deconv decoder로 전달
```

권장 class:

```python
class TemporalEnhancer(nn.Module):
    def forward(self, feat_16m, memory_bevs=None, pose=None):
        return enhanced_feat_16m, current_bev
```

검증:

```text
memory 없음: forward 성공
memory 1개: forward 성공
memory 3개: forward 성공
enhanced_feat_16m shape가 feat_16m과 같다.
```

기록:

```text
notes.md에 "temporal은 deconv 이전 1.6m feature에 들어간다"라고 적는다.
0.4m 이후에 temporal을 넣지 않는 이유도 적는다.
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
flow target
occlusion/unknown mask optional
```

데이터 구성:

```text
scene A: static cube, ego stationary
scene B: moving cube, ego stationary
scene C: static cube, ego moving
scene D: moving cube, ego moving
```

검증:

```text
cube가 t에서 t+1로 이동한 방향이 flow target과 일치한다.
static cube는 dynamic mask가 False다.
ego moving only scene에서 static object flow를 어떻게 정의할지 notes.md에 적는다.
```

중요:

```text
flow target은 coordinate convention이 매우 중요하다.
v1에서는 ego/current frame 기준 flow로 통일한다.
```

## D090 - DynamicFlowHead

만들 파일:

```text
phases/phase_07_temporal_flow/src/flow_head.py
phases/phase_07_temporal_flow/tests/test_flow_head.py
```

구현:

```python
class DynamicFlowHead(nn.Module):
    """
    input:  B x C x X x Y x Z
    output:
      dynamic_logit: B x 1 x X x Y x Z
      flow:          B x 3 x X x Y x Z
    """
```

검증:

```text
dynamic output shape 확인
flow output shape 확인
flow 값에 NaN이 없다.
flow head가 occupancy head와 독립적으로 꺼질 수 있다.
```

주의:

```text
flow는 모든 voxel에서 학습하면 안 된다.
occupied/dynamic/valid mask에서만 loss를 계산한다.
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

def flow_smooth_l1_loss(flow_pred, flow_gt, mask):
    ...

def flow_endpoint_error(flow_pred, flow_gt, mask):
    ...
```

mask 설계:

```text
flow_loss_mask = occupied_gt & dynamic_gt & valid_gt
```

검증:

```text
flow loss는 dynamic/occupied mask에서만 계산한다.
mask가 전부 False여도 NaN이 나지 않는다.
perfect flow prediction은 EPE가 0에 가깝다.
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
1. moving cube toy sequence를 고정한다.
2. TemporalEnhancer + DynamicFlowHead를 붙인다.
3. dynamic BCE와 flow SmoothL1을 같이 학습한다.
4. 300 iteration 내에서 overfit되는지 확인한다.
```

출력 로그:

```text
iter
loss_dynamic
loss_flow
dynamic_acc
flow_epe
```

완료 기준:

```text
moving cube의 flow 방향을 one-batch에서 맞춘다.
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
flow_bev_arrows_pred.png
flow_bev_arrows_gt.png
flow_error_map.png
```

구현 기준:

```text
1. Z축은 max 또는 occupied-weighted pooling으로 BEV에 투영한다.
2. arrow는 너무 조밀하게 그리지 않는다.
3. pred/gt arrow scale을 동일하게 둔다.
4. dynamic mask 위에 arrow를 overlay한다.
```

검증:

```text
toy moving cube에서 arrow 방향을 눈으로 확인할 수 있다.
static cube에는 arrow가 거의 없다.
```

## D094 - temporal ablation

실험:

```text
A0 no temporal
A1 simple temporal mean
A2 gated temporal
A3 temporal attention
```

기록할 항목:

```text
occupancy IoU
dynamic accuracy
flow endpoint error
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
gated fusion이 mean보다 좋고 latency가 작으면 gated를 v1 기본값으로 둔다.
temporal attention이 이득이 작거나 느리면 v1에서는 optional로 둔다.
no temporal 대비 flow EPE가 개선되는지 본다.
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
1. BEV memory shape
2. memory queue max_history
3. ego-motion warp convention
4. temporal fusion 기본 mode
5. flow target coordinate convention
6. flow loss mask
7. final_occnet_v1로 가져갈 파일 목록
```

최종 완료 기준:

```text
BEV warp identity/translation test가 통과한다.
TemporalEnhancer forward가 memory 없음/있음 모두 통과한다.
moving cube one-batch flow overfit이 된다.
flow visualization으로 방향을 확인할 수 있다.
```
