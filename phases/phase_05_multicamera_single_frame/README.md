# Phase 05 - Multi-camera Single-frame Occupancy

기간: D051-D065  
목표: multi-camera image tokens, canonical ray PE, 1.2m 3D query attention, 3D decoder(1.2m->0.6m->0.3m), occupancy head(2-갈래, dense 프로토타입)를 하나의 single-frame 모델로 연결한다.

## D051 - tiny config 작성

만들 파일:

```text
phases/phase_05_multicamera_single_frame/configs/tiny.yaml
```

내용:

```yaml
num_cameras: 4
image_size: [128, 256]
channels: 32
coarse_shape: [50, 17, 5]      # 1.2m
mid_shape: [100, 34, 10]       # 0.6m (dense, temporal/coarse occ)
fine_shape: [200, 68, 20]      # 0.3m (toy 프로토타입은 dense, 실제는 sparse)
# 전 축 격자=타깃, padding 없음 (Y=±10.2m=20.4m=68칸)
```

검증:

```text
config를 Python에서 yaml.safe_load로 읽을 수 있다.
```

## D052 - MultiCameraSyntheticDataset

만들 파일:

```text
phases/phase_05_multicamera_single_frame/src/dataset.py
```

구현:

```python
class MultiCameraSyntheticDataset(Dataset):
    # output:
    # images: N_cam x 3 x H x W
    # occ_target: 1 x X x Y x Z
    # valid_mask: 1 x X x Y x Z
```

검증:

```text
DataLoader batch:
  images B x N_cam x 3 x H x W
```

## D053 - camera geometry placeholder

만들 파일:

```text
phases/phase_05_multicamera_single_frame/src/camera_geometry.py
```

구현:

```python
def make_virtual_camera_rays(num_cameras, h, w):
    # simplified canonical rays

def make_camera_id(num_cameras, h, w):
    # camera id map
```

목표:

```text
처음부터 완전한 projection에 집착하지 않는다.
canonical ray PE placeholder를 먼저 만든다.
```

## D054 - multi-camera image encoder

만들 파일:

```text
phases/phase_05_multicamera_single_frame/src/image_encoder.py
```

구현:

```python
class MultiCameraImageEncoder(nn.Module):
    # shared TinyImageBackbone
    # input: B x N_cam x 3 x H x W
    # output tokens: B x (N_cam*Hf*Wf) x C
```

검증:

```text
camera 수가 바뀌면 token 수가 비례해서 바뀐다.
```

## D055 - multi-camera lifter

만들 파일:

```text
phases/phase_05_multicamera_single_frame/src/lifter.py
```

구현:

```python
class MultiCameraLifter(nn.Module):
    # packed image tokens
    # 1.2m 3D query (50 x 17 x 5)
    # vanilla cross-attention
    # output: B x C x 50 x 17 x 5
```

검증:

```text
output coarse 3D feature shape 확인
```

## D056 - SingleFrameOccNet

만들 파일:

```text
phases/phase_05_multicamera_single_frame/src/model.py
```

구현:

```python
class SingleFrameOccNet(nn.Module):
    image_encoder
    lifter                 # 1.2m
    two_stage_decoder      # 1.2m -> 0.6m -> 0.3m
    occupancy_head         # 0.6m coarse dense + 0.3m fine (toy는 dense 프로토타입)
```

출력:

```text
occ_logits
```

## D057 - model shape test

만들 파일:

```text
phases/phase_05_multicamera_single_frame/tests/test_model_shape.py
```

검증:

```text
B=1과 B=2에서 forward 성공
occ_logits shape 확인
NaN 없음
```

## D058 - training loop

만들 파일:

```text
phases/phase_05_multicamera_single_frame/scripts/train_single_frame.py
```

구현:

```text
dataset
model
loss
optimizer
one-batch overfit mode
```

완료 기준:

```text
loss가 감소한다.
```

## D059 - occupancy IoU metric

만들 파일:

```text
phases/phase_05_multicamera_single_frame/src/metrics.py
```

구현:

```python
def occupancy_iou(logits, target, valid_mask):
    ...
```

검증:

```text
perfect prediction IoU=1
```

## D060 - prediction visualization

만들 파일:

```text
phases/phase_05_multicamera_single_frame/scripts/visualize_prediction.py
```

출력:

```text
target z slices
prediction z slices
difference image
```

## D061 - real camera projection mini 실습

수정 파일:

```text
phases/phase_05_multicamera_single_frame/src/camera_geometry.py
```

구현:

```python
def project_points(K, T_cam_ego, points_ego):
    # ego xyz -> camera xyz -> pixel uv
```

검증:

```text
간단한 known point가 예상 pixel로 가는지 확인
```

## D062 - projection notes

수정 파일:

```text
phases/phase_05_multicamera_single_frame/notes.md
```

기록:

```text
intrinsic K
extrinsic T_cam_ego
ego point -> camera point -> pixel
왜 depth가 없으면 완전한 viewpoint 변환이 어려운지
```

## D063 - real dataset interface stub

만들 파일:

```text
phases/phase_05_multicamera_single_frame/src/real_dataset_stub.py
```

구현:

```python
class RealDatasetStub(Dataset):
    # __getitem__ return format만 정의
```

목표:

```text
나중에 실제 dataset을 붙여도 model 코드를 많이 바꾸지 않도록 interface를 정한다.
```

## D064 - single-frame overfit 안정화

작업:

```text
learning rate 조정
channel 수 조정
valid mask 확인
loss가 안 내려가면 attention block을 일시적으로 단순화
```

기록:

```text
notes.md에 성공한 hyperparameter 기록
```

## D065 - Phase 05 report

검증:

```bash
pytest phases/phase_05_multicamera_single_frame/tests -q
```

기록:

```text
multi-camera token shape
coarse 3D feature shape
single-frame occupancy 결과
```

