# Phase 05 - Multi-camera Single-frame Occupancy

기간: D051-D065  
목표: multi-camera image tokens, canonical ray PE, 1.2m 3D query attention, 3D decoder(1.2m->0.6m->0.3m), occupancy head(2-갈래, dense 프로토타입), auxiliary depth/free-space evidence head를 하나의 single-frame 모델로 연결한다.

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
    aux_depth_head         # v1 inference에도 필요한 depth/free-space evidence
```

출력:

```text
occ_coarse_logits
occ_fine_logits
aux_depth / depth_confidence / free_space_confidence
```

## D057 - model shape test

만들 파일:

```text
phases/phase_05_multicamera_single_frame/tests/test_model_shape.py
```

검증:

```text
B=1과 B=2에서 forward 성공
occ_coarse / occ_fine / aux_depth output shape 확인
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
architecture Section 19의 GT 출력을 받을 수 있는 key를 미리 고정한다.

필수/준필수 key:
  images / intrinsics / extrinsics_or_virtual_extrinsics / ego_pose
  occupancy (occupied / free / unknown)
  visibility / observed_free_evidence targets
  z_surface / surface_valid / traversability_cost / drop_risk
  dynamic mask / flow
  continuous SDF / surface-shell samples (optional, Sub-Voxel Shape / Queryable 학습용)
  dense or sparse metric depth / depth confidence
  free-space ray evidence
```

## D064 - Auxiliary Depth / Free-space Evidence Head (v1 inference-critical)

만들 파일:

```text
phases/phase_05_multicamera_single_frame/src/aux_depth_head.py
phases/phase_05_multicamera_single_frame/src/aux_depth_losses.py
phases/phase_05_multicamera_single_frame/tests/test_aux_depth_head.py
```

구현:

```text
image feature 또는 low-res multi-camera feature에서
low-res depth/range, depth_confidence, free_space_confidence를 예측한다.

이 head는 단순 training auxiliary가 아니라 추론 시에도 유지한다.
Section 10의 observed_free_evidence_0.3(child) /
observed_free_evidence_0.6(parent)는 이 head의 high-confidence ray/depth만 사용한다.
confidence가 낮으면 observed free가 아니라 unknown으로 보수 처리한다.
```

loss:

```python
def depth_loss(depth_pred, depth_gt, confidence): ...
def depth_conf_loss(conf_pred, depth_error_or_target): ...
def free_space_evidence_loss(free_space_logit, ray_free_target, valid_ray_mask): ...
```

검증:

```text
synthetic depth GT에서 L_depth / L_depth_conf / L_free_space_evidence가 감소한다.
hit 뒤쪽 / occluded / 미관측 구간을 free negative로 쓰지 않는다.
pred_free_shell free 즉답은 child-level observed-free evidence 없이는 금지됨을 unit test로 확인한다.
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
auxiliary depth/free-space evidence head의 출력 shape
observed-free evidence가 없으면 free로 반환하지 않는 보수 규칙
```
