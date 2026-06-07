# Daily 6-Month Implementation Plan for `occupancy_network_architecture.md`

이 문서는 하루 1시간, 평일 5일 기준 26주, 총 130일 동안 `occupancy_network_architecture.md`의 구조를 실제 구현 가능한 단계로 쌓아 올리는 계획이다.

주말은 공식 커리큘럼에 넣지 않는다. 주말은 밀린 날 보충, 시각화 개선, 논문 보충 읽기, 코드 정리에 사용한다.

---

## 0. 최종 목표

6개월 후 목표는 아래 v1 prototype이다.

```text
multi-camera image
-> image backbone + FPN/BiFPN-lite
-> 1.6m 3D query vanilla cross-attention
-> ViewFormer-style 1.6m BEV temporal memory
-> 3D deconv: 1.6m -> 0.8m -> 0.4m
-> structured 20cm occupancy head
-> road surface geometry head
-> dynamic / flow head
-> active mask + quota Top-K
-> sparse local feature anchor
-> packed adaptive Queryable MLP
```

20 FPS v1 원칙:

```text
금지:
  dense 0.2m feature volume
  dense 0.2m deconvolution
  모든 20cm voxel에 generic GridSample + MLP 호출
  full 3D temporal memory
  P2/P3 local image re-query in v1

필수:
  0.4m까지만 dense feature 유지
  20cm dense output은 structured sub-voxel channel head
  Queryable MLP는 selected voxel refinement에만 사용
  K_total / Q_total hard cap
  packed batched Queryable MLP
  one-batch overfit
```

---

## 1. Phase 디렉토리 구조

코드는 phase별로 관리한다. 각 phase는 독립 실험 폴더이고, 마지막 phase에서 reusable code를 `final_occnet_v1/`로 모은다.

```text
study_occupancy/
  occupancy_network_architecture.md
  tesla_occnet.md
  deconv
  plan.md

  phases/
    phase_01_pytorch_primitives/
      src/
      tests/
      scripts/
      notes.md

    phase_02_toy_occupancy/
      src/
      tests/
      scripts/
      notes.md

    phase_03_attention_lifting/
      src/
      tests/
      scripts/
      notes.md

    phase_04_deconv_structured_head/
      src/
      tests/
      scripts/
      notes.md

    phase_05_multicamera_single_frame/
      src/
      tests/
      scripts/
      configs/
      notes.md

    phase_06_surface_geometry/
      src/
      tests/
      scripts/
      notes.md

    phase_07_temporal_flow/
      src/
      tests/
      scripts/
      notes.md

    phase_08_dynamic_resolution_refinement/
      src/
      tests/
      scripts/
      notes.md

    phase_09_integration_evaluation/
      src/
      tests/
      scripts/
      configs/
      notes.md

    phase_10_runtime_final_report/
      scripts/
      reports/
      notes.md

  final_occnet_v1/
    src/occnet_v1/
    tests/
    configs/
    scripts/
    README.md
```

첫날 생성 명령:

```bash
mkdir -p phases/phase_{01_pytorch_primitives,02_toy_occupancy,03_attention_lifting,04_deconv_structured_head,05_multicamera_single_frame,06_surface_geometry,07_temporal_flow,08_dynamic_resolution_refinement,09_integration_evaluation,10_runtime_final_report}/{src,tests,scripts}
mkdir -p phases/phase_05_multicamera_single_frame/configs phases/phase_09_integration_evaluation/configs phases/phase_10_runtime_final_report/reports
mkdir -p final_occnet_v1/src/occnet_v1 final_occnet_v1/tests final_occnet_v1/configs final_occnet_v1/scripts
```

---

## 2. 전체 Phase 요약

```text
Phase 01, D001-D010:
  PyTorch tensor, Conv2d/3d, ConvTranspose3d, grid_sample

Phase 02, D011-D020:
  synthetic occupancy dataset, toy model, one-batch overfit

Phase 03, D021-D035:
  Q/K/V attention, image token, 3D query lifting

Phase 04, D036-D050:
  1.6m -> 0.4m deconv, structured 20cm occupancy head

Phase 05, D051-D065:
  multi-camera token packing, canonical ray PE, single-frame occupancy training

Phase 06, D066-D080:
  RoadBEV-style surface geometry head

Phase 07, D081-D095:
  ViewFormer-style temporal BEV memory, dynamic / flow head

Phase 08, D096-D110:
  active mask, quota Top-K, sparse anchor, packed Queryable MLP

Phase 09, D111-D120:
  final model integration, losses, metrics, visualization

Phase 10, D121-D130:
  runtime profiling, ablation, final report, next steps
```

---

## 3. 매일 1시간 운영법

매일 같은 패턴을 따른다.

```text
10분:
  어제 코드 실행
  failing test 확인

40분:
  오늘 파일 1~2개 작성 또는 수정

10분:
  pytest 또는 script 실행
  notes.md에 결과 기록
```

각 day의 완료 조건:

```text
1. 지정된 파일이 존재한다.
2. 지정된 test 또는 script가 실행된다.
3. shape 출력 또는 loss 출력이 기록된다.
4. 실패했으면 notes.md에 실패 이유를 적는다.
```

---

## 4. Phase 01: PyTorch Primitives

기간: D001-D010

목표:

```text
tensor shape 조작, Conv2d/3d, ConvTranspose3d, grid_sample을 직접 구현한다.
occupancy architecture에서 계속 쓰는 shape 감각을 만든다.
```

### D001: phase 디렉토리 생성과 notes 시작

만들 파일:

```text
phases/phase_01_pytorch_primitives/notes.md
```

할 일:

```text
phase 디렉토리 생성
Python import가 되는지 확인
notes.md에 "D001 시작" 기록
```

검증:

```bash
find phases -maxdepth 2 -type d | sort
```

### D002: tensor flatten / unflatten 구현

만들 파일:

```text
phases/phase_01_pytorch_primitives/src/tensor_utils.py
phases/phase_01_pytorch_primitives/tests/test_tensor_utils.py
```

구현:

```python
def flatten_hw(x): ...
def unflatten_hw(tokens, h, w): ...
def flatten_xyz(x): ...
def unflatten_xyz(tokens, x_size, y_size, z_size): ...
```

검증:

```bash
pytest phases/phase_01_pytorch_primitives/tests/test_tensor_utils.py
```

이해:

```text
B x C x H x W <-> B x HW x C
B x C x X x Y x Z <-> B x XYZ x C
```

### D003: permute / contiguous / reshape 실험

수정 파일:

```text
phases/phase_01_pytorch_primitives/tests/test_tensor_utils.py
```

할 일:

```text
permute 후 view가 실패하는 case 작성
contiguous 후 view가 성공하는 case 작성
reshape와 view 차이 notes.md에 기록
```

검증:

```bash
pytest phases/phase_01_pytorch_primitives/tests/test_tensor_utils.py -q
```

### D004: Tiny Conv2d backbone

만들 파일:

```text
phases/phase_01_pytorch_primitives/src/image_backbone.py
phases/phase_01_pytorch_primitives/tests/test_image_backbone.py
```

구현:

```python
class TinyImageBackbone(nn.Module):
    # input: B x 3 x 128 x 256
    # output: B x C x 16 x 32
```

검증:

```bash
pytest phases/phase_01_pytorch_primitives/tests/test_image_backbone.py
```

### D005: Conv3d block

만들 파일:

```text
phases/phase_01_pytorch_primitives/src/voxel_blocks.py
phases/phase_01_pytorch_primitives/tests/test_voxel_blocks.py
```

구현:

```python
class Conv3DBlock(nn.Module):
    # Conv3d + Norm + ReLU
```

검증:

```text
input:  B x C x 8 x 4 x 2
output: B x C2 x 8 x 4 x 2
```

### D006: ConvTranspose3d upsample

수정 파일:

```text
phases/phase_01_pytorch_primitives/src/voxel_blocks.py
phases/phase_01_pytorch_primitives/tests/test_voxel_blocks.py
```

구현:

```python
class Deconv3DBlock(nn.Module):
    # stride=2 ConvTranspose3d
```

검증:

```text
8 x 4 x 2 -> 16 x 8 x 4
```

### D007: 2D grid_sample 기본

만들 파일:

```text
phases/phase_01_pytorch_primitives/src/grid_sample_utils.py
phases/phase_01_pytorch_primitives/tests/test_grid_sample_utils.py
```

구현:

```python
def pixel_to_grid(pixel_xy, width, height, align_corners=False): ...
def sample_2d_feature(feat, pixel_xy): ...
```

검증:

```text
known feature map에서 특정 pixel 값을 정확히 sample
```

### D008: 3D trilinear sampling 준비

수정 파일:

```text
phases/phase_01_pytorch_primitives/src/grid_sample_utils.py
phases/phase_01_pytorch_primitives/tests/test_grid_sample_utils.py
```

구현:

```python
def xyz_to_grid(points_xyz, x_range, y_range, z_range, shape_xyz): ...
def sample_3d_feature(feat_3d, points_xyz): ...
```

검증:

```text
B x C x X x Y x Z에서 Q개 point sample
출력 shape: B x Q x C
```

### D009: voxel center 생성

만들 파일:

```text
phases/phase_01_pytorch_primitives/src/voxel_grid.py
phases/phase_01_pytorch_primitives/tests/test_voxel_grid.py
```

구현:

```python
def make_voxel_centers(x_range, y_range, z_range, voxel_size): ...
```

검증:

```text
x [-1, 1], voxel_size 1.0이면 center가 -0.5, 0.5인지 확인
```

### D010: Phase 01 mini report

만들 파일:

```text
phases/phase_01_pytorch_primitives/notes.md
```

할 일:

```text
지금까지 만든 함수 목록 정리
가장 헷갈린 shape 3개 기록
pytest 전체 실행
```

검증:

```bash
pytest phases/phase_01_pytorch_primitives/tests -q
```

---

## 5. Phase 02: Toy Occupancy

기간: D011-D020

목표:

```text
synthetic 3D occupancy label을 만들고, 작은 model이 one-batch overfit되는 경험을 만든다.
```

### D011: synthetic cube occupancy 생성

만들 파일:

```text
phases/phase_02_toy_occupancy/src/synthetic.py
phases/phase_02_toy_occupancy/tests/test_synthetic.py
```

구현:

```python
def make_empty_grid(shape): ...
def add_box(occ, center, size): ...
def make_box_scene(shape): ...
```

검증:

```text
occupied voxel count가 예상 범위인지 확인
```

### D012: free / occupied / unknown mask

수정 파일:

```text
phases/phase_02_toy_occupancy/src/synthetic.py
phases/phase_02_toy_occupancy/tests/test_synthetic.py
```

구현:

```python
def make_occ_free_unknown(occ): ...
```

검증:

```text
target 값:
  occupied = 1
  free = 0
  unknown = ignore mask
```

### D013: Tiny latent-to-occupancy model

만들 파일:

```text
phases/phase_02_toy_occupancy/src/tiny_model.py
phases/phase_02_toy_occupancy/tests/test_tiny_model.py
```

구현:

```python
class TinyLatentOccNet(nn.Module):
    # learnable B x C x X x Y x Z latent
    # Conv3d blocks
    # occupancy logits
```

검증:

```text
output shape: B x 1 x X x Y x Z
```

### D014: BCE loss with ignore mask

만들 파일:

```text
phases/phase_02_toy_occupancy/src/losses.py
phases/phase_02_toy_occupancy/tests/test_losses.py
```

구현:

```python
def masked_bce_with_logits(logits, target, valid_mask): ...
```

검증:

```text
unknown mask가 loss에 들어가지 않는지 확인
```

### D015: one-batch training script

만들 파일:

```text
phases/phase_02_toy_occupancy/scripts/train_one_batch.py
```

구현:

```text
1개 synthetic scene
TinyLatentOccNet
200 step train
loss print
```

검증:

```bash
python phases/phase_02_toy_occupancy/scripts/train_one_batch.py
```

완료 기준:

```text
loss가 확실히 감소
```

### D016: voxel visualization

만들 파일:

```text
phases/phase_02_toy_occupancy/src/visualize.py
phases/phase_02_toy_occupancy/scripts/visualize_scene.py
```

구현:

```python
def save_voxel_slices(occ, out_dir): ...
```

검증:

```text
z slice별 png 저장
```

### D017: class imbalance 관찰

수정 파일:

```text
phases/phase_02_toy_occupancy/src/losses.py
phases/phase_02_toy_occupancy/scripts/train_one_batch.py
```

구현:

```python
def focal_loss_with_logits(...): ...
```

검증:

```text
BCE와 Focal loss 둘 다 실행
notes.md에 차이 기록
```

### D018: simple dataset class

만들 파일:

```text
phases/phase_02_toy_occupancy/src/dataset.py
```

구현:

```python
class SyntheticVoxelDataset(Dataset):
    # random boxes, walls, spheres
```

검증:

```text
DataLoader batch shape 확인
```

### D019: tiny validation metric

만들 파일:

```text
phases/phase_02_toy_occupancy/src/metrics.py
```

구현:

```python
def occupancy_iou(logits, target, valid_mask, threshold=0.5): ...
def precision_recall(...): ...
```

검증:

```text
perfect prediction이면 IoU=1
all-zero prediction test
```

### D020: Phase 02 report

할 일:

```text
one-batch overfit 결과 이미지 저장
loss curve 저장
Month 1~2에서 가장 중요한 shape 정리
```

검증:

```bash
pytest phases/phase_02_toy_occupancy/tests -q
```

---

## 6. Phase 03: Attention Lifting

기간: D021-D035

목표:

```text
언어 번역 attention 이해를 image token과 3D query cross-attention으로 확장한다.
3D query가 image tokens를 읽어 coarse 3D feature를 만드는 tiny lifter를 구현한다.
```

### D021: scaled dot-product attention 구현

만들 파일:

```text
phases/phase_03_attention_lifting/src/attention.py
phases/phase_03_attention_lifting/tests/test_attention.py
```

구현:

```python
def scaled_dot_product_attention(q, k, v, mask=None): ...
```

검증:

```text
attention weight sum = 1
output shape = B x Nq x C
```

### D022: Multi-head CrossAttention

수정 파일:

```text
phases/phase_03_attention_lifting/src/attention.py
```

구현:

```python
class VanillaCrossAttention(nn.Module):
    # Q from query tokens
    # K,V from image tokens
```

검증:

```text
query B x 64 x C
image B x 512 x C
output B x 64 x C
```

### D023: REO 발췌 읽기

읽을 것:

```text
REO Abstract
Method overview figure
calibration-free / vanilla attention 관련 부분
```

작성:

```text
phases/phase_03_attention_lifting/notes.md
```

기록 질문:

```text
Q는 무엇인가?
K,V는 무엇인가?
왜 projection-first deformable attention과 다른가?
```

### D024: image token flatten

만들 파일:

```text
phases/phase_03_attention_lifting/src/image_tokens.py
phases/phase_03_attention_lifting/tests/test_image_tokens.py
```

구현:

```python
def image_to_tokens(feat): ...
def add_2d_sincos_pe(tokens, h, w): ...
```

검증:

```text
B x C x H x W -> B x HW x C
```

### D025: 3D query positional embedding

만들 파일:

```text
phases/phase_03_attention_lifting/src/query_grid.py
phases/phase_03_attention_lifting/tests/test_query_grid.py
```

구현:

```python
def make_3d_query_grid(shape_xyz): ...
class QueryEmbedding3D(nn.Module): ...
```

검증:

```text
8 x 4 x 2 -> 64 query tokens
```

### D026: TinyLifter forward

만들 파일:

```text
phases/phase_03_attention_lifting/src/lifter.py
phases/phase_03_attention_lifting/tests/test_lifter.py
```

구현:

```python
class TinyLifter(nn.Module):
    image -> TinyBackbone -> tokens
    query embeddings -> CrossAttention
    output -> B x C x X x Y x Z
```

검증:

```text
image B x 3 x 128 x 256
coarse feature B x C x 8 x 4 x 2
```

### D027: attention overfit toy

만들 파일:

```text
phases/phase_03_attention_lifting/scripts/train_attention_toy.py
```

구현:

```text
image에 단순 object 위치 encoding
3D query가 object occupancy를 맞추는 toy task
```

검증:

```text
1개 batch overfit
```

### D028: camera ray positional encoding 개념 코드

수정 파일:

```text
phases/phase_03_attention_lifting/src/image_tokens.py
```

구현:

```python
def make_dummy_ray_pe(h, w, c): ...
```

목표:

```text
처음에는 실제 intrinsic 없이 normalized ray direction 비슷한 PE를 만든다.
```

검증:

```text
ray PE shape가 image token shape와 맞는지 확인
```

### D029: camera id embedding

수정 파일:

```text
phases/phase_03_attention_lifting/src/image_tokens.py
```

구현:

```python
class CameraIDEmbedding(nn.Module): ...
```

검증:

```text
camera id가 다르면 token embedding이 달라지는지 확인
```

### D030: multi-camera token pack tiny

만들 파일:

```text
phases/phase_03_attention_lifting/tests/test_multicam_tokens.py
```

구현:

```python
def pack_multicam_tokens(tokens_per_cam): ...
```

검증:

```text
B x N_cam x HW x C -> B x (N_cam*HW) x C
```

### D031: cross-attention memory cost 기록

할 일:

```text
Nq, Nk를 바꿔 attention tensor 크기 계산
notes.md에 기록
```

기록 예:

```text
Nq=1976, Nk=multi-camera tokens
attention weight size = B x heads x Nq x Nk
```

### D032: attention output visualization

만들 파일:

```text
phases/phase_03_attention_lifting/scripts/visualize_attention.py
```

구현:

```text
특정 3D query가 보는 image token attention map 저장
```

검증:

```text
png 저장
```

### D033: residual + LayerNorm 추가

수정 파일:

```text
phases/phase_03_attention_lifting/src/attention.py
```

구현:

```python
class CrossAttentionBlock(nn.Module):
    attention + residual + norm + MLP
```

검증:

```text
shape 유지
gradient flow 확인
```

### D034: TinyLifter one-batch overfit

수정 파일:

```text
phases/phase_03_attention_lifting/scripts/train_attention_toy.py
```

할 일:

```text
CrossAttentionBlock 사용
loss 감소 확인
```

### D035: Phase 03 report

기록:

```text
번역 attention과 3D query attention의 차이
attention에서 가장 큰 memory cost
REO에서 가져온 아이디어
```

검증:

```bash
pytest phases/phase_03_attention_lifting/tests -q
```

---

## 7. Phase 04: Deconv and Structured 20cm Head

기간: D036-D050

목표:

```text
1.6m feature를 0.4m까지 deconv하고, 20cm dense output을 structured head로 만든다.
```

### D036: 1.6m / 0.8m / 0.4m grid 계산

만들 파일:

```text
phases/phase_04_deconv_structured_head/src/grid_config.py
phases/phase_04_deconv_structured_head/tests/test_grid_config.py
```

구현:

```python
def compute_logical_shapes(range_xyz, resolutions): ...
def compute_padded_shapes(coarse_shape, num_deconv): ...
```

검증:

```text
1.6m: 38 x 13 x 4
0.8m internal: 76 x 26 x 8
0.4m internal: 152 x 52 x 16
0.4m valid: 150 x 50 x 13
```

### D037: PanoOcc 발췌 읽기

읽을 것:

```text
PanoOcc architecture figure
coarse-to-fine decoder 설명
```

기록:

```text
deconv가 하는 일
왜 0.2m dense feature까지 만들지 않는지
```

### D038: TwoStageVoxelDecoder 구현

만들 파일:

```text
phases/phase_04_deconv_structured_head/src/voxel_decoder.py
phases/phase_04_deconv_structured_head/tests/test_voxel_decoder.py
```

구현:

```python
class TwoStageVoxelDecoder(nn.Module):
    # 38 x 13 x 4 -> 76 x 26 x 8 -> 152 x 52 x 16
```

### D039: valid crop / mask

수정 파일:

```text
phases/phase_04_deconv_structured_head/src/voxel_decoder.py
```

구현:

```python
def crop_valid_04m(feat): ...
```

검증:

```text
152 x 52 x 16 -> 150 x 50 x 13
```

### D040: StructuredSubvoxelHead 구현

만들 파일:

```text
phases/phase_04_deconv_structured_head/src/occupancy_heads.py
phases/phase_04_deconv_structured_head/tests/test_structured_head.py
```

구현:

```python
class StructuredSubvoxelHead(nn.Module):
    # B x C x X x Y x Z
    # -> B x 1 x 2X x 2Y x 2Z
```

### D041: channel reshape 정확성 test

수정 파일:

```text
phases/phase_04_deconv_structured_head/tests/test_structured_head.py
```

검증:

```text
8개 channel이 2x2x2 sub-voxel 위치로 정확히 들어가는지 synthetic value로 확인
```

### D042: occupancy head loss 연결

만들 파일:

```text
phases/phase_04_deconv_structured_head/src/losses.py
```

구현:

```python
def occupancy_loss(logits, target, valid_mask): ...
```

### D043: deconv + structured head end-to-end shape

만들 파일:

```text
phases/phase_04_deconv_structured_head/tests/test_end_to_end_shape.py
```

검증:

```text
1.6m feature
-> 0.4m valid feature
-> 20cm logits
```

### D044: tiny training with decoder

만들 파일:

```text
phases/phase_04_deconv_structured_head/scripts/train_decoder_toy.py
```

검증:

```text
one-batch overfit
```

### D045: 20 FPS guardrail test

수정 파일:

```text
phases/phase_04_deconv_structured_head/tests/test_structured_head.py
```

검증:

```text
StructuredSubvoxelHead가 20cm dense feature tensor를 만들지 않는지 확인
출력 channel은 logits뿐인지 확인
```

### D046: simple profiler

만들 파일:

```text
phases/phase_04_deconv_structured_head/scripts/profile_structured_head.py
```

측정:

```text
decoder latency
structured head latency
peak memory
```

### D047: decoder channel ablation

할 일:

```text
C=16, 32, 64 latency와 memory 비교
notes.md에 기록
```

### D048: 0.8m attention vs 1.6m attention query count 계산

만들 파일:

```text
phases/phase_04_deconv_structured_head/scripts/compute_query_counts.py
```

검증:

```text
1.6m query 수와 0.8m query 수 비교 출력
```

### D049: architecture 문서와 구현 비교

할 일:

```text
occupancy_network_architecture.md의 Section 7, 9와 현재 코드 비교
빠진 guardrail notes.md에 기록
```

### D050: Phase 04 report

검증:

```bash
pytest phases/phase_04_deconv_structured_head/tests -q
```

기록:

```text
20cm dense output과 20cm dense feature의 차이
```

---

## 8. Phase 05: Multi-camera Single-frame Occupancy

기간: D051-D065

목표:

```text
multi-camera token packing과 1.6m cross-attention lifting을 single-frame occupancy training에 연결한다.
```

### D051: phase_05 config 작성

만들 파일:

```text
phases/phase_05_multicamera_single_frame/configs/tiny.yaml
```

내용:

```yaml
num_cameras: 4
image_size: [128, 256]
channels: 32
coarse_shape: [38, 13, 4]
valid_04m_shape: [150, 50, 13]
```

### D052: MultiCameraSyntheticDataset

만들 파일:

```text
phases/phase_05_multicamera_single_frame/src/dataset.py
```

구현:

```python
class MultiCameraSyntheticDataset(Dataset):
    # N_cam images
    # occupancy target
```

검증:

```text
images: B x N_cam x 3 x H x W
target: B x 1 x 300 x 100 x 25 또는 tiny shape
```

### D053: camera geometry placeholder

만들 파일:

```text
phases/phase_05_multicamera_single_frame/src/camera_geometry.py
```

구현:

```python
def make_virtual_camera_rays(num_cameras, h, w): ...
def make_camera_id(num_cameras, h, w): ...
```

목표:

```text
처음에는 실제 projection보다 canonical ray PE placeholder를 사용한다.
```

### D054: multi-camera image encoder

만들 파일:

```text
phases/phase_05_multicamera_single_frame/src/image_encoder.py
```

구현:

```python
class MultiCameraImageEncoder(nn.Module):
    # encode each camera with shared backbone
    # pack tokens
```

### D055: multi-camera lifter

만들 파일:

```text
phases/phase_05_multicamera_single_frame/src/lifter.py
```

구현:

```python
class MultiCameraLifter(nn.Module):
    # packed image tokens
    # 1.6m 3D query
    # vanilla cross-attention
```

### D056: single-frame model assembly

만들 파일:

```text
phases/phase_05_multicamera_single_frame/src/model.py
```

구현:

```python
class SingleFrameOccNet(nn.Module):
    image encoder
    lifter
    decoder
    structured occupancy head
```

### D057: shape test

만들 파일:

```text
phases/phase_05_multicamera_single_frame/tests/test_model_shape.py
```

검증:

```text
batch 1, 2에서 forward 동작
```

### D058: training loop

만들 파일:

```text
phases/phase_05_multicamera_single_frame/scripts/train_single_frame.py
```

검증:

```text
loss print
one-batch overfit
```

### D059: occupancy IoU metric

만들 파일:

```text
phases/phase_05_multicamera_single_frame/src/metrics.py
```

구현:

```python
def occupancy_iou(...): ...
```

### D060: visualization

만들 파일:

```text
phases/phase_05_multicamera_single_frame/scripts/visualize_prediction.py
```

검증:

```text
target / prediction z-slice 저장
```

### D061: real camera projection mini 실습

수정 파일:

```text
phases/phase_05_multicamera_single_frame/src/camera_geometry.py
```

구현:

```python
def project_points(K, T_cam_ego, points_ego): ...
```

검증:

```text
간단한 known point projection 확인
```

### D062: projection notes

기록:

```text
intrinsic K
extrinsic T_cam_ego
ego point -> camera point -> pixel
```

목표:

```text
개념을 오래 끌지 않고 코드와 함께 정리
```

### D063: tiny real/sim data interface placeholder

만들 파일:

```text
phases/phase_05_multicamera_single_frame/src/real_dataset_stub.py
```

구현:

```python
class RealDatasetStub(Dataset):
    # 실제 dataset 연결 전 interface만 정의
```

### D064: single-frame overfit 안정화

할 일:

```text
learning rate 조정
channel 수 조정
loss가 내려가지 않으면 architecture를 단순화
```

### D065: Phase 05 report

검증:

```bash
pytest phases/phase_05_multicamera_single_frame/tests -q
```

기록:

```text
multi-camera token shape
1.6m query count
single-frame occupancy 결과
```

---

## 9. Phase 06: Road Surface Geometry

기간: D066-D080

목표:

```text
RoadBEV-style z_surface / valid / uncertainty head를 만든다.
```

### D066: RoadBEV 발췌 읽기

읽을 것:

```text
RoadBEV problem definition
road elevation output
loss / metric 부분
```

기록:

```text
occupancy와 surface height map의 차이
```

### D067: surface label 생성

만들 파일:

```text
phases/phase_06_surface_geometry/src/surface_labels.py
phases/phase_06_surface_geometry/tests/test_surface_labels.py
```

구현:

```python
def make_z_surface_from_occ(occ): ...
def make_surface_valid_mask(occ): ...
```

### D068: learned z pooling

만들 파일:

```text
phases/phase_06_surface_geometry/src/surface_head.py
phases/phase_06_surface_geometry/tests/test_surface_head.py
```

구현:

```python
class LearnedZPooling(nn.Module): ...
```

검증:

```text
B x C x X x Y x Z -> B x Cb x X x Y
```

### D069: SurfaceGeometryHead

수정 파일:

```text
phases/phase_06_surface_geometry/src/surface_head.py
```

구현:

```python
class SurfaceGeometryHead(nn.Module):
    # z_surface, valid_logit, uncertainty
```

### D070: surface losses

만들 파일:

```text
phases/phase_06_surface_geometry/src/surface_losses.py
```

구현:

```python
def surface_huber_loss(z_pred, z_gt, valid): ...
def surface_valid_bce(valid_logit, valid): ...
```

### D071: surface one-batch training

만들 파일:

```text
phases/phase_06_surface_geometry/scripts/train_surface_one_batch.py
```

검증:

```text
z_surface loss 감소
```

### D072: surface visualization

만들 파일:

```text
phases/phase_06_surface_geometry/scripts/visualize_surface.py
```

검증:

```text
height map png 저장
```

### D073: slope / normal derived output

만들 파일:

```text
phases/phase_06_surface_geometry/src/surface_derived.py
```

구현:

```python
def compute_slope_from_z(z_surface): ...
def compute_step_score_from_z(z_surface): ...
```

### D074: step / curb toy scene

수정 파일:

```text
phases/phase_06_surface_geometry/src/surface_labels.py
```

구현:

```text
flat ground
ramp
step
stair-like repeated step
```

### D075: occupancy-surface consistency toy

만들 파일:

```text
phases/phase_06_surface_geometry/src/consistency.py
```

구현:

```python
def surface_occ_consistency_loss(occ_logits, z_surface, valid): ...
```

### D076: surface head를 phase_05 model에 붙이기

만들 파일:

```text
phases/phase_06_surface_geometry/src/model_with_surface.py
```

구현:

```text
phase_05 SingleFrameOccNet 구조 + surface head
```

### D077: occupancy + surface joint train

만들 파일:

```text
phases/phase_06_surface_geometry/scripts/train_occ_surface.py
```

검증:

```text
L_occ, L_surface 둘 다 감소
```

### D078: surface uncertainty

수정 파일:

```text
phases/phase_06_surface_geometry/src/surface_head.py
```

구현:

```text
uncertainty output이 positive가 되도록 softplus 사용
```

### D079: surface metric

만들 파일:

```text
phases/phase_06_surface_geometry/src/surface_metrics.py
```

구현:

```python
def surface_mae(z_pred, z_gt, valid): ...
def valid_accuracy(valid_logit, valid): ...
```

### D080: Phase 06 report

기록:

```text
surface head가 occupancy와 다른 이유
z_surface / valid / uncertainty 결과
```

검증:

```bash
pytest phases/phase_06_surface_geometry/tests -q
```

---

## 10. Phase 07: Temporal Memory and Flow

기간: D081-D095

목표:

```text
ViewFormer-style 1.6m BEV memory를 구현하고, dynamic / flow head를 붙인다.
```

### D081: ViewFormer 발췌 읽기

읽을 것:

```text
ViewFormer temporal memory
ego-motion alignment
occupancy flow 관련 부분
```

기록:

```text
왜 3D memory가 아니라 BEV memory인가?
```

### D082: ZPool 구현

만들 파일:

```text
phases/phase_07_temporal_flow/src/temporal_memory.py
phases/phase_07_temporal_flow/tests/test_temporal_memory.py
```

구현:

```python
class ZPool(nn.Module): ...
```

검증:

```text
B x C x X x Y x Z -> B x Cb x X x Y
```

### D083: BEV ego-motion warp

수정 파일:

```text
phases/phase_07_temporal_flow/src/temporal_memory.py
```

구현:

```python
def warp_bev_feature(bev, pose_delta): ...
```

검증:

```text
identity pose면 output 동일
translation pose면 feature 이동
```

### D084: memory queue

구현:

```python
class BEVMemoryQueue:
    push(feature, pose)
    get_aligned(current_pose)
```

검증:

```text
N_history=3 유지
```

### D085: temporal fusion simple

구현:

```python
class TemporalFusion(nn.Module):
    # current BEV + aligned memory
    # simple mean / gated fusion
```

목표:

```text
처음에는 attention보다 simple fusion으로 시작
```

### D086: temporal attention

수정:

```text
TemporalFusion에 attention option 추가
```

검증:

```text
output shape 유지
```

### D087: Inject3D

구현:

```python
class InjectBEVTo3D(nn.Module):
    # BEV temporal context broadcast to Z
    # z embedding
    # residual add
```

### D088: temporal model integration

만들 파일:

```text
phases/phase_07_temporal_flow/src/model_temporal.py
```

구현:

```text
1.6m feature
-> temporal BEV memory
-> enhanced 1.6m feature
-> decoder
```

### D089: moving cube synthetic sequence

만들 파일:

```text
phases/phase_07_temporal_flow/src/sequence_dataset.py
```

구현:

```text
moving cube sequence
pose delta
flow target
```

### D090: DynamicFlowHead

만들 파일:

```text
phases/phase_07_temporal_flow/src/flow_head.py
phases/phase_07_temporal_flow/tests/test_flow_head.py
```

구현:

```python
class DynamicFlowHead(nn.Module):
    # dynamic logit
    # flow dx, dy, dz
```

### D091: flow loss

만들 파일:

```text
phases/phase_07_temporal_flow/src/flow_losses.py
```

구현:

```python
def dynamic_bce_loss(...): ...
def flow_smooth_l1_loss(flow_pred, flow_gt, mask): ...
```

### D092: flow one-batch overfit

만들 파일:

```text
phases/phase_07_temporal_flow/scripts/train_flow_one_batch.py
```

검증:

```text
moving cube flow 방향 학습
```

### D093: flow visualization

만들 파일:

```text
phases/phase_07_temporal_flow/scripts/visualize_flow.py
```

검증:

```text
BEV arrow plot 저장
```

### D094: temporal ablation

실험:

```text
no temporal
simple temporal mean
temporal attention
```

기록:

```text
loss / IoU / flow error 비교
```

### D095: Phase 07 report

검증:

```bash
pytest phases/phase_07_temporal_flow/tests -q
```

기록:

```text
BEV memory shape
flow head mask
가장 어려운 점
```

---

## 11. Phase 08: Dynamic-resolution Refinement

기간: D096-D110

목표:

```text
Active Mask + quota Top-K + sparse local anchor + packed Queryable MLP를 구현한다.
```

### D096: active score components

만들 파일:

```text
phases/phase_08_dynamic_resolution_refinement/src/active_mask.py
phases/phase_08_dynamic_resolution_refinement/tests/test_active_mask.py
```

구현:

```python
def boundary_score(occ_logits): ...
def uncertainty_score(occ_logits): ...
def dynamic_score(dynamic_logits): ...
```

### D097: surface / planner / near score

수정 파일:

```text
phases/phase_08_dynamic_resolution_refinement/src/active_mask.py
```

구현:

```python
def near_field_score(voxel_centers): ...
def planner_corridor_score(voxel_centers, trajectory): ...
def surface_score(z_surface, valid): ...
```

### D098: S_refine 통합

구현:

```python
def compute_refine_score(...): ...
```

검증:

```text
boundary / uncertainty / planner 영역 score가 높음
```

### D099: quota_topk

구현:

```python
def quota_topk(category_scores, quotas): ...
```

검증:

```text
quota별 selected count 확인
중복 제거 확인
```

### D100: distance / planner LOD budget

구현:

```python
def assign_query_budget(selected_voxels, distance, planner_score, dynamic_score): ...
```

검증:

```text
near/planner/dynamic은 M_i가 큼
far/static은 M_i가 작음
```

### D101: exposed face mask

구현:

```python
def compute_exposed_faces(occ_binary): ...
```

검증:

```text
free/unknown neighbor 방향이 exposed인지 확인
```

### D102: adaptive query generation

만들 파일:

```text
phases/phase_08_dynamic_resolution_refinement/src/query_generation.py
```

구현:

```python
def make_initial_queries(exposed_faces, budget): ...
```

### D103: pack variable queries

구현:

```python
def pack_query_lists(query_lists): ...
```

검증:

```text
q_local_packed: Q x 3
query_offsets: K + 1
```

### D104: SparseLocalAnchor

만들 파일:

```text
phases/phase_08_dynamic_resolution_refinement/src/sparse_anchor.py
```

구현:

```python
def gather_04m_features(feat, selected_idx): ...
class SparseLocalAnchor(nn.Module): ...
```

### D105: QueryableMLP

만들 파일:

```text
phases/phase_08_dynamic_resolution_refinement/src/queryable_mlp.py
```

구현:

```python
class QueryableMLP(nn.Module):
    # F_query + q_local + PE + coarse logits
```

### D106: packed execution test

만들 파일:

```text
phases/phase_08_dynamic_resolution_refinement/tests/test_queryable_mlp.py
```

검증:

```text
K=32
variable M_i
Q=sum(M_i)
MLP output Q x 1
```

### D107: boundary search

수정 파일:

```text
phases/phase_08_dynamic_resolution_refinement/src/query_generation.py
```

구현:

```python
def find_sign_change_pairs(query_points, occ_probs): ...
def refine_boundary_bisection(...): ...
```

### D108: fine overlay representation

만들 파일:

```text
phases/phase_08_dynamic_resolution_refinement/src/fine_overlay.py
```

구현:

```python
class FineOverlay:
    # selected voxel idx
    # local cuboids / local boundary samples
```

### D109: refinement toy training

만들 파일:

```text
phases/phase_08_dynamic_resolution_refinement/scripts/train_refinement_toy.py
```

검증:

```text
selected boundary voxel 내부 query loss 감소
```

### D110: Phase 08 report

검증:

```bash
pytest phases/phase_08_dynamic_resolution_refinement/tests -q
```

기록:

```text
K_total
Q_total
packed MLP latency
```

---

## 12. Phase 09: Integration and Evaluation

기간: D111-D120

목표:

```text
각 phase에서 만든 핵심 모듈을 final_occnet_v1로 모으고, end-to-end forward / loss / metric / visualization을 만든다.
```

### D111: final_occnet_v1 skeleton

만들 파일:

```text
final_occnet_v1/src/occnet_v1/__init__.py
final_occnet_v1/src/occnet_v1/model.py
final_occnet_v1/configs/tiny.yaml
```

할 일:

```text
각 phase에서 안정된 모듈 copy
import 정리
```

### D112: OccNetV1 forward

수정 파일:

```text
final_occnet_v1/src/occnet_v1/model.py
```

구현:

```python
class OccNetV1(nn.Module):
    def forward(batch):
        return {
          "occ_logits": ...,
          "surface": ...,
          "dynamic": ...,
          "flow": ...,
          "refine": ...
        }
```

### D113: final shape test

만들 파일:

```text
final_occnet_v1/tests/test_model_shape.py
```

검증:

```text
batch 1, 2 forward
all output keys
```

### D114: unified losses

만들 파일:

```text
final_occnet_v1/src/occnet_v1/losses.py
```

구현:

```python
L_total = L_occ + lambda_surface*L_surface + lambda_flow*L_flow + lambda_refine*L_refine
```

### D115: final training script

만들 파일:

```text
final_occnet_v1/scripts/train_tiny.py
```

검증:

```text
one-batch overfit
```

### D116: metrics

만들 파일:

```text
final_occnet_v1/src/occnet_v1/metrics.py
```

구현:

```text
occupancy IoU
surface MAE
flow endpoint error
active selection count
```

### D117: visualization package

만들 파일:

```text
final_occnet_v1/scripts/visualize_outputs.py
```

검증:

```text
occupancy slices
surface height
flow BEV arrows
active mask map
```

### D118: config ablation switches

수정 파일:

```text
final_occnet_v1/configs/tiny.yaml
```

추가:

```yaml
use_surface: true
use_temporal: true
use_flow: true
use_refinement: true
```

### D119: end-to-end small validation

할 일:

```text
small synthetic validation set
metrics table 출력
```

### D120: Phase 09 report

기록:

```text
동작하는 모듈
아직 불안정한 모듈
가장 큰 shape 문제
```

---

## 13. Phase 10: Runtime and Final Report

기간: D121-D130

목표:

```text
runtime profiling, ablation, 20 FPS 가능성 평가, final report를 작성한다.
```

### D121: profiler script

만들 파일:

```text
final_occnet_v1/scripts/profile_runtime.py
```

측정:

```text
backbone
attention
temporal
deconv
structured head
surface head
flow head
active mask
Top-K
packed Queryable MLP
```

### D122: K/Q latency table

실험:

```text
K=512, 1024, 2048
average M_i=4, 8, 16
```

기록:

```text
final_occnet_v1/reports/runtime_kq_table.md
```

### D123: memory profiling

할 일:

```text
torch.cuda.max_memory_allocated 기록
phase별 peak memory 기록
```

### D124: ablation A0-A3

실험:

```text
A0 occupancy only
A1 + surface
A2 + temporal
A3 + flow
```

### D125: ablation A4-A6

실험:

```text
A4 + active mask
A5 + sparse anchor
A6 + packed adaptive query
```

### D126: 20 FPS blocker 분석

기록:

```text
가장 느린 top 3 stage
가장 memory 큰 top 3 tensor
20 FPS 달성 가능성
```

### D127: ONNX export 준비

만들 파일:

```text
final_occnet_v1/scripts/export_onnx.py
```

목표:

```text
전체가 안 되면 backbone + structured head 일부라도 export
```

### D128: final README

만들 파일:

```text
final_occnet_v1/README.md
```

내용:

```text
model overview
how to train
how to visualize
known limitations
```

### D129: final report

만들 파일:

```text
phases/phase_10_runtime_final_report/reports/final_report.md
```

내용:

```text
구현 완료
미완료
metric
latency
visual examples
다음 단계
```

### D130: next 3-month plan

만들 파일:

```text
phases/phase_10_runtime_final_report/reports/next_3_months.md
```

후보:

```text
TensorRT FP16
P3-only local image re-query v1.5
flow-aware dynamic feature correction
real dataset label pipeline
```

---

## 14. 공통 검증 체크리스트

모든 phase에서 다음을 지킨다.

```text
shape test가 있는가?
one-batch overfit이 되는가?
NaN이 없는가?
batch size 1과 2 모두 동작하는가?
시각화가 있는가?
notes.md에 오늘 결과가 기록되었는가?
```

실패하면 다음 phase로 넘어가지 않는다.

---

## 15. 바이브 코딩 요청 템플릿

AI에게 요청할 때는 항상 다음 형식을 쓴다.

```text
목표:
  무엇을 만들지

파일:
  어느 파일에 만들지

입력 shape:
  B x ...

출력 shape:
  B x ...

금지:
  어떤 구현을 피해야 하는지

검증:
  어떤 pytest나 script가 통과해야 하는지
```

예시:

```text
phases/phase_04_deconv_structured_head/src/occupancy_heads.py에 StructuredSubvoxelHead를 만들어줘.
입력은 B x C x X x Y x Z이고 출력은 B x 1 x 2X x 2Y x 2Z야.
20cm dense feature volume은 만들면 안 돼.
Conv3d 1x1로 8개 sub-voxel logit channel을 만든 뒤 reshape해야 해.
tests/test_structured_head.py도 작성해줘.
```

---

## 16. 논문 읽기 순서 요약

```text
D023:
  REO Abstract + Method overview

D037:
  PanoOcc architecture + coarse-to-fine decoder

D066:
  RoadBEV problem definition + road elevation output

D081:
  ViewFormer temporal memory + occupancy flow
```

논문을 읽을 때는 다음만 먼저 본다.

```text
입력
출력
feature shape
loss
핵심 figure
내 코드에서 대응되는 module
```

---

## 17. 6개월 후 성공 기준

필수 성공 기준:

```text
synthetic dataset에서 end-to-end 학습 가능
one-batch overfit 가능
occupancy / surface / flow visualization 가능
active mask + quota Top-K 동작
packed Queryable MLP 동작
runtime profiling 가능
```

성공이면 좋은 기준:

```text
small config에서 20 FPS에 근접
ONNX export 일부 성공
real/sim dataset 연결 준비
```

가장 중요한 원칙:

```text
작은 모델을 끝까지 동작시킨다.
그 다음 한 축씩 실제 architecture에 가까워진다.
```
