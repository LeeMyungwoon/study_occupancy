# Daily 6-Month Implementation Plan for `occupancy_network_architecture.md`

이 문서는 하루 1시간, 평일 5일 기준 26주, 총 130일(+ 보강일 D081a / D101a / D101b 3일 = 133일) 동안 `occupancy_network_architecture.md`의 구조를 실제 구현 가능한 단계로 쌓아 올리는 계획이다. 보강일은 아키텍처의 4 volume head와 pre-temporal refinement를 빠짐없이 덮기 위해 끼워 넣은 짧은 day다.

주말은 공식 커리큘럼에 넣지 않는다. 주말은 밀린 날 보충, 시각화 개선, 논문 보충 읽기, 코드 정리에 사용한다.

---

## 0. 최종 목표

6개월 후 목표는 아래 v1 prototype이다.

```text
multi-camera image
-> image backbone + FPN/BiFPN-lite
-> 1.2m 3D query vanilla cross-attention
-> dense deconv: 1.2m -> 0.6m
-> 0.6m pre-temporal refinement (BEVDet4D 교훈: temporal 전 정리)
-> Tesla/PanoOcc-style 0.6m 3D temporal (z 유지, align+concat+3D residual conv, NOT attention)
-> coarse dense occupancy/visibility/mixed_surface_risk @ 0.6m (free/unknown + 프루닝 guide)
-> sparse deconv + recall-first 2단 게이트 prune
   (0.6m parent gate + 0.3m child score/prune + near-surface free shell tag)
-> 30cm sparse candidate feature F_0.3_sparse = prelim_kept
   (= child_score_keep_candidates ⊎ prelim_free_shell)
-> O_occ_fine on [N_kept] -> 6-way pred label + pred_heavy_mask
-> Tesla-style 4 volume heads
   (Occupancy / Occupancy Flow / Sub-Voxel Shape / 3D Semantics)
-> Surface Outputs head (z-flatten, geometry + geometry-derived risk)
-> exposed-face mask on pred_heavy_mask
-> packed Queryable interface
   (surface_shell_prob + planner collision_state, semantic query 없음)
```

20 FPS v1 원칙:

```text
금지:
  dense 0.3m feature volume (272k voxel)
  dense 0.3m deconvolution
  z-squeeze BEV temporal / temporal attention (Tesla/PanoOcc와 다름)
  full-res(0.3m) / full global temporal memory
  P2/P3 local image re-query in v1

필수:
  1.2m -> 0.6m까지만 dense feature 유지
  0.6m 3D temporal (z 유지, align+concat+3D residual conv)
  0.6m -> 0.3m은 sparse deconv + recall-first 2단 게이트
  high-confidence free만 drop, unknown/uncertain/mixed-surface-risk는 keep 또는 ignore
  keep ~0.5는 soft budget target일 뿐 mandatory recall keep이 우선
  prelim_kept와 최종 pred_* label을 분리
  Queryable MLP는 pred_heavy_mask branch(A)에서 surface_shell_prob만 예측
  planner collision은 surface_shell_prob 단독이 아니라 collision_state fallback으로 결정
  auxiliary depth/free-space evidence head는 v1 inference-critical
  K_total / Q_total hard cap
  packed batched Queryable MLP
  one-batch overfit
```

참고:

```text
architecture 문서의 최종 30cm 표현은
sparse deconv + 2단 게이트 프루닝(0.6m parent gate + 0.3m child score/prune)으로
F_0.3_sparse = prelim_kept를 만든다.
prelim_kept는 최종 label이 아니라 O_occ_fine을 실행할 sparse index set이며,
child_score_keep_candidates ⊎ prelim_free_shell로 구성된다.
near_surface_free_shell_tag는 prelim_free_shell index source가 아니라
최종 pred_free_shell / pred_free_kept를 가르는 semantic tag다.
(dense 0.3m feature volume은 만들지 않는다.)

이 plan의 Phase 04는 학습 초기엔 dense 프로토타입으로 0.3m occupancy를 만들고,
Phase 08에서 sparse deconv + 2단 게이트 prune으로 전환한다
(architecture 문서 Section 16 Stage 1의 권장 순서).

O_occ_fine 이후 kept voxel은
pred_occupied_surface / pred_boundary / pred_visibility_frontier /
pred_free_shell / pred_free_kept / pred_uncertain_kept로 완전 분할된다.
Flow / Sub-Voxel Shape / 3D Semantics는 pred_heavy_mask
(pred_occupied_surface | pred_boundary)의 [N_heavy]에서만 실행한다.

Queryable은 Tesla 2-MLP 중 occupancy MLP만 둔다.
단 이 MLP 출력은 planner collision probability가 아니라 surface_shell_prob이고,
planner는 collision_state fallback 규칙을 함께 사용한다.
semantic은 연속 query를 두지 않고 voxel-level 3D Semantics Head에서만 제공한다.
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

    phase_04_deconv_occupancy_head/
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

    phase_08_sparse_prune_queryable/
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
mkdir -p phases/phase_{01_pytorch_primitives,02_toy_occupancy,03_attention_lifting,04_deconv_occupancy_head,05_multicamera_single_frame,06_surface_geometry,07_temporal_flow,08_sparse_prune_queryable,09_integration_evaluation,10_runtime_final_report}/{src,tests,scripts}
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
  1.2m -> 0.6m -> 0.3m deconv, 0.3m occupancy head (dense 프로토타입)

Phase 05, D051-D065:
  multi-camera token packing, canonical ray PE, single-frame occupancy training,
  auxiliary depth/free-space evidence head

Phase 06, D066-D080:
  Tesla Surface Outputs-style dense BEV head
  (z_surface / slope / step / uncertainty / traversability_cost / drop_risk)

Phase 07, D081-D095 (+D081a):
  0.6m pre-temporal refinement(D081a),
  Tesla/PanoOcc-style 0.6m 3D temporal (align+concat+3D conv),
  Occupancy Flow Head 내부 O_motion_0.6 coarse seed + [N_heavy] readout 준비

Phase 08, D096-D110 (+D101a, D101b):
  sparse deconv + recall-first 2단 게이트 prune,
  prelim_kept / near_surface_free_shell_tag / 6-way pred label 분리,
  pred_heavy_mask, exposed-face mask,
  Sub-Voxel Shape Head(D101a), 3D Semantics Head(D101b),
  quota keep, sparse anchor, packed Queryable interface(surface_shell_prob + collision_state)

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

## 7. Phase 04: Deconv and 0.3m Occupancy Head (dense 프로토타입)

기간: D036-D050

목표:

```text
1.2m feature를 0.6m -> 0.3m까지 deconv하고, 0.3m occupancy를 dense 프로토타입으로 만든다.
(실제 크기에선 0.3m dense가 금지이므로, Phase 08에서 sparse deconv + 2단 게이트 prune으로
 전환한다. toy 단계에서는 학습 흐름 이해를 위해 dense 프로토타입을 먼저 만든다.)
```

### D036: 1.2m / 0.6m / 0.3m grid 계산

만들 파일:

```text
phases/phase_04_deconv_occupancy_head/src/grid_config.py
phases/phase_04_deconv_occupancy_head/tests/test_grid_config.py
```

구현:

```python
def compute_logical_shapes(range_xyz, resolutions): ...
def compute_padded_shapes(coarse_shape, num_deconv): ...
```

검증:

```text
range: X 60m(rear10~front50), Y 20.4m(±10.2, 격자=타깃), Z 6m(-2~+4)
1.2m: 50 x 17 x 5   = 4,250
0.6m: 100 x 34 x 10 = 34,000
0.3m: 200 x 68 x 20 = 272,000
(X·Y·Z 전부 격자=타깃, padding 없음)
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
왜 0.3m dense feature까지 만들지 않는지 (272k voxel -> sparse deconv + prune으로 대체)
coarse-to-fine + occupancy 프루닝(keep ratio)의 의미
```

### D038: TwoStageVoxelDecoder 구현

만들 파일:

```text
phases/phase_04_deconv_occupancy_head/src/voxel_decoder.py
phases/phase_04_deconv_occupancy_head/tests/test_voxel_decoder.py
```

구현:

```python
class TwoStageVoxelDecoder(nn.Module):
    # 50 x 17 x 5 -> 100 x 34 x 10 -> 200 x 68 x 20
    # (toy 단계는 dense 프로토타입. Phase 08에서 2단계째를 sparse deconv + prune으로 교체)
```

### D039: 좌표 규약 / grid=target 검증 (padding/crop 불필요)

수정 파일:

```text
phases/phase_04_deconv_occupancy_head/src/grid_config.py
```

작업:

```text
architecture Section 1.1 (Coordinate Conventions) 규약을 코드로 고정:
  origin (-10, -10.2, -2)m, half-open cell, center=origin+(idx+0.5)*s,
  point->index=floor((x-origin)/s), parent-child ratio 정확히 2.
전 축 격자=타깃이라 padding/valid crop 없음.
```

검증:

```text
X 60m=200, Y 20.4m=68, Z 6m=20 정수 떨어짐 assert.
voxel center / point->index round-trip 일치.
1.2m->0.6m->0.3m index mapping ratio 2 정확.
```

### D040: Occupancy Head (0.3m dense 프로토타입)

만들 파일:

```text
phases/phase_04_deconv_occupancy_head/src/occupancy_heads.py
phases/phase_04_deconv_occupancy_head/tests/test_occupancy_head.py
```

구현:

```python
class OccupancyHead(nn.Module):
    # 갈래 A: coarse dense @ 0.6m
    #   -> occupied/free/unknown logits + visibility_logit + mixed_surface_risk_logit
    # 갈래 B: fine @ 0.3m
    #   -> occupied logit + surface-band auxiliary logit
    # toy 단계는 dense 0.3m로 프로토타입 (Phase 08에서 sparse prelim_kept 위로 전환)
```

### D041: 2-갈래 occupancy 정확성 test

수정 파일:

```text
phases/phase_04_deconv_occupancy_head/tests/test_occupancy_head.py
```

검증:

```text
갈래 A: 0.6m dense에서 occupied/free/unknown/visibility/mixed_surface_risk logit shape
갈래 B: 0.3m에서 occupied logit + surface-band auxiliary logit shape
free/unknown은 0.6m, occupied/surface-band 후보는 0.3m이 책임진다는 분업을 synthetic으로 확인
```

### D042: occupancy head loss 연결

만들 파일:

```text
phases/phase_04_deconv_occupancy_head/src/losses.py
```

구현:

```python
def coarse_occupancy_loss(cls_logits, visibility_logit, mixed_surface_risk_logit, target): ...
def fine_occupancy_loss(occ_logit, surface_band_logit, target, valid_mask): ...
```

### D043: deconv + occupancy head end-to-end shape

만들 파일:

```text
phases/phase_04_deconv_occupancy_head/tests/test_end_to_end_shape.py
```

검증:

```text
1.2m feature
-> 0.6m feature
-> 0.3m valid feature
-> occupancy logits (갈래 A 0.6m + 갈래 B 0.3m)
```

### D044: tiny training with decoder

만들 파일:

```text
phases/phase_04_deconv_occupancy_head/scripts/train_decoder_toy.py
```

검증:

```text
one-batch overfit
```

### D045: 20 FPS guardrail test

수정 파일:

```text
phases/phase_04_deconv_occupancy_head/tests/test_occupancy_head.py
```

검증:

```text
toy 프로토타입임을 명시 (실제 크기 0.3m dense는 금지).
0.6m까지만 dense, 0.3m dense feature volume은 실제 크기에서 만들지 않음을 notes.md에 기록.
Phase 08에서 sparse deconv + 2단 게이트 prune으로 전환할 지점을 표시.
```

### D046: simple profiler

만들 파일:

```text
phases/phase_04_deconv_occupancy_head/scripts/profile_decoder.py
```

측정:

```text
decoder latency (1.2m->0.6m->0.3m)
occupancy head latency
peak memory (0.3m dense 프로토타입의 비용 -> sparse 전환 동기 확인)
```

### D047: decoder channel ablation

할 일:

```text
C=16, 32, 64 latency와 memory 비교
notes.md에 기록
```

### D048: 0.6m attention vs 1.2m attention query count 계산

만들 파일:

```text
phases/phase_04_deconv_occupancy_head/scripts/compute_query_counts.py
```

검증:

```text
1.2m query 수(4,250)와 0.6m query 수(34,000) 비교 출력
-> 왜 attention을 1.2m에서만 하는지 확인
```

### D049: architecture 문서와 구현 비교

할 일:

```text
occupancy_network_architecture.md의 Section 8(deconv/2단 게이트), 9(4 volume heads)와 현재 코드 비교
dense 0.3m 프로토타입과 실제 sparse deconv+prune의 차이를 notes.md에 기록
0.6m mixed_surface_risk가 Phase 08 parent gate recall 보호에 쓰인다는 점을 기록
fine surface-band auxiliary가 pred_boundary / pred_heavy_mask의 학습 발판임을 기록
```

### D050: Phase 04 report

검증:

```bash
pytest phases/phase_04_deconv_occupancy_head/tests -q
```

기록:

```text
0.3m dense 프로토타입과 (실제) sparse deconv+prune의 차이
free/unknown은 0.6m coarse, occupied/surface-band 후보는 0.3m sparse라는 2-해상도 hybrid
mixed_surface_risk / surface-band auxiliary를 왜 따로 두는지
```

---

## 8. Phase 05: Multi-camera Single-frame Occupancy

기간: D051-D065

목표:

```text
multi-camera token packing과 1.2m cross-attention lifting을 single-frame occupancy training에 연결한다.
또한 v1 추론에서 pred_free_shell free 즉답에 필요한
auxiliary depth/free-space evidence head의 최소 형태를 만든다.
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
coarse_shape: [50, 17, 5]      # 1.2m
mid_shape: [100, 34, 10]       # 0.6m (dense, temporal/coarse occ)
fine_shape: [200, 68, 20]      # 0.3m (toy 프로토타입은 dense, 실제는 sparse)
# 전 축 격자=타깃, padding 없음 (Y=±10.2m=20.4m=68칸)
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
target: B x 1 x 200 x 68 x 20 (0.3m) 또는 tiny shape
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
    # 1.2m 3D query (50 x 17 x 5)
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
    lifter (1.2m)
    decoder (1.2m -> 0.6m -> 0.3m)
    occupancy head (0.6m coarse dense + 0.3m fine; toy는 dense 프로토타입)
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

설계 주의:

```text
interface는 architecture 문서 Section 19의 GT 출력을
받을 수 있는 키 구조로 정의한다.

images / intrinsics / pose
occupancy (occupied / free / unknown)
visibility / observed_free_evidence targets
z_surface / surface_valid / traversability_cost / drop_risk
dynamic mask / flow
continuous SDF / surface-shell samples (optional, Sub-Voxel Shape / Queryable 학습용)
dense or sparse metric depth / depth confidence
free-space ray evidence
```

### D064: Auxiliary Depth / Free-space Evidence Head (v1 inference-critical)

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
synthetic depth GT에서 L_depth / L_depth_conf / L_free_space_evidence가 감소.
hit 뒤쪽 / occluded / 미관측 구간을 free negative로 쓰지 않는지 확인.
pred_free_shell free 즉답은 child-level observed-free evidence 없이는 금지됨을 unit test로 확인.
```

### D065: Phase 05 report

검증:

```bash
pytest phases/phase_05_multicamera_single_frame/tests -q
```

기록:

```text
multi-camera token shape
1.2m query count (4,250)
single-frame occupancy 결과
auxiliary depth/free-space evidence head의 출력 shape와 보수적 free 판정 규칙
```

---

## 9. Phase 06: Surface Outputs Head

기간: D066-D080

목표:

```text
Tesla Surface Outputs처럼 Volume Outputs와 분리된 dense BEV surface head를 만든다.
Road semantics(차선/주행구역)는 두지 않지만,
z_surface / slope / step / valid / uncertainty / traversability_cost / drop_risk를 예측한다.
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
occupancy volume과 별도 Surface Outputs head의 차이
road semantic은 제외하지만 geometry-derived risk는 포함하는 이유
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
def make_step_height_target(z_surface): ...
def make_traversability_cost_target(z_surface, slope, step): ...
def make_drop_risk_target(z_surface, valid): ...
```

### D068: z-flatten (높이를 채널로 보존, ZPool 금지)

만들 파일:

```text
phases/phase_06_surface_geometry/src/surface_head.py
phases/phase_06_surface_geometry/tests/test_surface_head.py
```

구현:

```python
class ZFlatten(nn.Module):
    # Surface Outputs Head는 0.6m dense feature를 입력으로 받는 별도 branch다.
    # 0.3m sparse volume head의 후처리가 아니다.
    # B x C x X x Y x Z -> B x (Z*C) x X x Y -> 1x1 conv -> B x Cb x X x Y
    # 주의: ZPool(평균/최대로 z를 누름) 금지.
    #   바닥 높이(z_surface)를 예측하는 head가 입력에서 높이 단서를 먼저 버리면 안 됨 (L3).
```

검증:

```text
B x C x X x Y x Z -> B x Cb x X x Y (단, 내부적으로 z를 채널로 펼쳐 정보 보존)
ZPool 대비 z 정보가 head 입력까지 전달되는지 synthetic으로 확인
```

### D069: SurfaceOutputsHead

수정 파일:

```text
phases/phase_06_surface_geometry/src/surface_head.py
```

구현:

```python
class SurfaceOutputsHead(nn.Module):
    # z_surface, valid_logit, uncertainty
    # slope_or_normal, step_height, traversability_cost, drop_risk
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
def surface_slope_or_normal_loss(...): ...
def step_height_loss(...): ...
def traversability_cost_loss(...): ...
def drop_risk_loss(...): ...
```

### D071: surface one-batch training

만들 파일:

```text
phases/phase_06_surface_geometry/scripts/train_surface_one_batch.py
```

검증:

```text
z_surface / valid / traversability_cost / drop_risk loss 감소
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
def compute_normal_from_z(z_surface): ...
def compute_step_score_from_z(z_surface): ...
def compute_geometry_risk_from_surface(z_surface, valid, slope, step): ...
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
phase_05 SingleFrameOccNet 구조 + 0.6m dense feature 기반 Surface Outputs Head
주의: Surface Outputs는 0.3m sparse volume head의 후처리가 아니라 별도 dense BEV branch.
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
def traversability_error(cost_pred, cost_gt, valid): ...
def drop_risk_auc(drop_logit, drop_gt, valid): ...
```

### D080: Phase 06 report

기록:

```text
surface head가 occupancy와 다른 이유
z_surface / valid / uncertainty / traversability_cost / drop_risk 결과
Surface Outputs는 0.3m sparse volume head가 아니라 별도 dense BEV branch임을 기록
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
Tesla/PanoOcc-style 0.6m 3D temporal(z 유지, align+concat+3D residual conv, NOT attention)을
구현하고, Occupancy Flow Head의 0.6m coarse motion seed(O_motion_0.6)를 만든다.
최종 flow는 Phase 08에서 O_occ_fine 이후 확정되는 pred_heavy_mask 위의 [N_heavy] readout/refine으로 낸다.
(ViewFormer z-squeeze BEV temporal은 architecture에서 폐기됨 -> 사용하지 않는다.)
```

### D081: PanoOcc 발췌 읽기 (temporal encoder)

읽을 것:

```text
PanoOcc temporal encoder (temporal align + temporal fuse)
ego-motion 3D alignment
Tesla AI Day Temporal Alignment 그림 (Spatial Frame Alignment -> Spatiotemporal Features stack)
```

기록:

```text
왜 BEV z-squeeze가 아니라 3D(z 유지)로 정렬+concat 하는가?
(높이별 motion / vz를 살리기 위해. attention이 아니라 align+concat+conv)
```

### D081a: 0.6m Pre-temporal Feature Refinement (architecture Section 6)

만들 파일:

```text
phases/phase_07_temporal_flow/src/pre_temporal.py
phases/phase_07_temporal_flow/tests/test_pre_temporal.py
```

구현:

```python
class PreTemporalRefine3D(nn.Module):
    # dense deconv(1.2m->0.6m) 직후, temporal 직전에 현재 frame 내부만 한 번 정리
    # light 3D conv block (depthwise 3x3x3 + pointwise 1x1x1 + residual/gating)
    # input/output: B x C x 100 x 34 x 10 (shape 유지, z 유지)
```

왜 필요한가:

```text
BEVDet4D 교훈: view transformer(1.2m attention) 직후 feature는 너무 coarse해서
temporal cue를 바로 쓰면 velocity error가 오른다(+11.9%).
temporal 전에 작은 encoder로 한 번 정리한다 (큰 모듈/추가 camera attention 아님).
위치: deconv(1.2m->0.6m) 직후, D082~D088 temporal 직전.
```

검증:

```text
input/output shape 동일 (B x C x 100 x 34 x 10, z 유지)
residual이라 초기엔 거의 identity에 가깝다.
backward 통과, NaN 없음.
이후 TemporalEnhancer(D088)는 F_0.6_refined를 입력으로 받는다.
```

### D082: 3D voxel memory align (z 유지, no z-squeeze)

만들 파일:

```text
phases/phase_07_temporal_flow/src/temporal_memory.py
phases/phase_07_temporal_flow/tests/test_temporal_memory.py
```

구현:

```python
# ZPool/z-squeeze 없음. 0.6m 3D feature(z 유지)를 그대로 다룬다.
def stack_history_3d(current, aligned_history): ...   # 채널 방향 concat 준비
```

검증:

```text
B x C x X x Y x Z 가 z를 유지한 채 다뤄지는지 확인 (BEV로 누르지 않음)
```

### D083: 3D ego-motion warp

수정 파일:

```text
phases/phase_07_temporal_flow/src/temporal_memory.py
```

구현:

```python
def warp_3d_feature(feat_3d, pose_delta): ...   # grid_sample 3D, ego-motion 정렬
```

검증:

```text
identity pose면 output 동일
translation pose면 3D feature가 이동
```

### D084: 3D memory queue

구현:

```python
class VoxelMemoryQueue:
    push(feature_3d, pose)
    get_aligned(current_pose)
```

검증:

```text
N_history=3 (현재 포함 4 frame, PanoOcc와 동일) 유지
```

설계 주의:

```text
architecture 문서 기준 memory는 0.6m 3D voxel(z 유지)로 저장/warp하고,
keyframe은 ~0.2s 시간 간격으로 띄엄띄엄 저장한다(0.6~0.8s 창).
queue 인터페이스에 (저장 해상도 0.6m, keyframe 간격, 1.2m fallback) 설정을 열어둔다.
```

### D085: temporal fusion (concat + 3D residual conv)

구현:

```python
class TemporalFusion3D(nn.Module):
    # X = concat(current_3d, aligned_history_3d)   # 채널 stack
    # out = current + Residual3DConv(X)            # attention 아님
```

목표:

```text
Tesla/PanoOcc식 align + concat + 3D residual conv로 융합한다.
(temporal attention / z-squeeze 금지)
```

### D086: streaming memory option (학습 효율)

수정:

```text
TemporalFusion3D에 ViewFormer streaming memory option 추가
(학습 시 과거 feature를 재계산하지 않고 캐시 -> 학습 효율, 추론 latency 변화 없음)
```

검증:

```text
output shape 유지, streaming on/off 결과 동일성 확인
```

### D087: 1.2m fallback 스위치 + 융합 마무리

구현:

```python
# 융합 해상도 스위치: 기본 0.6m 3D, Orin 예산 초과 시 1.2m 3D fallback
# (3D 유지는 동일, 해상도만 한 단계 내림)
```

검증:

```text
0.6m / 1.2m 두 해상도에서 fusion forward 동작
```

### D088: temporal model integration

만들 파일:

```text
phases/phase_07_temporal_flow/src/model_temporal.py
```

구현:

```text
0.6m 3D feature (z 유지)
-> 3D voxel memory align + concat + 3D residual conv
-> enhanced 0.6m 3D feature
-> O_motion_0.6 coarse motion seed (vx/vy/vz + dynamic seed)
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

### D090: Occupancy Flow Head coarse seed + readout skeleton

만들 파일:

```text
phases/phase_07_temporal_flow/src/flow_head.py
phases/phase_07_temporal_flow/tests/test_flow_head.py
```

구현:

```python
class OccupancyFlowHead(nn.Module):
    # branch A: F_0.6_temporal -> O_motion_0.6 dense coarse seed
    #   dynamic_seed_logit + coarse vx, vy, vz
    # branch B skeleton: pred_heavy_mask voxel에 seed + F_0.3_sparse를 gather해서
    #   final [N_heavy] dynamic_logit + vx, vy, vz readout/refine
    # Phase 07에서는 branch A와 gather/readout interface까지만 toy로 만든다.
    # 실제 pred_heavy_mask는 Phase 08 O_occ_fine 이후 연결한다.
```

### D091: flow loss

만들 파일:

```text
phases/phase_07_temporal_flow/src/flow_losses.py
```

구현:

```python
def dynamic_bce_loss(...): ...
def flow_coarse_seed_loss(o_motion_06, flow_gt_06, valid_flow_seed_mask): ...
def flow_fine_readout_loss(flow_pred_heavy, flow_gt_heavy, valid_flow_mask): ...
```

검증:

```text
L_flow_coarse_seed는 0.6m dense seed에 적용.
L_flow_fine_readout은 [N_heavy] readout에만 적용.
valid_flow_mask 밖(정적/저신뢰/occluded)은 ignore.
```

### D092: flow one-batch overfit

만들 파일:

```text
phases/phase_07_temporal_flow/scripts/train_flow_one_batch.py
```

검증:

```text
moving cube flow 방향 학습
0.6m O_motion_0.6 seed가 먼저 맞고, toy pred_heavy_mask gather/readout이 shape를 유지
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
3D concat + residual conv (채택)
+ streaming memory option
```

기록:

```text
loss / IoU / flow error 비교 (특히 vz가 살아나는지)
O_motion_0.6 coarse seed loss와 [N_heavy] readout loss를 분리 기록
```

### D095: Phase 07 report

검증:

```bash
pytest phases/phase_07_temporal_flow/tests -q
```

기록:

```text
3D voxel memory shape (z 유지)
O_motion_0.6 coarse seed shape
Occupancy Flow Head가 독립 5번째 head가 아니라 4 volume head 중 Flow Head의 내부 branch임
[N_heavy] final flow는 Phase 08 pred_heavy_mask 이후에만 의미 있음
가장 어려운 점
```

---

## 11. Phase 08: Sparse Deconv + 2단 게이트 Prune + Queryable

기간: D096-D110

목표:

```text
0.6m -> 0.3m을 sparse deconv + recall-first 2단 게이트로 만든다.

핵심 불변식:
  gate① parent: high-confidence free parent만 drop.
    occupied / mixed_surface_risk / unknown / uncertain / dynamic / planner-corridor는 keep 또는 ignore.
  gate② child: cheap child score head가 keep/free/surface/shell-tag score를 낸다.
  prelim_kept = child_score_keep_candidates ⊎ prelim_free_shell.
    이것은 최종 label이 아니라 O_occ_fine을 실행할 sparse index set이다.
  near_surface_free_shell_tag = prelim_free_shell_raw ∩ prelim_kept.
    이것은 최종 pred_free_shell semantic을 정하는 tag다.
  O_occ_fine 이후 6-way pred label로 완전 분할한다.
  pred_heavy_mask = pred_occupied_surface | pred_boundary.
    Flow / Sub-Voxel Shape / 3D Semantics / Queryable MLP branch A는 여기서만 실행한다.

Phase 04의 dense 0.3m 프로토타입을 이 sparse 경로로 교체한다.
```

### D096: sparse deconv + gate① (0.6m parent gate, 연산량)

만들 파일:

```text
phases/phase_08_sparse_prune_queryable/src/sparse_decoder.py
phases/phase_08_sparse_prune_queryable/tests/test_sparse_decoder.py
```

구현:

```python
def high_conf_free_parent(coarse_cls_logits, visibility_logit, mixed_surface_risk_logit, thresholds): ...
def parent_gate_06m(coarse_cls_logits, visibility_logit, mixed_surface_risk_logit, priority_context): ...
    # high-confidence free parent만 drop
    # occupied / mixed_surface_risk / unknown / uncertain parent는 keep 또는 ignore
    # 이 시점에는 O_occ_fine 전이라 pred_boundary가 아직 없음
class SparseVoxelDeconv(nn.Module): ...
    # keep된 parent만 0.3m child 후보로 sparse 업샘플
```

검증:

```text
clean observed-free parent의 child만 생성하지 않는지 확인.
unknown / low-visibility / mixed_surface_risk parent가 free로 잘리지 않는지 확인.
parent gate가 dense 0.6m feature 자체를 자르는 것이 아니라 sparse fine 가지에만 적용되는지 확인.
```

### D097: gate② child score + prelim_kept / shell-tag 분리

수정 파일:

```text
phases/phase_08_sparse_prune_queryable/src/sparse_decoder.py
```

구현:

```python
class CheapChildScoreHead(nn.Module):
    # input: sparse 0.3m child candidate features
    # output: keep_logit, free_logit, surface_logit, shell_tag_logit

def select_child_score_keep_candidates(child_scores, mandatory_keep, quotas, soft_budget): ...
def select_prelim_free_shell_raw(child_scores, near_surface_evidence, observed_free_evidence_03): ...
def make_prelim_kept(child_score_keep_candidates, prelim_free_shell_raw): ...
    prelim_free_shell = prelim_free_shell_raw - child_score_keep_candidates
    prelim_kept = child_score_keep_candidates union_disjoint prelim_free_shell
    near_surface_free_shell_tag = prelim_free_shell_raw & prelim_kept
    return prelim_kept, prelim_free_shell, near_surface_free_shell_tag
```

검증:

```text
prelim_kept = child_score_keep_candidates ⊎ prelim_free_shell 인지 확인.
prelim_free_shell은 중복 제거된 extra index set이고, semantic source가 아님을 test로 고정.
child_score_keep_candidates 안에 이미 있던 near-surface free child도
near_surface_free_shell_tag가 유지되는지 확인.
keep ~0.5는 soft budget target이고 mandatory recall keep이 우선인지 확인.
```

### D098: prune keep + query priority score 통합

구현:

```python
def surface_band_score(surface_logit_or_cheap_surface_score): ...
def uncertainty_score(occ_logits): ...
def mixed_surface_risk_score(mixed_surface_risk_logit): ...
def shell_tag_score(shell_tag_logit): ...
def dynamic_score(dynamic_logits): ...
def near_field_score(voxel_centers): ...
def planner_corridor_score(voxel_centers, trajectory): ...
def thin_or_high_gradient_score(...): ...
def compute_keep_priority(...): ...   # gate①/② keep 우선순위 + 이후 query budget 공용
```

검증:

```text
surface-band / mixed-risk / uncertainty / dynamic / planner 영역 score가 높음.
far-field / thin-object quota가 단순 top-k에 밀려 사라지지 않음.
priority score는 keep/drop을 돕는 cheap score이지 최종 pred label이 아님.
```

### D099: Pruning false-negative 방어 (v1 필수, architecture Section 8.2.1)

프루닝 false negative가 가장 위험(coarse가 얇은 물체 free 오판 시 0.3m 복구 불가, 비가역).
v1.5가 아니라 v1부터 7가지 전부 넣는다.

구현:

```python
# 1. recall-first gate (게이트① 임계값 recall 기준)
# 2. uncertainty keep (p~0.5/unknown/저신뢰 parent 무조건 keep)
def quota_keep(category_scores, quotas): ...   # 3. category quota
# 4. GT-guided warmup (초기 GT occupied parent 강제 keep)
# 5. soft / straight-through top-k (keep에 gradient)
# 6. gate② mandatory keep target
def gate2_keep_loss(keep_logit, keep_target, keep_weight): ...        # L_gate2_keep
# 7. gate② shell-tag supervision
def gate2_shell_tag_loss(shell_tag_logit, shell_tag_target, mask): ... # L_gate2_shell_tag
```

검증:

```text
prune_recall(GT occupied 생존율) 측정 - 핵심 지표
observed surface-band / near-surface observed free shell recall 측정
얇은 물체 toy에서 coarse/cheap score가 약해도 recall 유지
quota별 selected count / 카테고리별 최소 keep 보장
L_gate2_keep은 "살릴지", L_gate2_shell_tag는 "shell semantic tag인지"를 분리해 학습
중복 제거 확인
```

### D100: distance / planner LOD budget

구현:

```python
def make_lod_budget_policy(distance_bins, planner_weight, dynamic_weight, hard_caps): ...
def assign_query_budget_after_pred(pred_heavy_voxels, budget_policy, planner_score, dynamic_score): ...
```

검증:

```text
D100에서는 아직 pred_heavy_mask가 없으므로 budget policy만 독립 test.
D102 이후 pred_heavy_voxels에 적용해 M_i를 배정한다.
near/planner/dynamic은 M_i가 큼
far/static은 M_i가 작음
N_kept / K_total / Q_total hard cap 초과 시 낮은 priority부터 fine/query 실행을 줄이되,
제외된 voxel을 free로 간주하지 않고 coarse unknown/occupied-risk fallback으로 degrade
```

### D101: O_occ_fine 6-way pred label + exposed face mask

구현:

```python
class SparseFineOccupancyHead(nn.Module):
    # input: F_0.3_sparse on prelim_kept
    # output: occ_logit + surface_band_logit on [N_kept]

def make_six_way_pred_labels(occ_logit, surface_band_logit, visibility, near_surface_free_shell_tag): ...
    # pred_occupied_surface
    # pred_boundary
    # pred_visibility_frontier
    # pred_free_shell
    # pred_free_kept
    # pred_uncertain_kept

def compute_pred_heavy_mask(pred_labels): ...
    return pred_occupied_surface | pred_boundary

def compute_exposed_faces(pred_heavy_idx, pred_labels_6way, coarse_fallback): ...
    # neighbor가 pred_occupied_surface 또는 pred_boundary solid면 not exposed
    # neighbor가 pred_free_shell / pred_free_kept이면 exposed-to-free 후보
    # neighbor가 pred_visibility_frontier / pred_uncertain_kept이면 unknown/occlusion 후보
    # neighbor가 pruned이면 부모 0.6m coarse/visibility fallback으로 판정
```

검증:

```text
prelim_kept 전체가 6-way pred label 중 정확히 하나로 분할되는지 확인.
pred_heavy_mask = pred_occupied_surface | pred_boundary인지 확인.
near_surface_free_shell_tag가 있어도 O_occ_fine이 occupied로 보면 pred_occupied_surface가 되는지 확인.
child_score_keep_candidates에서 온 voxel이라도 fine이 free이고 shell tag가 있으면 pred_free_shell인지 확인.
exposed-face 계산이 단순 kept 기준이 아니라 6-way pred label 기준인지 확인.
```

### D101a: Sub-Voxel Shape Information Head (architecture Section 9.3)

만들 파일:

```text
phases/phase_08_sparse_prune_queryable/src/subvoxel_shape_head.py
phases/phase_08_sparse_prune_queryable/tests/test_subvoxel_shape_head.py
```

구현:

```python
class SubVoxelShapeHead(nn.Module):
    # input:  F_0.3_sparse[pred_heavy_mask] + exposed_face_mask(D101)
    # output (per pred_heavy voxel, [N_heavy]):
    #   exposed_face_logits: 6
    #   face_surface_offset: 6   (각 exposed face -> voxel 내부 surface까지 normalized [0,1])
    #   local_normal: 3
    #   shape_uncertainty: 1
    #   thinness_logit: 1
    #   shape_code: Cs (예: 8 or 16)   # Queryable MLP / anchor가 소비
```

face-aware execution (head 내부 동작):

```text
zero exposed face(내부) -> heavy local query 생략
1 face -> 그 normal 방향만 offset/boundary query
2 face -> edge, 3+ face -> corner/thin 후보(budget 증가)
모든 voxel을 같은 비용으로 query하지 않는다 (priority mask 사용).
```

검증:

```text
출력이 pred_heavy_mask(occupied surface + boundary) 위에서만 계산된다.
pred_free_shell / pred_free_kept / pred_visibility_frontier / pred_uncertain_kept / pruned는 제외.
shape_code가 D104 anchor / D105 Queryable branch A surface-shell MLP 입력으로 전달된다.
offset이 [0,1](또는 [-0.5,0.5]) 범위, normal이 단위 vector에 가깝다.
exposed face 없는 interior voxel은 query budget이 0에 가깝다.
```

기록:

```text
notes.md에 "Sub-Voxel Shape는 후처리가 아니라 4 volume head 중 하나"임을 적는다.
exposed-face mask(D101)는 이 head의 '실행 방식'이지 별도 head가 아니다.
```

### D101b: 3D Semantics Head (architecture Section 9.4)

만들 파일:

```text
phases/phase_08_sparse_prune_queryable/src/semantics_head.py
phases/phase_08_sparse_prune_queryable/tests/test_semantics_head.py
```

구현:

```python
class SemanticsHead(nn.Module):
    # input:  F_0.3_sparse[pred_heavy_mask]
    # output: [N_heavy] x N_class   (sparse, pred_heavy voxel별 semantic logit)
```

권장 class (실내+실외 비도로):

```text
ground/floor, wall/building, vehicle, pedestrian, cyclist,
curb/barrier, vegetation, other static
(free/unknown은 Occupancy Head 담당 -> semantics에서 제외/ignore)
```

검증:

```text
semantic은 pred_heavy_mask 위에서 실행하고, loss는 valid_semantic_mask에만 적용.
pred_boundary 전체에 강한 semantic label을 주지 않는다.
pred_free_shell / pred_free_kept / pred_visibility_frontier / pred_uncertain_kept 영역에 semantic을 강제하지 않는다.
출력 shape [N_heavy] x N_class, NaN 없음.
Queryable은 surface-shell occupancy용 branch A만 두므로 semantic 질의는 이 voxel-level head에서만 제공(N7-B).
```

기록:

```text
notes.md에 "Queryable MLP에는 semantic을 넣지 않고 이 head에서만 제공"(Tesla 2-MLP 중 occupancy만 채택)을 적는다.
```

### D102: adaptive query generation

만들 파일:

```text
phases/phase_08_sparse_prune_queryable/src/query_generation.py
```

구현:

```python
def make_initial_queries(pred_heavy_idx, exposed_faces, shape_uncertainty, budget): ...
    # Queryable MLP branch A(pred_heavy_mask)용 query만 생성
    # pred_free_shell / pred_free_kept는 observed-free fallback 분기라 MLP query를 만들지 않음
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
branch_id가 모두 A(pred_heavy_mask)인지 확인
```

### D104: SparseLocalAnchor

만들 파일:

```text
phases/phase_08_sparse_prune_queryable/src/sparse_anchor.py
```

구현:

```python
def gather_03m_features(feat_sparse, selected_pred_heavy_idx): ...
    # F_0.3_sparse는 prelim_kept의 sparse feature.
    # Queryable branch A에서는 pred_heavy voxel feature만 gather.
class SparseLocalAnchor(nn.Module): ...
```

### D105: Queryable Interface (surface_shell_prob + collision_state)

만들 파일:

```text
phases/phase_08_sparse_prune_queryable/src/queryable_mlp.py
phases/phase_08_sparse_prune_queryable/src/collision_state.py
```

구현:

```python
class QueryableSurfaceShellMLP(nn.Module):
    # branch A only: pred_heavy_mask voxel
    # input: F_query(F_0.3_sparse) + shape_code + exposed_face context + q_local + PE
    # output: surface_shell_prob / surface_shell_uncertainty
    # semantic query 없음. semantic은 voxel-level 3D Semantics Head에서만 제공.

def collision_state_from_fallback(query_point, pred_label, coarse_occ, visibility, observed_free_evidence_06, observed_free_evidence_03): ...
    # A: pred_heavy_mask -> MLP surface_shell_prob + coarse/visibility/solid fallback
    # B1: pred_free_shell -> child-level observed_free_evidence_03 + NOT solid_interior_veto 필요
    # B2: pred_free_kept  -> parent-level observed_free_evidence_06 + coarse_free_conf 필요
    # C: pred_visibility_frontier | pred_uncertain_kept -> coarse/visibility fallback
    # D: pruned -> coarse/visibility fallback
```

주의:

```text
surface_shell_prob를 collision probability로 직접 해석하지 않는다.
planner는 collision_state == observed_free일 때만 통과 가능으로 본다.
unknown / occluded / occupied-risk / uncertain은 conservative occupied 또는 high-cost다.
```

### D106: packed execution test

만들 파일:

```text
phases/phase_08_sparse_prune_queryable/tests/test_queryable_mlp.py
```

검증:

```text
K=32
variable M_i
Q=sum(M_i)
MLP output Q x 2 (surface_shell_prob, uncertainty)
branch A만 MLP를 실행하는지 확인
branch B(pred_free_shell/pred_free_kept)는 child/parent observed-free evidence가 있을 때만 free 즉답
branch C/D는 coarse/visibility fallback으로 빠지는지 확인
observed-free evidence가 없으면 free로 반환하지 않는지 확인
```

### D107: boundary search

수정 파일:

```text
phases/phase_08_sparse_prune_queryable/src/query_generation.py
```

구현:

```python
def find_sign_change_pairs(query_points, surface_shell_prob): ...
def refine_boundary_bisection(...): ...
```

주의:

```text
v1 Queryable output은 전역 dense 3D feature interpolation이 아니라 sparse feature 기반 piecewise-continuous surface-shell field다.
boundary search는 pred_heavy_mask branch A 안에서만 사용한다.
분기 B/C/D에는 MLP boundary search를 적용하지 않는다.
```

### D108: fine overlay representation

만들 파일:

```text
phases/phase_08_sparse_prune_queryable/src/fine_overlay.py
```

구현:

```python
class SparseSurfaceRepresentation:
    # prelim_kept index set
    # 6-way pred labels
    # pred_heavy_idx
    # near_surface_free_shell_tag
    # F_0.3_sparse feature
    # Sub-Voxel Shape code / offset / normal on [N_heavy]
    # Semantics on [N_heavy]
    # local boundary samples
```

설계 주의:

```text
이것은 architecture 문서의 최종 30cm sparse candidate 표현 위에 올라가는 runtime representation이다.
"prelim_kept -> O_occ_fine 6-way label -> pred_heavy branch A query / fallback branch B-C-D"
인터페이스로 추상화해, Queryable MLP / Sub-Voxel Shape Head / planner fallback이 공유하도록 한다.
(Gaussian refinement는 현재 architecture에 없다 -> 도입하지 않는다.)
```

### D109: Queryable / surface-shell toy training

만들 파일:

```text
phases/phase_08_sparse_prune_queryable/scripts/train_queryable_toy.py
```

검증:

```text
pred_heavy_mask branch A query에서 L_query_surface_shell / L_query_uncertainty 감소
observed-free local sample이 surface_shell_prob를 낮추는지 확인
solid interior / occluded / 미관측 query를 free negative로 강제하지 않는지 확인
```

### D110: Phase 08 report

검증:

```bash
pytest phases/phase_08_sparse_prune_queryable/tests -q
```

기록:

```text
gate① high-conf free parent drop 규칙
gate② prelim_kept / prelim_free_shell / near_surface_free_shell_tag 분리
6-way pred label과 pred_heavy_mask
L_gate2_keep / L_gate2_shell_tag 분리
exposed-face mask
Sub-Voxel Shape Head(D101a) 출력(offset/normal/shape_code/thinness)
3D Semantics Head(D101b) 출력([N_heavy])
K_total / Q_total
packed MLP latency
surface_shell_prob와 collision_state를 분리한 이유
이로써 4 volume head(Occupancy/Flow/Sub-Voxel Shape/3D Semantics)와 Queryable interface가 모두 구현됨
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
          "occ_coarse": ...,        # 0.6m occupied/free/unknown/visibility/mixed_surface_risk
          "aux_depth": ...,         # low-res depth/confidence/free-space evidence
          "o_motion_06": ...,       # Flow Head 내부 0.6m coarse motion seed
          "prelim_kept": ...,       # child_score_keep_candidates ⊎ prelim_free_shell
          "near_surface_free_shell_tag": ...,
          "occ_fine": ...,          # [N_kept] occ_logit + surface_band_logit
          "pred_labels_6way": ...,  # 6-way pred label
          "pred_heavy_mask": ...,   # pred_occupied_surface | pred_boundary
          "flow": ...,              # [N_heavy] vx,vy,vz + dynamic
          "subvoxel_shape": ...,    # [N_heavy] offset/normal/shape_code/thinness
          "semantics": ...,         # [N_heavy] voxel-level semantics
          "surface": ...,           # dense BEV Surface Outputs
          "queryable": ...,         # surface_shell_prob + collision_state interface
        }
# forward 내부 순서: ... -> dense deconv 1.2->0.6 -> pre-temporal refine(D081a)
#  -> 0.6m 3D temporal -> O_motion_0.6 + coarse occ/mixed risk
#  -> sparse deconv + gate①/② -> prelim_kept + near_surface_free_shell_tag
#  -> O_occ_fine -> 6-way pred label -> pred_heavy_mask
#  -> Flow/Shape/Semantics on [N_heavy] + Surface Outputs branch
#  -> Queryable branch A MLP + branch B/C/D collision_state fallback
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
prelim_kept와 pred_labels_6way의 개수 일치
Flow/Shape/Semantics 출력 첫 차원이 N_heavy인지 확인
Queryable MLP query가 pred_heavy_mask branch A에서만 생성되는지 확인
```

### D114: unified losses

만들 파일:

```text
final_occnet_v1/src/occnet_v1/losses.py
```

구현:

```python
L_total = (
    L_occ_coarse_cls
    + L_occ_coarse_visibility
    + L_coarse_mixed_surface_risk
    + L_gate2_keep
    + L_gate2_shell_tag
    + L_occ_fine
    + L_surface_band
    + L_heavy_recall
    + L_heavy_mask
    + L_flow_coarse_seed
    + L_flow_fine_readout
    + L_dynamic
    + L_shape_exposed_face
    + L_shape_offset
    + L_shape_normal
    + L_shape_uncertainty
    + L_query_surface_shell
    + L_query_uncertainty
    + L_depth
    + L_depth_conf
    + L_free_space_evidence
    + L_semantic
    + L_surface_valid
    + L_surface_z
    + L_surface_slope_or_normal
    + L_step_height
    + L_traversability_cost
    + L_drop_risk
    + L_surface_sparse_consistency
)
```

검증:

```text
각 loss의 정의역이 맞는지 확인:
  O_occ_fine / L_occ_fine은 [N_kept] 전체.
  Flow/Shape/Semantics loss는 [N_heavy]에서 실행하되 valid_*_mask로 supervise.
  Queryable loss는 branch A surface-shell query에만 적용.
  collision_state는 직접 CE로 학습하지 않고 fallback rule/unit test로 검증.
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
heavy mask recall
prune recall(GT occupied / surface-band / free-shell)
surface MAE
traversability / drop-risk metric
flow endpoint error
semantic accuracy(valid_semantic_mask)
gate selected count / N_kept / N_heavy
Queryable surface-shell calibration
collision_state conservative-free violation count
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
traversability / drop-risk map
flow BEV arrows
pred_heavy_mask / exposed-face mask map
6-way pred label map
Queryable branch A/B/C/D debug overlay
```

### D118: config ablation switches

수정 파일:

```text
final_occnet_v1/configs/tiny.yaml
```

추가:

```yaml
model:
  use_surface: true
  use_aux_depth: true
  use_temporal: true
  use_flow: true
  use_sparse_prune: true
  use_near_surface_free_shell: true
  use_gate2_keep_loss: true
  use_gate2_shell_tag_loss: true
  use_queryable_surface_shell: true
  use_collision_state_fallback: true
  use_semantic_head: true
  temporal_fallback_12m: false
runtime:
  k_total: 512
  q_total: 4096
  soft_keep_budget: 0.5
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
attention (1.2m)
pre-temporal refine (0.6m, D081a)
temporal (0.6m 3D)
dense deconv (1.2m->0.6m)
sparse deconv + 2단 게이트 prune (0.6m->0.3m)
occupancy head (coarse dense + sparse fine)
gate② child score + L_gate2_keep / L_gate2_shell_tag
6-way pred label + pred_heavy_mask generation
sub-voxel shape head (D101a)
3D semantics head (D101b)
Surface Outputs head (z-flatten + geometry-derived risk)
Occupancy Flow Head (O_motion_0.6 seed + [N_heavy] readout)
exposed-face mask
quota keep
auxiliary depth/free-space evidence head
packed Queryable branch A surface-shell MLP
collision_state fallback branch B/C/D
```

### D122: K/Q latency table

실험:

```text
K=512, 1024, 2048
average M_i=4, 8, 16
N_kept / N_heavy / Q_total을 함께 기록
branch A MLP latency와 branch B/C/D fallback latency를 분리
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
A1 + mixed_surface_risk + recall-first gate
A2 + auxiliary depth/free-space evidence
A3 + Surface Outputs head
```

### D125: ablation A4-A7

기록:

```text
phases/phase_10_runtime_final_report/reports/ablation_a4_a7.md
```

실험:

```text
A4 + temporal + O_motion_0.6
A5 + Occupancy Flow [N_heavy] readout
A6 + Sub-Voxel Shape / 3D Semantics on pred_heavy_mask
A7 + Queryable surface_shell_prob + collision_state fallback
```

### D126: 20 FPS blocker 분석

기록:

```text
가장 느린 top 3 stage
가장 memory 큰 top 3 tensor
N_kept / N_heavy / Q_total이 예산을 넘는 조건
fallback으로 품질을 낮춘 frame 수
20 FPS 달성 가능성
```

### D127: ONNX export 준비

만들 파일:

```text
final_occnet_v1/scripts/export_onnx.py
```

목표:

```text
전체가 안 되면 backbone + dense 부분(1.2m->0.6m + occupancy head) 일부라도 export
(sparse deconv는 TensorRT 네이티브 지원이 약하므로 dense 경로부터 export)
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
prelim_kept vs 6-way pred label 용어표
surface_shell_prob vs collision_state 용어표
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
N_kept / N_heavy / Q_total budget
prune recall / heavy recall
collision_state conservative-free violation
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
TensorRT FP16 + sparse conv 배포 최적화
  - sparse 3D conv(spconv/torchsparse)의 Orin 커스텀 커널/플러그인
  - 막히면 0.3m masked dense(ROI/frustum crop) fallback (architecture Section 8.4/13)
  - INT8 검토 (v1.5)
temporal v1.5 강화 (architecture Section 7)
  - 장기 누적 기억(Tesla Temporal Context, GRU/EMA식) 추가
  - 0.3m residual flow head (full 0.3m temporal은 계속 금지)
prune v1.5 강화 (architecture Section 8.2)
  - v1의 category quota / shell-tag supervision은 유지
  - per-class quota 비율 튜닝, hard-negative mining, active learning 보강
deformable image re-query (P3, 1 round) 실험 (선택)
flow-aware dynamic feature correction
(선택 연구) Gaussian refinement (GaussianFormer 계열)
  - 현재 architecture에는 없음. Phase 08 Queryable branch A surface-shell MLP와 ablation 비교용 연구 트랙
camera-only GT auto-labeling pipeline (architecture 문서 Section 19)
  - Stage A: MapAnything / MASt3R 기반 metric 재구성
    (vanilla DUSt3R 직접 사용 금지: scale 모호 / pair 단위 추론 /
     정적 가정 / 기지 calibration 미활용)
  - SAM2 + tracking 동적 분리, ray casting free/unknown
  - Stage B: 3DGS refinement -> 연속 SDF (sub-voxel supervision)
  - 검증: nuScenes에서 Occ3D LiDAR GT와 거리별 오차 정량 비교 후
    자체 데이터 적용
Basalt VIO 연동 (architecture 문서 Section 20)
  - v1: pose / 중력 정렬, landmark sparse depth supervision
  - v1.5: landmark 입력 주입 + dropout, outlier dynamic 힌트,
    landmark ray free-space 증거
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
prelim_kept와 pred_* label을 섞어 쓰지 않았는가?
Flow/Shape/Semantics가 [N_heavy]에서만 실행되는가?
free로 반환하는 모든 경로에 observed-free evidence가 있는가?
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
phases/phase_08_sparse_prune_queryable/src/sparse_decoder.py에
sparse deconv + 2단 게이트 prune을 만들어줘.
입력은 0.6m coarse occupancy/visibility/mixed_surface_risk logits와 0.6m feature야.
게이트①: high-confidence free parent만 0.3m child 생성을 생략해.
         unknown / uncertain / mixed_surface_risk parent는 free로 자르지 마.
게이트②: cheap child score head가 keep/free/surface/shell-tag score를 내고,
         child_score_keep_candidates와 prelim_free_shell_raw를 만든 뒤
         prelim_kept = child_score_keep_candidates ⊎ prelim_free_shell을 만들어.
         near_surface_free_shell_tag는 prelim_free_shell_raw ∩ prelim_kept로 유지해.
출력은 prelim_kept index + F_0.3_sparse feature + near_surface_free_shell_tag야.
dense 0.3m feature volume은 만들면 안 돼.
tests/test_sparse_decoder.py도 작성해줘.
```

---

## 16. 논문 읽기 순서 요약

```text
D023:
  REO Abstract + Method overview

D037:
  PanoOcc architecture + coarse-to-fine sparse decoder + occupancy 프루닝

D066:
  RoadBEV problem definition + road elevation output
  (Road semantics는 참고만, v1 Surface Outputs는 geometry-derived risk 중심)

D081:
  PanoOcc temporal encoder (align+concat+3D conv, z 유지)
  + Tesla AI Day Temporal Alignment 그림
  (ViewFormer는 streaming memory / flow supervision만 참고)
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
prelim_kept / 6-way pred label / pred_heavy_mask 동작
L_gate2_keep + L_gate2_shell_tag 동작
packed Queryable branch A surface-shell MLP와 collision_state fallback 동작
auxiliary depth/free-space evidence가 observed-free 판정에 연결됨
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
