# Phase 09 - Integration and Evaluation

기간: D111-D120  
목표: 각 phase에서 안정된 모듈을 `final_occnet_v1`로 모으고, end-to-end forward, loss, metric, visualization을 만든다.

이 phase는 "새로운 아이디어를 더 넣는 기간"이 아니라 "작동하는 v1을 만드는 기간"이다.  
여기서는 성능이 조금 부족해도 괜찮다. 대신 shape contract, config, loss, metric, visualization, ablation switch가 모두 일관되게 동작해야 한다.

v1 통합 원칙:

```text
1. free/unknown은 0.6m coarse dense, occupied 표면은 0.3m sparse deconv + 2단 게이트 prune
   (+ near-surface free shell, L2). 0.3m dense feature volume은 만들지 않는다.
2. QueryableMLP는 occupancy 전용. semantic은 voxel-level 3D Semantics Head에서만.
3. temporal은 0.6m 3D(z 유지, align+concat+3D residual conv, NOT attention)로 유지한다.
4. surface는 z-flatten 기반 z_surface / valid / uncertainty를 core output으로 둔다 (geometry only).
5. flow는 per-voxel(vx,vy,vz), dynamic/occupied/valid mask에서만 loss를 계산한다.
6. 모든 optional branch는 config로 켜고 끌 수 있어야 한다.
```

## 통합 대상 모듈

```text
phase_01:
  tensor shape helper
  Conv3d / ConvTranspose3d / grid_sample 이해 코드

phase_02:
  toy occupancy dataset
  occupancy loss / metric baseline

phase_03:
  vanilla cross attention lifting
  1.2m coarse 3D feature 생성

phase_04:
  1.2m -> 0.6m -> 0.3m deconv decoder
  2-갈래 occupancy head (0.6m coarse dense + 0.3m fine)

phase_05:
  multicamera canonical preprocessing
  calibration-free vanilla attention path

phase_06:
  SurfaceGeometryHead (z-flatten, geometry only)
  surface losses / metrics

phase_07:
  0.6m pre-temporal refinement (BEVDet4D 교훈)
  0.6m 3D temporal (align+concat+3D residual conv, z 유지)
  DynamicFlowHead (vx,vy,vz)

phase_08:
  sparse deconv + 2단 게이트 prune (+ near-surface free shell)
  exposed-face mask / quota keep
  SubVoxelShapeHead (offset/normal/shape_code/thinness)
  SemanticsHead (voxel-level 3D semantics)
  SparseLocalAnchor
  QueryableMLP (occupancy 전용)
  SparseSurfaceRepresentation
```

## D111 - final_occnet_v1 skeleton

만들 파일:

```text
final_occnet_v1/src/occnet_v1/__init__.py
final_occnet_v1/src/occnet_v1/model.py
final_occnet_v1/src/occnet_v1/config.py
final_occnet_v1/configs/tiny.yaml
final_occnet_v1/configs/v1_debug.yaml
```

작업:

```text
1. 각 phase에서 안정된 모듈만 final_occnet_v1로 복사한다.
2. import path를 final_occnet_v1 기준으로 정리한다.
3. phase별 실험용 code는 final로 무리하게 가져오지 않는다.
4. config dataclass 또는 yaml loader를 만든다.
```

권장 package 구조:

```text
final_occnet_v1/src/occnet_v1/
  __init__.py
  config.py
  model.py
  backbone.py
  lifting.py
  decoder.py            # dense 1.2m->0.6m + sparse 0.6m->0.3m (2단 게이트)
  heads.py
  temporal.py           # 0.6m 3D align+concat+conv
  surface.py            # z-flatten
  sparse_queryable.py   # sparse prune + queryable MLP (occupancy 전용)
  losses.py
  metrics.py
  visualization.py
```

`tiny.yaml` 최소 내용:

```yaml
model:
  voxel_size_final: 0.3
  voxel_size_feature: 0.6
  voxel_size_attention: 1.2
  x_range: [-10.0, 50.0]
  y_range: [-10.0, 10.0]
  z_range: [-2.0, 4.0]
  channels: 64
  use_surface: true
  use_temporal: true
  use_flow: true
  use_sparse_prune: true
  near_surface_free_shell: true   # L2
  temporal_fallback_12m: false    # Orin 예산 초과 시 true
runtime:
  k_total: 512
  q_total: 4096
  keep_ratio: 0.5
```

완료 기준:

```text
python -c "import occnet_v1"가 성공한다.
config를 load해서 model을 만들 수 있다.
tiny config는 빠른 debug용으로 작게 유지한다.
```

## D112 - OccNetV1 forward

수정 파일:

```text
final_occnet_v1/src/occnet_v1/model.py
```

구현:

```python
class OccNetV1(nn.Module):
    def forward(self, batch):
        return {
            "occ_coarse": ...,      # 0.6m free/unknown/occupied/visibility
            "occ_fine": ...,        # 0.3m sparse kept (occupied 표면)
            "surface": ...,         # z-flatten
            "dynamic": ...,
            "flow": ...,            # vx,vy,vz
            "subvoxel_shape": ...,  # offset/normal/shape_code/thinness (Sub-Voxel Shape Head)
            "semantics": ...,       # voxel-level 3D Semantics Head
            "queryable": ...,       # occupancy 전용
        }
```

forward 순서:

```text
1. images -> image backbone
2. image features -> vanilla cross attention lifting at 1.2m
3. dense deconv: 1.2m -> 0.6m
4. 0.6m pre-temporal refinement (BEVDet4D 교훈)
5. 0.6m 3D temporal (align+concat+3D residual conv, z 유지) + per-voxel flow
6. coarse dense occupancy/visibility @ 0.6m (free/unknown + 프루닝 게이트 기준)
7. sparse deconv + 2단 게이트 prune (게이트①0.6m parent + 게이트②0.3m child) + near-surface free shell
8. 4 volume heads: occupancy fine / flow / Sub-Voxel Shape / 3D Semantics (sparse kept)
   + surface head (z-flatten, 별도 브랜치)
9. exposed-face mask + packed QueryableMLP (occupancy 전용; pruned 영역은 coarse 즉답)
```

batch key 권장:

```text
images
intrinsics
extrinsics_or_virtual_extrinsics
ego_pose
history(optional)
targets(optional)
```

검증:

```text
targets가 없어도 inference forward가 된다.
targets가 있으면 loss 계산용 output이 충분히 나온다.
use_sparse_prune=false이면 0.3m dense 프로토타입 경로로 폴백한다.
```

실패 체크:

```text
모든 branch가 서로 다른 coordinate convention을 쓰면 통합이 깨진다.
voxel origin, axis order, shape convention을 model.py 상단 주석 또는 config에 고정한다.
```

## D113 - final shape test

만들 파일:

```text
final_occnet_v1/tests/test_model_shape.py
```

테스트할 것:

```text
B=1 forward
B=2 forward
use_surface false
use_temporal false
use_flow false
use_sparse_prune false (dense 0.3m 프로토타입 fallback)
near_surface_free_shell false
all true
```

예상 output shape:

```text
occ_coarse:
  B x C_occ x 100 x 34 x 10   (0.6m: free/unknown/occupied/visibility)

occ_fine:
  sparse kept voxels (0.3m, occupied 표면 + near-surface free shell)
  -> [N_kept] x 1 (sparse 표현) 또는 dense 프로토타입 시 200 x 68 x 20

surface.z_surface:
  B x 1 x 100 x 34   (0.6m BEV, z-flatten; 필요 시 0.3m upsample)

dynamic / flow:
  motion은 0.6m 3D (100 x 34 x 10), flow는 vx,vy,vz
  출력은 0.3m sparse kept voxel로 broadcast
```

권장:

```text
v1 debug에서는 memory 절약을 위해 tiny shape도 허용한다.
real config에서는 0.6m=100x34x10 / 0.3m=200x68x20 shape contract를 문서화한다.
(X·Y·Z 전부 격자=타깃, padding 없음)
```

검증:

```text
모든 output key 존재
NaN 없음
shape mismatch 없음
backward 가능한 loss subset 존재
```

주의:

```text
shape test는 학습 성능을 보지 않는다.
shape contract를 깨지 않기 위한 안전망이다.
```

## D114 - unified losses

만들 파일:

```text
final_occnet_v1/src/occnet_v1/losses.py
final_occnet_v1/tests/test_losses.py
```

구현:

```python
def compute_losses(outputs, targets, cfg):
    return {
        "loss_total": ...,
        "loss_occ_coarse": ...,   # 0.6m free/unknown/occupied
        "loss_occ_fine": ...,     # 0.3m kept 표면
        "loss_prune": ...,        # keep/drop 학습 (occupied GT가 keep 타깃)
        "loss_surface_z": ...,
        "loss_surface_valid": ...,
        "loss_dynamic": ...,
        "loss_flow": ...,         # vx,vy,vz
        "loss_subvoxel": ...,     # offset/normal/exposed-face (Sub-Voxel Shape Head)
        "loss_semantic": ...,     # 3D Semantics Head (kept occupied voxel)
        "loss_query": ...,        # Queryable MLP occupancy
    }
```

권장 loss:

```text
L_occ:
  BCE 또는 focal BCE
  unknown/invalid mask 제외

L_surface_z:
  Huber, valid surface cell에서만 계산

L_surface_valid:
  BCE

L_dynamic:
  BCE, valid/occupied 영역 기준

L_flow:
  SmoothL1(vx,vy,vz), occupied & dynamic & valid mask에서만 계산

L_prune:
  keep/drop 학습, occupied GT가 keep 타깃 (프루닝 비가역 -> recall 우선)

L_query:
  selected query point occupancy BCE
  optional boundary BCE
```

기본 weight:

```yaml
loss:
  occ_coarse: 1.0
  occ_fine: 1.0
  prune: 0.5
  surface_z: 1.0
  surface_valid: 0.5
  dynamic: 0.5
  flow: 0.5
  subvoxel: 0.5
  semantic: 0.5
  query: 0.5
```

검증:

```text
각 loss term이 scalar이다.
unknown mask가 적용된다.
mask가 비어도 NaN이 나지 않는다.
disabled branch loss는 0 또는 생략된다.
```

## D115 - final training script

만들 파일:

```text
final_occnet_v1/scripts/train_tiny.py
```

구현:

```text
1. config load
2. synthetic tiny dataset 생성
3. model 생성
4. optimizer 생성
5. forward
6. compute_losses
7. backward
8. gradient clipping optional
9. checkpoint save optional
```

출력 로그:

```text
iter
loss_total
loss_occ
loss_surface_z
loss_dynamic
loss_flow
loss_query
gpu_memory(optional)
```

검증:

```text
one-batch overfit
loss term별 print
loss_total이 감소
```

주의:

```text
이 script는 real training script가 아니라 통합 smoke train이다.
여기서 안 돌아가면 final model 통합이 아직 안 된 것이다.
```

## D116 - metrics

만들 파일:

```text
final_occnet_v1/src/occnet_v1/metrics.py
final_occnet_v1/tests/test_metrics.py
```

구현:

```python
def occupancy_iou(pred_logits, gt, valid=None, threshold=0.5): ...
def occupancy_precision_recall(pred_logits, gt, valid=None): ...
def surface_mae(z_pred, z_gt, valid): ...
def surface_valid_accuracy(valid_logit, valid): ...
def flow_endpoint_error(flow_pred, flow_gt, mask): ...
def active_selection_stats(active_output): ...
```

필수 metric:

```text
occupancy IoU
occupied precision / recall
surface MAE
surface valid accuracy
flow endpoint error
K_total selected count
Q_total query count
```

검증:

```text
perfect prediction에서 metric이 이상적 값에 가깝다.
all-empty case에서 division by zero가 나지 않는다.
active stats가 K/Q budget을 출력한다.
```

## D117 - visualization package

만들 파일:

```text
final_occnet_v1/src/occnet_v1/visualization.py
final_occnet_v1/scripts/visualize_outputs.py
```

출력:

```text
occupancy z slices
occupancy BEV max projection
surface height map
surface uncertainty map
flow BEV arrows (vx,vy) + vz slice
active/exposed-face mask map
sparse surface view (kept 표면 + near-surface free shell)
```

구현 기준:

```text
1. output directory를 인자로 받는다.
2. pred/gt가 있으면 side-by-side로 저장한다.
3. 좌표 축 방향을 모든 그림에서 동일하게 유지한다.
4. active/sparse-surface overlay는 occupancy 위에 얹어서 보여준다.
```

검증:

```text
train_tiny.py의 한 batch output을 visualize할 수 있다.
각 이미지 파일이 실제로 생성된다.
shape가 틀린 경우 조용히 넘어가지 말고 assert한다.
```

## D118 - config ablation switches

수정 파일:

```text
final_occnet_v1/configs/tiny.yaml
final_occnet_v1/configs/v1_debug.yaml
final_occnet_v1/src/occnet_v1/config.py
final_occnet_v1/tests/test_config_switches.py
```

추가:

```yaml
model:
  use_surface: true
  use_temporal: true
  use_flow: true
  use_sparse_prune: true
  near_surface_free_shell: true       # L2
  temporal_fallback_12m: false        # Orin 예산 초과 시 true
  sparse_prune:
    keep_ratio: 0.5
    k_total: 512
    q_total: 4096
```

검증:

```text
각 switch를 false로 해도 forward가 깨지지 않는다.
use_surface=false이면 surface loss/metric이 계산되지 않는다.
use_flow=false이면 flow head/loss가 꺼진다.
use_sparse_prune=false이면 0.3m dense 프로토타입 경로로 폴백한다.
near_surface_free_shell=false이면 표면-only로 keep한다.
temporal_fallback_12m=true이면 temporal을 1.2m 3D로 내린다 (z 유지).
```

중요:

```text
ablation switch는 Phase 10 runtime 분석의 기반이다.
켜고 끌 수 없으면 어떤 branch가 느린지 알 수 없다.
```

## D119 - end-to-end small validation

작업:

```text
1. small synthetic validation set을 만든다.
2. train_tiny.py로 짧게 학습한다.
3. validation metric table을 출력한다.
4. visualization을 저장한다.
5. ablation switch 하나씩 꺼서 forward와 metric을 확인한다.
```

만들 파일:

```text
final_occnet_v1/scripts/validate_tiny.py
phases/phase_09_integration_evaluation/reports/validation_tiny.md
```

기록할 표:

```text
config
occ_iou
surface_mae
flow_epe
K_total
Q_total
notes
```

완료 기준:

```text
train loss 외에도 validation metric을 볼 수 있다.
visualization path가 report에 기록된다.
```

주의:

```text
toy validation metric이 높다고 실제 SOTA라는 뜻은 아니다.
여기서는 pipeline이 end-to-end로 닫혔는지 확인한다.
```

## D120 - Phase 09 report

만들 파일:

```text
phases/phase_09_integration_evaluation/notes.md
phases/phase_09_integration_evaluation/reports/integration_report.md
```

기록:

```text
1. 동작하는 모듈
2. 불안정한 모듈
3. shape contract 표
4. config switch 표
5. loss term 표
6. metric 표
7. visualization 예시 경로
8. 가장 큰 shape 문제
9. Phase 10에서 profiling할 후보
10. final_occnet_v1로 유지할 기본 config
```

최종 검증:

```bash
pytest final_occnet_v1/tests -q
python final_occnet_v1/scripts/train_tiny.py --config final_occnet_v1/configs/tiny.yaml
python final_occnet_v1/scripts/validate_tiny.py --config final_occnet_v1/configs/tiny.yaml
python final_occnet_v1/scripts/visualize_outputs.py --config final_occnet_v1/configs/tiny.yaml
```

완료 기준:

```text
final_occnet_v1 model import가 된다.
end-to-end forward가 된다.
loss 계산과 backward가 된다.
metric과 visualization이 나온다.
ablation switch가 동작한다.
```
