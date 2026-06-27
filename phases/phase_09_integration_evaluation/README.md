# Phase 09 - Integration and Evaluation

기간: D111-D120  
목표: 각 phase에서 안정된 모듈을 `final_occnet_v1`로 모으고, end-to-end forward, loss, metric, visualization을 만든다.

이 phase는 새로운 구조를 더 발명하는 기간이 아니다.
`occupancy_network_architecture.md`와 `plan.md`에서 확정한 v1 구조를 하나의 모델 계약으로 닫는 기간이다.

## v1 통합 원칙

```text
1. 0.3m dense feature volume은 만들지 않는다.
2. dense feature는 1.2m -> 0.6m까지만 유지한다.
3. 0.6m coarse branch는 occupied/free/unknown/visibility/mixed_surface_risk를 낸다.
4. 0.6m -> 0.3m은 sparse deconv + recall-first 2단 게이트로 만든다.
5. prelim_kept는 최종 label이 아니라 O_occ_fine 실행 대상 sparse index set이다.
6. prelim_kept = child_score_keep_candidates ⊎ prelim_free_shell 이다.
7. near_surface_free_shell_tag = prelim_free_shell_raw ∩ prelim_kept 이며,
   최종 pred_free_shell / pred_free_kept를 가르는 semantic tag다.
8. O_occ_fine은 [N_kept] 전체에서 6-way label을 만든다.
9. Flow / Sub-Voxel Shape / 3D Semantics는 pred_heavy_mask의 [N_heavy]에서만 실행한다.
10. Queryable MLP는 branch A(pred_heavy_mask)에서 surface_shell_prob만 예측한다.
11. planner collision은 surface_shell_prob 단독이 아니라 collision_state fallback으로 결정한다.
12. Surface Outputs Head는 0.6m dense feature 기반 별도 BEV branch다.
13. auxiliary depth/free-space evidence head는 v1 추론에서 observed-free 판정에 필요하다.
```

핵심 index / label 관계:

```text
0.6m dense feature
-> sparse deconv + gate① high-confidence free parent drop
-> gate② child score
-> child_score_keep_candidates
-> prelim_free_shell
-> prelim_kept
-> O_occ_fine on [N_kept]
-> 6-way pred label:
     pred_occupied_surface
     pred_boundary
     pred_visibility_frontier
     pred_free_shell
     pred_free_kept
     pred_uncertain_kept
-> pred_heavy_mask = pred_occupied_surface | pred_boundary
-> Flow / Sub-Voxel Shape / 3D Semantics on [N_heavy]
-> Queryable branch A from pred_heavy_mask surface anchors
```

## 통합 대상 모듈

```text
phase_01:
  tensor shape helper
  Conv3d / ConvTranspose3d / grid_sample 이해 코드

phase_02:
  toy occupancy dataset
  occupancy loss / metric baseline
  one-batch overfit 습관

phase_03:
  vanilla cross-attention lifting
  1.2m 3D query feature

phase_04:
  1.2m -> 0.6m dense decoder
  0.3m dense occupancy prototype
  coarse occupied/free/unknown/visibility/mixed_surface_risk
  fine occupied + surface-band auxiliary prototype

phase_05:
  multi-camera token packing
  canonical ray PE / calibration-light path
  auxiliary depth/free-space evidence head

phase_06:
  SurfaceOutputsHead
  z_surface / slope / step / valid / uncertainty
  traversability_cost / drop_risk

phase_07:
  0.6m pre-temporal refinement
  0.6m 3D temporal align + concat + residual 3D conv
  Occupancy Flow Head의 O_motion_0.6 coarse seed

phase_08:
  sparse deconv + recall-first 2단 게이트
  prelim_kept / near_surface_free_shell_tag
  O_occ_fine 6-way label
  pred_heavy_mask
  exposed-face mask
  Sub-Voxel Shape Head on [N_heavy]
  3D Semantics Head on [N_heavy]
  Queryable surface_shell_prob + collision_state interface
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

권장 package 구조:

```text
final_occnet_v1/src/occnet_v1/
  __init__.py
  config.py
  model.py
  backbone.py
  lifting.py
  decoder.py             # dense 1.2m->0.6m + sparse 0.6m->0.3m
  gates.py               # gate① parent + gate② child score/shell tag
  occupancy.py           # coarse branch + O_occ_fine
  aux_depth.py           # depth/confidence/free-space evidence
  temporal.py            # 0.6m 3D align+concat+residual conv
  flow.py                # O_motion_0.6 + [N_heavy] readout/refine
  shape.py               # Sub-Voxel Shape Head
  semantics.py           # 3D Semantics Head
  surface.py             # dense BEV Surface Outputs Head
  queryable.py           # surface_shell_prob branch A
  collision_state.py     # branch B/C/D fallback rules
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
  y_range: [-10.2, 10.2]
  z_range: [-2.0, 4.0]
  channels: 64
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
runtime:
  k_total: 512
  q_total: 4096
  soft_keep_budget: 0.5
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

구현할 output 계약:

```python
class OccNetV1(nn.Module):
    def forward(self, batch):
        return {
            "occ_coarse": ...,                  # 0.6m occupied/free/unknown/visibility/mixed_surface_risk
            "aux_depth": ...,                   # depth/confidence/free-space evidence
            "o_motion_06": ...,                 # Occupancy Flow Head coarse seed
            "prelim_kept": ...,                 # child_score_keep_candidates ⊎ prelim_free_shell
            "near_surface_free_shell_tag": ..., # prelim_free_shell_raw ∩ prelim_kept
            "occ_fine": ...,                    # [N_kept] occ/surface-band/fine logits
            "pred_labels_6way": ...,            # [N_kept] 6-way label
            "pred_heavy_mask": ...,             # [N_kept] bool, occupied_surface | boundary
            "flow": ...,                        # [N_heavy] vx, vy, vz + dynamic
            "subvoxel_shape": ...,              # [N_heavy] offset/normal/shape_code/thinness/exposed-face
            "semantics": ...,                   # [N_heavy] voxel-level 3D semantics
            "surface": ...,                     # dense BEV Surface Outputs
            "queryable": ...,                   # surface_shell_prob + uncertainty + collision_state
        }
```

forward 순서:

```text
1. images -> image backbone / FPN or BiFPN-lite
2. image features -> vanilla cross-attention lifting at 1.2m
3. dense deconv: 1.2m -> 0.6m
4. 0.6m pre-temporal refinement
5. 0.6m 3D temporal: ego-align + concat + residual 3D conv, z 유지
6. O_motion_0.6 coarse seed
7. coarse occupancy/visibility/mixed_surface_risk @ 0.6m
8. auxiliary depth/free-space evidence -> observed_free_evidence
9. sparse deconv + gate① high-confidence free parent drop
10. gate② child score -> child_score_keep_candidates / prelim_free_shell_raw
11. prelim_kept와 near_surface_free_shell_tag 생성
12. O_occ_fine on [N_kept] -> 6-way pred label
13. pred_heavy_mask = pred_occupied_surface | pred_boundary
14. Flow / Sub-Voxel Shape / 3D Semantics on [N_heavy]
15. Surface Outputs Head on 0.6m dense feature
16. exposed-face mask / sparse anchors
17. Queryable branch A surface_shell_prob
18. branch B/C/D collision_state fallback
```

batch key 권장:

```text
images
intrinsics
extrinsics_or_virtual_extrinsics
ego_pose
history(optional)
targets(optional)

targets:
  occupancy occupied/free/unknown
  visibility / observed_free_evidence
  depth / depth_confidence / free-space ray evidence
  surface z/valid/slope/step/traversability/drop_risk
  dynamic / flow
  sub-voxel offset/normal/exposed-face
  semantics + valid_semantic_mask
  surface-shell query samples
```

검증:

```text
targets가 없어도 inference forward가 된다.
targets가 있으면 loss 계산용 output이 충분히 나온다.
prelim_kept와 pred_labels_6way의 첫 차원 수가 같다.
Flow/Shape/Semantics의 첫 차원은 N_heavy다.
N_heavy <= N_kept가 항상 성립한다.
Queryable branch A는 pred_heavy_mask anchor에서만 query를 만든다.
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
use_surface=false
use_aux_depth=false
use_temporal=false
use_flow=false
use_sparse_prune=false
use_near_surface_free_shell=false
use_queryable_surface_shell=false
use_collision_state_fallback=false
all true
```

예상 output shape 계약:

```text
occ_coarse:
  B x C_coarse x 100 x 34 x 10
  C_coarse includes occupied/free/unknown/visibility/mixed_surface_risk

prelim_kept:
  [N_kept] sparse 0.3m indices, batch index 포함

near_surface_free_shell_tag:
  [N_kept] bool or 0/1 tag

occ_fine / pred_labels_6way:
  [N_kept] x C_fine
  [N_kept]

pred_heavy_mask:
  [N_kept] bool

flow / subvoxel_shape / semantics:
  first dimension = N_heavy

surface:
  B x C_surface x 100 x 34
  z_surface / valid / uncertainty / slope_or_normal / step_height /
  traversability_cost / drop_risk

queryable:
  Q_branch_A <= Q_total
  output contains surface_shell_prob, uncertainty, collision_state
```

검증:

```text
모든 output key 존재
NaN 없음
shape mismatch 없음
disabled branch의 output/loss 처리 규칙 명확
backward 가능한 loss subset 존재
```

## D114 - unified losses

만들 파일:

```text
final_occnet_v1/src/occnet_v1/losses.py
final_occnet_v1/tests/test_losses.py
```

구현할 loss 항목:

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

정의역 체크:

```text
L_occ_coarse_*:
  0.6m dense grid 전체. unknown/invalid는 target별 mask 처리.

L_gate2_keep:
  child score가 keep해야 하는 child를 떨어뜨리지 않게 하는 recall-first loss.

L_gate2_shell_tag:
  near-surface free shell 후보를 occupied/surface keep과 혼동하지 않게 하는 별도 loss.

L_occ_fine / L_surface_band:
  [N_kept] 전체에서 계산.

L_heavy_recall / L_heavy_mask:
  pred_occupied_surface | pred_boundary가 GT surface/boundary를 놓치지 않는지 계산.

L_flow_coarse_seed:
  O_motion_0.6 dense/coarse seed에 적용.

L_flow_fine_readout / L_dynamic:
  [N_heavy]에서 계산. valid_flow_mask / dynamic target 필요.

L_shape_*:
  [N_heavy]에서 계산. exposed-face / offset / normal / uncertainty target 필요.

L_semantic:
  [N_heavy]에서 계산. valid_semantic_mask가 없으면 ignore.

L_query_*:
  Queryable branch A surface-shell query에만 적용.
  planner collision_state는 직접 CE로 학습하지 않고 fallback rule/unit test로 검증.

L_depth / L_depth_conf / L_free_space_evidence:
  auxiliary depth/free-space evidence head에 적용.
  hit 뒤쪽 / occluded / 미관측 공간을 free positive로 쓰지 않는다.

L_surface_*:
  0.6m dense BEV Surface Outputs Head에 적용.
```

검증:

```text
각 loss term이 scalar이다.
mask가 비어도 NaN이 나지 않는다.
disabled branch loss는 0 또는 생략된다.
collision_state fallback은 unit test로 conservative-free violation을 검증한다.
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
loss_occ_coarse
loss_gate2_keep
loss_gate2_shell_tag
loss_occ_fine
loss_heavy_recall
loss_flow
loss_shape
loss_semantic
loss_query_surface_shell
loss_depth
loss_surface
N_kept
N_heavy
Q_total
gpu_memory(optional)
```

검증:

```text
one-batch overfit
loss term별 print
loss_total이 감소
N_kept / N_heavy / Q_total이 runtime cap을 넘지 않는다.
```

## D116 - metrics

만들 파일:

```text
final_occnet_v1/src/occnet_v1/metrics.py
final_occnet_v1/tests/test_metrics.py
```

필수 metric:

```text
occupancy IoU
occupied precision / recall
6-way label confusion
prune recall: GT occupied / boundary / free-shell
heavy mask recall
surface MAE
surface valid accuracy
traversability error
drop-risk AUC
flow endpoint error on valid_flow_mask
dynamic precision / recall
semantic accuracy on valid_semantic_mask
gate selected count
N_kept / N_heavy / Q_total
Queryable surface-shell calibration
collision_state conservative-free violation count
```

검증:

```text
perfect prediction에서 metric이 이상적 값에 가깝다.
all-empty case에서 division by zero가 나지 않는다.
N_heavy <= N_kept, Q_branch_A <= Q_total을 metric에서 같이 기록한다.
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
6-way pred label map
prelim_kept / near_surface_free_shell_tag overlay
pred_heavy_mask / exposed-face mask map
surface height map
surface uncertainty map
traversability / drop-risk map
flow BEV arrows (vx, vy) + vz slice
semantic sparse map
Queryable branch A surface-shell query overlay
collision_state branch B/C/D debug overlay
```

검증:

```text
train_tiny.py의 한 batch output을 visualize할 수 있다.
각 이미지 파일이 실제로 생성된다.
좌표 축 방향이 모든 그림에서 동일하다.
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

검증:

```text
각 switch를 false로 해도 forward가 깨지지 않는다.
use_surface=false이면 surface loss/metric이 계산되지 않는다.
use_aux_depth=false이면 observed-free는 보수적으로 unknown 처리된다.
use_flow=false이면 flow readout/loss가 꺼진다.
use_semantic_head=false이면 semantic loss/metric이 꺼진다.
use_sparse_prune=false이면 tiny/dense prototype fallback으로만 테스트한다.
use_near_surface_free_shell=false이면 pred_free_shell 없이 pred_free_kept/uncertain으로 처리한다.
temporal_fallback_12m=true이면 temporal을 1.2m 3D로 내리되 z는 유지한다.
```

중요:

```text
soft_keep_budget은 목표 예산이지 mandatory recall keep보다 우선하지 않는다.
high-confidence free parent만 drop한다.
unknown/uncertain/mixed_surface_risk parent는 keep 또는 ignore다.
```

## D119 - end-to-end small validation

만들 파일:

```text
final_occnet_v1/scripts/validate_tiny.py
phases/phase_09_integration_evaluation/reports/validation_tiny.md
```

작업:

```text
1. small synthetic validation set을 만든다.
2. train_tiny.py로 짧게 학습한다.
3. validation metric table을 출력한다.
4. visualization을 저장한다.
5. ablation switch를 하나씩 꺼서 forward와 metric을 확인한다.
```

기록할 표:

```text
config
occ_iou
occupied_recall
heavy_recall
prune_recall
surface_mae
traversability_error
drop_risk_auc
flow_epe
semantic_acc
surface_shell_calibration
collision_state_violation
N_kept
N_heavy
Q_total
visualization_path
notes
```

완료 기준:

```text
train loss 외에도 validation metric을 볼 수 있다.
visualization path가 report에 기록된다.
collision_state fallback이 위험한 free 판정을 만들지 않는지 숫자로 확인한다.
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
4. output key 표
5. config switch 표
6. loss term 표와 정의역
7. metric 표
8. visualization 예시 경로
9. N_kept / N_heavy / Q_total 예산
10. 가장 큰 shape 문제
11. Phase 10에서 profiling할 stage
12. final_occnet_v1로 유지할 기본 config
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
prelim_kept / 6-way pred label / pred_heavy_mask 관계가 깨지지 않는다.
```
