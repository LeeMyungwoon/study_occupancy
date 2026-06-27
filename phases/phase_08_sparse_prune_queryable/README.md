# Phase 08 - Sparse Deconv + 2단 게이트 Prune + Queryable

기간: D096-D110

목표: `0.6m -> 0.3m`을 sparse deconv + recall-first 2단 게이트로 만들고, `prelim_kept -> O_occ_fine -> 6-way pred label -> pred_heavy_mask` 흐름 위에 Sub-Voxel Shape / 3D Semantics / Queryable interface를 붙인다.

이 phase는 Phase 04의 dense 0.3m 프로토타입을 실제 v1 sparse 경로로 교체하는 단계다.

## 핵심 불변식

```text
gate① parent:
  high-confidence free parent만 drop.
  occupied / mixed_surface_risk / unknown / uncertain / dynamic / planner-corridor는 keep 또는 ignore.

gate② child:
  cheap child score head가 keep/free/surface/shell-tag score를 낸다.

prelim_kept:
  child_score_keep_candidates ⊎ prelim_free_shell.
  이것은 최종 label이 아니라 O_occ_fine을 실행할 sparse index set이다.

near_surface_free_shell_tag:
  prelim_free_shell_raw ∩ prelim_kept.
  이것은 final pred_free_shell / pred_free_kept를 가르는 semantic tag다.

O_occ_fine 이후:
  pred_occupied_surface
  pred_boundary
  pred_visibility_frontier
  pred_free_shell
  pred_free_kept
  pred_uncertain_kept
  위 6개 label로 prelim_kept 전체를 완전 분할한다.

pred_heavy_mask:
  pred_occupied_surface | pred_boundary.
  Flow / Sub-Voxel Shape / 3D Semantics / Queryable MLP branch A는 여기서만 실행한다.
```

## 이 phase가 끝나면 할 수 있어야 하는 것

```text
1. high-confidence free parent만 drop하는 gate①을 만들 수 있다.
2. gate② cheap child score로 child_score_keep_candidates / prelim_free_shell_raw를 만들 수 있다.
3. prelim_kept와 near_surface_free_shell_tag를 분리해 관리할 수 있다.
4. L_gate2_keep과 L_gate2_shell_tag를 분리해 supervise할 수 있다.
5. O_occ_fine 이후 6-way pred label과 pred_heavy_mask를 만들 수 있다.
6. exposed face를 6-way pred label + coarse fallback 기준으로 계산할 수 있다.
7. Sub-Voxel Shape Head와 3D Semantics Head를 [N_heavy]에서만 실행할 수 있다.
8. Queryable MLP는 branch A(pred_heavy_mask)에서 surface_shell_prob만 예측하게 할 수 있다.
9. branch B/C/D는 collision_state fallback으로 보수 판정할 수 있다.
10. N_kept / N_heavy / K_total / Q_total budget을 분리해서 기록할 수 있다.
```

## D096 - sparse deconv + gate①

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

class SparseVoxelDeconv(nn.Module):
    # keep된 parent만 0.3m child 후보로 sparse 업샘플
```

검증:

```text
clean observed-free parent의 child만 생성하지 않는다.
unknown / low-visibility / mixed_surface_risk parent가 free로 잘리지 않는다.
0.6m dense feature 자체를 자르는 것이 아니라 sparse fine 가지에만 gate가 적용된다.
```

## D097 - gate② child score + prelim_kept / shell-tag 분리

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
prelim_kept = child_score_keep_candidates ⊎ prelim_free_shell.
prelim_free_shell은 중복 제거된 extra index set이고 semantic source가 아니다.
child_score_keep_candidates 안에 이미 있던 near-surface free child도 near_surface_free_shell_tag가 유지된다.
keep ~0.5는 soft budget target이고 mandatory recall keep이 우선이다.
```

## D098 - prune keep + query priority score 통합

```python
def surface_band_score(surface_logit_or_cheap_surface_score): ...
def uncertainty_score(occ_logits): ...
def mixed_surface_risk_score(mixed_surface_risk_logit): ...
def shell_tag_score(shell_tag_logit): ...
def dynamic_score(dynamic_logits): ...
def near_field_score(voxel_centers): ...
def planner_corridor_score(voxel_centers, trajectory): ...
def thin_or_high_gradient_score(...): ...
def compute_keep_priority(...): ...
```

검증:

```text
surface-band / mixed-risk / uncertainty / dynamic / planner 영역 score가 높다.
far-field / thin-object quota가 단순 top-k에 밀려 사라지지 않는다.
priority score는 keep/drop을 돕는 cheap score이지 최종 pred label이 아니다.
```

## D099 - Pruning false-negative 방어

v1 필수 항목:

```python
# 1. recall-first gate
# 2. uncertainty keep
def quota_keep(category_scores, quotas): ...  # 3. category quota
# 4. GT-guided warmup
# 5. soft / straight-through top-k
def gate2_keep_loss(keep_logit, keep_target, keep_weight): ...        # L_gate2_keep
def gate2_shell_tag_loss(shell_tag_logit, shell_tag_target, mask): ... # L_gate2_shell_tag
```

검증:

```text
prune_recall(GT occupied 생존율)을 측정한다.
observed surface-band / near-surface observed free shell recall을 측정한다.
L_gate2_keep은 "살릴지", L_gate2_shell_tag는 "shell semantic tag인지"를 분리해 학습한다.
중복 voxel이 여러 category에 있어도 최종 index는 unique다.
```

## D100 - distance / planner LOD budget

```python
def make_lod_budget_policy(distance_bins, planner_weight, dynamic_weight, hard_caps): ...
def assign_query_budget_after_pred(pred_heavy_voxels, budget_policy, planner_score, dynamic_score): ...
```

검증:

```text
D100에서는 아직 pred_heavy_mask가 없으므로 budget policy만 독립 test한다.
D102 이후 pred_heavy_voxels에 적용해 M_i를 배정한다.
N_kept / K_total / Q_total hard cap 초과 시 낮은 priority부터 fine/query 실행을 줄인다.
제외된 voxel을 free로 간주하지 않고 coarse unknown/occupied-risk fallback으로 degrade한다.
```

## D101 - O_occ_fine 6-way pred label + exposed face mask

```python
class SparseFineOccupancyHead(nn.Module):
    # input: F_0.3_sparse on prelim_kept
    # output: occ_logit + surface_band_logit on [N_kept]

def make_six_way_pred_labels(occ_logit, surface_band_logit, visibility, near_surface_free_shell_tag): ...
def compute_pred_heavy_mask(pred_labels): ...
def compute_exposed_faces(pred_heavy_idx, pred_labels_6way, coarse_fallback): ...
```

검증:

```text
prelim_kept 전체가 6-way pred label 중 정확히 하나로 분할된다.
pred_heavy_mask = pred_occupied_surface | pred_boundary.
near_surface_free_shell_tag가 있어도 O_occ_fine이 occupied로 보면 pred_occupied_surface가 된다.
child_score_keep_candidates에서 온 voxel이라도 fine이 free이고 shell tag가 있으면 pred_free_shell이다.
exposed-face 계산은 단순 kept 기준이 아니라 6-way pred label 기준이다.
```

## D101a - Sub-Voxel Shape Information Head

```python
class SubVoxelShapeHead(nn.Module):
    # input:  F_0.3_sparse[pred_heavy_mask] + exposed_face_mask
    # output: [N_heavy] x channels
    # exposed_face_logits / face_surface_offset / local_normal /
    # shape_uncertainty / thinness_logit / shape_code
```

검증:

```text
출력은 pred_heavy_mask(occupied surface + boundary) 위에서만 계산된다.
pred_free_shell / pred_free_kept / pred_visibility_frontier / pred_uncertain_kept / pruned는 제외된다.
shape_code가 SparseLocalAnchor / Queryable MLP branch A 입력으로 전달된다.
```

## D101b - 3D Semantics Head

```python
class SemanticsHead(nn.Module):
    # input:  F_0.3_sparse[pred_heavy_mask]
    # output: [N_heavy] x N_class
```

검증:

```text
semantic은 pred_heavy_mask 위에서 실행하고, loss는 valid_semantic_mask에만 적용한다.
pred_boundary 전체에 강한 semantic label을 주지 않는다.
pred_free_shell / pred_free_kept / pred_visibility_frontier / pred_uncertain_kept 영역에 semantic을 강제하지 않는다.
Queryable은 semantic query를 지원하지 않는다.
```

## D102 - adaptive query generation

만들 파일:

```text
phases/phase_08_sparse_prune_queryable/src/query_generation.py
```

```python
def make_initial_queries(pred_heavy_idx, exposed_faces, shape_uncertainty, budget): ...
    # Queryable MLP branch A(pred_heavy_mask)용 query만 생성
    # pred_free_shell / pred_free_kept는 observed-free fallback 분기라 MLP query를 만들지 않음
```

검증:

```text
query는 branch A(pred_heavy_mask)용으로만 생성된다.
pred_free_shell / pred_free_kept / pred_visibility_frontier / pred_uncertain_kept는 MLP query를 만들지 않는다.
```

## D103 - pack variable queries

```python
def pack_query_lists(query_lists): ...
```

검증:

```text
q_local_packed: Q x 3
query_offsets: K + 1
branch_id가 모두 A(pred_heavy_mask)다.
```

## D104 - SparseLocalAnchor

만들 파일:

```text
phases/phase_08_sparse_prune_queryable/src/sparse_anchor.py
```

```python
def gather_03m_features(feat_sparse, selected_pred_heavy_idx): ...
    # F_0.3_sparse는 prelim_kept의 sparse feature.
    # Queryable branch A에서는 pred_heavy voxel feature만 gather.

class SparseLocalAnchor(nn.Module): ...
```

검증:

```text
SparseLocalAnchor는 pred_heavy_mask voxel feature만 Queryable branch A context로 모은다.
prelim_kept 전체를 무조건 queryable MLP에 넣지 않는다.
```

## D105 - Queryable Interface

만들 파일:

```text
phases/phase_08_sparse_prune_queryable/src/queryable_mlp.py
phases/phase_08_sparse_prune_queryable/src/collision_state.py
```

구현:

```python
class QueryableSurfaceShellMLP(nn.Module):
    # branch A only: pred_heavy_mask voxel
    # input: F_query + shape_code + exposed_face context + q_local + PE
    # output: surface_shell_prob / surface_shell_uncertainty

def collision_state_from_fallback(query_point, pred_label, coarse_occ, visibility, observed_free_evidence_06, observed_free_evidence_03): ...
    # A: pred_heavy_mask -> MLP surface_shell_prob + fallback
    # B1: pred_free_shell -> observed_free_evidence_03 + NOT solid_interior_veto 필요
    # B2: pred_free_kept  -> observed_free_evidence_06 + coarse_free_conf 필요
    # C: pred_visibility_frontier | pred_uncertain_kept -> coarse/visibility fallback
    # D: pruned -> coarse/visibility fallback
```

검증:

```text
MLP output은 Q x 2(surface_shell_prob, uncertainty).
surface_shell_prob를 collision probability로 직접 해석하지 않는다.
planner는 collision_state == observed_free일 때만 통과 가능으로 본다.
observed-free evidence가 없으면 free로 반환하지 않는다.
```

## D106 - packed execution test

만들 파일:

```text
phases/phase_08_sparse_prune_queryable/tests/test_queryable_mlp.py
```

검증:

```text
K=32
variable M_i
Q=sum(M_i)
MLP output Q x 2(surface_shell_prob, uncertainty)
branch A만 MLP를 실행한다.
branch B(pred_free_shell/pred_free_kept)는 child/parent observed-free evidence가 있을 때만 free 즉답한다.
branch C/D는 coarse/visibility fallback으로 빠진다.
observed-free evidence가 없으면 free로 반환하지 않는다.
```

## D107 - boundary search

```python
def find_sign_change_pairs(query_points, surface_shell_prob): ...
def refine_boundary_bisection(...): ...
```

검증:

```text
boundary search는 pred_heavy_mask branch A 안에서만 사용한다.
v1 Queryable output은 전역 dense 3D feature interpolation이 아니라 sparse feature 기반 piecewise-continuous surface-shell field다.
분기 B/C/D에는 MLP boundary search를 적용하지 않는다.
```

## D108 - fine overlay representation

만들 파일:

```text
phases/phase_08_sparse_prune_queryable/src/fine_overlay.py
```

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

검증:

```text
prelim_kept -> O_occ_fine 6-way label -> pred_heavy branch A query / fallback branch B-C-D 흐름을 한 representation에 담는다.
Queryable MLP / Sub-Voxel Shape Head / planner fallback이 같은 sparse representation을 공유한다.
Gaussian refinement는 현재 architecture에 없으므로 도입하지 않는다.
```

## D109 - Queryable / surface-shell toy training

만들 파일:

```text
phases/phase_08_sparse_prune_queryable/scripts/train_queryable_toy.py
```

검증:

```text
pred_heavy_mask branch A query에서 L_query_surface_shell / L_query_uncertainty가 감소한다.
observed-free local sample이 surface_shell_prob를 낮추는지 확인한다.
solid interior / occluded / 미관측 query를 free negative로 강제하지 않는다.
```

## D110 - Phase 08 report

검증:

```bash
pytest phases/phase_08_sparse_prune_queryable/tests -q
python phases/phase_08_sparse_prune_queryable/scripts/train_queryable_toy.py
```

기록할 내용:

```text
1. gate① high-conf free parent drop 규칙
2. gate② prelim_kept / prelim_free_shell / near_surface_free_shell_tag 분리
3. 6-way pred label과 pred_heavy_mask
4. L_gate2_keep / L_gate2_shell_tag 분리
5. exposed-face mask
6. Sub-Voxel Shape Head 출력([N_heavy])
7. 3D Semantics Head 출력([N_heavy])
8. surface_shell_prob와 collision_state를 분리한 이유
9. N_kept / N_heavy / K_total / Q_total
10. final_occnet_v1로 가져갈 파일 목록
```
