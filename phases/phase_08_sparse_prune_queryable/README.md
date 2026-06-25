# Phase 08 - Sparse Deconv + 2단 게이트 Prune + Queryable

기간: D096-D110  
목표: `0.6m -> 0.3m`을 sparse deconv + 2단 게이트 prune(gate① 0.6m parent + gate② 0.3m child)으로 만들고, near-surface free shell(L2)을 함께 keep한다. 그 위에 exposed-face mask / quota keep / sparse anchor / packed Queryable MLP(occupancy 전용)를 붙인다.

이 phase는 Phase 04의 **dense 0.3m 프로토타입을 sparse 경로로 교체**하는 단계다. (architecture Section 8, 10, 16)

전체 관계:

```text
0.6m dense feature + coarse occupancy
-> 게이트① (0.6m parent): occupied/boundary parent만 0.3m로 확장 (free/unknown은 생성 안 함)
-> sparse deconv: 남은 parent만 0.3m로 업샘플
-> 게이트② (0.3m child): 빈 child prune(keep~0.5) + 표면 인접 free 1겹 keep (L2)
-> F_0.3_sparse (kept = occupied 표면 + near-surface free shell)
-> 4 volume heads (sparse) + exposed-face mask + packed Queryable MLP(occupancy 전용)
```

중요한 구분:

```text
2단 게이트 = 비용 + 선명도:
  게이트① (연산량): 빈 영역 0.3m children을 "생성조차 안 함" (272k materialize 회피)
  게이트② (선명도/anti-dilation): 빈 child 제거 -> 표면이 얇고 또렷

near-surface free shell (L2, 기본 ON):
  표면 인접 free 1겹을 keep -> 좁은 통로 통과 판단을 0.3m로 정밀하게
  비용: kept가 표면 voxel 대비 ~1.5~2배 (공짜 아님), heavy head는 표면만

0.3m dense feature volume:
  실제 크기에선 만들지 않는다 (272k voxel).
```

## 이 phase가 끝나면 할 수 있어야 하는 것

```text
1. 0.6m coarse occupancy로 occupied/boundary parent만 0.3m로 확장할 수 있다 (게이트①).
2. 확장된 child를 keep~0.5로 prune하고 near-surface free 1겹을 keep할 수 있다 (게이트②, L2).
3. category별 quota로 keep을 보호할 수 있다 (얇은 물체/원거리/dynamic).
4. exposed face를 kept-AND-occupied 기준으로 계산할 수 있다 (N1).
5. Sub-Voxel Shape Head(offset/normal/shape_code/thinness)를 kept occupied voxel 위에 만들 수 있다 (D101a).
6. 3D Semantics Head를 kept occupied voxel 위에 만들 수 있다 (D101b).
7. selected voxel마다 query 개수 M_i를 다르게 배정할 수 있다.
8. variable-length query list를 Q x D packed tensor로 만들 수 있다.
9. QueryableMLP(occupancy 전용)를 for-loop 없이 batch로 실행할 수 있다.
10. pruned 영역 query는 부모 0.6m coarse로 보수적 즉답할 수 있다.
```

## D096 - sparse deconv + 게이트① (0.6m parent gate, 연산량)

만들 파일:

```text
phases/phase_08_sparse_prune_queryable/src/sparse_decoder.py
phases/phase_08_sparse_prune_queryable/tests/test_sparse_decoder.py
```

구현:

```python
def parent_gate_06m(coarse_occ_logits, threshold):
    """
    coarse occupancy로 occupied/boundary parent만 남긴다.
    free/unknown parent의 0.3m children은 생성하지 않는다 (272k materialize 회피).
    """

class SparseVoxelDeconv(nn.Module):
    # 남은 parent만 0.3m로 sparse 업샘플
```

검증:

```text
free/unknown parent는 0.3m child를 만들지 않는다.
occupied/boundary parent만 확장된다.
전체 0.3m grid(272k)를 dense로 materialize하지 않는다.
```

## D097 - 게이트② (0.3m child prune, 선명도) + near-surface free shell (L2)

수정 파일:

```text
phases/phase_08_sparse_prune_queryable/src/sparse_decoder.py
```

구현:

```python
def child_prune_03m(child_occ_logits, keep_ratio=0.5):
    """
    확장된 child 중 빈 child를 제거 -> 표면을 또렷하게 (anti-dilation).
    """

def add_near_surface_free_shell(kept, occ_flag):
    """
    표면 인접 free child 1겹을 함께 keep (기본 ON, L2).
    config flag(near_surface_free_shell)로 off 가능.
    """
```

검증:

```text
빈 child가 제거되어 표면이 얇게 유지된다 (anti-dilation).
표면 인접 free 1겹이 keep된다 (좁은 통로 통과 판단용).
kept = occupied 표면 + near-surface free shell 임을 확인 (occupied/free flag로 구분).
```

기록:

```text
notes.md에 "왜 0.3m을 dense로 만들면 안 되는가 = 비용 + 표면 흐려짐(dilation)"을 적는다.
near-surface free shell의 비용(kept ~1.5~2배)도 적는다.
```

## D098 - prune keep + query priority score 통합

수정 파일:

```text
phases/phase_08_sparse_prune_queryable/src/active_mask.py
phases/phase_08_sparse_prune_queryable/tests/test_active_mask.py
```

구현:

```python
def boundary_score(occ_logits): ...
def uncertainty_score(occ_logits): ...
def dynamic_score(dynamic_logits): ...
def near_field_score(voxel_centers): ...
def planner_corridor_score(voxel_centers, trajectory): ...
def compute_keep_priority(...):
    # 게이트② keep 우선순위 + 이후 query budget에 공용으로 쓰는 score
```

검증:

```text
boundary / uncertainty / dynamic / planner 영역 score가 높다.
far-field는 keep을 더 높여 원거리 물체를 보호한다 (프루닝 비가역, recall 우선).
score scale을 0~1로 정규화한 뒤 합친다 (특정 component가 전부 먹지 않게).
```

주의:

```text
keep priority는 최종 occupancy가 아니다.
"어떤 표면 voxel을 살릴지 + 어디를 더 자세히 query할지"를 정하는 score다.
```

## D099 - Pruning false-negative 방어 (v1 필수, architecture Section 8.2.1)

**프루닝 false negative가 가장 위험한 실패 모드다.** coarse(게이트①)가 얇은 물체를 free로 오판하면 0.3m에서 복구 불가(비가역). 충돌 안전에 직결되므로 v1.5가 아니라 **v1부터 전부** 넣는다.

수정 파일:

```text
phases/phase_08_sparse_prune_queryable/src/active_mask.py
phases/phase_08_sparse_prune_queryable/src/sparse_decoder.py
phases/phase_08_sparse_prune_queryable/tests/test_prune_recall.py
```

구현 (v1 필수, 5가지):

```python
# 1. recall-first gate: 게이트① 임계값을 recall 기준 (occupied 놓침 << free 더 keep)
# 2. uncertainty keep: coarse p~0.5 / unknown / 저신뢰 parent는 무조건 keep
def quota_keep(category_scores, quotas, total_budget): ...
#    3. category quota: near/far-lowconf/dynamic/planner/thin/random 최소 keep 보장
# 4. GT-guided warmup: 학습 초기 GT occupied parent 강제 keep -> 점진 전환
# 5. soft / straight-through top-k: hard top-k는 gradient 없음 -> keep에 gradient 흘림
```

검증:

```text
prune_recall (GT occupied가 살아남은 비율)을 측정한다 (핵심 지표).
얇은 물체 toy(난간/막대)에서 coarse가 약해도 recall이 유지되는지 확인.
category별 selected count가 quota를 넘지 않고, 카테고리별 최소 keep이 보장된다.
soft/straight-through top-k로 gate에 gradient가 흐른다.
중복 voxel이 여러 category에 있어도 최종 index는 unique이다.
```

## D100 - distance / planner LOD budget

수정 파일:

```text
phases/phase_08_sparse_prune_queryable/src/active_mask.py
```

구현:

```python
def assign_query_budget(selected_voxels, distance, planner_score, dynamic_score):
    """
    selected voxel마다 query 개수 M_i를 정한다.
    """
```

권장 rule:

```text
near + planner + dynamic: M_i 큼
far / static:             M_i 작음
random probe:             M_i 작게
sum(M_i) == Q_total 이 hard cap을 넘지 않게 clamp
```

검증:

```text
near/planner/dynamic voxel은 M_i가 크다.
far/static voxel은 M_i가 작다.
sum(M_i)가 Q_total을 넘으면 낮은 priority부터 M_i를 줄인다.
```

기록:

```text
notes.md에 K_total(선택 voxel 수)과 Q_total(MLP query point 수)의 차이를 적는다.
```

## D101 - exposed face mask (kept AND occupied, N1)

만들 파일:

```text
phases/phase_08_sparse_prune_queryable/src/query_generation.py
phases/phase_08_sparse_prune_queryable/tests/test_exposed_faces.py
```

구현:

```python
def compute_exposed_faces(kept_idx, occ_flag, coarse_occ):
    """
    neighbor가 kept AND occupied -> not exposed (occupied 표면끼리 맞닿음)
    neighbor가 kept free shell    -> exposed-to-free (fine 확정 노출면)
    neighbor가 pruned             -> 부모 0.6m coarse로 free/unknown/occupied 구분
    face order: +x, -x, +y, -y, +z, -z
    """
```

의미 / N1 주의:

```text
L2로 near-surface free shell도 kept이므로, 판정은 "kept인가"가 아니라
"kept이면서 occupied인가"여야 한다.
그렇지 않으면 표면이 free shell과 맞닿는 면(= 진짜 노출면)을 빠뜨린다.
```

검증:

```text
표면이 kept free shell과 맞닿는 면이 exposed로 잡힌다 (N1).
pruned neighbor는 부모 coarse occupancy로 판정된다.
occupied interior는 exposed face가 적다.
```

## D101a - Sub-Voxel Shape Information Head (architecture Section 9.3)

만들 파일:

```text
phases/phase_08_sparse_prune_queryable/src/subvoxel_shape_head.py
phases/phase_08_sparse_prune_queryable/tests/test_subvoxel_shape_head.py
```

구현:

```python
class SubVoxelShapeHead(nn.Module):
    """
    input:  F_0.3_sparse kept (occupied 표면) voxel feature + exposed_face_mask(D101)
    output (per kept occupied voxel):
      exposed_face_logits: 6
      face_surface_offset: 6   # 각 exposed face -> voxel 내부 surface까지 normalized
      local_normal: 3
      shape_uncertainty: 1
      thinness_logit: 1
      shape_code: Cs           # 예: 8 or 16. D104 anchor / D105 QueryableMLP가 소비
    """
```

face-aware execution (head 내부 동작):

```text
zero exposed face(내부) -> heavy local query 생략
1 face -> 그 normal 방향만 offset/boundary query
2 face -> edge, 3+ face -> corner/thin 후보 (budget 증가)
모든 voxel을 같은 비용으로 query하지 않는다 (priority mask 사용).
```

검증:

```text
출력이 kept occupied voxel 위에서만 계산된다 (free shell/pruned 제외).
shape_code가 D104 anchor / D105 QueryableMLP 입력으로 전달된다.
offset이 [0,1](또는 [-0.5,0.5]) 범위, normal이 단위 vector에 가깝다.
interior voxel은 query budget이 0에 가깝다.
```

기록:

```text
"Sub-Voxel Shape는 후처리가 아니라 4 volume head 중 하나"임을 notes.md에 적는다.
exposed-face mask(D101)는 이 head의 '실행 방식'이지 별도 head가 아니다.
```

## D101b - 3D Semantics Head (architecture Section 9.4)

만들 파일:

```text
phases/phase_08_sparse_prune_queryable/src/semantics_head.py
phases/phase_08_sparse_prune_queryable/tests/test_semantics_head.py
```

구현:

```python
class SemanticsHead(nn.Module):
    """
    input:  F_0.3_sparse kept (occupied 표면) voxel feature
    output: [N_kept] x N_class   # sparse, kept voxel별 semantic logit
    """
```

권장 class (실내 + 실외 비도로):

```text
ground/floor, wall/building, vehicle, pedestrian, cyclist,
curb/barrier, vegetation, other static
(free/unknown은 Occupancy Head 담당 -> semantics에서 제외/ignore)
```

검증:

```text
semantic은 occupied/surface-near voxel 위에서만 계산/loss 적용.
free/unknown 영역에 semantic을 강제하지 않는다 (noisy supervision 방지).
출력 shape [N_kept] x N_class, NaN 없음.
```

기록:

```text
"Queryable MLP에는 semantic을 넣지 않고 이 head에서만 제공"(Tesla 2-MLP 중
occupancy만 채택, N7-B)을 notes.md에 적는다.
```

## D102 - adaptive query generation

만들 파일:

```text
phases/phase_08_sparse_prune_queryable/src/query_generation.py
phases/phase_08_sparse_prune_queryable/tests/test_query_generation.py
```

구현:

```python
def make_initial_queries(exposed_faces, budget, voxel_size=0.3):
    """
    selected 0.3m voxel 내부/면의 local query point를 만든다.
    """
```

query 후보:

```text
voxel center
exposed face center
exposed face 주변 edge points
corner point
surface shell point
```

검증:

```text
one exposed face는 해당 normal 방향 query가 많다.
여러 face가 exposed되면 query가 더 다양해진다.
M_i budget보다 많으면 priority로 잘라낸다.
```

## D103 - variable query packing

수정 파일:

```text
phases/phase_08_sparse_prune_queryable/src/query_generation.py
phases/phase_08_sparse_prune_queryable/tests/test_query_packing.py
```

구현:

```python
def pack_query_lists(query_lists):
    # list of Mi x 3 -> Q x 3, offsets K+1

def unpack_query_outputs(outputs, offsets):
    # Q x D -> list[Mi x D]
```

검증:

```text
q_local_packed: Q x 3
query_offsets: K + 1
offsets[-1] == Q
unpack(pack(x))가 원래 list 길이와 일치한다.
```

금지:

```text
for voxel in selected_voxels:
    QueryableMLP(...)
```

이유:

```text
Orin에서 작은 MLP를 Python loop로 K번 실행하면 latency가 크게 증가한다.
Q개 query를 하나의 batch로 묶어서 실행한다.
```

## D104 - SparseLocalAnchor

만들 파일:

```text
phases/phase_08_sparse_prune_queryable/src/sparse_anchor.py
phases/phase_08_sparse_prune_queryable/tests/test_sparse_anchor.py
```

구현:

```python
def gather_03m_features(feat, selected_idx):
    """
    F_0.3_sparse의 kept voxel feature를 모은다.
    return: K x C
    """

class SparseLocalAnchor(nn.Module):
    # selected 0.3m feature를 queryable local context로 변환
```

anchor에 넣을 정보:

```text
F_0.3_sparse feature
coarse occupancy logits (부모 0.6m)
surface distance
dynamic probability
normalized voxel center PE
shape code / offset / normal (Sub-Voxel Shape Head)
```

검증:

```text
K selected voxel -> K x C anchor feature
전체 0.3m dense feature를 만들지 않는다.
selected_idx가 batch를 포함해도 shape가 맞다.
```

## D105 - QueryableMLP (occupancy 전용)

만들 파일:

```text
phases/phase_08_sparse_prune_queryable/src/queryable_mlp.py
phases/phase_08_sparse_prune_queryable/tests/test_queryable_mlp.py
```

구현:

```python
class QueryableMLP(nn.Module):
    """
    input:
      anchor_feature: Q x C_anchor   (F_0.3_sparse + shape code)
      q_local:        Q x 3
      q_pe:           Q x C_pe
      coarse_logits:  Q x C_coarse

    output:
      occ_logit:      Q x 1
      uncertainty:    Q x 1
    (occupancy 전용. semantic은 voxel-level 3D Semantics Head에서만 -
     Tesla 2-MLP 중 occupancy MLP만 채택)
    """
```

pruned 영역 처리:

```text
pruned voxel 근처 query는 MLP를 돌리지 않고 부모 0.6m coarse occupancy로 보수적 즉답:
  high-conf free -> free / unknown -> unknown / occupied·uncertain -> occupied
(프루닝됨 != free; solid interior를 free로 오답하지 않게)
```

기록:

```text
notes.md에 "QueryableMLP는 모든 0.3m voxel을 예측하는 기본 head가 아니다"라고 적는다.
기본 occupancy는 Occupancy Head가, 임의 좌표 평가만 Queryable MLP가 담당한다.
```

## D106 - packed execution test

수정 파일:

```text
phases/phase_08_sparse_prune_queryable/tests/test_queryable_mlp.py
```

테스트 상황:

```text
K=32
M_i가 [2, 4, 8, 16]로 섞임
Q=sum(M_i)
anchor: K x C
q_local_packed: Q x 3
```

검증:

```text
QueryableMLP output Q x 1 (occupancy)
Python loop 없이 한 번에 실행된다.
backward가 통과한다.
pruned 영역 query는 coarse 즉답 경로로 빠진다.
```

추가 benchmark:

```text
K=512, avg M=4
K=1024, avg M=8
K=2048, avg M=8
```

## D107 - boundary search

수정 파일:

```text
phases/phase_08_sparse_prune_queryable/src/query_generation.py
```

구현:

```python
def find_sign_change_pairs(query_points, occ_probs):
    # occupied/free probability가 바뀌는 query pair를 찾는다.

def refine_boundary_bisection(query_a, query_b, mlp_fn, steps=3):
    # 두 query 사이에서 boundary 위치를 이분 탐색한다.
```

검증:

```text
occupied/free sign change가 있는 pair를 찾는다.
boundary가 없는 pair는 refine 대상에서 제외된다.
bisection step을 늘리면 boundary point가 더 안정된다.
```

주의:

```text
bisection은 비용이 있으므로 boundary confidence가 높은 소수 후보에만 적용한다.
```

## D108 - sparse surface representation

만들 파일:

```text
phases/phase_08_sparse_prune_queryable/src/sparse_surface.py
phases/phase_08_sparse_prune_queryable/tests/test_sparse_surface.py
```

구현:

```python
class SparseSurfaceRepresentation:
    # kept voxel idx (occupied 표면 + near-surface free shell)
    # F_0.3_sparse feature + Sub-Voxel Shape code / offset / normal
    # query points / occ logits / boundary points / confidence
```

설계 주의:

```text
이것은 architecture의 최종 30cm 표현(sparse deconv + 2단 게이트 prune +
near-surface free shell) 위에 올라가는 sparse surface representation이다.
"selected voxel -> local context(shape code/offset) -> 임의 점 occupancy 평가"
인터페이스로 추상화해, Queryable MLP / Sub-Voxel Shape Head가 공유하도록 한다.
(Gaussian refinement는 현재 architecture에 없다 -> 도입하지 않는다.
 선택 연구로 두려면 같은 인터페이스로 D130 ablation에서만 비교.)
```

검증:

```text
SparseSurfaceRepresentation을 만들고 selected voxel별로 조회할 수 있다.
empty도 안전하게 처리된다.
dense map으로 rasterize하는 helper는 optional (시각화/planner용).
```

## D109 - sparse decoder + queryable toy training

만들 파일:

```text
phases/phase_08_sparse_prune_queryable/scripts/train_sparse_toy.py
```

작업:

```text
1. 0.6m coarse occupancy + feature와 0.3m occupancy toy target을 만든다.
2. 게이트①/게이트②로 sparse 0.3m kept voxel을 만든다 (+ near-surface free shell).
3. kept voxel 위에서 occupancy / QueryableMLP를 학습한다.
4. dense 0.3m 프로토타입(Phase 04)과 sparse 결과를 비교한다.
```

출력 로그:

```text
iter
loss_occ_fine
loss_query_occ
kept_voxel_count (표면 + free shell)
Q_total
prune_recall (GT occupied가 살아남은 비율)
```

완료 기준:

```text
kept voxel 위 occupancy loss가 감소한다.
GT occupied voxel이 과도하게 prune되지 않는다 (recall 우선).
K/Q budget이 설정값을 넘지 않는다.
```

실패 체크:

```text
prune_recall이 낮으면 keep_ratio를 높이거나 quota를 강화한다.
loss가 안 줄면 query point와 GT occupancy sampling 좌표계를 확인한다.
```

## D110 - Phase 08 report

검증:

```bash
pytest phases/phase_08_sparse_prune_queryable/tests -q
python phases/phase_08_sparse_prune_queryable/scripts/train_sparse_toy.py
```

기록 파일:

```text
phases/phase_08_sparse_prune_queryable/notes.md
```

기록할 내용:

```text
1. 2단 게이트(parent/child)의 두 목적 (연산량 + 선명도)
2. near-surface free shell(L2)의 효과와 비용 (~1.5~2배)
3. exposed face의 kept-AND-occupied 기준 (N1)
4. Sub-Voxel Shape Head(D101a) 출력 (offset/normal/shape_code/thinness)
5. 3D Semantics Head(D101b) 출력
6. quota keep 카테고리
7. K_total과 Q_total 기본값
8. QueryableMLP input/output (occupancy 전용)
9. pruned 영역 coarse 즉답 규칙 (프루닝됨 != free)
10. final_occnet_v1로 가져갈 파일 목록
-> 이 phase로 4 volume head(Occupancy/Flow/Sub-Voxel Shape/3D Semantics)가 모두 완성된다.
```

최종 완료 기준:

```text
게이트①/게이트② sparse decoder 테스트가 통과한다.
near-surface free shell on/off가 동작한다.
Sub-Voxel Shape Head / 3D Semantics Head가 kept occupied voxel 위에서 동작한다.
quota keep / query packing 테스트가 통과한다.
QueryableMLP packed forward/backward가 된다 (occupancy 전용).
sparse decoder + queryable toy training loss가 감소한다.
```
