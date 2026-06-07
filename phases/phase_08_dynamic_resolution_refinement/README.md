# Phase 08 - Dynamic-resolution Refinement

기간: D096-D110  
목표: `Active Mask Head`, quota Top-K, sparse local feature anchor, packed Queryable MLP를 구현한다.

이 phase는 네 아키텍처에서 가장 중요한 "동적 해상도 복셀" 부분이다.  
기본 20cm dense occupancy는 이미 structured sub-voxel head로 만든다고 가정한다. 여기서 만드는 refinement는 전체 공간을 다시 20cm feature volume으로 만드는 것이 아니라, 중요한 위치만 더 자세히 물어보는 sparse 보조 경로다.

전체 관계:

```text
0.4m dense feature
-> structured 2x2x2 occupancy head
-> 20cm dense occupancy map

0.4m dense feature + 20cm occupancy + surface + flow
-> active mask
-> selected 0.4m cells
-> sparse local anchor
-> adaptive query points
-> packed Queryable MLP
-> fine overlay / refined boundary
```

중요한 구분:

```text
20cm occupancy:
  모든 20cm voxel에 대해 occupied/free/unknown을 빠르게 출력한다.

20cm feature:
  v1에서는 전체 공간에 대해 만들지 않는다.
  선택된 cell 내부에서만 local anchor와 query point feature를 만든다.

refinement:
  dense occupancy를 대체하지 않는다.
  boundary, dynamic object, planner corridor 같은 중요한 곳을 더 정밀하게 보정한다.
```

## 이 phase가 끝나면 할 수 있어야 하는 것

```text
1. occupancy/surface/flow 결과로 refinement 후보 점수를 만들 수 있다.
2. category별 quota로 K_total을 넘지 않게 selected voxel을 고를 수 있다.
3. selected voxel마다 query 개수 M_i를 다르게 배정할 수 있다.
4. 노출된 face 기준으로 query point를 생성할 수 있다.
5. variable-length query list를 Q x D packed tensor로 만들 수 있다.
6. QueryableMLP를 for-loop 없이 batch로 실행할 수 있다.
7. dense 20cm occupancy와 sparse fine overlay를 분리해서 저장할 수 있다.
```

## D096 - active score components

만들 파일:

```text
phases/phase_08_dynamic_resolution_refinement/src/active_mask.py
phases/phase_08_dynamic_resolution_refinement/tests/test_active_mask.py
```

구현:

```python
def boundary_score(occ_logits):
    """
    occupied/free 경계 근처 voxel에 높은 점수.
    """

def uncertainty_score(occ_logits):
    """
    p=0.5 근처 또는 unknown logit이 높은 곳에 높은 점수.
    """

def dynamic_score(dynamic_logits):
    """
    움직일 가능성이 높은 voxel에 높은 점수.
    """
```

구현 팁:

```text
boundary_score:
  occ prob의 local gradient 또는 neighbor 차이를 사용한다.

uncertainty_score:
  sigmoid(logit)가 0.5에 가까울수록 높게 한다.
  예: 1 - abs(p - 0.5) * 2

dynamic_score:
  dynamic probability를 그대로 쓰되, occupied 영역과 곱하는 variant도 만든다.
```

검증:

```text
occupied interior보다 occupied/free boundary score가 높다.
p=0.5 근처 uncertainty score가 높다.
p=0.0 또는 p=1.0 근처 uncertainty score는 낮다.
dynamic logit이 큰 voxel의 dynamic score가 높다.
```

주의:

```text
active score는 최종 occupancy가 아니다.
"더 자세히 볼 후보"를 고르는 점수다.
```

## D097 - surface / planner / near score

수정 파일:

```text
phases/phase_08_dynamic_resolution_refinement/src/active_mask.py
```

구현:

```python
def near_field_score(voxel_centers, near_radius=10.0):
    ...

def planner_corridor_score(voxel_centers, trajectory, corridor_width=1.5):
    ...

def surface_score(z_surface, valid, voxel_centers):
    ...
```

각 score의 의미:

```text
near_field_score:
  ego vehicle 가까운 영역은 안전상 더 중요하다.

planner_corridor_score:
  계획 경로 주변은 반드시 자세히 봐야 한다.

surface_score:
  바닥 surface 근처의 voxel은 traversability와 충돌 판단에 중요하다.
```

검증:

```text
ego 근처 voxel score가 멀리 있는 voxel보다 높다.
planner trajectory 주변 score가 corridor 밖보다 높다.
surface z 근처 voxel score가 surface에서 먼 voxel보다 높다.
invalid surface cell은 surface_score가 낮다.
```

기록:

```text
notes.md에 active mask가 "occupied만 keep"하면 위험한 이유를 적는다.
특히 unknown, surface, planner 주변을 남겨야 하는 이유를 적는다.
```

## D098 - S_refine 통합

수정 파일:

```text
phases/phase_08_dynamic_resolution_refinement/src/active_mask.py
```

구현:

```python
def normalize_score(score, eps=1e-6):
    ...

def compute_refine_score(
    boundary,
    uncertainty,
    dynamic,
    planner,
    surface,
    near,
    weights=None,
):
    ...
```

기본 weight:

```text
boundary:    1.0
uncertainty: 0.8
dynamic:     1.0
planner:     1.2
surface:     0.8
near:        0.6
```

검증:

```text
score component를 끄고 켤 수 있다.
weight를 0으로 두면 해당 component 영향이 사라진다.
모든 score가 0이어도 NaN이 나지 않는다.
```

실패 체크:

```text
score scale이 제각각이면 특정 component가 전부 먹어버린다.
각 score를 0~1 범위로 정규화한 뒤 합친다.
```

## D099 - quota_topk

수정 파일:

```text
phases/phase_08_dynamic_resolution_refinement/src/active_mask.py
phases/phase_08_dynamic_resolution_refinement/tests/test_quota_topk.py
```

구현:

```python
def quota_topk(category_scores, quotas, total_budget):
    """
    category별 quota만큼 top-k 선택.
    중복 voxel 제거.
    total_budget을 넘지 않게 clamp.
    """
```

기본 quota:

```text
K_total = 2048
K_boundary = 768
K_near = 384
K_planner = 384
K_dynamic = 256
K_uncertain = 192
K_random_probe = 64
```

검증:

```text
category별 selected count가 quota를 넘지 않는다.
전체 selected 수가 K_total 이하이다.
중복 voxel이 여러 category에 있어도 최종 selected index는 unique이다.
score가 전부 같은 경우에도 deterministic하게 동작한다.
```

주의:

```text
단순 global top-k만 쓰면 planner corridor나 random probe가 사라질 수 있다.
quota는 안전성과 다양성을 위한 장치다.
```

## D100 - distance / planner LOD budget

수정 파일:

```text
phases/phase_08_dynamic_resolution_refinement/src/active_mask.py
```

구현:

```python
def assign_query_budget(selected_voxels, distance, planner_score, dynamic_score):
    """
    selected voxel마다 query 개수 M_i를 정한다.
    return: K 길이의 integer tensor
    """
```

권장 rule:

```text
near + planner + dynamic:
  M_i = 16

near boundary:
  M_i = 8

far uncertain:
  M_i = 4

random probe:
  M_i = 2 또는 4
```

검증:

```text
near/planner/dynamic voxel은 M_i가 크다.
far/static voxel은 M_i가 작다.
sum(M_i) == Q_total이 hard cap을 넘지 않는다.
Q_total cap을 넘으면 낮은 priority voxel부터 M_i를 줄인다.
```

기록:

```text
notes.md에 K_total과 Q_total의 차이를 적는다.
K_total은 선택된 coarse cell 수이고, Q_total은 실제 MLP query point 수다.
```

## D101 - exposed face mask

수정 파일:

```text
phases/phase_08_dynamic_resolution_refinement/src/query_generation.py
phases/phase_08_dynamic_resolution_refinement/tests/test_exposed_faces.py
```

구현:

```python
def compute_exposed_faces(occ_binary, unknown_mask=None):
    """
    output: X x Y x Z x 6 face mask 또는 K x 6 face mask
    face order: +x, -x, +y, -y, +z, -z
    """
```

의미:

```text
exposed face는 occupied voxel이 free/unknown 이웃과 맞닿는 면이다.
한쪽 면만 보이면 그 면 주변에 query를 더 많이 날린다.
```

검증:

```text
free neighbor 방향이 exposed로 표시된다.
unknown neighbor 방향도 exposed로 표시할 수 있다.
occupied interior는 exposed face가 적다.
isolated occupied voxel은 face 6개가 exposed다.
```

주의:

```text
exposed face는 20cm dense occupancy 또는 coarse occupancy에서 계산할 수 있다.
v1에서는 먼저 20cm dense occupancy를 0.4m cell 단위로 요약해서 사용한다.
```

## D102 - adaptive query generation

만들 파일:

```text
phases/phase_08_dynamic_resolution_refinement/src/query_generation.py
phases/phase_08_dynamic_resolution_refinement/tests/test_query_generation.py
```

구현:

```python
def make_initial_queries(exposed_faces, budget, voxel_size=0.4):
    """
    selected 0.4m cell 내부의 local query point를 만든다.
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
corner처럼 여러 face가 exposed되면 query가 더 다양해진다.
M_i budget보다 많은 query를 만들면 priority 기준으로 잘라낸다.
```

이해:

```text
이 부분이 tesla_occnet.md에서 이야기한 "보이는 면만 가변적으로 더 묻기"에 해당한다.
전체 voxel 내부를 균등하게 묻는 것이 아니라, 관측/경계가 있는 면을 중심으로 query를 배치한다.
```

## D103 - variable query packing

수정 파일:

```text
phases/phase_08_dynamic_resolution_refinement/src/query_generation.py
phases/phase_08_dynamic_resolution_refinement/tests/test_query_packing.py
```

구현:

```python
def pack_query_lists(query_lists):
    """
    list of Mi x 3 -> Q x 3, offsets K+1
    """

def unpack_query_outputs(outputs, offsets):
    """
    Q x D -> list[Mi x D]
    """
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
Jetson Orin AGX에서 작은 MLP를 Python loop로 K번 실행하면 latency가 크게 증가한다.
반드시 Q개 query를 하나의 batch로 묶어서 실행한다.
```

## D104 - SparseLocalAnchor

만들 파일:

```text
phases/phase_08_dynamic_resolution_refinement/src/sparse_anchor.py
phases/phase_08_dynamic_resolution_refinement/tests/test_sparse_anchor.py
```

구현:

```python
def gather_04m_features(feat, selected_idx):
    """
    feat: B x C x X x Y x Z
    selected_idx: K x 4 또는 B,K,3
    return: K x C
    """

class SparseLocalAnchor(nn.Module):
    """
    selected 0.4m feature를 queryable local context로 변환한다.
    """
```

anchor에 넣을 정보:

```text
0.4m dense feature
coarse occupancy logits
surface distance
dynamic probability
normalized voxel center PE
```

검증:

```text
K selected voxel -> K x C anchor feature
전체 20cm dense feature를 만들지 않는다.
selected_idx가 batch를 포함해도 shape가 맞다.
```

주의:

```text
SparseLocalAnchor는 sparse deconv를 꼭 의미하지 않는다.
v1에서는 먼저 selected feature + 작은 MLP/conv로 local anchor를 만든다.
나중에 필요하면 sparse convolution/deconv backend로 교체할 수 있다.
```

## D105 - QueryableMLP

만들 파일:

```text
phases/phase_08_dynamic_resolution_refinement/src/queryable_mlp.py
phases/phase_08_dynamic_resolution_refinement/tests/test_queryable_mlp.py
```

구현:

```python
class QueryableMLP(nn.Module):
    """
    input:
      anchor_feature: Q x C_anchor
      q_local:        Q x 3
      q_pe:           Q x C_pe
      coarse_logits:  Q x C_coarse

    output:
      occ_logit:      Q x 1
      unknown_logit:  Q x 1
      boundary_logit: Q x 1
    """
```

구현 순서:

```text
1. selected voxel의 K x C anchor를 만든다.
2. offsets를 이용해 anchor를 Q개 query로 repeat_interleave한다.
3. q_local positional encoding을 붙인다.
4. coarse occ/surface/dynamic 정보를 붙인다.
5. 한 번의 MLP forward로 Q x output을 만든다.
```

검증:

```text
input Q x C_in
output Q x C_out
Q가 0인 edge case에서도 안전하게 처리한다.
```

기록:

```text
notes.md에 "QueryableMLP는 모든 20cm voxel을 예측하는 기본 head가 아니다"라고 적는다.
기본 dense occupancy는 structured head가 담당한다.
```

## D106 - packed execution test

수정 파일:

```text
phases/phase_08_dynamic_resolution_refinement/tests/test_queryable_mlp.py
```

테스트 상황:

```text
K=32
M_i가 [2, 4, 8, 16] 등으로 섞임
Q=sum(M_i)
anchor: K x C
q_local_packed: Q x 3
```

검증:

```text
QueryableMLP output Q x 3
Python loop 없이 한 번에 실행된다.
backward가 통과한다.
```

추가 benchmark:

```text
K=512, avg M=4
K=1024, avg M=8
K=2048, avg M=8
```

기록:

```text
latency를 notes.md에 적는다.
PyTorch CPU/GPU 환경에 따라 절대 시간은 달라도, loop 방식보다 packed 방식이 빠른지 확인한다.
```

## D107 - boundary search

수정 파일:

```text
phases/phase_08_dynamic_resolution_refinement/src/query_generation.py
```

구현:

```python
def find_sign_change_pairs(query_points, occ_probs):
    """
    occupied/free probability가 바뀌는 query pair를 찾는다.
    """

def refine_boundary_bisection(query_a, query_b, mlp_fn, steps=3):
    """
    두 query 사이에서 boundary 위치를 이분 탐색한다.
    """
```

검증:

```text
occupied/free sign change가 있는 pair를 찾는다.
boundary가 없는 pair는 refine 대상에서 제외된다.
bisection step을 늘리면 boundary point가 더 안정된다.
```

주의:

```text
bisection은 비용이 있으므로 모든 selected voxel에 하지 않는다.
boundary confidence가 높은 소수 후보에만 적용한다.
```

## D108 - fine overlay representation

만들 파일:

```text
phases/phase_08_dynamic_resolution_refinement/src/fine_overlay.py
phases/phase_08_dynamic_resolution_refinement/tests/test_fine_overlay.py
```

구현:

```python
class FineOverlay:
    """
    dense occupancy 위에 얹는 sparse refinement 결과.
    """
```

저장할 정보:

```text
selected 0.4m voxel index
query points
query occ logits
boundary points
confidence
optional exposed face id
```

목표:

```text
전체 map을 dense fine voxel로 바꾸지 않는다.
sparse overlay로만 저장한다.
planner나 visualization이 필요할 때 dense occupancy와 함께 조회한다.
```

검증:

```text
FineOverlay를 만들고 다시 selected voxel별로 조회할 수 있다.
empty overlay도 안전하게 처리된다.
overlay를 dense map으로 rasterize하는 helper를 optional로 만든다.
```

## D109 - refinement toy training

만들 파일:

```text
phases/phase_08_dynamic_resolution_refinement/scripts/train_refinement_toy.py
```

작업:

```text
1. coarse 0.4m feature와 dense 20cm occupancy toy target을 만든다.
2. boundary가 있는 selected voxel만 Active Mask로 고른다.
3. QueryableMLP가 selected voxel 내부 query point의 occupancy를 맞추게 학습한다.
4. structured head baseline과 refinement overlay 결과를 비교한다.
```

출력 로그:

```text
iter
loss_query_occ
loss_boundary
selected_K
Q_total
boundary_accuracy
```

완료 기준:

```text
selected boundary voxel 내부 query loss가 감소한다.
dense occupancy는 그대로 유지되고 overlay만 추가된다.
K/Q budget이 설정값을 넘지 않는다.
```

실패 체크:

```text
loss가 줄지 않으면 query point와 gt occupancy sampling 좌표계를 확인한다.
K가 너무 작으면 boundary 후보가 빠질 수 있다.
Q가 너무 크면 runtime 목표를 망칠 수 있다.
```

## D110 - Phase 08 report

검증:

```bash
pytest phases/phase_08_dynamic_resolution_refinement/tests -q
python phases/phase_08_dynamic_resolution_refinement/scripts/train_refinement_toy.py
```

기록 파일:

```text
phases/phase_08_dynamic_resolution_refinement/notes.md
```

기록할 내용:

```text
1. active score component별 의미
2. 기본 quota 표
3. K_total과 Q_total 기본값
4. exposed face 기반 query 생성 방식
5. QueryableMLP input/output shape
6. packed execution latency
7. dense 20cm occupancy와 sparse fine overlay의 차이
8. final_occnet_v1로 가져갈 파일 목록
```

최종 완료 기준:

```text
quota Top-K 테스트가 통과한다.
query packing/unpacking 테스트가 통과한다.
QueryableMLP packed forward/backward가 된다.
refinement toy training loss가 감소한다.
FineOverlay로 결과를 저장하고 조회할 수 있다.
```
