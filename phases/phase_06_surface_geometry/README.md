# Phase 06 - Road Surface Geometry

기간: D066-D080  
목표: `Road Surface Geometry Head`를 구현한다. v1 core는 `z_surface`, `valid`, `uncertainty`에 집중한다.

이 phase의 핵심은 3D occupancy와 surface를 분리해서 이해하는 것이다.  
occupancy는 `X x Y x Z` 공간 안에서 각 voxel이 차 있는지 보는 출력이고, surface는 각 `X, Y` 위치에서 "주행 가능한 바닥 높이가 어디인가"를 예측하는 BEV 출력이다.

v1에서 surface는 다음 세 가지를 예측한다.

```text
z_surface:     B x 1 x X x Y  # 각 BEV cell의 바닥 높이
valid_logit:   B x 1 x X x Y  # 이 cell에 의미 있는 surface가 있는지
uncertainty:   B x 1 x X x Y  # 높이 예측이 얼마나 불확실한지
```

v1에서 slope, normal, step은 별도 neural head로 만들지 않는다.  
먼저 `z_surface`에서 미분/차분으로 유도한다. 이렇게 해야 head 수가 폭발하지 않고, "surface의 본체는 높이장(height field)"이라는 감각을 유지할 수 있다.

## 이 phase가 끝나면 할 수 있어야 하는 것

```text
1. 3D occupancy target에서 2D surface height target을 만들 수 있다.
2. 0.4m 또는 1.6m 3D feature에서 BEV surface map을 예측할 수 있다.
3. flat / ramp / curb / stair toy scene에서 z_surface, valid, step score를 볼 수 있다.
4. occupancy loss와 surface loss를 같이 학습해도 NaN 없이 loss가 감소한다.
5. surface uncertainty를 active mask/refinement 후보로 넘길 수 있다.
```

## D066 - RoadBEV / road surface 개념 정리

읽을 것:

```text
RoadBEV problem definition
road elevation / road geometry output
loss / metric 부분
가능하면 FastRSR 또는 유사 road surface reconstruction 논문 abstract + method overview
```

기록 파일:

```text
phases/phase_06_surface_geometry/notes.md
```

기록할 질문:

```text
1. occupancy volume과 surface height map은 무엇이 다른가?
2. 왜 surface는 300 x 100 x 25가 아니라 300 x 100 BEV output인가?
3. "차 있음/비어 있음"과 "이 위치의 바닥 높이"는 어떤 경우에 다르게 동작하는가?
4. 계단, 연석, 경사로는 occupancy에서는 어떻게 보이고 surface에서는 어떻게 보이는가?
5. v1에서 road semantics를 별도 class로 예측하지 않고 geometry 중심으로 시작하는 이유는 무엇인가?
```

완료 기준:

```text
notes.md에 위 5개 질문에 대한 본인 언어 설명이 있다.
surface output이 occupancy output과 같은 문제가 아니라는 점을 설명할 수 있다.
```

## D067 - surface label 생성

만들 파일:

```text
phases/phase_06_surface_geometry/src/surface_labels.py
phases/phase_06_surface_geometry/tests/test_surface_labels.py
```

구현할 함수:

```python
def make_z_surface_from_occ(occ, z_values, mode="lowest_occupied"):
    """
    occ: B x 1 x X x Y x Z 또는 B x X x Y x Z
    z_values: Z 위치의 실제 높이값
    return: z_surface, valid
    """

def make_surface_valid_mask(occ, min_occupied_count=1):
    """
    각 X,Y column에 surface를 정의할 수 있는지 판단한다.
    """
```

구현 순서:

```text
1. occ tensor shape convention을 notes.md에 먼저 적는다.
2. flat floor synthetic occ를 만든다.
3. 각 X,Y column에서 가장 낮은 occupied z를 surface로 선택한다.
4. occupied가 하나도 없는 column은 valid=False로 둔다.
5. batch dimension이 있는 경우와 없는 경우를 모두 테스트한다.
```

검증:

```text
flat floor scene에서 모든 valid cell의 z_surface가 동일해야 한다.
empty scene에서는 valid가 전부 False여야 한다.
single pillar scene에서는 pillar column에만 valid가 True가 된다.
```

주의:

```text
실제 dataset에서는 "가장 낮은 occupied"가 항상 road surface는 아니다.
예를 들어 차량 아래 shadow/occlusion, noise, overhang이 있을 수 있다.
이번 toy 구현은 surface head와 loss를 이해하기 위한 최소 label generator다.
```

## D068 - learned z pooling

만들 파일:

```text
phases/phase_06_surface_geometry/src/surface_head.py
phases/phase_06_surface_geometry/tests/test_surface_head.py
```

구현:

```python
class LearnedZPooling(nn.Module):
    """
    입력:  B x C x X x Y x Z
    출력:  B x Cb x X x Y

    Z축 정보를 단순 max/mean으로 없애지 않고,
    작은 attention 또는 1x1x1 conv 기반 가중합으로 BEV feature를 만든다.
    """
```

최소 구현:

```text
1. Conv3d로 channel을 줄인다.
2. Z축 score를 만든다.
3. softmax(score, dim=Z)로 z weight를 만든다.
4. feature * weight를 Z축으로 sum한다.
```

검증:

```text
입력 shape:  B=2, C=32, X=38, Y=13, Z=4
출력 shape:  B=2, Cb=32, X=38, Y=13
Z dimension이 사라진다.
gradient backward가 통과한다.
```

기록:

```text
notes.md에 max pooling, mean pooling, learned pooling의 차이를 적는다.
learned pooling은 surface 위치를 찾는 데 더 유리할 수 있지만, 완전히 공짜는 아니다.
```

## D069 - SurfaceGeometryHead

수정 파일:

```text
phases/phase_06_surface_geometry/src/surface_head.py
```

구현:

```python
class SurfaceGeometryHead(nn.Module):
    """
    input:  B x C x X x Y x Z
    output:
      z_surface:   B x 1 x X x Y
      valid_logit: B x 1 x X x Y
      uncertainty: B x 1 x X x Y
    """
```

구현 순서:

```text
1. LearnedZPooling으로 3D feature를 BEV feature로 바꾼다.
2. 2D Conv block 2개 정도로 BEV context를 섞는다.
3. z_surface head, valid_logit head, uncertainty_raw head를 분리한다.
4. uncertainty에는 아직 softplus를 적용하지 않아도 된다. D078에서 정리한다.
```

검증:

```text
z_surface: B x 1 x X x Y
valid_logit: B x 1 x X x Y
uncertainty: B x 1 x X x Y
모든 출력에 NaN이 없다.
```

실패 체크:

```text
Z축을 flatten해서 X,Y와 섞어버리지 않는다.
surface는 BEV output이므로 최종 출력에 Z dimension이 남으면 안 된다.
```

## D070 - surface losses

만들 파일:

```text
phases/phase_06_surface_geometry/src/surface_losses.py
phases/phase_06_surface_geometry/tests/test_surface_losses.py
```

구현:

```python
def surface_huber_loss(z_pred, z_gt, valid, delta=0.5):
    """
    valid=True인 cell에서만 z loss를 계산한다.
    """

def surface_valid_bce(valid_logit, valid):
    """
    surface 존재 여부를 BCE로 학습한다.
    """

def surface_uncertainty_loss(z_pred, z_gt, valid, uncertainty):
    """
    선택 사항.
    큰 오차에는 큰 uncertainty를 허용하고,
    작은 오차에는 uncertainty가 커지지 않도록 regularize한다.
    """
```

검증:

```text
invalid cell은 z loss에 영향을 주면 안 된다.
valid가 전부 False인 batch에서도 NaN이 나면 안 된다.
z_pred == z_gt이면 z loss가 거의 0이다.
```

기록:

```text
notes.md에 "valid BCE"와 "z regression"이 왜 둘 다 필요한지 적는다.
```

## D071 - surface one-batch training

만들 파일:

```text
phases/phase_06_surface_geometry/scripts/train_surface_one_batch.py
```

작업:

```text
1. synthetic flat surface feature/target을 만든다.
2. synthetic ramp surface feature/target을 만든다.
3. SurfaceGeometryHead만 단독으로 학습한다.
4. 200 iteration 정도에서 one-batch overfit을 확인한다.
```

출력할 로그:

```text
iter
loss_z
loss_valid
z_mae
valid_acc
```

완료 기준:

```text
flat scene에서 z_mae가 빠르게 감소한다.
ramp scene에서 완벽하지 않아도 z_surface가 기울어진 형태로 변한다.
valid BCE가 감소한다.
```

실패 체크:

```text
loss가 감소하지 않으면 먼저 label generator를 시각화한다.
surface head가 너무 작은지 보기 전에 target이 맞는지 확인한다.
```

## D072 - surface visualization

만들 파일:

```text
phases/phase_06_surface_geometry/scripts/visualize_surface.py
```

출력:

```text
z_surface_pred.png
z_surface_gt.png
z_error.png
valid_pred.png
valid_gt.png
uncertainty.png
```

구현 기준:

```text
1. matplotlib imshow를 사용한다.
2. X축은 전후방, Y축은 좌우 방향으로 일관되게 둔다.
3. 색상 범위는 pred/gt가 동일해야 한다.
4. invalid cell은 회색 또는 투명 처리한다.
```

검증:

```text
flat, ramp, step toy scene을 각각 저장한다.
이미지만 봐도 z_surface가 무엇을 의미하는지 알 수 있어야 한다.
```

## D073 - derived slope / normal / step

만들 파일:

```text
phases/phase_06_surface_geometry/src/surface_derived.py
phases/phase_06_surface_geometry/tests/test_surface_derived.py
```

구현:

```python
def compute_slope_from_z(z_surface, dx=0.4, dy=0.4):
    """
    z_surface의 x/y 방향 차분으로 slope magnitude를 계산한다.
    """

def compute_normal_from_z(z_surface, dx=0.4, dy=0.4):
    """
    height field의 normal vector를 근사한다.
    """

def compute_step_score_from_z(z_surface, threshold=0.25):
    """
    인접 cell 높이 차이가 큰 곳을 step/curb 후보로 본다.
    """
```

이해:

```text
v1에서는 slope/normal/step을 별도 head로 만들지 않는다.
z_surface에서 유도한다.
이렇게 해야 head가 너무 많아지는 것을 막고, surface geometry의 중심을 z height로 유지할 수 있다.
```

검증:

```text
flat surface: slope와 step score가 거의 0
ramp surface: slope는 일정하고 step score는 낮음
single curb: curb 경계에서 step score가 높음
```

## D074 - step / curb / stair toy scene

수정 파일:

```text
phases/phase_06_surface_geometry/src/surface_labels.py
```

추가할 toy scene:

```text
flat ground
ramp
single curb
stair-like repeated step
hole / invalid region
```

구현 팁:

```text
1. toy scene은 label generator 검증용이므로 복잡할 필요가 없다.
2. X,Y grid 위에 z_surface_gt를 먼저 만든다.
3. z_surface_gt를 occupancy volume으로 변환하는 helper를 만든다.
4. 다시 occupancy volume에서 z_surface를 복원해 gt와 비교한다.
```

검증:

```text
step 위치에서 step_score가 높다.
hole region에서는 valid=False가 나온다.
stair scene에서는 여러 step boundary가 검출된다.
```

기록:

```text
notes.md에 "계단도 이동 가능 영역이 될 수 있는가?"에 대한 v1 관점을 적는다.
권장 답:
계단은 단순 road semantic이 아니라 surface geometry + traversability 판단 문제다.
v1은 계단을 직접 class로 분류하기보다 z_surface/step_score/uncertainty로 표현한다.
```

## D075 - occupancy-surface consistency

만들 파일:

```text
phases/phase_06_surface_geometry/src/consistency.py
phases/phase_06_surface_geometry/tests/test_consistency.py
```

구현:

```python
def surface_occ_consistency_loss(occ_logits, z_surface, valid, z_values):
    """
    예측 surface z 주변에 occupancy boundary가 있어야 한다는 약한 제약.
    """
```

간단한 구현 방향:

```text
1. z_surface를 각 X,Y column의 연속 z index로 변환한다.
2. 해당 z 근처의 occ probability를 sampling한다.
3. surface 근처는 occupied/free 경계에 가까워야 한다는 보조 loss를 만든다.
4. 처음에는 작은 lambda로만 사용한다.
```

주의:

```text
이 loss를 너무 강하게 걸면 occupancy와 surface가 서로를 망칠 수 있다.
v1에서는 "보조 regularization"으로만 둔다.
```

검증:

```text
z_surface가 occupancy 바닥 위치와 일치하면 loss가 작다.
z_surface를 일부러 위아래로 shift하면 loss가 커진다.
```

## D076 - phase_05 model에 surface 붙이기

만들 파일:

```text
phases/phase_06_surface_geometry/src/model_with_surface.py
phases/phase_06_surface_geometry/tests/test_model_with_surface.py
```

구현:

```text
phase_05 SingleFrameOccNet 구조
+ SurfaceGeometryHead
```

권장 forward output:

```python
{
    "occ_logits": occ_logits,
    "surface": {
        "z_surface": z_surface,
        "valid_logit": valid_logit,
        "uncertainty": uncertainty,
    },
}
```

검증:

```text
forward output에 occ_logits와 surface outputs가 같이 존재한다.
occ_logits shape는 기존 phase_05와 동일하다.
surface output은 B x 1 x X x Y이다.
```

실패 체크:

```text
surface head 때문에 occupancy head shape가 바뀌면 안 된다.
surface는 추가 branch이지 occupancy decoder를 대체하지 않는다.
```

## D077 - occupancy + surface joint training

만들 파일:

```text
phases/phase_06_surface_geometry/scripts/train_occ_surface.py
```

구현:

```text
L_total = L_occ
        + lambda_z * L_surface_z
        + lambda_valid * L_surface_valid
        + lambda_consistency * L_consistency
```

처음 loss weight:

```text
lambda_z = 1.0
lambda_valid = 0.5
lambda_consistency = 0.05
```

완료 기준:

```text
L_occ와 L_surface가 모두 NaN 없이 감소한다.
surface loss를 켰을 때 occupancy loss가 갑자기 발산하지 않는다.
```

기록:

```text
notes.md에 loss weight를 바꿨을 때 관찰한 내용을 적는다.
```

## D078 - uncertainty positive output

수정 파일:

```text
phases/phase_06_surface_geometry/src/surface_head.py
phases/phase_06_surface_geometry/src/surface_losses.py
```

구현:

```text
uncertainty raw output에 softplus 적용
minimum epsilon 추가
```

예시:

```python
uncertainty = F.softplus(uncertainty_raw) + 1e-4
```

검증:

```text
uncertainty 값이 항상 양수이다.
uncertainty가 loss를 NaN으로 만들지 않는다.
```

이해:

```text
uncertainty는 "이 cell을 믿어도 되는가"를 active mask에 넘기는 신호가 될 수 있다.
높은 uncertainty cell은 refinement 후보가 될 수 있다.
```

## D079 - surface metrics

만들 파일:

```text
phases/phase_06_surface_geometry/src/surface_metrics.py
phases/phase_06_surface_geometry/tests/test_surface_metrics.py
```

구현:

```python
def surface_mae(z_pred, z_gt, valid):
    ...

def surface_rmse(z_pred, z_gt, valid):
    ...

def valid_accuracy(valid_logit, valid):
    ...

def step_f1(step_score, step_gt):
    ...
```

검증:

```text
perfect prediction은 MAE=0에 가깝다.
invalid cell은 MAE/RMSE에서 제외된다.
valid_accuracy는 threshold 기준으로 계산된다.
```

기록:

```text
v1 최종 평가에서는 surface MAE, valid accuracy를 필수 metric으로 가져간다.
step_f1은 toy scene 분석용으로 둔다.
```

## D080 - Phase 06 report

검증:

```bash
pytest phases/phase_06_surface_geometry/tests -q
python phases/phase_06_surface_geometry/scripts/train_surface_one_batch.py
python phases/phase_06_surface_geometry/scripts/train_occ_surface.py
```

기록 파일:

```text
phases/phase_06_surface_geometry/notes.md
```

기록할 내용:

```text
1. surface head가 occupancy head와 다른 이유
2. z_surface / valid / uncertainty의 의미
3. flat / ramp / step / stair toy scene 결과
4. surface loss weight 기본값
5. active mask에 넘길 surface signal
6. final_occnet_v1로 가져갈 파일 목록
```

최종 완료 기준:

```text
surface 단독 one-batch overfit이 된다.
occupancy + surface joint forward가 된다.
surface visualization이 저장된다.
테스트가 통과한다.
```
