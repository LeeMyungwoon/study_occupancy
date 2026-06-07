# Phase 02 - Toy Occupancy

기간: D011-D020  
목표: synthetic 3D occupancy label을 만들고, 작은 모델이 one-batch overfit되는 경험을 만든다.

이 phase는 "데이터 -> 모델 -> loss -> metric -> visualization"의 전체 훈련 루프를 아주 작은 문제에서 끝까지 경험하는 단계다.

## 완료 기준

```bash
pytest phases/phase_02_toy_occupancy/tests -q
python phases/phase_02_toy_occupancy/scripts/train_one_batch.py
```

## D011 - synthetic cube occupancy 생성

만들 파일:

```text
phases/phase_02_toy_occupancy/src/synthetic.py
phases/phase_02_toy_occupancy/tests/test_synthetic.py
phases/phase_02_toy_occupancy/notes.md
```

구현:

```python
def make_empty_grid(shape):
    # shape: X, Y, Z
    # return: 1 x X x Y x Z occupancy tensor

def add_box(occ, center, size):
    # occ grid 안에 axis-aligned box를 occupied로 채움

def make_box_scene(shape):
    # empty grid + random/simple box
```

검증:

```text
occupied voxel count가 0보다 크다.
box size를 키우면 occupied voxel count가 증가한다.
```

## D012 - free / occupied / unknown mask

수정 파일:

```text
phases/phase_02_toy_occupancy/src/synthetic.py
phases/phase_02_toy_occupancy/tests/test_synthetic.py
```

구현:

```python
def make_occ_free_unknown(occ, unknown_ratio=0.0):
    # target: occupied=1, free=0
    # valid_mask: free/occupied만 True, unknown은 False
```

검증:

```text
unknown_ratio=0이면 valid_mask가 전부 True
unknown_ratio>0이면 일부 voxel이 loss에서 제외됨
```

이해:

```text
unknown은 free가 아니다.
관측되지 않은 공간을 free로 학습시키면 occupancy가 위험해진다.
```

## D013 - Tiny latent-to-occupancy model

만들 파일:

```text
phases/phase_02_toy_occupancy/src/tiny_model.py
phases/phase_02_toy_occupancy/tests/test_tiny_model.py
```

구현:

```python
class TinyLatentOccNet(nn.Module):
    # learnable latent: 1 x C x X x Y x Z
    # Conv3d blocks
    # output: B x 1 x X x Y x Z logits
```

검증:

```text
B=2, X=16, Y=8, Z=4 기준 output shape 확인
```

## D014 - masked BCE loss

만들 파일:

```text
phases/phase_02_toy_occupancy/src/losses.py
phases/phase_02_toy_occupancy/tests/test_losses.py
```

구현:

```python
def masked_bce_with_logits(logits, target, valid_mask):
    # unknown은 loss에서 제외
```

검증:

```text
valid_mask가 False인 voxel의 target을 바꿔도 loss가 변하지 않아야 한다.
```

## D015 - one-batch training script

만들 파일:

```text
phases/phase_02_toy_occupancy/scripts/train_one_batch.py
```

구현:

```text
1개 synthetic scene 생성
TinyLatentOccNet 생성
Adam optimizer
200 step train
10 step마다 loss 출력
```

검증:

```bash
python phases/phase_02_toy_occupancy/scripts/train_one_batch.py
```

완료 기준:

```text
loss가 명확히 감소한다.
감소하지 않으면 learning rate, target shape, mask를 확인한다.
```

## D016 - voxel visualization

만들 파일:

```text
phases/phase_02_toy_occupancy/src/visualize.py
phases/phase_02_toy_occupancy/scripts/visualize_scene.py
```

구현:

```python
def save_voxel_slices(occ, out_dir):
    # z slice별 png 저장
```

검증:

```bash
python phases/phase_02_toy_occupancy/scripts/visualize_scene.py
```

완료 기준:

```text
target과 prediction slice png를 눈으로 확인할 수 있다.
```

## D017 - focal loss와 class imbalance 관찰

수정 파일:

```text
phases/phase_02_toy_occupancy/src/losses.py
phases/phase_02_toy_occupancy/scripts/train_one_batch.py
```

구현:

```python
def focal_loss_with_logits(logits, target, valid_mask, gamma=2.0, alpha=0.25):
    ...
```

작업:

```text
BCE와 focal loss를 번갈아 사용한다.
occupied voxel이 매우 적을 때 loss 차이를 notes.md에 기록한다.
```

## D018 - SyntheticVoxelDataset

만들 파일:

```text
phases/phase_02_toy_occupancy/src/dataset.py
```

구현:

```python
class SyntheticVoxelDataset(Dataset):
    # random boxes
    # random walls
    # optional sphere-like blob
```

검증:

```text
DataLoader에서 batch가 나온다.
target, valid_mask shape가 일치한다.
```

## D019 - occupancy metric

만들 파일:

```text
phases/phase_02_toy_occupancy/src/metrics.py
phases/phase_02_toy_occupancy/tests/test_metrics.py
```

구현:

```python
def occupancy_iou(logits, target, valid_mask, threshold=0.5):
    ...

def precision_recall(logits, target, valid_mask, threshold=0.5):
    ...
```

검증:

```text
perfect prediction이면 IoU=1
all-free prediction에서 occupied recall이 낮아야 함
```

## D020 - Phase 02 report

수정 파일:

```text
phases/phase_02_toy_occupancy/notes.md
```

작업:

```text
1. one-batch overfit loss 기록
2. target/prediction visualization 경로 기록
3. BCE와 focal loss 차이 기록
4. IoU metric 결과 기록
```

검증:

```bash
pytest phases/phase_02_toy_occupancy/tests -q
```

