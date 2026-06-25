# Phase 04 - Deconv and 0.3m Occupancy Head (dense 프로토타입)

기간: D036-D050  
목표: 1.2m coarse feature를 0.6m -> 0.3m까지 deconv하고, 0.3m occupancy를 dense 프로토타입으로 만든다.

핵심 원칙:

```text
실제 크기에선 0.3m dense feature volume(272k voxel)은 금지다.
toy 단계에서는 학습 흐름 이해를 위해 dense 프로토타입을 먼저 만든다.
Phase 08에서 sparse deconv + 2단 게이트 prune으로 전환한다 (architecture Section 16 Stage 1).
free/unknown은 0.6m coarse dense, occupied 표면은 0.3m이 담당한다 (2-해상도 hybrid).
```

## D036 - grid shape 계산

만들 파일:

```text
phases/phase_04_deconv_occupancy_head/src/grid_config.py
phases/phase_04_deconv_occupancy_head/tests/test_grid_config.py
```

구현:

```python
def compute_logical_shapes(range_xyz, resolutions):
    ...

def compute_padded_shapes(coarse_shape, num_deconv):
    ...
```

검증:

```text
range: X 60m(rear10~front50), Y 20.4m(±10.2, 격자=타깃), Z 6m(-2~+4)
1.2m: 50 x 17 x 5   = 4,250
0.6m: 100 x 34 x 10 = 34,000
0.3m: 200 x 68 x 20 = 272,000
(X·Y·Z 전부 격자=타깃, padding 없음)
```

## D037 - PanoOcc 발췌 읽기

읽을 것:

```text
PanoOcc architecture figure
coarse-to-fine sparse decoder + occupancy 프루닝 설명
```

기록:

```text
deconv가 하는 일
왜 0.3m dense feature까지 만들지 않는지 (272k voxel -> sparse deconv + prune으로 대체)
coarse-to-fine + occupancy 프루닝(keep ratio)의 의미
```

## D038 - TwoStageVoxelDecoder

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

검증:

```bash
pytest phases/phase_04_deconv_occupancy_head/tests/test_voxel_decoder.py -q
```

## D039 - 좌표 규약 / grid=target 검증 (padding/crop 불필요)

수정 파일:

```text
phases/phase_04_deconv_occupancy_head/src/grid_config.py
phases/phase_04_deconv_occupancy_head/tests/test_grid_config.py
```

작업:

```text
architecture Section 1.1 (Coordinate Conventions)을 읽고 grid 규약을 코드로 고정한다:
  origin = (-10, -10.2, -2) m, half-open cell, cell center = origin + (idx+0.5)*s
  point -> index = floor((x - origin)/s)
  parent-child: 1.2m(i) -> 0.6m(2i,2i+1) -> 0.3m(4i..4i+3), ratio 정확히 2
전 축 격자=타깃이라 padding도 valid crop도 없다 (이전의 crop_valid_03m 불필요).
```

검증:

```text
X 60m=200, Y 20.4m=68, Z 6m=20 모두 정수로 떨어진다 (assert).
voxel center / point->index round-trip이 일치한다.
1.2m->0.6m->0.3m index mapping이 ratio 2로 정확히 맞는다.
```

## D040 - Occupancy Head (0.3m dense 프로토타입)

만들 파일:

```text
phases/phase_04_deconv_occupancy_head/src/occupancy_heads.py
phases/phase_04_deconv_occupancy_head/tests/test_occupancy_head.py
```

구현:

```python
class OccupancyHead(nn.Module):
    # 갈래 A: coarse dense @ 0.6m -> occupied/free/unknown/visibility logits
    # 갈래 B: fine @ 0.3m -> occupied 표면 정밀 logit
    # toy 단계는 dense 0.3m로 프로토타입 (Phase 08에서 sparse kept voxel 위로 전환)
```

설계 원리:

```text
free/unknown(빈 공간, 저주파)은 0.6m coarse dense에서 담당한다.
occupied 표면(고주파)만 0.3m로 정밀하게 푼다.
이렇게 나눠야 dense 0.3m 비용을 피하면서 정밀 표면을 얻는다.
```

## D041 - 2-갈래 occupancy 정확성 테스트

수정 파일:

```text
phases/phase_04_deconv_occupancy_head/tests/test_occupancy_head.py
```

테스트:

```text
갈래 A: 0.6m dense에서 free/unknown/occupied/visibility logit shape 확인
갈래 B: 0.3m에서 occupied logit shape 확인
free/unknown은 0.6m, occupied 표면은 0.3m이 책임진다는 분업을 synthetic value로 확인
```

완료 기준:

```text
2-갈래 occupancy의 역할 분담을 설명할 수 있다.
```

## D042 - occupancy loss

만들 파일:

```text
phases/phase_04_deconv_occupancy_head/src/losses.py
phases/phase_04_deconv_occupancy_head/tests/test_losses.py
```

구현:

```python
def occupancy_loss(logits, target, valid_mask):
    # BCE or focal, unknown은 ignore mask
```

## D043 - decoder + occupancy head end-to-end shape

만들 파일:

```text
phases/phase_04_deconv_occupancy_head/tests/test_end_to_end_shape.py
```

검증:

```text
input:
  B x C x 50 x 17 x 5  (1.2m)

output:
  갈래 A: 0.6m occupancy (100 x 34 x 10)
  갈래 B: 0.3m occupancy (200 x 68 x 20, 전 축 격자=타깃)
```

## D044 - decoder toy training

만들 파일:

```text
phases/phase_04_deconv_occupancy_head/scripts/train_decoder_toy.py
```

작업:

```text
learnable coarse feature (1.2m)
-> decoder (1.2m -> 0.6m -> 0.3m)
-> occupancy head (2-갈래)
-> synthetic target overfit
```

완료 기준:

```text
one-batch overfit 성공
```

## D045 - 20 FPS guardrail test

수정 파일:

```text
phases/phase_04_deconv_occupancy_head/tests/test_occupancy_head.py
```

테스트:

```text
toy 프로토타입임을 명시 (실제 크기 0.3m dense는 금지).
0.6m까지만 dense, 0.3m dense feature volume은 실제 크기에서 만들지 않음을 notes.md에 기록.
Phase 08에서 sparse deconv + 2단 게이트 prune으로 전환할 지점을 표시.
```

## D046 - simple profiler

만들 파일:

```text
phases/phase_04_deconv_occupancy_head/scripts/profile_decoder.py
```

측정:

```text
decoder latency (1.2m->0.6m->0.3m)
occupancy head latency
peak memory (0.3m dense 프로토타입 비용 -> sparse 전환 동기 확인)
```

완료 기준:

```text
latency 숫자가 notes.md에 기록된다.
```

## D047 - channel ablation

작업:

```text
C=16, 32, 64로 decoder/head latency 비교
```

기록:

```text
notes.md에 channel 수와 latency/memory 표 작성
```

## D048 - 0.6m vs 1.2m attention query count

만들 파일:

```text
phases/phase_04_deconv_occupancy_head/scripts/compute_query_counts.py
```

출력:

```text
1.2m query count (4,250)
0.6m query count (34,000)
ratio
```

이해:

```text
왜 attention을 1.2m에서만 하는지 숫자로 이해한다.
```

## D049 - architecture 문서와 구현 비교

작업:

```text
occupancy_network_architecture.md Section 8(deconv/2단 게이트), 9(4 volume heads)를 읽는다.
dense 0.3m 프로토타입과 실제 sparse deconv+prune의 차이를 notes.md에 정리한다.
```

## D050 - Phase 04 report

검증:

```bash
pytest phases/phase_04_deconv_occupancy_head/tests -q
```

기록:

```text
1. deconv shape (1.2m -> 0.6m -> 0.3m)
2. 2-갈래 occupancy head shape
3. 0.3m dense 프로토타입과 sparse deconv+prune의 차이
4. free/unknown(0.6m coarse) vs occupied 표면(0.3m)의 2-해상도 hybrid
5. channel ablation 결과
```
