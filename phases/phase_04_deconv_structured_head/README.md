# Phase 04 - Deconv and Structured 20cm Head

기간: D036-D050  
목표: 1.6m coarse feature를 0.4m dense feature까지 키우고, structured sub-voxel head로 20cm dense occupancy logits를 만든다.

핵심 원칙:

```text
20cm dense output은 만든다.
20cm dense feature volume은 만들지 않는다.
모든 20cm voxel에 generic MLP를 호출하지 않는다.
```

## D036 - grid shape 계산

만들 파일:

```text
phases/phase_04_deconv_structured_head/src/grid_config.py
phases/phase_04_deconv_structured_head/tests/test_grid_config.py
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
1.6m logical: 38 x 13 x 4
0.8m internal: 76 x 26 x 8
0.4m internal: 152 x 52 x 16
0.4m valid: 150 x 50 x 13
```

## D037 - PanoOcc 발췌 읽기

읽을 것:

```text
PanoOcc architecture figure
coarse-to-fine decoder 설명
```

기록:

```text
deconv가 하는 일
왜 20cm dense feature까지 가면 무거운지
```

## D038 - TwoStageVoxelDecoder

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

검증:

```bash
pytest phases/phase_04_deconv_structured_head/tests/test_voxel_decoder.py -q
```

## D039 - valid crop / valid mask

수정 파일:

```text
phases/phase_04_deconv_structured_head/src/voxel_decoder.py
phases/phase_04_deconv_structured_head/tests/test_voxel_decoder.py
```

구현:

```python
def crop_valid_04m(feat):
    # 152 x 52 x 16 -> 150 x 50 x 13
```

검증:

```text
crop 후 shape가 정확해야 한다.
```

## D040 - StructuredSubvoxelHead

만들 파일:

```text
phases/phase_04_deconv_structured_head/src/occupancy_heads.py
phases/phase_04_deconv_structured_head/tests/test_structured_head.py
```

구현:

```python
class StructuredSubvoxelHead(nn.Module):
    # input:  B x C x X x Y x Z
    # output: B x 1 x 2X x 2Y x 2Z
```

구현 방식:

```text
Conv3d 1x1로 8개 sub-voxel logit channel 생성
8 channel을 2 x 2 x 2 위치로 reshape
```

금지:

```text
20cm dense feature tensor를 만들지 않는다.
모든 20cm voxel에 MLP를 따로 호출하지 않는다.
```

## D041 - channel reshape 정확성 테스트

수정 파일:

```text
phases/phase_04_deconv_structured_head/tests/test_structured_head.py
```

테스트:

```text
8개 channel에 서로 다른 숫자를 넣는다.
reshape 후 2x2x2 sub-voxel 위치가 기대와 같은지 확인한다.
```

완료 기준:

```text
structured head의 channel-to-subvoxel mapping을 설명할 수 있다.
```

## D042 - occupancy loss

만들 파일:

```text
phases/phase_04_deconv_structured_head/src/losses.py
phases/phase_04_deconv_structured_head/tests/test_losses.py
```

구현:

```python
def occupancy_loss(logits, target, valid_mask):
    # BCE or focal
```

## D043 - decoder + structured head end-to-end shape

만들 파일:

```text
phases/phase_04_deconv_structured_head/tests/test_end_to_end_shape.py
```

검증:

```text
input:
  B x C x 38 x 13 x 4

output:
  B x 1 x 300 x 100 x 26 또는 valid crop 후 target shape
```

주의:

```text
Z는 13에서 26으로 커질 수 있으므로 target 25에 맞게 crop/mask한다.
```

## D044 - decoder toy training

만들 파일:

```text
phases/phase_04_deconv_structured_head/scripts/train_decoder_toy.py
```

작업:

```text
learnable coarse feature
-> decoder
-> structured head
-> synthetic target overfit
```

완료 기준:

```text
one-batch overfit 성공
```

## D045 - 20 FPS guardrail test

수정 파일:

```text
phases/phase_04_deconv_structured_head/tests/test_structured_head.py
```

테스트:

```text
StructuredSubvoxelHead가 output logits 외에 20cm feature tensor를 반환하지 않는지 확인한다.
```

기록:

```text
notes.md에 "20cm output과 20cm feature의 차이"를 적는다.
```

## D046 - simple profiler

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

## D048 - 0.8m vs 1.6m attention query count

만들 파일:

```text
phases/phase_04_deconv_structured_head/scripts/compute_query_counts.py
```

출력:

```text
1.6m query count
0.8m query count
ratio
```

이해:

```text
왜 attention을 1.6m에서 하는지 숫자로 이해한다.
```

## D049 - architecture 문서와 구현 비교

작업:

```text
occupancy_network_architecture.md Section 7, 9를 읽는다.
현재 구현과 같은 점/다른 점을 notes.md에 정리한다.
```

## D050 - Phase 04 report

검증:

```bash
pytest phases/phase_04_deconv_structured_head/tests -q
```

기록:

```text
1. deconv shape
2. structured head shape
3. 20 FPS guardrail
4. channel ablation 결과
```

