# Phase 03 - Attention Lifting

기간: D021-D035  
목표: 언어 번역 attention 이해를 image token과 3D query cross-attention으로 확장한다.

이 phase의 핵심 문장:

```text
번역:
  decoder token이 source sentence token을 본다.

occupancy:
  3D voxel query가 multi-camera image token을 본다.
```

## D021 - scaled dot-product attention

만들 파일:

```text
phases/phase_03_attention_lifting/src/attention.py
phases/phase_03_attention_lifting/tests/test_attention.py
```

구현:

```python
def scaled_dot_product_attention(q, k, v, mask=None):
    # q: B x H x Nq x D
    # k: B x H x Nk x D
    # v: B x H x Nk x D
```

검증:

```text
attention weight의 마지막 차원 합이 1인지 확인
output shape가 B x H x Nq x D인지 확인
```

## D022 - VanillaCrossAttention

수정 파일:

```text
phases/phase_03_attention_lifting/src/attention.py
```

구현:

```python
class VanillaCrossAttention(nn.Module):
    # Q from query_tokens
    # K,V from image_tokens
    # output: B x Nq x C
```

검증:

```text
query: B x 64 x C
image: B x 512 x C
output: B x 64 x C
```

## D023 - REO 발췌 읽기

읽을 것:

```text
REO Abstract
Method overview figure
calibration-free / vanilla attention 관련 부분
```

기록 파일:

```text
phases/phase_03_attention_lifting/notes.md
```

기록 질문:

```text
1. REO에서 query는 무엇을 의미하는가?
2. image feature는 K,V로 어떻게 쓰이는가?
3. projection-first deformable attention과 무엇이 다른가?
4. 내 구조에서는 BEV query가 아니라 3D voxel query로 확장한다는 점을 어떻게 이해할까?
```

## D024 - image token flatten

만들 파일:

```text
phases/phase_03_attention_lifting/src/image_tokens.py
phases/phase_03_attention_lifting/tests/test_image_tokens.py
```

구현:

```python
def image_to_tokens(feat):
    # B x C x H x W -> B x HW x C

def add_2d_sincos_pe(tokens, h, w):
    # token에 2D position encoding 추가
```

검증:

```text
B x 32 x 16 x 32 -> B x 512 x 32
```

## D025 - 3D query positional embedding

만들 파일:

```text
phases/phase_03_attention_lifting/src/query_grid.py
phases/phase_03_attention_lifting/tests/test_query_grid.py
```

구현:

```python
def make_3d_query_grid(shape_xyz):
    # X x Y x Z x 3 normalized coordinate

class QueryEmbedding3D(nn.Module):
    # fixed grid index -> learnable query token
```

검증:

```text
8 x 4 x 2 grid -> 64 query tokens
```

## D026 - TinyLifter

만들 파일:

```text
phases/phase_03_attention_lifting/src/lifter.py
phases/phase_03_attention_lifting/tests/test_lifter.py
```

구현:

```python
class TinyLifter(nn.Module):
    # image -> TinyBackbone -> tokens
    # 3D query embeddings
    # VanillaCrossAttention
    # output: B x C x X x Y x Z
```

검증:

```text
image: B x 3 x 128 x 256
coarse feature: B x C x 8 x 4 x 2
```

## D027 - attention toy overfit

만들 파일:

```text
phases/phase_03_attention_lifting/scripts/train_attention_toy.py
```

구현:

```text
synthetic image에 object 위치를 표시한다.
3D query attention이 object occupancy를 맞추게 한다.
```

완료 기준:

```text
1개 batch에서 loss가 감소한다.
```

## D028 - dummy ray positional encoding

수정 파일:

```text
phases/phase_03_attention_lifting/src/image_tokens.py
```

구현:

```python
def make_dummy_ray_pe(h, w, c):
    # normalized u, v에서 ray-like feature 생성
```

검증:

```text
ray PE shape가 B 없이 HW x C 또는 1 x HW x C로 만들어진다.
```

## D029 - camera id embedding

수정 파일:

```text
phases/phase_03_attention_lifting/src/image_tokens.py
```

구현:

```python
class CameraIDEmbedding(nn.Module):
    # camera index -> embedding
```

검증:

```text
camera 0 token과 camera 1 token이 다른 embedding을 받는다.
```

## D030 - multi-camera token packing

만들 파일:

```text
phases/phase_03_attention_lifting/tests/test_multicam_tokens.py
```

구현:

```python
def pack_multicam_tokens(tokens_per_cam):
    # B x N_cam x HW x C -> B x (N_cam*HW) x C
```

검증:

```text
N_cam=4, HW=512이면 token 수는 2048이다.
```

## D031 - attention memory cost 계산

수정 파일:

```text
phases/phase_03_attention_lifting/notes.md
```

작업:

```text
Nq=1976일 때 attention weight 크기를 계산한다.
Nk가 camera 수와 image feature 해상도에 따라 어떻게 커지는지 기록한다.
```

이해:

```text
왜 spatial attention을 1.2m coarse grid에서 해야 하는가?
```

## D032 - attention map visualization

만들 파일:

```text
phases/phase_03_attention_lifting/scripts/visualize_attention.py
```

구현:

```text
특정 3D query의 attention weight를 image feature grid로 reshape해서 png 저장
```

검증:

```text
attention map png가 생성된다.
```

## D033 - CrossAttentionBlock

수정 파일:

```text
phases/phase_03_attention_lifting/src/attention.py
```

구현:

```python
class CrossAttentionBlock(nn.Module):
    # attention
    # residual
    # LayerNorm
    # feed-forward MLP
```

검증:

```text
input query shape와 output query shape가 동일하다.
```

## D034 - TinyLifter one-batch overfit

수정 파일:

```text
phases/phase_03_attention_lifting/scripts/train_attention_toy.py
```

작업:

```text
CrossAttentionBlock으로 바꾼 뒤 one-batch overfit을 다시 확인한다.
```

완료 기준:

```text
loss가 감소하고 attention output이 NaN이 아니다.
```

## D035 - Phase 03 report

수정 파일:

```text
phases/phase_03_attention_lifting/notes.md
```

기록:

```text
1. 번역 attention과 3D query attention의 차이
2. Q/K/V 각각의 의미
3. attention에서 가장 큰 memory cost
4. REO에서 가져온 아이디어
```

검증:

```bash
pytest phases/phase_03_attention_lifting/tests -q
```

