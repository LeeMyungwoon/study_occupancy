# Phase 01 - PyTorch Primitives

기간: D001-D010  
목표: occupancy 모델을 만들기 전에 tensor shape, Conv2d, Conv3d, ConvTranspose3d, `grid_sample`, voxel center 생성에 익숙해진다.

이 phase에서 중요한 것은 "멋진 모델"이 아니라 shape를 정확히 다루는 감각이다. 이후 모든 phase에서 여기서 만든 함수와 테스트 습관을 반복한다.

## 완료 기준

```text
pytest phases/phase_01_pytorch_primitives/tests -q
```

위 명령이 통과해야 한다.

## D001 - phase 시작과 notes 작성

목표:

```text
phase_01 폴더가 정상인지 확인하고, 매일 기록할 notes.md를 만든다.
```

만들 파일:

```text
phases/phase_01_pytorch_primitives/notes.md
```

작업:

```text
1. src, tests, scripts 폴더가 있는지 확인한다.
2. notes.md를 만들고 아래 내용을 적는다.
   - D001 시작
   - 오늘 확인한 폴더 구조
   - 이번 phase의 목표
```

검증:

```bash
find phases/phase_01_pytorch_primitives -maxdepth 2 -type d | sort
```

완료 기준:

```text
notes.md에 D001 기록이 있다.
```

## D002 - tensor flatten / unflatten

목표:

```text
CNN feature map과 token 표현 사이를 오갈 수 있게 한다.
```

만들 파일:

```text
phases/phase_01_pytorch_primitives/src/tensor_utils.py
phases/phase_01_pytorch_primitives/tests/test_tensor_utils.py
```

구현:

```python
def flatten_hw(x):
    # B x C x H x W -> B x (H*W) x C

def unflatten_hw(tokens, h, w):
    # B x (H*W) x C -> B x C x H x W

def flatten_xyz(x):
    # B x C x X x Y x Z -> B x (X*Y*Z) x C

def unflatten_xyz(tokens, x_size, y_size, z_size):
    # B x (X*Y*Z) x C -> B x C x X x Y x Z
```

테스트:

```text
1. random tensor를 flatten 후 unflatten했을 때 원래 shape와 같은지 확인한다.
2. 값도 동일한지 torch.allclose로 확인한다.
3. 2D와 3D 모두 테스트한다.
```

검증:

```bash
pytest phases/phase_01_pytorch_primitives/tests/test_tensor_utils.py -q
```

이해할 것:

```text
attention에서는 image feature가 token이 된다.
3D voxel feature도 token으로 바꿔 attention에 넣을 수 있다.
```

## D003 - permute, contiguous, reshape 실험

목표:

```text
PyTorch에서 shape 변환이 왜 자주 깨지는지 이해한다.
```

수정 파일:

```text
phases/phase_01_pytorch_primitives/tests/test_tensor_utils.py
phases/phase_01_pytorch_primitives/notes.md
```

작업:

```text
1. x.permute(...).view(...)가 실패하는 예제를 만든다.
2. x.permute(...).contiguous().view(...)가 성공하는 예제를 만든다.
3. reshape가 view보다 유연하게 동작하는 예제를 만든다.
4. notes.md에 contiguous가 필요한 이유를 3줄로 정리한다.
```

검증:

```bash
pytest phases/phase_01_pytorch_primitives/tests/test_tensor_utils.py -q
```

완료 기준:

```text
permute 후 memory layout이 바뀐다는 것을 설명할 수 있다.
```

## D004 - Tiny Conv2d backbone

목표:

```text
이미지를 작은 feature map으로 줄이는 backbone을 직접 만든다.
```

만들 파일:

```text
phases/phase_01_pytorch_primitives/src/image_backbone.py
phases/phase_01_pytorch_primitives/tests/test_image_backbone.py
```

구현:

```python
class TinyImageBackbone(nn.Module):
    # input:  B x 3 x 128 x 256
    # output: B x C x 16 x 32
```

권장 구조:

```text
Conv2d stride 2
ReLU
Conv2d stride 2
ReLU
Conv2d stride 2
ReLU
```

테스트:

```text
B=2, C=32 기준 output shape가 B x 32 x 16 x 32인지 확인한다.
```

검증:

```bash
pytest phases/phase_01_pytorch_primitives/tests/test_image_backbone.py -q
```

이해할 것:

```text
원본 이미지를 바로 attention에 넣으면 token 수가 너무 많다.
backbone은 이미지를 압축해서 token 수를 줄인다.
```

## D005 - Conv3d block

목표:

```text
3D voxel feature를 처리하는 기본 Conv3d block을 만든다.
```

만들 파일:

```text
phases/phase_01_pytorch_primitives/src/voxel_blocks.py
phases/phase_01_pytorch_primitives/tests/test_voxel_blocks.py
```

구현:

```python
class Conv3DBlock(nn.Module):
    # Conv3d + normalization + ReLU
```

테스트:

```text
input:
  B x 16 x 8 x 4 x 2

output:
  B x 32 x 8 x 4 x 2
```

검증:

```bash
pytest phases/phase_01_pytorch_primitives/tests/test_voxel_blocks.py -q
```

이해할 것:

```text
Conv2d는 이미지 평면을 본다.
Conv3d는 X,Y,Z voxel 공간을 본다.
```

## D006 - ConvTranspose3d upsample

목표:

```text
coarse 3D feature를 더 촘촘한 3D feature로 키우는 deconv block을 만든다.
```

수정 파일:

```text
phases/phase_01_pytorch_primitives/src/voxel_blocks.py
phases/phase_01_pytorch_primitives/tests/test_voxel_blocks.py
```

구현:

```python
class Deconv3DBlock(nn.Module):
    # ConvTranspose3d stride=2
```

테스트:

```text
input:
  B x C x 8 x 4 x 2

output:
  B x C2 x 16 x 8 x 4
```

검증:

```bash
pytest phases/phase_01_pytorch_primitives/tests/test_voxel_blocks.py -q
```

이해할 것:

```text
deconv는 정보를 새로 마법처럼 만드는 것이 아니다.
coarse feature를 더 fine한 grid로 펼치고 학습된 filter로 보간/복원한다.
```

## D007 - 2D grid_sample

목표:

```text
feature map에서 임의 pixel 위치의 feature를 뽑는다.
```

만들 파일:

```text
phases/phase_01_pytorch_primitives/src/grid_sample_utils.py
phases/phase_01_pytorch_primitives/tests/test_grid_sample_utils.py
```

구현:

```python
def pixel_to_grid(pixel_xy, width, height, align_corners=False):
    # pixel coordinate -> normalized grid coordinate [-1, 1]

def sample_2d_feature(feat, pixel_xy):
    # feat: B x C x H x W
    # pixel_xy: B x Q x 2
    # output: B x Q x C
```

테스트:

```text
값이 좌표와 같은 known feature map을 만든다.
특정 pixel을 sample했을 때 예상 값이 나오는지 확인한다.
```

검증:

```bash
pytest phases/phase_01_pytorch_primitives/tests/test_grid_sample_utils.py -q
```

## D008 - 3D trilinear sampling

목표:

```text
3D voxel feature에서 임의 3D point의 feature를 뽑는다.
```

수정 파일:

```text
phases/phase_01_pytorch_primitives/src/grid_sample_utils.py
phases/phase_01_pytorch_primitives/tests/test_grid_sample_utils.py
```

구현:

```python
def xyz_to_grid(points_xyz, x_range, y_range, z_range, shape_xyz):
    # ego/world xyz -> normalized 3D grid coordinate

def sample_3d_feature(feat_3d, points_xyz):
    # feat_3d: B x C x X x Y x Z
    # points_xyz: B x Q x 3
    # output: B x Q x C
```

테스트:

```text
B=1, Q=5로 시작한다.
출력 shape가 B x Q x C인지 확인한다.
```

검증:

```bash
pytest phases/phase_01_pytorch_primitives/tests/test_grid_sample_utils.py -q
```

이해할 것:

```text
나중에 Queryable MLP가 selected point에서 feature를 얻을 때 이 기능이 필요하다.
```

## D009 - voxel center 생성

목표:

```text
occupancy grid의 각 voxel center 좌표를 만든다.
```

만들 파일:

```text
phases/phase_01_pytorch_primitives/src/voxel_grid.py
phases/phase_01_pytorch_primitives/tests/test_voxel_grid.py
```

구현:

```python
def make_voxel_centers(x_range, y_range, z_range, voxel_size):
    # output: X x Y x Z x 3
```

테스트:

```text
x_range = [-1, 1], voxel_size = 1이면 center는 -0.5, 0.5가 되어야 한다.
```

검증:

```bash
pytest phases/phase_01_pytorch_primitives/tests/test_voxel_grid.py -q
```

## D010 - Phase 01 report

목표:

```text
Phase 01에서 만든 도구가 모두 동작하는지 확인하고 정리한다.
```

수정 파일:

```text
phases/phase_01_pytorch_primitives/notes.md
```

작업:

```text
1. 만든 파일 목록 정리
2. 가장 헷갈린 shape 3개 기록
3. grid_sample에서 align_corners가 헷갈린 점 기록
4. pytest 전체 실행
```

검증:

```bash
pytest phases/phase_01_pytorch_primitives/tests -q
```

완료 기준:

```text
Phase 02로 넘어가도 될 만큼 tensor shape와 sampling 함수가 안정적이다.
```

