# d001
- notes.md 파일 생성
- scripts/src/tests 디렉토리 생성
- 이번 phase의 목표: occupancy 모델 구현 전에 tensor shape, Conv2d, Conv3d, ConvTranspose3d, grid_sample, voxel center 생성에 익숙해지는 것
- 가상환경 세팅
   - python3 -m venv ~/venv/study_occupancy
   - pip install --upgrade pip
   - python -m pip install --upgrade pip setuptools wheel
   - pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
   - pip install numpy scipy matplotlib opencv-python pillow pyyaml tqdm einops pytest ipython tensorboard
   - pip install tqdm
- 검증
   - find phases final_occnet_v1 -maxdepth 3 -type d | sort 실행 완료
   - .gitkeep 추가 완료
   - ~/venv/study_occupancy/bin/python import 확인: venv imports ok

# d002
- tensor_utils.py 생성
- flatten_hw / unflatten_hw 구현
- flatten_xyz / unflatten_xyz 구현
- B x C x H x W <-> B x HW x C 변환 테스트
- B x C x X x Y x Z <-> B x XYZ x C 변환 테스트
- pytest 실행 결과: 2 passed

# d003
- permute / contiguous / reshape 실험
- pytest 실행 결과: 5 passed
- 이해한 점:
   - permute는 값을 복사하지 않고 축 순서와 stride를 바꿔 tensor를 다르게 보게 한다 -> 실제 메모리 구조를 변경하는것이 아니다.(약간 관점만 바꾼 느낌), non-contiguous 상태가 된다
   - view는 tensor가 contiguous한 memory layout일 때만 안전하게 shape를 바꿀 수 있다
   - reshape는 view보다 유연하고, non-contiguous tensor에서는 필요하면 내부적으로 copy해서 shape를 바꿀 수 있다.

# d004
- TinyImageBackbone 구현
- image_backbone.py 생성
- test_image_backbone.py 생성
- input: B x 3 x 128 x 256
- output: B x 32 x 16 x 32
- Conv2d stride=2를 3번 사용해서 spatial 크기를 128 x 256 -> 16 x 32 로 줄임
- channel은 3 -> 16 -> 32 -> 32 로 변경됨
- pytest 실행 결과: 1 passed
- 이해한 점:
   - stride=2는 H, W를 줄인다
   - backbone은 원본 이미지를 압축해서 attention token 수를 줄인다
   - out_channels는 각 위치의 feature 개수를 바꾼다
   - 기본적인 PyTorch nn사용법, class 구조 익힘
