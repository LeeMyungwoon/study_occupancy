# d001
- notes.md 파일 생성
- scripts/src/tests 디렉토리 생성
- 이번 phase의 목표: occupancy 모델 구현 전에 tensor shape, Conv2d, Conv3d, ConvTranspose3d, grid_sample, voxel center 생성에 익숙해지는 것
- 가상환경 세팅
   - python3 -m venv ~/venv/study_occupancy
   - python -m pip install --upgrade pip setuptools wheel
   - pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
   - pip install numpy scipy matplotlib opencv-python pillow pyyaml tqdm einops pytest ipython tensorboard

