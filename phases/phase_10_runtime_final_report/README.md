# Phase 10 - Runtime and Final Report

기간: D121-D130  
목표: runtime profiling, ablation, 20 FPS 가능성 평가, final report를 작성한다.

이 phase의 목적은 "모델이 돌아간다"에서 끝나는 것이 아니다.  
Jetson Orin AGX에서 20 FPS에 접근하려면 어떤 stage가 병목인지, 어떤 branch를 줄여야 하는지, 어떤 구조는 v1.5로 미뤄야 하는지 숫자로 판단해야 한다.

## 20 FPS 기준

```text
target FPS: 20
frame budget: 50 ms
권장 p50 latency: <= 45 ms
허용 p95 latency: <= 55~60 ms
```

## v1 runtime 원칙

```text
1. 0.3m dense feature volume은 만들지 않는다.
2. dense feature는 1.2m -> 0.6m까지만 유지한다.
3. 0.6m -> 0.3m은 sparse deconv + recall-first 2단 게이트로만 간다.
4. high-confidence free parent만 drop한다.
5. unknown / uncertain / mixed_surface_risk parent는 keep 또는 ignore다.
6. soft_keep_budget은 latency 목표일 뿐 mandatory recall keep보다 우선하지 않는다.
7. prelim_kept와 최종 6-way pred label을 분리한다.
8. Flow / Sub-Voxel Shape / 3D Semantics는 pred_heavy_mask의 [N_heavy]에서만 실행한다.
9. Queryable MLP는 branch A(pred_heavy_mask) surface_shell_prob만 예측한다.
10. collision_state fallback branch B/C/D를 별도 측정한다.
11. auxiliary depth/free-space evidence head는 observed-free 판정에 필요하므로 runtime에 포함한다.
12. profiling은 평균만 보지 않고 p50/p95/max를 같이 본다.
13. PyTorch eager에서 20 FPS가 안 나와도 stage별 병목을 정확히 기록한다.
```

금지:

```text
dense 0.3m feature volume
dense 0.3m deconvolution
full-resolution 0.3m temporal memory
z-squeeze BEV temporal / temporal attention
P2/P3 local image re-query in v1
surface_shell_prob만 보고 planner free를 확정하는 규칙
```

## Runtime Stage 정의

Phase 10에서는 모든 실험이 같은 stage 이름을 사용해야 한다.

```text
image_backbone
vanilla_cross_attention_12m
dense_deconv_12_to_06
pre_temporal_refine_06m
temporal_3d_06m
o_motion_06_seed
coarse_occupancy_06
aux_depth_free_space
sparse_deconv_06_to_03
gate1_parent_free_drop
gate2_child_score
prelim_kept_build
near_surface_free_shell_tag
occ_fine_6way
pred_heavy_mask_build
flow_readout_heavy
subvoxel_shape_heavy
semantics_heavy
surface_outputs_bev
exposed_face_mask
sparse_anchor_branch_a
packed_queryable_surface_shell
collision_state_fallback
visualization_optional
```

항상 함께 기록할 budget:

```text
N_parent_06
N_child_candidates
N_kept
N_heavy
K_total
Q_branch_A
Q_total
soft_keep_budget
peak_memory_mb
```

## D121 - profiler script

만들 파일:

```text
final_occnet_v1/scripts/profile_runtime.py
final_occnet_v1/src/occnet_v1/profiling.py
```

구현:

```python
class StageTimer:
    def __enter__(self): ...
    def __exit__(self, *args): ...

def profile_model(model, batch, warmup=20, iters=100):
    ...
```

측정 stage:

```text
image_backbone
vanilla_cross_attention_12m
dense_deconv_12_to_06
pre_temporal_refine_06m
temporal_3d_06m
o_motion_06_seed
coarse_occupancy_06
aux_depth_free_space
sparse_deconv_06_to_03
gate1_parent_free_drop
gate2_child_score
prelim_kept_build
near_surface_free_shell_tag
occ_fine_6way
pred_heavy_mask_build
flow_readout_heavy
subvoxel_shape_heavy
semantics_heavy
surface_outputs_bev
exposed_face_mask
sparse_anchor_branch_a
packed_queryable_surface_shell
collision_state_fallback
```

측정 규칙:

```text
1. CUDA 사용 시 torch.cuda.synchronize()를 stage 전후에 넣는다.
2. warmup iteration은 기록하지 않는다.
3. p50, p95, mean, max를 모두 출력한다.
4. batch size, image resolution, N_kept, N_heavy, K_total, Q_total을 report에 같이 기록한다.
5. branch가 꺼져 있으면 "disabled"로 기록하고 total에서 제외한다.
```

출력 예시:

```text
stage                          p50_ms  p95_ms  mean_ms  max_ms
image_backbone                  12.3    14.1    12.6     16.0
vanilla_cross_attention_12m       5.8     6.4     5.9      7.1
...
packed_queryable_surface_shell    2.4     3.1     2.5      4.0
collision_state_fallback          0.2     0.3     0.2      0.4
total                            43.5    51.2    44.1     58.0
```

완료 기준:

```text
stage별 p50/p95 latency를 출력한다.
total latency도 출력한다.
결과를 markdown 파일로 저장할 수 있다.
```

## D122 - N/K/Q latency table

만들 파일:

```text
phases/phase_10_runtime_final_report/reports/runtime_kq_table.md
```

실험:

```text
K_total = 512, 1024, 2048
average M_i = 4, 8, 16
Q_total = K_total * average M_i
N_kept / N_heavy는 실제 forward 결과로 기록
```

기록할 표:

```text
config
N_kept
N_heavy
K_total
avg_M
Q_branch_A
Q_total
gate2_child_score_ms
pred_heavy_mask_ms
sparse_anchor_branch_a_ms
packed_queryable_surface_shell_ms
collision_state_fallback_ms
total_ms
peak_memory_mb
```

판단 기준:

```text
K_total=2048, avg_M=8이 20 FPS budget 안에 들어오면 v1 기본 후보.
K_total=2048, avg_M=16이 너무 느리면 10 FPS queryable 또는 planner-triggered mode로 둔다.
K_total=512에서만 빠르면 surface coverage와 collision_state 품질을 같이 본다.
```

주의:

```text
N_kept, N_heavy, K_total, Q_total을 혼동하지 않는다.
Flow/Shape/Semantics latency는 N_heavy에 주로 비례한다.
Queryable MLP latency는 Q_branch_A 또는 Q_total에 주로 비례한다.
collision_state fallback은 MLP가 아니라 rule/evidence lookup 비용으로 따로 본다.
```

## D123 - memory profiling

만들 파일:

```text
phases/phase_10_runtime_final_report/reports/memory_profile.md
```

구현/작업:

```text
torch.cuda.reset_peak_memory_stats()
torch.cuda.max_memory_allocated()
stage별 주요 tensor shape 기록
```

기록할 항목:

```text
model parameter memory
activation peak memory
input image feature memory
1.2m feature memory
0.6m dense feature memory
0.6m temporal memory
O_motion_0.6 memory
coarse occupancy/visibility/mixed_surface_risk memory
aux depth/free-space evidence memory
0.3m sparse prelim_kept feature memory
near_surface_free_shell_tag memory
6-way pred label memory
pred_heavy_mask memory
flow [N_heavy] memory
subvoxel shape [N_heavy] memory
semantics [N_heavy] memory
Surface Outputs BEV memory
queryable packed query memory
collision_state buffer memory
```

shape/memory 표 예시:

```text
name
shape
dtype
numel
estimated_mb
notes
```

중요 체크:

```text
0.3m sparse occupancy/index/output은 허용된다.
0.3m dense feature volume은 금지다.
0.6m dense feature가 가장 큰 dense 3D feature tensor 후보인지 확인한다.
near-surface free shell이 N_kept를 얼마나 늘리는지 기록한다.
pred_heavy_mask가 N_heavy를 충분히 줄이는지 기록한다.
```

완료 기준:

```text
가장 큰 tensor top 5가 report에 있다.
peak memory가 config별로 기록되어 있다.
dense 0.3m feature volume이 생성되지 않았음을 확인한다.
```

## D124 - ablation A0-A3

만들 파일:

```text
phases/phase_10_runtime_final_report/reports/ablation_a0_a3.md
```

실험:

```text
A0 occupancy only
A1 + mixed_surface_risk + recall-first gate
A2 + auxiliary depth/free-space evidence
A3 + Surface Outputs head
```

각 실험에서 기록:

```text
config switch
occ_iou
occupied_recall
prune_recall
free_shell_precision
collision_state_violation
surface_mae
traversability_error
drop_risk_auc
N_kept
N_heavy
total_latency_p50
total_latency_p95
peak_memory
visualization path
짧은 관찰
```

판단 포인트:

```text
A0:
  coarse/fine occupancy만으로 baseline이 닫히는가?

A1:
  mixed_surface_risk와 recall-first gate가 occupied/surface drop을 줄이는가?
  soft_keep_budget을 조금 넘더라도 recall을 보호하는가?

A2:
  observed-free evidence가 pred_free_shell / collision_state free 판정을 보수적으로 만드는가?
  occluded/hit 뒤쪽을 free로 오판하지 않는가?

A3:
  Surface Outputs latency가 작은가?
  traversability/drop-risk가 occupancy와 별도 가치가 있는가?
```

완료 기준:

```text
A0~A3 표가 있다.
각 ablation마다 metric, latency, memory가 같이 기록된다.
```

## D125 - ablation A4-A7

만들 파일:

```text
phases/phase_10_runtime_final_report/reports/ablation_a4_a7.md
```

실험:

```text
A4 + temporal + O_motion_0.6
A5 + Occupancy Flow [N_heavy] readout
A6 + Sub-Voxel Shape / 3D Semantics on pred_heavy_mask
A7 + Queryable surface_shell_prob + collision_state fallback
```

각 실험에서 기록:

```text
config switch
occ_iou
heavy_recall
flow_epe
dynamic_f1
shape_offset_error
shape_normal_error
semantic_acc
surface_shell_calibration
collision_state_violation
N_kept
N_heavy
Q_branch_A
Q_total
latency_p50
latency_p95
memory
visualization path
```

판단 포인트:

```text
A4:
  temporal과 O_motion_0.6 seed가 dynamic/flow 안정성에 도움이 되는가?
  0.6m 3D temporal latency가 budget 안에 들어오는가?

A5:
  [N_heavy] flow readout이 dense flow보다 충분히 싸게 동작하는가?
  valid_flow_mask에서 flow EPE가 의미 있게 줄어드는가?

A6:
  pred_heavy_mask가 Shape/Semantics 실행 대상을 충분히 줄이는가?
  boundary/surface recall을 해치지 않는가?

A7:
  surface_shell_prob가 surface shell query에 calibration되어 있는가?
  collision_state fallback이 conservative-free violation을 줄이는가?
  branch A MLP와 branch B/C/D fallback latency가 분리되어 있는가?
```

완료 기준:

```text
A4~A7 표가 있다.
N_heavy와 Q_total budget을 넘는 설정이 명확히 표시되어 있다.
기본 20 FPS perception을 망치는 branch는 v1.5 또는 multi-rate 후보로 분리한다.
```

## D126 - 20 FPS blocker 분석

만들 파일:

```text
phases/phase_10_runtime_final_report/reports/20fps_blockers.md
```

기록:

```text
가장 느린 stage top 3
가장 memory 큰 tensor top 3
p95 latency를 키우는 stage
N_kept / N_heavy / Q_total이 예산을 넘는 조건
fallback으로 품질을 낮춘 frame 수
20 FPS 달성 가능성
줄여야 할 channel / image resolution / N_kept / N_heavy / K_total / Q_total
multi-rate로 내릴 branch
```

판단 규칙:

```text
total p50 <= 45ms and p95 <= 60ms:
  20 FPS 후보로 유지.

total p50 <= 50ms but p95 > 60ms:
  jitter/peak 원인을 줄인다.

total p50 > 50ms:
  branch 축소 또는 multi-rate 필요.
```

가능한 조치:

```text
image_backbone이 느림:
  input resolution 축소, backbone 경량화, TensorRT 우선순위.

cross_attention이 느림:
  1.2m grid 크기 확인, channel 축소, head 수 축소.

temporal이 느림:
  0.6m 3D -> 1.2m 3D fallback (z 유지), history N 축소.

sparse_deconv/gate가 느림:
  sparse kernel 최적화, channel 축소, mixed_surface_risk threshold 조정.
  단 high-confidence free only drop 원칙은 유지.

N_kept가 너무 큼:
  gate② calibration, category quota, shell-tag supervision 강화.
  near-surface free shell을 완전히 끄기 전에 observed-free precision부터 확인.

N_heavy가 너무 큼:
  pred_boundary calibration, exposed-face mask, heavy quota 조정.
  GT surface/boundary recall 저하는 허용하지 않는다.

queryable이 느림:
  K_total/Q_total 축소, avg M_i 축소, 10 FPS multi-rate.
  collision_state fallback은 유지해 planner 안전성을 보존한다.
```

완료 기준:

```text
어떤 stage를 줄여야 하는지 우선순위 1,2,3으로 정리되어 있다.
20 FPS 실패 원인이 "느림"이 아니라 구체적인 stage와 budget으로 설명된다.
```

## D127 - ONNX export 준비

만들 파일:

```text
final_occnet_v1/scripts/export_onnx.py
phases/phase_10_runtime_final_report/reports/onnx_export_notes.md
```

목표:

```text
전체 export가 어려우면 부분 export라도 한다.
우선순위:
1. image backbone
2. dense path: 1.2m lifting output -> 0.6m deconv -> coarse occupancy
3. Surface Outputs Head
4. auxiliary depth/free-space evidence head
5. O_motion_0.6 seed
6. packed Queryable surface_shell_prob MLP
```

체크할 것:

```text
dynamic shape 사용 여부
grid_sample export 가능 여부
Top-K / scatter / gather export 가능 여부
sparse deconv custom op 필요 여부
collision_state fallback rule을 ONNX 밖에서 처리할지 여부
TensorRT에서 지원 안 되는 연산
```

완료 기준:

```text
최소 1개 submodule ONNX export 성공.
실패한 module은 실패 이유를 report에 기록.
```

주의:

```text
ONNX export가 실패해도 Phase 10이 실패한 것은 아니다.
목적은 Jetson/TensorRT로 가기 전에 어떤 연산이 위험한지 찾는 것이다.
```

## D128 - final README

만들 파일:

```text
final_occnet_v1/README.md
```

내용:

```text
1. model overview
2. architecture diagram text
3. expected input/output
4. config 설명
5. prelim_kept vs 6-way pred label 용어표
6. pred_heavy_mask / [N_heavy] 실행 범위
7. surface_shell_prob vs collision_state 용어표
8. observed-free evidence와 pred_free_shell 조건
9. how to run shape test
10. how to train tiny
11. how to validate tiny
12. how to visualize
13. how to profile
14. known limitations
15. runtime notes
```

반드시 적을 아키텍처 요약:

```text
camera image features
-> vanilla cross attention at 1.2m
-> dense deconv 1.2m to 0.6m
-> 0.6m pre-temporal refinement
-> 0.6m 3D temporal (align+concat+3D residual conv, z 유지)
-> O_motion_0.6 coarse seed
-> coarse dense occupancy/visibility/mixed_surface_risk @ 0.6m
-> auxiliary depth/free-space evidence
-> sparse deconv + recall-first 2단 게이트 prune
-> prelim_kept + near_surface_free_shell_tag
-> O_occ_fine on [N_kept] -> 6-way pred label
-> pred_heavy_mask
-> Flow / Sub-Voxel Shape / 3D Semantics on [N_heavy]
-> Surface Outputs Head on 0.6m dense feature
-> exposed-face mask
-> Queryable branch A surface_shell_prob
-> collision_state fallback branch B/C/D
```

완료 기준:

```text
새로 보는 사람이 README만 보고 test/train/profile 명령을 실행할 수 있다.
최신 architecture 용어와 다른 옛 용어가 남아 있지 않다.
```

## D129 - final report

만들 파일:

```text
phases/phase_10_runtime_final_report/reports/final_report.md
```

내용:

```text
1. 최종 구현 개요
2. occupancy output 결과: 0.6m coarse + 0.3m sparse fine
3. prelim_kept / 6-way pred label / pred_heavy_mask 결과
4. auxiliary depth/free-space evidence 결과
5. Surface Outputs 결과
6. temporal / O_motion_0.6 / flow 결과
7. Sub-Voxel Shape / Semantics 결과
8. Queryable surface_shell_prob + collision_state fallback 결과
9. metric table
10. latency table
11. memory table
12. ablation summary
13. N_kept / N_heavy / Q_total budget
14. prune recall / heavy recall
15. collision_state conservative-free violation
16. 20 FPS 가능성 판단
17. 미완료/위험 요소
18. 다음 단계
```

20 FPS 판단 문장 예시:

```text
현재 PyTorch eager 기준 p50은 XXms, p95는 YYms이다.
Jetson Orin AGX TensorRT FP16 최적화 전 기준으로는 20 FPS를 보장하지 못한다/가능성이 있다.
가장 큰 병목은 A, B, C이며, v1에서 줄여야 할 것은 channel / N_heavy / Q_total이다.
```

완료 기준:

```text
final_report.md에 metric, latency, memory, ablation이 모두 있다.
주관적 느낌이 아니라 숫자로 판단한다.
```

## D130 - next 3-month plan

만들 파일:

```text
phases/phase_10_runtime_final_report/reports/next_3_months.md
```

후보:

```text
TensorRT FP16 + sparse conv 배포 최적화
  - sparse 3D conv(spconv/torchsparse)의 Orin custom kernel/plugin 검토
  - 막히면 0.3m masked dense ROI/frustum crop fallback 검토
  - INT8 검토

temporal v1.5 강화
  - 장기 누적 기억(Tesla Temporal Context, GRU/EMA식) 추가
  - 0.3m residual flow head 실험
  - full 0.3m temporal memory는 계속 금지

prune v1.5 강화
  - v1의 category quota / shell-tag supervision 유지
  - per-class quota 비율 튜닝
  - hard-negative mining
  - active learning 보강

deformable image re-query 실험
  - P3, 1 round만 선택 실험
  - v1 기본 runtime에는 넣지 않는다.

flow-aware dynamic feature correction

Gaussian refinement 연구 트랙
  - 현재 v1 architecture에는 없음
  - Queryable branch A surface-shell MLP와 ablation 비교용으로만 검토

camera-only GT auto-labeling pipeline
  - metric reconstruction
  - dynamic object separation
  - ray casting free/unknown
  - continuous SDF / surface-shell supervision

Basalt VIO 연동
  - v1: pose / gravity alignment, landmark sparse depth supervision
  - v1.5: landmark input injection + dropout, outlier dynamic hint,
    landmark ray free-space evidence
```

우선순위 예시:

```text
Priority 1:
  runtime path 고정
  TensorRT/ONNX 위험 연산 제거
  Jetson profiling
  N_kept / N_heavy / Q_total budget 안정화

Priority 2:
  real dataset label pipeline
  observed-free evidence 품질 개선
  surface supervision 강화
  flow target 정리

Priority 3:
  P3 local image re-query
  stronger sparse refinement
  larger backbone 실험
  Gaussian refinement 비교
```

완료 기준:

```text
다음 3개월에 할 일을 우선순위 1, 2, 3으로 정리한다.
각 우선순위에는 "왜 지금 해야 하는지"와 "완료 기준"이 있다.
```

## Phase 10 최종 체크리스트

```text
README:
  final_occnet_v1/README.md 존재

reports:
  runtime_kq_table.md
  memory_profile.md
  ablation_a0_a3.md
  ablation_a4_a7.md
  20fps_blockers.md
  onnx_export_notes.md
  final_report.md
  next_3_months.md

scripts:
  profile_runtime.py
  export_onnx.py

핵심 판단:
  20 FPS 가능성
  가장 큰 병목
  v1에서 유지할 branch
  v1.5로 미룰 branch
  N_kept / N_heavy / Q_total budget
  collision_state conservative-free violation
```
