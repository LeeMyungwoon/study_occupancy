# Phase 10 - Runtime and Final Report

기간: D121-D130  
목표: runtime profiling, ablation, 20 FPS 가능성 평가, final report를 작성한다.

이 phase의 목적은 "모델이 돌아간다"에서 끝나는 것이 아니다.  
20 FPS 목표를 기준으로 어떤 stage가 병목인지, 어떤 branch를 줄여야 하는지, Jetson Orin AGX에 올리기 전에 무엇을 바꿔야 하는지 판단하는 것이다.

20 FPS 기준:

```text
target FPS: 20
frame budget: 50 ms
권장 p50 latency: <= 45 ms
허용 p95 latency: <= 55~60 ms
```

v1 runtime 원칙:

```text
1. 20cm dense feature volume은 만들지 않는다.
2. dense occupancy는 structured sub-voxel head로 만든다.
3. QueryableMLP는 selected K/Q budget 안에서만 실행한다.
4. P2/P3 local image re-query는 v1에서 제외한다.
5. temporal memory는 1.6m BEV memory로 제한한다.
6. profiling은 평균만 보지 않고 p50/p95/max를 같이 본다.
7. PyTorch eager에서 20 FPS가 안 나와도 stage별 병목을 정확히 기록한다.
```

## Runtime Stage 정의

Phase 10에서는 모든 실험이 같은 stage 이름을 사용해야 한다.

```text
image_backbone
vanilla_cross_attention_16m
temporal_memory
deconv_16_to_08
deconv_08_to_04
structured_occupancy_head
surface_head
flow_head
active_mask
quota_topk
sparse_anchor
packed_queryable_mlp
visualization_optional
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
vanilla_cross_attention_16m
temporal_memory
deconv_16_to_08
deconv_08_to_04
structured_occupancy_head
surface_head
flow_head
active_mask
quota_topk
sparse_anchor
packed_queryable_mlp
```

측정 규칙:

```text
1. CUDA 사용 시 torch.cuda.synchronize()를 stage 전후에 넣는다.
2. warmup iteration은 기록하지 않는다.
3. p50, p95, mean, max를 모두 출력한다.
4. batch size, image resolution, K_total, Q_total을 report에 같이 기록한다.
```

출력 예시:

```text
stage                         p50_ms  p95_ms  mean_ms  max_ms
image_backbone                 12.3    14.1    12.6     16.0
vanilla_cross_attention_16m      5.8     6.4     5.9      7.1
...
total                          43.5    51.2    44.1     58.0
```

완료 기준:

```text
stage별 p50/p95 latency를 출력한다.
total latency도 출력한다.
결과를 markdown 파일로 저장할 수 있다.
```

## D122 - K/Q latency table

만들 파일:

```text
phases/phase_10_runtime_final_report/reports/runtime_kq_table.md
```

실험:

```text
K=512, 1024, 2048
average M_i=4, 8, 16
Q_total = K * average M_i
```

기록할 표:

```text
K_total
avg_M
Q_total
active_mask_ms
quota_topk_ms
sparse_anchor_ms
packed_mlp_ms
total_ms
peak_memory_mb
```

판단 기준:

```text
K=2048, avg_M=8이 20 FPS budget 안에 들어오면 v1 기본 후보.
K=2048, avg_M=16이 너무 느리면 10 FPS refinement 또는 planner-triggered mode로 둔다.
K=512에서만 빠르면 active region이 너무 제한적일 수 있으므로 quality와 같이 본다.
```

주의:

```text
K_total과 Q_total을 혼동하지 않는다.
MLP latency는 대부분 Q_total에 비례한다.
Top-K와 anchor gather는 대부분 K_total에 비례한다.
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
1.6m feature memory
0.4m dense feature memory
20cm occ_logits memory
surface output memory
flow output memory
refinement packed query memory
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
20cm dense occupancy logits는 괜찮지만,
20cm dense feature volume은 금지다.
0.4m dense feature가 가장 큰 feature tensor 후보인지 확인한다.
flow를 20cm dense로 만들 경우 memory가 커지는지 확인한다.
```

완료 기준:

```text
가장 큰 tensor top 5가 report에 있다.
peak memory가 config별로 기록되어 있다.
```

## D124 - ablation A0-A3

실험:

```text
A0 occupancy only
A1 + surface
A2 + temporal
A3 + flow
```

만들 파일:

```text
phases/phase_10_runtime_final_report/reports/ablation_a0_a3.md
```

각 실험에서 기록:

```text
config switch
occ_iou
surface_mae
flow_epe
total_latency_p50
total_latency_p95
peak_memory
visualization path
짧은 관찰
```

판단 포인트:

```text
A1:
  surface를 켰을 때 latency 증가가 작은가?
  surface MAE가 의미 있게 나오는가?

A2:
  temporal을 켰을 때 occupancy/flow가 안정되는가?
  latency 증가가 20 FPS budget에 들어오는가?

A3:
  flow head가 memory/latency를 얼마나 추가하는가?
  dynamic toy sequence에서 flow EPE가 줄어드는가?
```

완료 기준:

```text
A0~A3 표가 있다.
각 ablation마다 metric과 latency가 같이 기록된다.
```

## D125 - ablation A4-A6

실험:

```text
A4 + active mask
A5 + sparse anchor
A6 + packed adaptive query
```

만들 파일:

```text
phases/phase_10_runtime_final_report/reports/ablation_a4_a6.md
```

각 실험에서 기록:

```text
K_total
Q_total
selected category count
boundary quality
query loss
latency
memory
visualization path
```

판단 포인트:

```text
A4:
  active mask가 boundary/planner/dynamic/uncertain 영역을 골고루 고르는가?

A5:
  sparse anchor가 QueryableMLP input으로 충분한 context를 제공하는가?

A6:
  packed adaptive query가 quality를 올리면서 budget 안에 남는가?
```

완료 기준:

```text
refinement를 켰을 때 dense occupancy output은 그대로 유지된다.
fine overlay가 별도 output으로 나온다.
K/Q budget을 넘지 않는다.
```

주의:

```text
refinement가 quality를 올리지 못하면 v1에서 K/Q를 줄이거나 10 FPS multi-rate branch로 둔다.
기본 20 FPS perception을 망치면 안 된다.
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
20 FPS 달성 가능성
줄여야 할 channel / K / Q / image resolution
멀티레이트로 내릴 branch
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
  1.6m grid 크기 확인, channel 축소, head 수 축소.

deconv가 느림:
  channel 축소, ConvTranspose3d kernel/stride 재검토.

flow가 느림:
  flow head를 0.4m grid 또는 10 FPS로 제한.

refinement가 느림:
  K_total/Q_total 축소, avg M_i 축소, 10 FPS multi-rate.
```

완료 기준:

```text
어떤 stage를 줄여야 하는지 우선순위가 1,2,3으로 정리되어 있다.
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
2. decoder + structured occupancy head
3. surface head
4. packed QueryableMLP
```

체크할 것:

```text
dynamic shape 사용 여부
grid_sample export 가능 여부
Top-K export 가능 여부
custom op 필요 여부
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
5. how to run shape test
6. how to train tiny
7. how to validate tiny
8. how to visualize
9. how to profile
10. known limitations
11. runtime notes
```

반드시 적을 아키텍처 요약:

```text
camera image features
-> vanilla cross attention at 1.6m
-> ViewFormer-style BEV temporal memory
-> deconv 1.6m to 0.8m
-> deconv 0.8m to 0.4m
-> structured 20cm occupancy head
-> surface / flow / active mask
-> sparse selected QueryableMLP refinement
```

완료 기준:

```text
새로 보는 사람이 README만 보고 test/train/profile 명령을 실행할 수 있다.
```

## D129 - final report

만들 파일:

```text
phases/phase_10_runtime_final_report/reports/final_report.md
```

내용:

```text
1. 최종 구현 개요
2. occupancy output 결과
3. surface output 결과
4. temporal/flow 결과
5. dynamic-resolution refinement 결과
6. metric table
7. latency table
8. memory table
9. ablation summary
10. 20 FPS 가능성 판단
11. 미완료/위험 요소
12. 다음 단계
```

20 FPS 판단 문장 예시:

```text
현재 PyTorch eager 기준 p50은 XXms, p95는 YYms이다.
Jetson Orin AGX TensorRT FP16 최적화 전 기준으로는 20 FPS를 보장하지 못한다/가능성이 있다.
가장 큰 병목은 A, B, C이며, v1에서 줄여야 할 것은 K_total/Q_total/channel이다.
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
TensorRT FP16
Jetson Orin AGX 실제 profiling
P3-only local image re-query v1.5
flow-aware dynamic feature correction
real dataset label pipeline
RoadBEV/FastRSR-style surface supervision 강화
nuScenes/Occ3D/OpenOccupancy benchmark 대응
```

우선순위 예시:

```text
Priority 1:
  runtime path 고정
  TensorRT/ONNX 위험 연산 제거
  Jetson profiling

Priority 2:
  real dataset label pipeline
  surface supervision 강화
  flow target 정리

Priority 3:
  P2/P3 local image re-query
  더 강한 sparse refinement
  larger backbone 실험
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
  ablation_a4_a6.md
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
```
