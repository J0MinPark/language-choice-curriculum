# GPU 메모리 검증 정정

2026-09-15: 사용자 승인에 따라 장치 검증의 메모리 비교 기준을 정정한다.
연구 설계, 모델, GPU 지정, 예산, 동결 자료에는 변경이 없다.

물리 GPU 2의 UUID는 `GPU-de18b86c-419a-795a-0667-c7ae03bf800f`이며,
NVIDIA FB total 97,887 MiB, reserved 638 MiB, CUDA total 97,249 MiB를
확인했다. 차이는 정확히 예약 메모리와 일치했다. 첫 재현성 실행은
`BLOCKED_GPU2_PROPERTY_MISMATCH`로 학습 전에 중단됐고 예산 원장에
`FAILED_BEFORE_TRAINING`으로 0.6505153956823051초가 기록됐다.

NVIDIA의 예약 메모리 정의는 드라이버 또는 펌웨어의 시스템용 예약량이다:
https://docs.nvidia.com/deploy/nvml-api/structnvmlMemory__v2__t.html

검증은 이미 식별된 UUID를 직접 질의해 `memory.total - memory.reserved`와
CUDA total을 비교한다. 기존 1 MiB 반올림 허용량은 늘리지 않는다.
빈 결과, 다중 행, 다른 UUID, 변경된 total, 숫자가 아닌 예약량, 음수 또는
total 이상 예약량은 차단한다. free/used 값으로 총량을 추측하지 않는다.
장치 마스크, 물리 UUID, 프로세스 귀속, 독점 사용, 이름, BF16 검증은 유지한다.
예약량 보고가 지원되지 않아도 무시하거나 CPU/다른 GPU로 전환하지 않는다.
