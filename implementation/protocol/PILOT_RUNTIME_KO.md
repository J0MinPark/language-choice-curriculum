# GPU2 파일럿 실행기

`python -m implementation.pilot_cli`가 의미 중심 v4.1.2 실행 진입점이다.
기존 `pilot.run_production_pilot`은 옛 v4.0 상태 형식의 호환용 차단기로 보존한다.
이 문서는 구현 설명이며 실행 성공이나 연구 결과를 주장하지 않는다.

## 실행 범위

새 `pilot-execution-freeze-v1`은 승인된 동결/egg/공백 정정/모델/학습량을
그대로 이어받는다. 원자료 해시를 학습 레코드까지 전달하며, 기존에 정의한
개념 ID 정렬·인접 쌍 규칙으로 결과 이전 B36 pairing을 고정한다.
어원은 필수가 아니다. main은 계속 금지한다.

독립 root 4101~4104의 INIT→CORPUS→T0→T1→T2→T3를 순서대로 실행한다.
첫 root의 T3 readiness/measurement가 실패하면 다른 root를 시작하지 않는다.
후속 root의 실패도 중단한다. 모두 통과한 경우에만 각 T3에서 한 번의
H_BASE LR 전환 후 동일 전체 상태를 A/B로 복원한다. 학습 자료와
개념별 노출 순서는 기존 schedule/records 검증기를 사용한다.

2,000,000 코퍼스 토큰 전체(마지막 부분 배치 포함), 각 lexical 3,000회,
H 각 300회가 고정 종료점이다. 200회 간격 KO/RD/dev 진단과 단계 종점의
활성 입력언어 RD dev/test 평가를 저장한다. 미학습 언어는 학습/요청에
넣지 않지만 확률 후보 집합은 항상 4언어이다. CLOZE는 이번 실행기에
포함되지 않은 보조분석이며 NOT_RUN이다. B36 대비는 개념별 h/r을
보고하며 main predictor나 어원 분석을 실행하지 않는다.

## 중단·복원과 서버 실행

모든 GPU 작업은 기존 물리 GPU2 UUID 확인, 프로젝트 독점 잠금,
예산 원장 아래에서 수행한다. 실행 전체에도 별도 중복 실행 잠금이 있다.
30분 GPU 예약 구간마다 필요하면 안전한 경계에서 저장·검증 후 다시
예산을 예약한다. 총 24 GPUh/각 root 6h 한도와 기존 사용량을 보존한다.
비정상 프로세스 종료 시 미정산 예약은 보수적으로 전량 계상한다.

모델/optimizer/moments/scheduler/모든 RNG/loader cursor/lineage를 저장한다.
재개에는 동일 코드·데이터·CPU 및 GPU 검수 증거와 `--resume`가 필요하다.
SIGTERM/SIGINT는 optimizer 경계에서 저장하거나 이미 저장된 평가 전
체크포인트를 보존한 뒤 종료한다. SIGKILL/전원 차단은 마지막 완료된
체크포인트 이후의 작업을 잃을 수 있다. 불완전 디렉터리를 자동 채택하지 않는다.

`--launch`는 nohup + 새 서버 session + 파일 로그 + 닫힌 표준입력으로
실행한다. 접속 PC 종료와 SSH 종료에 종속되지 않는다. 서버 전원 종료를
견디거나 서버 재부팅 시 자동 재시작하는 서비스는 아니다.
오류가 나면 자동으로 기준을 낮추거나 무한 재시도하지 않는다.

`--pause-after 4101:T0:200`은 실제 종료점을 변경하지 않는 운영상 일시정지다.
그 상태를 T0 완료로 표시하지 않는다. 준비 시험에 이 옵션을 사용할 수 있다.

## 보존과 보고

run 디렉터리의 `journal/000000.json`부터 이어지는 write-once 해시 체인을
재구성해 다음 단계를 결정한다. `run_summary_*.json`은 그 시점의 단일
보고 원천이고, scores/records/trace/checkpoint의 파일 해시를 포함한다.
콘솔 문구를 PASS 증거로 사용하지 않는다.

각 단계 종점 체크포인트는 모두 보존한다. 주기적 중간 전체 상태는 최신
두 개를 보존하고, 후속 체크포인트 및 평가가 완료된 뒤 이번 실행이 만든
나머지 중간 state.pt만 제거한다. 해당 manifest, commit marker, 점수,
trace, 해시는 모두 남기고 STATE_PRUNED 이벤트를 기록한다. 제거된
중간 가중치 자체는 복구본이 없으며 최신 상태/단계 종점으로 재개한다.
기존 동결/과거 실행/원자료는 삭제하지 않는다. 디스크 5 GiB 비상 여유를
침범할 가능성이 있으면 학습 업데이트 전에 중단한다. 모든 root/branch
실행 가능 여부는 실제 속도·준비 기준·남은 디스크·GPU 예산에 달려 있다.

## 명령

프로젝트 루트에서 실행한다. 경로는 실제 검수 산출물로 지정한다.

```sh
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 .venv/bin/python -m implementation.pilot_cli \
  --freeze work/freeze/PILOT/experiment_freeze.json \
  --implementation-manifest work/integrity/RUN/implementation_manifest.json \
  --cpu-checks work/reports/CPU/cpu_checks.json \
  --replay work/reports/REPLAY.json \
  --run-directory work/runs/PILOT --launch
```

재개는 같은 인자에 `--resume`를 추가한다. PID·서버명·로그 위치는
`launch_*.json`에 기록한다. STARTING은 검수 또는 학습 성공이 아니다.
