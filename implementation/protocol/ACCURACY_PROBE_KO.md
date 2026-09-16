# 정확도 개선안의 제한 비교 검증

5,400회 연장 초안은 보류되었으며 실행 동결 생성도 차단한다.
현재 진입점은 `python -m implementation.accuracy_probe_cli`이다.
새 `accuracy-controlled-360-v1` 동결은 원래 실패한 T3 종점의 정확한
run summary hash에서 체크포인트를 결정한다. 원본은 쓰지 않는다.

먼저 read-only `accuracy_diagnostic`에서 KO 입력 train1/train2와 dev1/dev2를
동일 greedy 규칙으로 비교했다. 학습 프롬프트의 중국어 93.33%/95%,
프랑스어 95%/96.67%와 검증 프롬프트 사이의 격차가 확인됐다.
이것은 순수 학습량 부족을 입증하지 않으며 prompt 일반화 문제를 시사한다.

## 사전에 고정한 비교

동일 T3 full state(model/optimizer/scheduler/RNG)를 각각 복원한다.
각 arm은 동일 360 updates × 32 examples, 동일 T3 target/input/mode/wrapper
카운트와 record ID 순서를 쓴다. 기존 3,000회 이후의 추가 단계다.

1. constant: 기존 LR 3e-4, 기존 train1/train2.
2. low_lr: LR만 1e-4로 감소.
3. prompt_mix: LR 1e-4, KO 입력의 약 절반을 짧은/순서 변형 지시문으로 교체.

변형 대상은 record ID 해시로 정하며 오답 ID/점수를 쓰지 않는다.
뜻풀이와 정답, 요청언어는 보존한다. 새 형태는 “{언어}로 답하세요”를
뜻풀이 앞 또는 뒤에 놓는다. ANY에는 언어 지정 없이 “한 단어로 답하세요”를
사용한다. 원래 dev/test 문구와의 완전 일치는 생성 시 차단한다.
이는 dev 격차를 관찰한 뒤의 탐색 설계이며 완전히 독립적인 검증이 아니다.

각 arm의 180/360회 full state를 보존하고 180회에서 실제 디스크 복원을 수행한다.
공통 GPU replay는 연속 10회 대 두 복원 실행의 정확한 일치를 별도 확인한다.
최종 360회에서만 기존 KO/RD/dev readiness와 ANY measurement를 평가한다.
test를 설정 선택에 쓰지 않고, best checkpoint/기준 완화/정답 추가는 없다.

후보 조건: baseline과 constant 양쪽보다 중국어·프랑스어 각각 120관측 중
2개 이상 개선, KO/EN의 각 dev cell은 양쪽 대비 최대 1/60만 하락 허용.
이 조건은 탐색 후보 선별이지 통계적 유의성이나 연구 통과 기준이 아니다.
후속 연구에는 기존 모든 readiness/measurement gate가 여전히 필요하다.
어떤 arm도 자동으로 장기 재학습/H/다른 root/main으로 확대하지 않는다.

## 운영

새 동결, exact-code CPU suite, 새 GPU replay 증거가 모두 필요하다.
GPU2 UUID/프로젝트 잠금/기존 예산 누계와 5 GiB 디스크 비상 여유를 지킨다.
모든 산출물은 현재 jm/work 안에 저장한다. 다른 저장장치/서버 설정은 건드리지 않는다.
새 directory만 허용하며 오류 시 중단하고 완료된 arm 및 체크포인트를 보존한다.
이 제한 비교 CLI는 비정상 중단 후 자동 재시도하지 않는다.
판정과 보고의 원천은 해당 run의 단일 `run_summary.json`이다.
