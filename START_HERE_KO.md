# 새 시작 v4.0.0

## 이 ZIP 하나를 새 폴더에 풀어 사용한다
이 폴더의 `CLAUDE_START_V4_KO.txt`를 새 Claude Code 세션에 입력한다. 옛 프로젝트는 지우거나 이동하지 않아도 된다. 새 weights·code·venv·git·run directory를 사용한다.

**먼저 해결할 외부 조건:** 국립국어원 API 인증키 또는 적법한 원본 스냅샷. 인증키 없이 새 폴더를 만들어도 자료를 받을 수 없다. 공식 인증키 발급 안내:
https://krdict.korean.go.kr/eng/openApi/openApiRegister

키는 Claude 채팅에 붙이지 않고, **사용자가 자신의 로컬/서버 터미널에서** 입력한다. 아래 Bash 명령은 입력값을 화면과 history에 드러내지 않는다. 셸 tracing이 켜져 있으면 먼저 끈다. 같은 터미널에서 Claude Code를 실행해야 환경을 상속받는다.

```bash
cd lexical_freshstart_v4   # ZIP을 푼 상위 폴더에서
set +x
read -rsp 'KRDICT API key: ' KRDICT_API_KEY; printf '\n'
export KRDICT_API_KEY
python3 bootstrap.py
```

키를 아직 발급받지 못했다면 `python3 bootstrap.py`만 실행해 CPU 시험과 차단 상태를 확인할 수 있다. `BLOCKED_CREDENTIALS`와 exit code 10은 GPU 고장이 아니라 사용자 인증이 필요한 상태다. 비밀키를 저장소·스크린샷·로그에 넣지 않는다.

## 달라진 실행 범위
- 기존 실행/지표 충돌 감사는 이번 일에 포함하지 않는다.
- 한국어→영어→중국어→프랑스어와 어원 검증은 유지한다.
- 자료·검수·재현성 gate를 통과하면 **제한 파일럿은 허용**된다. v3의 audit_only/GPU0을 재사용하지 않는다.
- 본시험, 교정정책, 인간 로그 학습, 모델 전이는 비활성이다. 필수 사람이미검수자료는 자동승인하지 않는다.
- 모든 수치/판정은 run별 JSON에서 생성한다. 다른 집계식의 숫자를 같은 Spearman으로 적지 않는다.

## 이미 있는 코드와 Claude가 구현할 코드
| 구분 | 포함 상태 |
|---|---|
| 전달 hash, 무키 preflight, 제한 API 수집기 | 실행 가능. 실제 API 인증 응답은 아직 미검증 |
| 표현 사건/Q/Z/shared 처리, 기본 일정, 준비·측정 계산/보고 | CPU 참조 코드 및 합성 시험 포함 |
| 실제 자료의 의미·어원 검수 | 미완료; 사용자/연구자 검수 필요 |
| 한국어 corpus 확보, tokenizer, full LM trainer/scorer, 실제 GPU resume | Claude가 implementation/에 구현·시험해야 함 |
| 실제 GPU 모델 학습 결과 | 이 ZIP에 없음 |

### 기존 명령 예
```bash
python3 -m freshstart --help
python3 -m freshstart fetch --queries templates/queries.txt --out work/raw/krdict_run01
python3 -m freshstart data-check --concepts work/annotations/concepts.jsonl \
  --etymology work/annotations/etymology_en_fr.jsonl --out work/reports/data_audit.json
python3 -m freshstart schedule-smoke --out work/reports/schedule_smoke.json
```
위 schedule-smoke는 가짜 ID로 일정 수학을 검사하는 CPU 시험이며 실제 학습이 아니다. API 수집은 `COLLECTED_UNREVIEWED`로 끝나며 곧바로 승인자료가 되지 않는다.

## 첫 번째로 받을 보고
`work/reports/run_summary.json`와 그것에서 생성한 `.md`.
최초에는 인증/수집/자료QA/코드구현/재현성의 실제 상태를 보고한다. 실행된 모델이 없으면 점수는 NOT_RUN이다. 최초 모델부터 모든 준비·측정이 통과한 후에만 나머지 roots와 H를 확대한다.

연구계획서 5쪽은 사람이 읽는 요약이고, 명세와 config가 구현 기준이다. 변경은 새 revision과 hash로 남긴다. 이 파일럿이 유의미한 결과나 학회 채택을 보장하지 않는다.

원본 스냅샷 경로가 이미 있다면 `python3 bootstrap.py --snapshot-manifest work/raw/source_manifest.json`으로 존재 여부를 확인할 수 있다. 이것만으로 출처·내용 QA가 승인되는 것은 아니다. preflight 시도는 고유 파일명으로 저장하며 과거 상태를 덮어쓰지 않는다. 재집계 결과는 새로운 --out 경로를 사용한다.
