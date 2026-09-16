# v4.1.2 입력·답안 경계 교정

사용자 jm02의 승인에 따라 결과 관측 전 질문 형식을 교정한다.
`AMENDMENT_V4_1_2_PROMPT_BOUNDARY.json`이 이전/신규 평가 계획과 승인 범위를
고정한다. 엔진 JSON schema `4.0.0-r1`은 유지하고 연구 절차 revision은 4.1.2로
기록한다. 기존 파일과 4.1.1 의미·표현 annotation freeze는 보존한다.

## 변경과 확인

한국어·영어·프랑스어의 6개 wrapper × 2개 요청 방식, 총 36개 template 끝의
ASCII 공백만 제거한다. 중국어 template 12개는 그대로다. 뜻풀이, 정답,
언어 요청 문구, 내부 공백, 답 끝 newline, 데이터, tokenizer weights, 모델,
빈도·일정·readiness·measurement 기준은 바꾸지 않는다. 학습/개발/test 모두
같은 규칙을 사용하고 모든 root/branch에 동일하게 적용한다.

기존 Byte BPE는 prefix 끝의 공백을 답 첫 글자와 합쳐 별도 토큰을 만들었다.
예: prefix의 `Ġ`가 `egg`를 붙인 전체 입력에서는 `Ġe`가 되어 prefix token
동일성 조건이 깨졌다. 수정 후 전체 7,200개 조건에서 token prefix가 안정적이고
답안 사건이 중복 없이 prefix-free임을 검사한다. 단어 선택 확률 자체가 이전과
동일하다는 주장은 하지 않는다. 이는 고정할 입력 형식의 전향적 교정이다.

실제 검증 결과는 다음 write-once 파일에 기록된다.

- `work/reports/semantic_tokenizer_v4_1_2_jm02/gate.json`
- `work/reports/semantic_tokenizer_v4_1_2_jm02/token_boundary_audit.json`
- `work/reports/semantic_tokenizer_v4_1_2_jm02/corpus_reverification.json`

tokenizer는 원래 한국어 코퍼스 및 tokenizer policy와의 연결을 검증한다.
lexical boundary 감사만 해시로 고정한 신규 평가 계획을 사용한다. 임의의
override 파일이나 다른 tokenizer 부모 계획은 허용하지 않는다.

## 실행

```bash
.venv/bin/python -m implementation.src.semantic_tokenizer_gate \
  --annotation-freeze work/freeze/semantic_v4_1_1_jm02/annotation_freeze.json \
  --tokenizer-manifest work/tokenizer/wiki40b_ko_bpe_v4r1/tokenizer_manifest.json \
  --evaluation-plan implementation/config/evaluation_plan_v4_1_2.json \
  --output-dir work/reports/NEW_UNIQUE_TOKENIZER_GATE_DIRECTORY
```

실패 시 exit code 2이며 자동 다음 단계 진입은 금지한다. 통과해도 이 명령은
GPU를 시작하지 않는다. 현재 기존 `prepare_data.verify_experiment_freeze` 및
`train.run_gpu2_replay`는 이전 annotation/evaluation 경로에 결속되어 있으며,
`pilot.run_production_pilot`의 production orchestration은 `NOT_IMPLEMENTED`다.
새 연구 버전의 experiment freeze와 실행기 연결을 구현·시험한 뒤 GPU replay와
첫 root를 진행해야 한다. 기존 실행 경로에 신규 artifact 이름만 바꾸어 넣지 않는다.
