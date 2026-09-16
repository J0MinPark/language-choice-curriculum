# v4.1.1 의미·표현 동결 절차

`implementation.src.semantic_annotation_freeze`는 원자료 재검증, 검수 연결표,
명시적 승인 입력, annotation freeze 게시·감사를 제공한다. 기존 v4.0 동결기를
변경하거나 이전 동결 파일을 재해석하지 않는다. 이 모듈의 동결 결과는 아직
trainer/tokenizer의 입력 어댑터에 연결되지 않았다.

## 원자료 연결

정정 artifact와 v4.1.1 개정 해시를 검증하고, 그 artifact가 지시하는 고정
selection plan의 source manifest를 재검증한다. 실제 KRDICT XML을 다시 파싱해
candidate hash, source option, 네 언어 뜻풀이, exact span과 canonical answer를
대조한다. 38번 EN은 원문 `hen's egg`의 `[6:9]`를 선택하며 OEWN 고정 근거를
추가 연결한다. 공유 문자열은 membership 목록으로 보존한다.

## 사용자 확인 범위

현재 확인용 파일:

- `work/annotations/semantic_review_v4_1_1_jm02/review_summary.md`
- `work/annotations/semantic_review_v4_1_1_jm02/review_packet.json`
- `work/annotations/semantic_review_v4_1_1_jm02/approval_template.json`

기존 jm02 검토와 egg 정정 판단은 기록되어 있다. v4.1 §1의
“production human evidence ... 완성된 review CSV 또는 freeze로 자동 승격하지
않는다”는 규칙에 따라, 새 packet에 결속된 정식 승인 기록을 별도로 받는다.
이 확인은 어원이나 추가 60개 개념의 검수가 아니라, 표시된 60개 개념의 기존
의미·표현 판단을 원자료 근거와 연결해 정식 동결에 사용하는 범위이다.

소프트웨어는 승인된 packet SHA/bytes, 60개 review subject hash, jm02, 날짜,
의미·표현·출처 연결 확인을 요구한다. 신원 인증은 주장하지 않는다. template의
PENDING/빈 reviewer/false 확인 필드는 자동 승인으로 채우지 않는다. 사용자가
이 정확한 범위를 승인하면 그 결정을 새 파일에 기록할 수 있다. AI가 자체
검수한 것을 인간 검수로 기록해서는 안 된다.

## 실행

```bash
.venv/bin/python -m implementation.src.semantic_annotation_freeze prepare \
  --correction work/annotations/registered_expression_correction_v4_1_1_jm02/selection_correction.json \
  --output-dir work/annotations/NEW_UNIQUE_REVIEW_DIRECTORY

.venv/bin/python -m implementation.src.semantic_annotation_freeze freeze \
  --packet work/annotations/semantic_review_v4_1_1_jm02/review_packet.json \
  --approval work/annotations/EXPLICIT_APPROVAL_RECORD.json \
  --out work/freeze/NEW_UNIQUE_SEMANTIC_FREEZE.json

.venv/bin/python -m implementation.src.semantic_annotation_freeze audit \
  --freeze work/freeze/NEW_UNIQUE_SEMANTIC_FREEZE.json
```

freeze JSON은 write-once/read-only로 생성된다. 감사는 packet과 approval의 바이트
해시, 원자료 재파싱, 승인 범위를 다시 확인하며 내용만 바꾸고 재해시한 결과도
거부한다. 어원과 family/synonym 판단은 이 annotation gate에 포함하지 않는다.

annotation gate 통과는 학습 허가 완료가 아니다. tokenizer/feature 어댑터와
동결, experiment freeze, CPU/GPU replay, production pilot runner 연결이 남는다.
이 절차만으로 GPU를 시작하거나 `main_enabled`를 변경하지 않는다.
