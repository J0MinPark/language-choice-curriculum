# 연구·구현 기준 v4.1.0 | 학습 이력 중심 전향 개정

2026-09-15 · project_id `lexical-freshstart-4lang-history-v4_1`

상태: `APPROVED_DESIGN_PENDING_IMPLEMENTATION`

이 문서는 원본 배포의 `spec/RESEARCH_SPEC_V4_KO.md`와 `spec/pilot.json`을
수정하지 않고 적용하는 v4.1 전향 개정이다. 원본에서 이 문서가 명시적으로
대체하지 않은 보안, 예산, 모델, 학습, 점수화, 재현성 및 실행 권한 조항은
그대로 유지된다. 충돌 시 v4.1 문서와 `implementation/config/pilot_v4_1.json`이
v4.1 실행에 한해 우선한다. 두 revision의 artifact나 결과를 같은 run으로
혼합하지 않는다.

이번 변경 전에 실제 모델 학습 결과나 평가 결과는 관측되지 않았다. 이 진술이
거짓임이 확인되면 실행을 중단하고 변경 시점을 다시 분류한다. 이 개정은 사용자
식별자 `jm02`의 설계 승인을 기록하지만 신원 인증, 전자서명 또는 독립적인 인간
검수 완료를 주장하지 않는다.

## 0. 제목, 질문과 범위

연구 제목:

> **Learning History Shapes Lexical Language Choice in Multilingual Language Models**

핵심 연구 질문:

> 동일한 네 언어 표현을 이미 학습한 모델에서, 개념별 EN–FR 추가 노출 순서가
> 언어 비지정 상황의 KO/EN/ZH/FR 등록 표현 선택을 변화시키는가?

장기 예측 질문:

> LM은 이미 학습했지만 predictor가 보지 않은 개념에서, 상세한 노출 순서가
> 빈도·최근성·초기 선호·입력 조건·표면 및 토큰 특성을 넘어 선택 변화를
> 예측하는가?

가설은 다음처럼 분리한다.

1. `H1_ABILITY_CHOICE`: 요청 시 등록 표현을 산출하는 능력 `A_l`과 언어를
   지정하지 않았을 때의 조건부 선택 `Q_l`은 서로 다를 수 있다.
2. `H2_COUNTERBALANCED_HISTORY`: 총 target-slot 노출과 branch별 공통 tail이
   같아도 EN→FR과 FR→EN의 과거 순서는 paired history contrast `h`와 `r`에
   흔적을 남길 수 있다.
3. `H3_HELDOUT_PREDICTION`: 전체 순서 이력은 강한 요약 baseline보다
   predictor-held-out 개념의 선택 변화 예측 MAE를 개선할 수 있다. 이 가설은
   현재 60개 B36 파일럿에서 판정하지 않으며 상태는 `NOT_RUN`이다.

어원은 primary 연구 질문, 데이터 gate, annotation freeze, tokenizer gate,
readiness, measurement 및 H 진입 조건에서 제외한다. 문헌으로 확인된 어휘 관계는
향후 선택적 탐색 분석에서만 사용할 수 있다. 어원 자료가 없거나 불완전해도 core
실험을 차단하지 않는다.

## 1. 변경 전 상태와 artifact 계승 규칙

- `model_results_observed_before_change=false`이다. 실제 모델 checkpoint, 점수,
  readiness 결과 또는 H 결과를 본 뒤 만든 개정이 아니다.
- v4.0 annotation freeze나 experiment freeze를 v4.1 PASS로 재해석하거나 이름만
  바꾸지 않는다. v4.1은 새 schema, config hash, annotation freeze와 experiment
  freeze를 만든다.
- 기존 원자료를 사용할 수 있는 유일한 경우는 trusted upstream KRDICT collection의
  정확한 path 대상이 SHA-256과 byte count에 모두 일치하고, v4.1 검증기가 그
  payload로 candidate를 다시 구성해 동일 source identity를 확인하는 경우다.
- 경로명, 비슷한 내용, 행 수 또는 과거 PASS만으로 호환성을 추정하지 않는다.
  호환성은 `UPSTREAM_EXACT_HASH_ONLY`이다.
- 기존 AI advisory와 `jm02` feedback intake는 감사 이력으로 보존할 수 있지만,
  production human evidence, 신원 인증, 완성된 review CSV 또는 freeze로 자동
  승격하지 않는다.

## 2. 데이터와 primary 검수 gate

### 2.1 의미와 등록 표현

KO/EN/ZH/FR의 네 등록 표현은 동일한 KRDICT 의미에 연결되어야 한다. snapshot,
`target_code`, `sense_order`, candidate hash, source option, exact source span 및
canonical answer를 결과 전에 고정한다. 결측 표현을 LLM 번역으로 채우지 않는다.

각 선택 개념에 대해 다음이 모두 필요하다.

- 네 언어 표현과 뜻풀이의 의미 정렬 승인
- 선택한 네 등록 표현 각각의 분절과 답안 적합성 승인
- 원자료 hash 및 exact span과 선택값의 기계적 재대조
- reviewer ID와 날짜가 exact candidate/term hash에 결속된 기록
- 동일 문자열의 복수언어 membership 보존

이 기록은 책임 있는 검토자의 명시적 판단이어야 한다. 소프트웨어는 reviewer
identity가 실제 누구인지 인증하지 않으며 `identity_authentication_claimed=false`를
기록한다. 구조 검사나 AI advisory만으로 인간 승인을 만들지 않는다.

외부 어원 사전은 core evidence가 아니다. KRDICT raw payload와 exact locator에
결속된 인간 의미·용어 승인을 primary source evidence로 허용한다. 자연스러움이나
sense가 의심되는 표현에는 추가 사전/말뭉치 근거 또는 사전 정의된 교체 절차를
요구할 수 있지만, 모든 표현에 별도의 어원 증거를 강제하지 않는다.

### 2.2 파일럿 coverage

필수 최소치는 다음과 같다.

- primary 검수가 완료된 개념 정확히 60개
- 네 등록 표현이 서로 다른 문자열 사건으로 식별되는 개념 40개 이상
- 모든 선택 개념의 네 등록 표현이 승인되어야 함

`min_related`와 `min_distinct_routes`는 삭제한다. 관련표현, 별도경로, shared 및
UNRESOLVED의 수는 core gate가 아니다. 이 기준 미달은 `BLOCKED_DATA_COVERAGE`
또는 `BLOCKED_TERM_QUALITY`이며 어원 미검수는 blocker가 아니다.

현재 고정 선정의 selection ordinal 38, review order 195,
candidate `krdict-b100064abad6c836:60487:1`의 영어 `hen's egg`는 의미 정렬과
별개인 등록 표현 품질 유보로 남는다. 다음 중 하나가 exact hash-bound 기록으로
완료되기 전에는 core annotation freeze를 만들지 않는다.

1. 해당 표현을 그대로 등록할 타당성을 검토자가 명시적으로 승인한다.
2. 모델 결과를 보기 전에 사전 정의된 reserve 교체 규칙으로 다른 source-bound
   후보를 선택하고, 새 cohort hash와 변경 이유를 기록한다.

`egg`로 조용히 바꾸거나 결과를 본 뒤 유리한 대안을 선택하지 않는다.

## 3. 표면·토큰 특성

표면 및 토큰 특성은 어원의 대용물이 아닌 관측 가능한 primary baseline/covariate다.
네 언어의 순서 없는 여섯 쌍 `ko-en`, `ko-zh`, `ko-fr`, `en-zh`, `en-fr`,
`zh-fr`을 모두 계산한다.

문자 특성:

- exact registered expression에 NFC와 기존 공백 정리만 적용
- 각 표현의 Unicode code-point 길이
- 각 언어쌍의 `1 - Levenshtein(a,b)/max(len(a),len(b))`
- 같은 canonical 문자열은 1.0으로 기록하고 제거하지 않음

토큰 특성:

- 실제 KO-only frozen tokenizer 사용
- registered expression만 tokenization
- 각 표현의 token 길이
- 각 언어쌍 token-ID 집합의 Jaccard
- prompt, instruction, padding, special token 및 답 종료 newline 제외

feature artifact는 tokenizer freeze 이후, 첫 모델 결과 이전에 생성한다. concept
hash, tokenizer hash, metric version, 입력 문자열과 산출값을 함께 결속하고 여섯
쌍 또는 네 표현 중 하나라도 누락되면 tokenizer/feature gate를 통과시키지 않는다.
KO-only byte BPE의 overlap은 tokenizer representation overlap이며 역사적 관계나
발음 유사성으로 해석하지 않는다.

## 4. 누수 방지 단위

어원 `family_id`를 core에서 삭제한다. 미래 predictor split은 독립적인
`leakage_component_id`를 사용한다. 다음 edge의 연결 성분 전체를 한 split에 둔다.

- 같은 KRDICT target/headword 계열
- 서로 다른 concept가 어느 언어에서든 같은 canonical 등록 표현을 공유함
- 검토자가 명시한 synonym 또는 semantic duplicate
- H에서 하나의 counterbalance pair로 함께 배정됨

기계적으로 확정 가능한 edge는 자동 생성하고 사람 판단이 필요한 동의 관계는
별도로 기록한다. 이 component 검수는 현재 S/H 파일럿 학습의 core data gate가
아니지만 predictor fit 전에는 필수다. 불완전하면 core 학습이 아니라
`BLOCKED_PREDICTOR_SPLIT`로 판정한다. 어원 자료가 없어도 leakage component를
구성할 수 있어야 한다.

## 5. 학습 일정과 estimand

S의 `random INIT → KO/T0 → EN/T1 → ZH/T2 → FR/T3`, 독립 roots, 고정 단계
종점, active-language 제약과 readiness 규칙은 v4.0을 유지한다.

H는 T3의 전체 상태를 A/B로 복제한 뒤 EN–FR의 추가 노출 순서만 뒤집는다.
B36, KO/ZH anchor, 공통 four-language tail, branch당 9,600 target 예시 및
content multiset 검사는 유지한다.

“최종 노출을 맞춘다”는 말은 A와 B가 언어별로 동일한 공통 tail과 마지막
target 위치를 갖는다는 뜻이다. tail 자체가 KO→EN→ZH→FR이므로 EN과 FR의
절대 마지막 step이 서로 같다는 뜻은 아니다. 총빈도와 공통 tail은 branch 간
통제되지만 과거 순서와 시간가중 노출은 의도한 처치다.

공유 모델에서 개념들이 서로 영향을 줄 수 있으므로 estimand는 배정된 전체
일정의 paired effect다. 특정 단어의 고립된 직접효과나 표면 특성의 인과효과로
해석하지 않는다. H에서 직접 조작하는 것은 EN–FR 순서뿐이며 네 언어 전체의
도입 순서를 개념마다 조작했다고 쓰지 않는다.

## 6. primary 평가와 분석

v4.0의 등록 산출 `S`, 요청언어 능력 `A_l`, 등록된 다른 언어 산출 `C_l`,
raw continuation probability `p`, 등록질량 `Z`, membership-aware `Q_M` 정의를
유지한다. `Q`는 닫힌 등록표현 집합 안에서의 조건부 선택이지 자연 발화 전체의
언어 선택과 동일하지 않다.

식별 cohort의 paired effect는 다음과 같다.

`h_(s,c,l) = g_(s,c) * (Q_A - Q_B)` 및 `sum_l h = 0`

EN–FR contrast는 `r = (h_fr - h_en) / 2`다. primary pilot은 다음을 보고한다.

- root별 평균 `h`와 `r`, concept별 분포와 root 간 변동
- T3 초기 `Q/p/Z/A`, 언어별 문자·token 길이
- 여섯 쌍 문자·token 유사도와 효과의 기술적 관계
- shared-string cohort와 four-string identifiable cohort의 분리 결과
- fixed dev wrapper 재현성과 readiness gate

표면·토큰 특성은 무작위 배정되지 않았으므로 그 계수나 층별 차이를 원인으로
단정하지 않는다. 4 roots와 60 concepts는 예측 우월성이나 보편적 효과를
확정하기 위한 표본으로 취급하지 않는다.

## 7. predictor 분석은 현재 NOT_RUN

현재 B36 하나에서는 총빈도와 많은 일정 요약이 상수이고 predictor용 독립
concept가 부족하다. 따라서 현재 파일럿에서 predictor를 fit하거나 held-out
우월성을 판정하지 않는다.

- `prediction.enabled_for_current_pilot=false`
- 상태: `NOT_RUN_PREDICTION_BY_DESIGN`
- “새 개념”은 향후 LM-unseen concept가 아니라 predictor-held-out concept를 뜻함
- score, final Q, loss 또는 test outcome을 feature로 사용하지 않음

향후 별도 preregistration은 B12/B18 등 다중 일정, 더 큰 concept 집합과 다음
고정 model ladder를 포함해야 한다.

1. zero 또는 train median
2. T3 초기 `Q/p/Z/A`, target 빈도, 마지막 노출, 사전 정의된 decay summaries,
   input context, 문자·token 길이와 여섯 쌍 유사도
3. 같은 회귀기와 튜닝 예산에서 full sequence/order features 추가

target은 held-out wrapper 평균 4차원 `h`이며 root, 새 concept, 새 일정의 일반화를
각각 보고한다. split은 `leakage_component_id` 단위로 수행한다. 60개 파일럿의
결과를 본 뒤 threshold나 feature set을 소급 변경하지 않는다.

## 8. 선택적 문헌 어휘 관계

문헌으로 확인한 차용·공통 원천·계승·별도 경로 같은 관계는 optional exploratory
artifact로만 허용한다. 현재 파일럿에서는 비활성이고 상태는
`NOT_RUN_EXPLORATORY_LEXICAL_RELATION_BY_DESIGN`이다. 기존 8건의 어원 유보를
해소할 필요가 없다.

향후 이 분석을 실행하려면 결과를 보기 전에 별도 revision에서 label, evidence,
sense alignment, 최소 coverage와 각 집단 최소치, complete-case cohort 및 model
comparison을 동결한다. 누락이나 `UNRESOLVED`를 `DISTINCT`, 0 또는 기준집단으로
코딩하지 않는다. relation이 없는 경우 core 결과는 그대로 진행하고 탐색 분석만
`NOT_RUN_EXPLORATORY_LEXICAL_RELATION`으로 남긴다. 서로 다른 complete-case
표본의 MAE를 같은 표본의 개선처럼 비교하지 않는다.

## 9. 실행 gate와 보고

순서는 다음과 같다.

1. 원본 배포 무결성 및 v4.1 protocol/config/amendment hash 확인
2. exact-hash upstream source 재검증
3. 60개 의미·네 등록표현 품질 검수와 source-bound 기록
4. 새 v4.1 annotation freeze
5. KO-only tokenizer와 여섯 쌍 surface/token feature artifact
6. corpus materialization, CPU integration, 실제 GPU replay
7. 첫 root S, T3 readiness와 measurement
8. 예정 roots가 모두 통과한 경우에만 H

어원 artifact의 부재는 1–8 중 어느 gate도 막지 않는다. ordinal 38 용어 품질은
3의 blocker다. predictor와 documented lexical relation은 이번 실행에서 각각
`NOT_RUN_PREDICTION_BY_DESIGN`,
`NOT_RUN_EXPLORATORY_LEXICAL_RELATION_BY_DESIGN`으로 보고한다.

run JSON은 base spec/config와 v4.1 spec/config/amendment의 SHA-256과 byte count,
실제 사용한 source/freeze/code/tokenizer hash, identity authentication claim=false,
legacy freeze reuse=false, gate별 상태, GPU 누계와 미실행 이유를 담는다. Markdown과
최종 보고는 그 단일 JSON에서 생성한다. `APPROVED_DESIGN_PENDING_IMPLEMENTATION`은
설계 개정 승인이지 data QA, 구현, freeze, GPU replay 또는 파일럿 완료가 아니다.
