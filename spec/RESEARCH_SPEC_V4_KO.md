# 연구·구현 기준 v4.0.0 | 완전 새 시작
2026-09-14 · project_id `lexical-freshstart-4lang-ety-v4`

**유일 기준은 이 문서와 `spec/pilot.json`이다.** 과거 v1/v2/v3, 옛 AGENTS.md, t0.pt, 보고서, 학습 결과를 읽거나 이어받지 않는다. 새 원본 패키지는 보존하고 구현은 이 폴더의 `implementation/`에 작성한다. 해시는 전달 무결성 검사이지 과학적 타당성 보증이 아니다.

## 0. 이번 실행의 목적과 권한
[D] 질문: 한국어 원천으로 초기학습한 AI가 영어·중국어·프랑스어의 같은 개념 표현을 순차적으로 배웠을 때, 학습 이력과 어원·표기·토큰 관계는 등록 표현의 산출과 선택에 어떤 정보를 주는가?

[H] 이후 확인할 가설: 어휘 관계와 학습 이력을 함께 쓰면, 초기 편향·노출 빈도·최근성·입력 조건만 쓴 예측기보다 새 개념의 선택 변화를 잘 예측한다. 이 문서에 실제 실험 결과는 없다.

이번은 **자료 확보 → 의미·어원 검수 → 실제 학습 코드 구현/시험 → 첫 root 준비·측정 → 최대 4개 root 파일럿**이다. v3의 GPU 권한 0을 가져오지 않는다. 이 v4는 필수 조건 통과 후 제한 파일럿을 허용하며, 본시험은 계속 금지한다. 인증키 없는 상태에서는 데이터 수집이 진행되지 않는다. 새 폴더로 바꿔도 이 조건은 없어지지 않는다.

[F] 자료·평가 근거는 `docs/SOURCES_KO.md`. [D] 수식/일정/표본/임계값은 연구 설계. [H] 미검증 설명. [R] run ID·코드/데이터 hash·실제 로그가 있는 측정. [U] 사용자가 보고했으나 이 환경에서 원격 확인하지 않은 값. 유형을 섞지 않는다.

**범위에서 제외:** 옛 Spearman 감사, 인간 학습 로그 전수조사, 학생 시뮬레이터 타당성 증명, 복습 정책/RL, 패칭, 거짓짝 벤치마크, 발음·한자 정보의 추가 입력, 대형 모델 전이, 모든 언어 순열. 에듀테크는 후속 동기다. 사람 자료 부재는 이 AI 파일럿의 차단 이유가 아니며, AI 결과가 사람의 학습효과를 증명하지도 않는다.

## 1. 시작 조건과 보안
1. `python3 verify_package.py` → `python3 -m unittest discover -s tests -v` → `python3 -m freshstart preflight`.
2. 인증은 `KRDICT_API_KEY` 환경변수로만 받는다. 사용자에게 채팅으로 키를 보내라고 하지 않는다. shell trace, 전체 환경 dump, 비밀값·키 포함 URL을 출력/저장/git commit하지 않는다.
3. 키가 없으면 `BLOCKED_CREDENTIALS`: 발급/로컬 설정 안내를 한 번 남기고 요청을 반복하지 않는다. 공개 사이트/API의 인증을 우회하지 않는다. 사용자가 적법하게 확보한 원본 스냅샷을 제공하면 provenance/schema/라이선스를 감사해 입력할 수 있으나 출처불명 CSV를 정답으로 인정하지 않는다.
4. 홈페이지와 공식 문서를 확인한 것, payload를 받은 것, 스키마가 맞는 것, 사람이 의미를 검수한 것, 학습 준비가 된 것은 서로 다른 상태다.
5. 홈 전체 탐색, 다른 에이전트의 세션 기록 조회, 과거 폴더 이동·삭제, 타인 프로세스 종료를 하지 않는다. 새 프로젝트 안에서만 작업한다.
6. 본 패키지는 CPU 참조 도구와 완전한 구현 명세다. **완성된 모델 학습기/데이터셋/가중치는 포함하지 않는다.** Claude가 `implementation/`에 실제 trainer/scorer를 구현해야 한다. 없는 명령을 실행한 것처럼 보고하지 않는다.

## 2. 데이터: KO/EN/ZH/FR 고정
### 2.1 의미 정렬
[F, R1] 국립국어원 한국어기초사전 검색 API는 의미별 번역을 제공한다. 요청 `translated=y`, 언어 `en=1`, `fr=3`, `zh=11`. `target_code`, `sense_order`에 원본 스냅샷 ID를 붙여 연구 개념 ID를 만든다. WordNet ILI라고 부르지 않는다.

- `templates/queries.txt`는 수집용으로 작성한 검색어일 뿐 표준 벤치마크·대표 표본·검증 어원 목록이 아니다. 명시한 exact query만 수집한다. 전수 검색, wildcard, pagination 의미를 추측하지 않는다.
- 동봉 `freshstart fetch`는 질의당 3개 언어 첫 결과를 최대 100개 entry까지 받는 제한 수집기다. 실제 total보다 반환이 작으면 불완전으로 표시한다. 검증 전에 실제 응답을 본 적 있는 것처럼 주장하지 않는다. 600요청 상한, timeout, 오류 시 중단, 원본 XML+SHA 저장. API 429/일한도/인증오류에서 무한 재시도하지 않는다.
- 한국어 표제어/정의 및 세 외국어 표현/정의가 있는 명사 의미를 후보로 한다. 결측을 LLM 번역으로 채우지 않는다. 같은 스냅샷 수집의 언어별 KO 정의/품사/의미 ID가 불일치하면 격리한다.
- 각 언어의 **등록 표현 1개**를 결과를 보기 전에 검수·고정한다. 쉼표 구분이 항상 동의어 목록이라고 가정하지 않는다. 대안 표현은 원본에 보존하되 결과를 보고 정답 집합에 추가하지 않는다.
- NFC와 공백 정리만 기본 정규화. 프랑스어 악센트, 한글, 한자, 간체·번체, 병음을 자동 통합하지 않는다. 공유 문자열은 버리지 않고 언어 membership으로 기록한다.
- 자료 검수는 연구자가 수행하거나 별도 책임 있는 검수 절차를 따른다. 에이전트가 `reviewer=human`을 만들어 넣지 않는다. `APPROVED_BY_RESEARCHER`는 로그가 있는 실제 승인이다. 구조 검사만으로 의미·어원 검수가 끝났다고 하지 않는다.
- 같은 뜻풀이를 다른 wrapper로 감싸는 것만으로 새 의미 문맥 일반화라고 주장하지 않는다. 모델 학습에 사용한 개념과 별도로 predictor의 train/dev/test 개념 집단을 분리한다.

### 2.2 어원과 형태
[F, R3] Wiktextract/Kaikki의 어원 필드는 근거 후보를 제공한다. 원본 출처·버전·라이선스·해당 sense를 대조한다. 공식 raw 링크를 사용할 때 다운로드 크기를 확인하고 압축 스트리밍한다. 모든 dump를 무작정 받지 않는다. 연구자가 제공한 소규모 근거 스냅샷도 가능하다.

[D] 각 개념의 EN–FR 관계를 주분석으로 검수:
`BORROWING_DOCUMENTED`, `SHARED_SOURCE_DOCUMENTED`, `DISTINCT_ROUTES_REVIEWED`, `UNRESOLVED`.
- DISTINCT는 명시한 어원 경로/역사 범위에서 별도로 검토됐다는 뜻이지 모든 시대에 무관하다는 증명이 아니다.
- 근거가 없으면 UNRESOLVED. 철자 유사도, 언어족, AI 설명으로 역사적 관계를 확정하지 않는다. 어원 관계가 다른 뜻에만 해당하면 해당 sense에는 쓰지 않는다.
- 표기 유사도: NFC 문자열의 `1 - Levenshtein/max(length)`.
- 토큰 유사도: 실제 KO-only frozen tokenizer가 만든 표현-only token들의 Jaccard와 각각의 길이. padding/지시/종료 토큰은 제외한다.
- 같은 네 표현의 6개 언어쌍 중 KO–ZH 등을 기술할 수 있으나, 이번 필수 주석은 EN–FR이다. 나머지 미검수 쌍은 UNRESOLVED이며 자동으로 같은 어원/다른 어원을 부여하지 않는다.
- 어원/표기/토큰의 분포와 공통 지지영역을 보고한다. 사실상 완전히 공선적이면 독립 어원 기여를 식별했다고 주장하지 않는다. 어원을 무작위 배정한 실험이 아니다.
- 같은 어휘 어원 family·동의어·동일 의미가 predictor의 서로 다른 split으로 누출되지 않도록 연결 성분을 묶는다. 거대한 언어족 자체를 family ID로 쓰지 않는다.

[D] 파일럿 운영 최소치: 검수한 개념 60개, 그중 네 표현이 문자열로 식별되는 개념 ≥40개, 식별 코호트 EN–FR 관련표현 ≥12개, 검토된 별도 경로 ≥12개. 표본은 결과 이전에 확정한다. 이것은 검정력 근거가 아닌 자료·측정 파일럿 설계값이다. 공유/UNRESOLVED의 비율과 배제 사유를 보고한다. 부족하면 `BLOCKED_DATA_COVERAGE`; 조용히 언어/기준을 낮추지 않는다.

### 2.3 한국어 출발 자료
[F, R2] Wiki40B/ko 공식 train split을 한국어 원천으로 사용한다. 데이터 전용 CPU 환경에서 공식 로더/파일을 확보하고 release, split, doc ID, license, SHA를 기록한다. GPU TensorFlow를 trainer 환경에 설치할 이유는 없다.
[D] 파일럿은 KO corpus에서 결정적으로 선택한 고정 subset으로 tokenizer를 학습하고, 같은 frozen tokenizer로 계산한 최대 2,000,000 model tokens를 corpus pretraining에 사용한다. 데이터 순서를 문서 ID/hash로 고정하고 버려진/남은 토큰을 로그한다. 이 수량은 품질 보증이 아니다. KO원천에도 외국어 문자열이 존재할 수 있다. '완전한 한국어 모국어 능력'이나 '다른 언어를 전혀 모름'이라고 쓰지 않는다.

## 3. 실제 구현과 학습 단계
### 3.1 모델과 고정 종료점
[D] 시작 구조: GPT2형 10층/폭512/8head/FFN2048/context256/vocab16384/dropout0. pretrained weights를 불러오지 않고 config로 random INIT한다. 토크나이저는 KO corpus train에서만 학습한 byte-level BPE, 전체 byte 지원 확인 후 고정. 미래 언어의 정답/정의/어원으로 tokenizer를 학습하지 않는다.

- 실제 파라미터 수, GPU·CUDA·torch/transformers/tokenizers 버전과 precision, optimizer 설정을 기록한다.
- spec/pilot.json의 초기 설계값: corpus 토큰 상한 후 각 lexical stage 3,000 optimizer updates, effective batch32, AdamW lr3e-4, H lr1e-4. 200update마다 dev 진단. **최종 평가 체크포인트는 고정 단계 종점**이다. best dev로 종점을 대체하지 않는다. 최초90%도달은 비용 참고값일 뿐이다.
- corpus는 통상적인 causal next-token loss. lexical은 response token CE를 예시별 평균한 뒤 batch 평균. HuggingFace 기본 token 평균 loss를 그대로 사용해 다른 가중을 주지 않는다.
- 길이 초과 시 정답이나 정의를 묵시적으로 잘라 학습하지 않는다. QA 단계에서 길이 검사를 완료하고, 범위 수정은 최초 실제 학습 전 버전 변경으로 기록한다.
- pilot 설정은 최초 학습 전 freeze. 결과 이후 예산/표본/문항/모델 변경은 새 탐색 revision이며, 성공으로 소급 덮어쓰기하지 않는다.

### 3.2 S: 처음 배우는 경로
각 root 4101,4102,4103,4104는 독립적으로:
`random INIT → KO corpus + KO lexical/T0 → EN 추가/T1 → ZH 추가/T2 → FR 추가/T3`.
T2는 3언어 중간 관측점, T3는 4언어 종점. 한 root의 준비 모델을 복제한 것을 다른 독립 초기화로 세지 않는다.

- stage의 active 언어만 input/target/request에 넣는다. 요청하지 않은 미래언어 성적은 offline 평가에서 계산 가능하지만 gradient/조기종료/스케줄 조정으로 되돌리지 않는다.
- T1 target 비율 EN1/2,KO1/2; T2 ZH1/2,KO1/4,EN1/4; T3 FR1/2,KO/EN/ZH 각1/6. 정수 카운트를 미리 만든 supercycle로 구현한다. 최종 일부 batch까지 실제 count를 보고하며 정확히 동일하다고 과장하지 않는다.
- active 입력언어를 target와 독립적으로 균형 배치하고, target별 ANY/requested를 동일 비율로 만든다. 학습 자료는 RD train1/train2 wrappers만. 지시 이름의 언어도 입력언어에 맞추되 미래언어 이름을 쓰지 않는다.
- S는 복습 비율·업데이트·도입언어가 함께 바뀌는 recipe의 관측이다. 순수 언어수의 인과효과가 아니다. ADD/REPLAY 대조는 이번 최소 파일럿에 없다.
- T0~T3 주비교는 **같은 KO 정의, 같은 평가 표현집합 U={ko,en,zh,fr}**. 아직 배우지 않은 언어도 후보집합에만 존재하며, 언어를 늘릴 때 분모를 바꾸지 않는다.

### 3.3 순서와 준비 gate
실제 순서: 자료·검수 완료 → toy CPU 통합 → 실제 GPU 재현성 검사 → 첫 root의 T0/T1/T2/T3 → T3 dev 준비/측정 → 다른 예정 roots의 S → 모두 통과 시 H.
첫 root가 준비/측정에서 차단되면 다른 roots와 H를 확대하지 않는다. 이미 관측한 S 성적과 비용은 보존한다. 예산 안에서 stage를 끝냈더라도 T3의 각 요청언어 A가 준비 기준에 미달하면 `BLOCKED_READINESS`다. 후속 root 중 실패하면 모두를 보고하고 H 확대를 중지한다. 성공 root만 골라 독립4개라고 부르지 않는다.

[D] 준비 기준은 KO input/RD/dev에서 A_KO,A_EN,A_ZH,A_FR가 **각각** ≥.90. 대각셀/전체평균은 대체하지 않는다. 이는 등록 명명 과제의 기준이고 전체 언어 능력 기준이 아니다. T0~T2는 해당 active언어의 같은 조건을 보고한다. 이 단계 점수가 낮다고 임의로 과거 언어를 삭제하지 않는다.

### 3.4 체크포인트와 재현성
[R7] 동일 환경의 model/optimizer/moments/scheduler/scaler/Python·NumPy·Torch CPU/CUDA RNG/각 generator/loader 위치/accumulation boundary/globalstep을 모두 보존한다. optimizer=None으로 weights만 로드하면 실패다.
- 각 root 초기 weights hash가 서로 다르고 parent root/phase가 맞는지 검사한다. tokenizer 공유는 가능, 준비 weights 공유는 불가.
- 단계 LR변환은 양 branch에 공통 적용한 뒤 fork한다. H용 optimizer를 새로 만들지 않는다.
- 동일 상태 A/A'의 동일 10steps가 loss/실제 records/model/optimizer에서 일치하는지 검사한다. 연속실행 대 save-resume도 검사한다. evaluation이 학습 RNG를 소비하지 않게 분리한다.
- 같은 GPU에서 A/B 순차 실행을 기본으로 한다. GPU가 2대여도 근거 없이 공유 메모리로 합쳐진다고 가정하지 않는다. roots 병렬화는 첫 root 통과 후만 허용하며 GPU시간은 합산한다.
- 다른 hardware/library에서 비트동일성은 보장하지 않는다. 반복 실행 통과 여부를 실제 로그로 보고한다. CPU 수식시험이 GPU 통합시험을 대체하지 않는다.

## 4. H: 습득 후 EN–FR 추가 노출 이력
T3 전체 상태를 A/B로 복제. 처음 외국어 도입순서를 바꾸는 실험이 아니라 **이미 배운 표현에 대한 추가 노출순서**다.

- 사전 pair마다 두 개념에 g=+1/-1을 root별 무작위 배정. A의 g+는 EN36→FR36, g-는 FR36→EN36. B에서 역전.
- mutable 2round마다 KO1round+ZH1round를 양쪽에 동일 삽입: 각 anchor36회.
- 공통 tail4cycles×(KO,EN,ZH,FR). 최종 언어당40회, 개념당160회, 60개면 **branch당9,600 target 예시**. round는 optimizer step이 아니다.
- 원래 record(content/input/mode/wrapper/canonical answer)를 branch 전에 생성하고 순서만 바꾼다. record ID가 같다고 실제 content도 같다고 가정하지 말고 hash/Counter로 확인한다.
- pair를 같은 batch에 두고 각 round의 batch 계획을 고정한다. anchor·tail·마지막 target optimizer step·전체 step은 동일해야 한다. padding/effective batch가 같다고 실제 유효 token 수까지 같다고 쓰지 않는다.
- input gloss에 정답 단어가 우연히 나타나는 incidental exposure와 target-slot exposure를 구분해서 기록한다. '40회'는 통제한 target-slot 횟수이지 생애 전체의 단어 노출 수가 아니다.
- 공유 모델에서 여러 개념이 동시에 변화한다. 추정치는 배정된 전체 일정의 효과이며 특정 개념만의 독립 직접효과나 어원의 인과효과가 아니다.

이번은 B36 하나만 실행한다. **B36 안에서는 여러 이력 특징이 상수일 수 있으므로 이 파일럿만으로 이력 특징의 일반화 우월성을 판정하지 않는다.** 향후 다중 일정(B12/B18 등)과 예측용 대규모 개념 split은 별도 본시험 등록 후 시행한다.

## 5. 평가 지표와 정의
### 5.1 등록 산출 [R4를 참고한 D 확장]
언어별 등록표현 1개. y의 언어 membership M(y)는 같은 문자열이면 복수언어다.
- `S_m`: mode m에서 greedy 첫줄이 등록 표현 union에 있는 비율.
- `A_l`: 언어 l 요구에서 출력 membership이 l을 포함하는 비율.
- `C_l`: 등록된 표현이지만 membership에 요구 l이 없는 비율.
- 그 외는 UNREGISTERED/EMPTY로 분리; 분모에서 삭제하지 않는다. 등록 밖 동의어를 실제 의미 오류/망각이라고 단정하지 않는다. ANY에는 정답 언어라는 것이 없으므로 언어선택 자체를 오류화하지 않는다.

### 5.2 확률 사건 [R5는 likelihood 구현 근거, Q/Z는 D]
`p(y)=P(canonical_tokens(y+'\n') | prefix,ANY)`.
`Z=sum_{distinct registered y} p(y)`.
`Q_M=sum_{y: membership(y)=M}p(y)/Z`.

prefix를 토큰화한 결과와 전체 prefix+답의 prefix 토큰이 같아야 한다. 전체 답변+종료기호 logprob 합산, 첫토큰만 사용하거나 길이정규화 점수를 확률로 취급하는 것은 금지한다. token 사건은 중복 없고 prefix-free여야 한다. 다른 가능한 segmentation 전체를 적분했다고 하지 않는다.

공유문자열은 Z에 딱1회 합산. 네 표현이 모두 식별되는 코호트에서만 Q_l이 완전한4차원 언어벡터다. 공유군은 Q_membership·공유질량·원시p/Z로 보고하고4언어MAE에0을 채워 끼워넣지 않는다. 동일 철자 쌍을 삭제하고 전체 동계어로 일반화하지 않는다.

logsumexp, 원시p,logZ,Z,생성S/A를 함께 저장. 유한logZ가 expunderflow로0이 된 것과 모든logp=-inf인 것을 구분. 후자는 Q undefined이다. 낮은Z에서 높은Q가 가능하다. 후보 집합만 늘어 Q가 감소한 것을 망각이라 부르지 않는다.

### 5.3 측정 재현성: 하나의 집계식
동일 root/T3/KO/RD/ANY/dev/동일 identifiable concept set에서 dev1과dev2를 concept_id로 정확히 대응한다. root/phase/언어/형식을 섞거나 list 순서로 align하지 않는다. 중복/누락은 에러다.
각 출력언어별 Spearman, 평균|Q_dev1-Q_dev2|, Q의SD/범위/동률/NaN, wrapper별 Z<.05 비율을 출력한다. [D] 어떤 언어의 rho<.60 또는 undefined, 어떤 언어의 절대차>.15, 어떤 wrapper에서 lowZ비율≥.10이면 `BLOCKED_MEASUREMENT`. 이 문턱은 보편적 타당도 기준이 아니라 탐색 운영 규칙이며 결과를 보고 낮추지 않는다.

언어별 수치를 하나의 보기 좋은 median으로 치환하지 않는다. 낮은rank와작은절대차는 함께 가능하며 Q신호범위가 작을 수 있다. 실패원인은 구현오류/제한된변동/실제prompt의존으로 나누어 진단한다. **JSON 한 개에서 보고서·판정·표를 생성**한다. 소수자리 반올림 전에 판정하며, 서로 다른셀 값을 같은 Spearman이라고 병기하지 않는다. RD의test1/test2, 다른입력언어와CLOZE는 고정된보조분석이고 좋게나온형식으로주분석을바꾸지않는다.

### 5.4 이력 효과와 어원 집단
식별코호트에서 `h_(s,c,l)=g_(s,c)*(Q_A-Q_B)`; sum_l h=0.
EN–FR 대비 `r=(h_fr-h_en)/2`. root별 개념 평균과 어원 집단 대비를 보고한다. 전체평균r이0이어도 개념별변동이 없다는 뜻은 아니다. 긍정/대조/unknown/shared 개수와 층별 초기A/Q/Z·길이·유사도를 같이 제시한다.

root는독립초기화단위이고개념/prompt/언어는교차·반복관측. root별SD, 개념간분산, paired-root보존split-half를구분한다.4roots의2/2분할은3개이며독립실험3개가아니다. raw상관을보편적noiseceiling이라하지않고MAE를상관으로나누지않는다.

향후 능력유지주장은 pairedTOST[R8]로 사전범위±.02(설계제안)를 검사: primaryKO/RD의A4개+S5개와T3→A,T3→B,A↔B 대비=27개. 네언어식별코호트와shared군분리. 모두통과한범위에서만aggregate능력유지라고쓴다.4root파일럿의비유의를동등성으로쓰지않는다.

## 6. 예측 본시험은 이번에 실행하지 않는다
[F, R9] MAE는 표준예측오차. [D] target는testwrapper평균4차원h, `MAE_vec=mean_root mean_concept mean_language |h-hhat|`.
비교군은0/trainmedian, 초기T3Q/p/Z/A+빈도·지수/멱감쇠최근성+문자/토큰길이+input context, 여기에 이력만/표기·토큰만/검수어원추가/이력×유사성·어원 결합을 단계적으로 더한다. 동일회귀기·동일튜닝예산·동일검수집단을 사용한다. 어원효과는 표면유사성통제모델보다 나아지는지로검사하되 원인으로단정하지않는다.
동의어·어원family 연결cluster 분할, LM이 학습한 개념 중 predictor가 보지 않은 개념에 평가. 새개념/새일정/새root는각각보고. 최종Q/loss/test결과를특징으로사용금지. 60개 파일럿으로 우월성을 확정하지 않는다.
[D] 이후 목표≈600개와독립testcluster≥100개의확보·반복수·효과범위를파일럿후검토. MAE절대.01/상대10%/gain95%CI하한>0은본시험전동결할제안값일뿐이번에합격판정하지않는다. main_enabled=false를자동변경하지않는다.

## 7. 상태와 실행 예산
- 자료/코드 검증 실패: 정확한 원인코드와 다음 한 행동. API 재시도만 반복하지 않는다.
- 학습구현 미완료: `NOT_IMPLEMENTED`. 파일럿 실행 안 됨: `NOT_RUN`.
- 첫root S는 data QA, freeze, tokenizer boundary, CPU integration, 실제GPUreplay가필수. H는모든예정root T3 readiness+measurement 통과후만가능.
- 실행중오류·중단은원자료와checkpoint보존. 결과 JSON은run별로write-once;수정재분석은새버전파일. markdown은JSON에서재생성한다.
- [D] campaign상한24GPUh, root상한6h. [U] 기존0.953GPUh를사용자보고누계로구분기록하여같은campaign잔여량에서차감한다. 추가사용이발견되면누적한다.2GPU동시1h=2GPUh. 새폴더라도비용을몰래0으로하지않는다. 예산증액은사용자승인필요. 데이터다운로드기본상한6GiB,원본안전저장,공유자원무단종료금지.
- 문서/사전/검수/code/replay/준비/측정 각gate PASS와 `PILOT_COMPLETE`는 별개. **GO 하나로 모든 상태를 합치지 않는다.** 프로세스 exit0은연구타당성판정이아니다.

## 8. Claude 구현 산출물과 필수 integration tests
`implementation/src/{prepare_data,build_tokenizer,train,score,checkpoint,report}.py`, CLI, requirements.lock, 코드commit; `work/` 아래 sources/annotations/freeze/checkpoints/evaluations/reports/ledger.
1. 데이터 hash/key/source실패/한언어결측→학습진입거부.
2. KO→EN→ZH→FR lineage; 미래언어input/target사용거부;독립INIT4개.
3. weights-onlyfork·다른optimizer·다른loader·평가RNG소비를negative test로검출.
4. 실제tokenizer prefix/종료/공유표현사건;실제모델의teacher-forcing logits값을작은수동예제와대조.
5. 학습recordcontent multiset/anchor/tail/laststep;round와update수분리.
6. 준비false에서H진입실패;어느한언어85%·평균90%를통과시키지않음.
7. 서로다른root/input/checkpoint를섞은Spearman집계거부;같은JSON을쓰는CLI/표/markdown판정일치.
8. 같은원자료행순서를섞어도통계불변;상수Q의rho는undefined;shared군을언어식별군에혼합거부.
9. key없는콜lector실제HTTP요청0;가짜fixture로dataQA통과시도거부.
10. GPU중단/resume와누적예산검사;프로세스가끝나도measurement차단상태유지.

실제자료가없어도fixture로구현단위시험은가능하나이를연구결과/어원증거로쓰지않는다. 사용자가 검수해야 할 항목은 하나의 pending_review.jsonl/요약파일로 모아 요청한다. 자동승인을 만들지 않는다.

## 9. 완료 보고
`work/reports/run_summary.json`이유일한요약원천이다. spec/config/data/code SHA,실제읽은자료,검수수,구현/실행/미실행,GPU누적,단계별gate,다음필요행동을담는다. 여기서 `run_summary.md`를생성한다. 과거0.539/0.7007을현재결과로복사하지않는다. 데이터스키마/코드시험이끝났다고학습완료라고하지않는다.
