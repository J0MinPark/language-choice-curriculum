# 출처·인용 범위·검증 상태
2026-09-14 확인. 공식 문서/논문 HTML/공개 저장소 경로를 확인했다.
작업 컨테이너의 실제 HTTP 수집은 DNS 오류로 실패했으며 API키는 제공되지 않았다.
아래 경로의 문서 확인은 payload·라이선스 원문·전체 자료·GPU 검증을 대신하지 않는다.

## R1. 국립국어원 한국어기초사전 API
https://krdict.korean.go.kr/eng/openApi/openApiInfo
https://krdict.korean.go.kr/eng/openApi/openApiRegister
인용 범위: 키 필요, XML 검색/의미별 번역 필드, 영어1/프랑스어3/중국어11. 파생 4언어 개념집합의 확보 수나 정답성은 이 문서에서 보장하지 않는다. 이 데이터는 직접 구축할 통제세트의 원천이지 이미 완성된4언어벤치마크가 아니다.

## R2. Google/TensorFlow Datasets: Wiki40B
https://www.tensorflow.org/datasets/catalog/wiki40b
인용 범위: wiki40b/ko와train/validation/test분할 존재. 한국어모국어능력이나단일언어순수성을보장하지않는다. 실제사용파일의라이선스/버전은수집시확인한다.

## R3. Wiktextract / Kaikki
https://kaikki.org/dictionary/rawdata.html
https://github.com/tatuylonen/wiktextract
Ylonen (2022), Wiktextract: Wiktionary as Machine-Readable Structured Data (LREC).
인용 범위: 구조화된 사전 추출물과 어원 근거 후보. sense별관계의진실/네언어정렬은추가검수영역이다. deprecatedpostprocessed링크를대신추측하지않는다.

## R4. Xu et al. (2024), On the Tip of the Tongue
https://arxiv.org/html/2402.14404
https://github.com/ningyuxu/tip_of_tongue
인용 범위: 정의→표현 reverse-dictionary, greedy첫줄, 등록정답·동의어일치 평가(§2.1).
우리의한국어기준4언어순차학습,언어당등록1표현,공유표현사건과Q/Z는본연구확장이다. 단순등록일치를완전한개념지식보존검사로해석하지않는다.

## R5. EleutherAI lm-evaluation-harness
https://github.com/EleutherAI/lm-evaluation-harness
https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/api/model.py
인용 범위: context/continuation likelihood API/구현 관례. Q/Z,공유표현membership,언어간과거이력효과를이코드가정의한다고쓰지않는다. 실제scorer는고정버전API와작은수동예제로대조한다.

## R6. SciPy spearmanr
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html
인용 범위: 순위상관,상수입력의정의불가. .60경고선은이문서의표준이아니다. 동봉stdlib구현은동률평균순위를사용하며SciPy교차검증을시험에포함한다.

## R7. PyTorch reproducibility
https://docs.pytorch.org/docs/stable/notes/randomness.html
고정 버전 참고: https://raw.githubusercontent.com/pytorch/pytorch/v2.8.0/docs/source/notes/randomness.rst
인용 범위: RNG/비결정성통제,환경간완전재현보장의한계. 프로젝트전체checkpoint계보·gate는우리설계이며실제GPU에서검사해야한다.

## R8. Lakens (2017), Equivalence Tests: A Practical Primer
https://pubmed.ncbi.nlm.nih.gov/28736600/
paired 구현: https://www.statsmodels.org/stable/generated/statsmodels.stats.weightstats.ttost_paired.html
인용 범위: 사전동등성범위의TOST. ±2%p와27대비는우리과제의설계값이다. 비유의성으로동등성을결론내리지않는다.

## R9. scikit-learn MAE
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html
인용 범위: 평균절대오차. 4차원이력벡터/1%p·10%개선기준/독립root·cluster재표집은본연구설계다.

## R10. Tsoukala et al. (2021), Simulating Code-switching Using a Neural Network Model of Bilingual Sentence Production
https://link.springer.com/article/10.1007/s42113-020-00088-6
관련 계보: 학습이력과언어혼합을다루는모델연구. 동일인학습자데이터/우리의학습알고리즘은아니다. v4재시작은새최초성을확보했다는주장이아니다.

# 사실과 제안의 경계
[F] 위자료의기능·문헌방법. [D] 우리지표·수집목록·코호트·일정·임계값·예산.
[H] 이력×어원이선택을예측한다는가설. [R] 실제로그/출처hash가있는결과만.
[U] 기존0.953GPUh는사용자보고값이며현재환경실측이아니다.
라이선스·API정책은수집시다시확인하고원자료조건을보존한다. 데이터를패키지에재배포하지않았다.
