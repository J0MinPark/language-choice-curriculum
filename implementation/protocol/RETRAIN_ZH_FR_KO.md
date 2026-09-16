# 중국어·프랑스어 재학습 탐색 revision

**보류된 초안: 실행 불가.** 사용자가 학습량 증가의 근거 검토를 요청했으므로
동결 생성이 차단되며, 현재는 ACCURACY_PROBE_KO.md의 짧은 비교만 시행한다.

사용자의 중국어·프랑스어 정확도 개선 구현·재학습 요청에 따른
`zh-fr-balanced-5400-v1`이다. 기존 동결·실패 결과·원자료는 변경하지 않는다.
기존 첫 root T3 dev 결과를 본 뒤 설계한 탐색이며 사전등록 본시험이 아니다.

기존 T2 dev 중국어는 85%/78.33%, T3에서는 83.33%/75%였다.
T3 중국어 오답 25개 관측 중 24개는 미등록 출력, 1개는 종료 실패였다.
프랑스어 오답 12개 관측 중 11개는 미등록 출력, 1개는 다른 언어였다.
깨진 byte 문자, 글자 누락·반복이 관찰되었다. 이는 원인 확정이 아니며,
단순히 학습량을 늘리면 개선된다는 보장도 아니다.

## 고정 변경

- root 4101의 기존 T1 종점 전체 상태(model/optimizer/scheduler/RNG/progress)를
  정확한 state/manifest/freeze/code 해시로 검증해 새 run에 복제한다.
  기존 파일은 쓰지 않는다. 새 독립 초기화로 계산하지 않는다.
- T2/T3만 각각 5,400 updates × 32 examples로 재학습한다.
- 각 개념 × target × input × mode × train wrapper의 전체 조합을 층화한다.
  T2는 60회, T3는 30회 완전 Cartesian cycle이다. 기존 target 비율,
  ANY/requested 1:1, train1/train2 1:1, active input 균형은 유지한다.
- 각 단계 첫 3,000회는 LR 3e-4, 나머지 2,400회는 1e-4.
  optimizer를 재초기화하지 않는다. 재개 시에도 동일 update 경계에 적용한다.
- 새 토크나이저, 추가 원자료, 정답 교체, 오답별 가중, 평가 프롬프트 학습은 없다.
  dev/test 프롬프트와 greedy 평가, 확률 정의, 모든 gate 임계값을 보존한다.
- 200회마다 dev 진단, 5,400회 고정 종점 평가. best-dev 선택이나 자동 연장 없음.
- 이 revision은 T3 평가 후 끝난다. 다른 root, H, main을 자동 확대하지 않는다.
  개선 여부는 기존 대비 실제 dev 결과에서만 판정하고 test로 설정을 조정하지 않는다.

## 안전 및 제한

기존 GPU2 UUID 잠금·24 GPUh campaign·6h/root 예산과 누적 비용을 그대로 사용한다.
현재 jm 경로만 사용하며 데이터 디스크/서버 설정은 변경하지 않는다.
5 GiB 비상 공간을 유지하고 공간 부족·GPU 예산·최종 gate 미달이면 중단한다.
서버 nohup 실행과 전체 상태 재개는 기존 실행기를 재사용한다.
출력 run summary의 exploratory_revision 및 FULL_STATE_IMPORT가 출처를 구분한다.
이 실행으로 정확도가 개선되어도 prompt 안정성이나 연구 가설 검증 성공과는 별개다.
