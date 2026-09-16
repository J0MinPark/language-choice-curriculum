# Claude가 새로 구현할 부분
참조 코드나 합성시험이 실제모델훈련을대신하지않는다. 기존훈련weights/옛설정을복사하지않는다.

1. 데이터 병합기: raw manifest/XML을snapshot/target/sense로연결하고PENDING검수표생성. source hash및원문출처보존. 승인없이학습JSON생성금지.
2. Corpus/토크나이저: Wiki40B/ko공식train의고정subset,KO-onlybyte-levelBPE,동결hash,길이검사.
3. 모델/훈련: HF GPT2Config randominit; corpusloss와lexicalresponse평균loss분리;독립root4개;단계별active입력/target;고정종점.
4. 실제상태저장/복구: optimizer/RNG/loader/accumulation포함.동일10step및중간resume양성/음성시험.
5. 실제scorer: greedyfirstline + 정확한continuationtokenloglikelihood;normalization/canonicalevent구분;shared·underflow·누락회귀시험.
6. gate연결: 자료/freeze/replay없으면S거부;모든root의T3준비·측정이없으면H거부;main무조건비활성.
7. 결과: run별immutableJSON→markdown.작업시간/VRAM/기존비용기록.잘못된셀이누락될때집계통과금지.
8. 인터페이스검증: 실제--help와실행로그로존재하는명령만START/README에추가. CPU fixture통과와실제GPU통과를분리.

학습기소스자체도연구산출물이다. 미완성이면NOT_IMPLEMENTED로표기하되완료한코드/시험은보존한다. 인증정보나검수가없으면추가요청하되없는자료로대체하지않는다.
