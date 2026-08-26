# ClaimMate 개발 가이드라인

## 프로젝트 개요
ClaimMate는 소비자의 비정형 상담/피해 글을 입력받아 **Pandas/NumPy 정제 → Scikit-learn
분류/예측 → OpenAI LLM Agent 인사이트 리포트** 파이프라인을 수행하는 풀스택 AI 웹 애플리케이션이다.
전체 명세는 [docs/SPEC_프로젝트개발명세서.md](docs/SPEC_프로젝트개발명세서.md) 참고.

## 아키텍처
1. `components/DisputeForm.tsx` — 사용자가 상담 내용/금액/일자를 입력
2. `app/api/analyze/route.ts` — `api/ml_predict.py`를 실행하여 Pandas 전처리
   (`api/ml_engine/cleaner.py`) + Scikit-learn 추론(`api/ml_engine/predictor.py`) 수행.
   로컬 개발에서는 subprocess로, Vercel 배포 환경에서는 같은 배포 내의
   `/api/ml_predict` 서버리스 함수를 내부 HTTP 호출로 중계한다. (Python 스크립트를
   `app/api/analyze`와 같은 `/api/analyze` 경로에 두면 빌드 시 Next 함수와 Python 함수가
   같은 출력 슬롯을 두고 충돌해 인접 함수까지 오염되므로 반드시 경로를 분리해야 한다.)
3. `app/api/agent/route.ts` — ML 결과를 받아 `lib/legalSearch.ts`(RAG 검색)로 관련
   법령 조항 Top-2를 찾은 뒤, OpenAI(`gpt-4o-mini`, Structured JSON Output)로
   법적 근거/인용 조항(`referenced_clauses`)/예상 환급/실행 계획/내용증명 생성.
   `OPENAI_API_KEY` 미설정 시 `lib/fallbackAgent.ts`의 규칙 기반 폴백으로 자동
   대체된다(이 경우 RAG 검색도 수행하지 않음).
4. `components/ResultReport.tsx` 외 컴포넌트가 최종 리포트를 렌더링(인용 법령
   조항은 아코디언 UI로 표시)

## 로컬 개발
```bash
npm install
python -m pip install -r api/requirements.txt
cp .env.example .env.local   # OPENAI_API_KEY 입력 (선택, 없으면 폴백 리포트 사용)
npm run dev
```
- `PYTHON_BIN` 환경 변수로 Python 인터프리터 경로를 지정할 수 있다 (기본: `python` → `python3` 순 탐색).
- `python api/ml_predict.py`는 stdin으로 JSON을 받아 stdout으로 JSON을 반환하는 CLI로도 동작하며,
  Vercel Python Runtime의 `handler` 클래스로도 동작한다(배포 시 `/api/ml_predict` 경로로 노출됨).
- **`npm run dev`와 `npm run build`를 동시에 실행하지 말 것.** 둘 다 `.next/`에 동시에 쓰기 때문에
  캐시가 깨질 수 있다(깨졌다면 dev 서버를 끄고 `.next/`를 삭제한 뒤 `npm run dev`로 재시작).

## ML 학습 데이터 (실제 공공데이터)
`data/raw/`에 원본 데이터 3종이 있다: 공정거래위원회 "소비자 민원학습데이터 모범상담 사례" CSV
((사건번호, 상담제목, 상담내용, 답변내용) 4개 컬럼), 한국소비자원 "품목별 피해구제 사례" XML
(Excel 2003 SpreadsheetML 형식, (일련번호, 품목, 출처, 제목, 질문, 답변) 6개 컬럼), 한국소비자원
"소비자상담 표준답변" CSV((번호, 품목명, 구분, 질문, 답변) 5개 컬럼). 앞의 둘은 카테고리 라벨을
그대로 신뢰하지 않는다 — XML의 "품목" 라벨은 ClaimMate 카테고리와 깨끗하게 1:1 매핑되지 않는다
(예: XML의 "금융/보험" 안에 상조회사 폐업·유사투자자문 사기·리볼빙 수수료처럼 보험과 무관한
사례가 다수 섞여 있고, "관광/운송"엔 택배/포장이사 분쟁이, "주거/시설"엔 대형마트 상해 사례까지
섞여 있음을 실제로 확인했다). 그래서 이 둘은 `python api/ml_engine/build_case_bank.py`가
제목/본문 키워드 매칭(`CATEGORY_KEYWORDS`/`_assign_category()`, 두 소스에 동일하게 적용)으로
카테고리를 부여한다. 반면 표준답변 CSV의 "품목명"은 TV/헬스장/아파트처럼 단일 품목 단위로
깨끗하게 라벨링돼 있어(실제로 수백 건을 열어봐도 품목명과 상담 내용이 어긋나는 사례를 못 찾음)
`STD_ITEM_CATEGORY_MAP`으로 직접 매핑한다(키워드 추측 불필요). 세 소스를 합쳐
`api/ml_engine/case_bank_data.json`(약 1,399건)으로 정제한다. `predictor.py`는
이 JSON을 `SYNTHETIC_CASES`(카테고리별 최소 표본 확보용 수작성 예시, 특히 데이터가 희소한
체육시설/헬스장·화장품/미용·부동산/임대차·식품)와 합쳐 `CASE_BANK`를 구성한다.
- 표준답변 CSV는 "구분" 필드(예: "계약해제·해지_위약금 공제_소비자 귀책_개시일 이전")가
  분쟁유형을 이미 구조화된 형태로 담고 있어 `dispute_type`으로 그대로 쓴다. 다만 이 필드값이
  여러 품목(예: 필라테스/헬스장)에 걸쳐 재사용되는 경우가 있어 `dispute_type` 자체의 중복은
  흔하다(같은 상담 내용까지 중복되는 것은 아니므로 정상이며, 실제 중복 제거는 위 (제목, 답변
  앞부분) 기준으로 여전히 동작한다). "안전"으로 시작하는 구분(화상·주차사고 등 신체/재산 피해
  배상)은 위해위험 CSV와 같은 이유로 제외한다.
- `_load_raw_df()`는 `data/raw/*.csv` 전체가 아니라 파일명에 "모범상담"이 포함된 CSV만 골라
  읽는다 — `data/raw/`에 검토 전 대용량 CSV(위해위험 소비자상담 데이터, 13만 건 규모, 아직
  `data/staging/`에서 별도 정제·검토 중)가 함께 있어도 잘못 집지 않게 하기 위함이다.
- `CATEGORY_KEYWORDS["부동산/임대차"]`에 원래 "중개"라는 짧은 키워드가 있었는데, XML 데이터를
  추가하면서 "결혼중개업체", "대부중개수수료"처럼 전혀 무관한 사례까지 "중개" 부분 문자열
  매칭으로 부동산 카테고리에 잘못 들어가는 게 실제로 발견됐다. "공인중개사"로 교체해 해결했다
  (이 카테고리는 표본이 원래도 적어 실데이터 매칭이 0건이 될 수 있는데, 그 경우도 오분류로
  오염되는 것보다 낫다고 판단해 `SYNTHETIC_CASES`에만 의존하도록 뒀다).
- XML 원본 자체에 같은 상담이 금액 표기(예: "990000원" vs "990,000원")만 다르게 중복 수록된
  경우가 다수 있어(초기 병합 시 911건 중 약 120건이 사실상 중복이었음), `build()`가 (제목,
  답변 앞 80자) 기준으로 중복을 걸러낸다(`_append()`).
- 원본 파일을 교체하거나 `CATEGORY_KEYWORDS`를 수정하면 `build_case_bank.py`를 다시 실행해
  `case_bank_data.json`을 재생성해야 한다.
- `predictor.py`는 매 요청(subprocess)마다 재학습하면 느리므로(~7초) 학습된 모델을
  `api/ml_engine/_model_cache.joblib`에 캐시한다. 케이스뱅크가 바뀌면(파일 mtime/건수 변경)
  캐시 키가 달라져 자동으로 재학습된다. 강제로 지우려면 `_model_cache.joblib`를 삭제한다.
- **알려진 한계**: `_assign_category()`는 카테고리당 키워드 점수 최고점 하나에만 배정하는
  단순 방식이라, 여러 카테고리 키워드가 동시에 등장하는 복합 사례(예: "무료 노트북을 준다고 해서
  가입한 상조서비스" → 실제로는 상조 청약철회 사례인데 "노트북" 키워드 때문에 전자제품으로
  분류됨)에서 여전히 오배정이 발생할 수 있다. 대용량 CSV(위해위험 데이터)는 이 문제가 훨씬
  심해서(도메인 자체가 다름) 별도 스테이징 검토 없이는 반영하지 않기로 했다 — 자세한 내용은
  `data/staging/csv_candidate_report.md`(및 2차 정제본 `_v2`) 참고. 스테이징 검토 중
  "요가"가 "필요가"의 부분 문자열로, "상조"가 "상조공제조합" 같은 무관한 기관명의 일부로
  잘못 매칭되는 사례도 발견됐다(짧은 한글 키워드가 한국어 특성상 단어 경계 없이 다른 단어
  안에 우연히 포함되는 문제) — 현재 `case_bank_data.json`(원본 CSV+XML 규모가 작아 아직
  이 패턴이 표면화되지 않음)에는 영향이 없는 것으로 확인했지만, 원본 데이터 규모를 키울
  때는 유의해야 한다.

## RAG 법령 지식베이스 (docs/SPEC_RAG.md)
`lib/knowledge_base/legal_kb.json`에 14개 업종 + 일반 법령 조항 청크(약 49건, `law_name`/
`clause_summary`/`refund_formula`/`keywords` 스키마)가 정의되어 있다. 원래 14개 업종 중 7개
(의료/병원·보험·자동차·상조/결혼서비스·부동산/임대차·전자제품·식품)는 업종당 2건뿐이었는데,
실제 상담 데이터에서 반복 확인됐지만 근거가 아예 없던 시나리오(전자제품 렌탈 계약 해지,
성형외과 수술 취소 계약금, 보험금 지급 지연이자, 결혼중개업체 허위정보, 전세사기 피해자
지원, 중고차 사고이력 미고지, 배달앱 오배달)를 업종당 1건씩 추가해 보강했다. 이를
`scripts/build_legal_kb.py`(OpenAI `text-embedding-3-small`로 임베딩, `OPENAI_API_KEY`
필요)로 빌드하면 `lib/knowledge_base/legal_kb_embedded.json`이 생성된다.
- `lib/legalSearch.ts`는 이 임베딩 JSON을 읽어 사용자 상담 텍스트를 쿼리 임베딩한 뒤
  코사인 유사도(+ML 예측 카테고리 일치 가중치)로 Top-2 조항을 찾아
  `app/api/agent/route.ts`의 프롬프트에 `[REFERENCE LEGAL CONTEXT]`로 주입한다. LLM은
  이 조항만 인용하도록 강제되며(hallucination 금지 규칙), 결과는 `referenced_clauses`
  필드로 반환되어 `ResultReport.tsx`의 아코디언 UI에 표시된다.
- `legal_kb_embedded.json`이 없거나(빌드 전) `OPENAI_API_KEY`가 없거나 임베딩 API 호출이
  실패하면 `retrieveLegalClauses()`가 `null`을 반환해 기존 정적 `legalKnowledge.ts` 기반
  동작으로 무중단 폴백한다(`referenced_clauses`는 빈 배열).
- `legal_kb.json`을 수정(법령 개정 반영 등)하면 `python scripts/build_legal_kb.py`를
  다시 실행해 임베딩을 재생성해야 한다. 이 스크립트는 로컬 1회성 개발 도구라
  `api/requirements.txt`(Vercel 배포 번들)에는 포함하지 않으며, 실행 전
  `pip install openai`가 별도로 필요하다.
- 새 조항을 추가한 뒤 `retrieveLegalClauses()`를 직접 호출해 확인해보면(ML 분류 결과를
  거치지 않고 category를 고정해서 테스트) 검색 자체는 정상 동작한다. 하지만 실제
  `/api/analyze` → `/api/agent` 전체 파이프라인으로 테스트하면, 예를 들어 "정수기 렌탈
  중도해지" 상담이 `predictor.py`에서 (전자제품 학습 표본이 적어) "통신/인터넷"으로
  잘못 분류되면서 새로 추가한 `KB_ELECTRONICS_03`(렌탈 계약 해지) 대신 통신 관련 조항이
  검색되는 걸 확인했다. 즉 RAG에 조항을 추가해도 그 앞단인 ML 카테고리 분류가 틀리면
  실제로는 검색되지 않을 수 있다 — `legalSearch.ts`가 `ml.confidence`가 낮을 때 category를
  배제하는 안전장치는 이미 있지만(35% 미만), 이번 사례처럼 confidence가 애매하게 높게
  나오면서 카테고리 자체가 틀리는 경우까지는 막지 못한다.
- RAG 검색은 순수 코사인 유사도만 쓰면 같은 카테고리 내에서 무관한 조항이 더 높은
  점수를 받는 경우가 있어(예: "천재지변" 사례가 "폐업" 조항보다 낮은 순위로 밀림),
  청크의 `keywords`가 분쟁유형/상담 텍스트에 실제로 등장하면 가중치를 더하는
  하이브리드(임베딩+키워드) 방식을 쓴다.

## success_rate 재산정 (ML 수치가 낮을 때 법령 근거로 Rerank/Override)
공공데이터 표본 부족(특히 체육시설/헬스장·화장품/미용·부동산/임대차·식품·여행/숙박처럼
실 사례가 적은 카테고리)으로, 실제로는 환급 사유가 명확한데도 ML의 `success_rate`
(분류 신뢰도 + 유사사례 평균 가중치)가 낮게 나오는 경우가 있다. `app/api/agent/route.ts`는
이를 보정하기 위해, RAG로 법령 조항이 검색되면 LLM이 항상
`[REFERENCE LEGAL CONTEXT]`의 조항을 사용자가 명시한 사실관계와 하나하나 대조해 법적
타당성을 직접 판단하고(`legal_success_estimate`), 그 값이 ML 값보다
`MIN_LEGAL_OVERRIDE_MARGIN`(10%p) 이상 높을 때만 `success_rate`를 그 값으로
대체한다(`resolveSuccessRate()`, 상향 전용). 조건이 안 맞거나 LLM이 근거 부족으로 ML 값과
동일한 값을 반환하면 항상 ML 산출값을 그대로 쓴다(과장 방지). 최종
값의 출처는 `success_rate_basis`(`"ml_similarity"` | `"legal_reasoning"`) 필드로 표시되며,
`ResultReport.tsx`가 `legal_reasoning`일 때 게이지 아래에 "법령 근거 기반 추정" 배지를
보여준다.
- 규칙 기반 폴백(`fallbackAgent.ts`)은 OpenAI를 호출하지 않으므로 이 재산정을 수행하지
  않고 항상 `success_rate_basis: "ml_similarity"`다. 다만 결제/계약일(`date`)이 있으면
  `lib/dateUtils.ts`의 `computeElapsedDays()`(원래 `route.ts`에만 있던 걸 두 파일이 공유하도록
  분리)로 경과일수를 계산해, 7일 이내면 +10, 90일 초과면 -10을 ML `success_rate`에 더하는
  완만한 보정을 적용한다(`applyDateAdjustment()`, 5~97로 clamp). 카테고리별 구체적인
  법령 조건(예: 헬스장 위약금의 정확한 산정식) 판단은 여전히 LLM만 할 수 있으므로, 이
  보정은 "최근일수록 대체로 유리하다"는 범용 경향만 반영하는 근사치다 — `legal_reasoning`
  기반 재산정과 섞이지 않도록 `success_rate_basis`는 그대로 `ml_similarity`로 둔다.
  `estimated_refund`에도 "(결제/계약일로부터 N일 경과)"를 덧붙여 폴백 모드에서도 날짜가
  화면에 반영되게 했다(`legal_success_reasoning`은 `isLegalGrade`일 때만 UI에 노출되므로
  폴백에서는 어차피 안 보인다 — 그래서 날짜 정보를 항상 보이는 `estimated_refund` 쪽에 붙였다).
- 경계값 불연속 수정: "ML success_rate < 50%"라는 고정 임계값 대신, `legal_success_estimate`가
  ML 값보다 `MIN_LEGAL_OVERRIDE_MARGIN`(10) 이상 높을 때만 override하는 단방향(상향만) 규칙으로
  바꿨다 — 49.9%와 50.4%처럼 근거 차이가 거의 없는데 임계값 하나로 결과가 완전히 갈리는
  문제를 없앴다.
- **알려진 한계**: 한 사안에 여러 조항이 검색될 때(예: 콘도미니엄 전용 조항 KB_TRAVEL_04와
  일반 여행업 조항 KB_TRAVEL_01이 함께 검색되는 경우), 검색 순위상 더 관련성 높은 조항이
  1번으로 제공되어도 LLM이 더 일반적인 조항을 인용하는 경향이 반복 관찰되었다("1번을
  우선하라"는 지시로도 완전히 해결되지 않음). 즉 여러 조항이 경쟁할 때 어떤 조항이
  최종 인용되는지는 아직 신뢰도가 낮다 — 특정 업종 전용 조항을 추가할 때는 이 한계를
  감안해야 한다(추후 개선 여지: 경쟁 조항이 있으면 Top-1만 전달하거나, 검색 단계에서
  더 구체적인 조항이 확실히 앞서도록 가중치를 강화하는 방법 등).
- hallucination 방지 핵심 기준(프롬프트 튜닝으로 확인됨): 사용자 입력을 "① 본인의 상황에
  대한 사실"(무엇을 했는지/안 했는지, 언제·왜 취소했는지 — 이건 사용자가 말한 대로 사실로
  인정)과 "② 사용자 본인이 스스로 주장하는 법령/정책 내용"(예: "n일 전 취소하면 무료로
  알고 있다" — 조항 원문이 그 구체적 수치를 직접 뒷받침하지 않는 한 사실로 인정하지 않음)
  으로 구분해야 한다. 이 구분 없이 "명확하면 재산정하라"는 식으로만 지시하면 두 가지
  실패 모드가 번갈아 발생했다: (a) 사용자의 미확인 주장까지 사실로 오인해 근거 없이
  override하거나, (b) 반대로 지시가 복잡해지면 모델이 과도하게 보수적으로 변해 명백히
  정당한 사안까지 재산정을 거부함. 프롬프트는 간결하게 유지하되 이 ①/② 구분만은 명시해야
  한다. 또한 `legal_success_estimate`는 "위약금율/공제율"이 아니라 "소비자 주장이 받아들여질
  확률"이라는 걸 명시하지 않으면, 모델이 조항 속 숫자(예: "위약금 10%")를 그대로 성공확률에
  넣는 오류가 관찰되었다(temperature를 0.4→0.2로 낮춰 이 재산정의 실행 간 일관성도 확보함).
- 등급(`certainty_level`) 시스템: 처음엔 LLM에게 "법이 명확히 유리하면 80~95 같은 값을
  넣으라"고만 지시했는데, "미개봉+7일 이내=100% 환급"처럼 reasoning 자체가 전혀 망설임
  없는 사건도 훨씬 애매한 사건과 똑같이 85%가 나오는 문제가 실사용 중 발견됐다(단일
  예시 범위에 모델이 그대로 앵커링). 지금은 LLM이 먼저 `certainty_level`("매우 높음" |
  "높음" | "조정 필요" | "구제 어려움")과 `is_legally_clear`(불리언, "매우 높음"일 때만
  true)를 판단하고, `legal_success_estimate`는 그 등급에 대응하는 고정 범위
  (`CERTAINTY_RANGES`: 매우 높음=95~99, 높음=75~90, 조정 필요=40~65, 구제 어려움=10~30)
  안에서만 산정하도록 프롬프트로 강제한다. `resolveSuccessRate()`가 LLM이 준 숫자를 그
  범위로 다시 clamp해 등급과 숫자가 항상 일치하도록 보장한다. `ResultReport.tsx`는
  `certainty_level`이 있으면 게이지 중앙에 "85.0%" 같은 정밀해 보이는 숫자 대신 등급
  텍스트(예: "매우 높음")와 참고 범위("95~99%")를 보여주고, 등급별 색상 배지를 표시한다.
- `date`(결제/계약 일자) 반영: 이전에는 이 필드가 `ml.input.date`로 저장만 되고 ML
  분류/성공확률 계산과 LLM 프롬프트 어디에도 실질적으로 쓰이지 않았다("오늘 날짜" 자체가
  프롬프트에 없어 LLM이 경과일수를 계산할 수 없었음). `buildPrompt()`가 이제
  `computeElapsedDays()`로 계약일로부터 경과일수를 계산해 "오늘 날짜"와 함께 프롬프트에
  명시하고, 이를 "7일 이내 청약철회" 같은 시간 기준 법령 조건 판단에 쓰도록 지시한다.
  같은 상담 문장이라도 날짜만 다르면(예: 3일 전 vs 55일 전) `success_rate` 재산정 여부가
  실제로 달라지는 것을 확인했다. ML 분류 자체(`predictor.py`)에는 여전히 `date`가
  전달되지 않는다 — 반영 범위는 LLM 리포트 생성 단계로 한정된다.
- **ML 저신뢰도 경고 주입**: `ml.confidence < ML_LOW_CONFIDENCE_WARNING_THRESHOLD`(35%)이면
  `predictor.py`가 K-Means 군집 전체로 검색을 넓히는데, 짧고 일반적인 입력에서는 완전히
  무관한 `dispute_type`/`similar_cases`가 나올 수 있다(실사례로 확인: "이어폰 환불 가능?"
  에 "해외구매대행 신발 사이즈" 사례가 매칭됨). 이 경우 LLM이 그 잘못된 dispute_type을
  사실로 착각해 legal_success_reasoning에 사용자가 말한 적 없는 내용(예: "해외 사업자라
  국내법 적용 어려움")을 끌어오면서, 동시에 `legal_basis`/`estimated_refund`(고정 참고자료
  기반, RAG와 무관)는 정상적으로 "전액 환급 가능"이라고 써서 같은 리포트 안에서
  `certainty_level`/`success_rate`와 정면으로 모순되는 문제가 있었다. `buildPrompt()`가
  이 임계값 미만이면 `[ML 분석 결과]`에 "분류기 신뢰도가 매우 낮습니다" 경고를 주입해,
  dispute_type/유사사례를 사실로 취급하지 말고 오직 [사용자 입력] 원문만 근거로 삼도록
  강제한다. 시스템 프롬프트에도 legal_basis/estimated_refund와 certainty_level 계열이
  서로 모순되면 안 된다는 정합성 규칙을 별도로 추가했다.
  - **후속 수정**: 위 조치만으로는 부족했다 — `ml.category` 자체가 저신뢰도 오분류일 때
    (1) `lib/legalSearch.ts`가 RAG 검색 쿼리에 `[ml.category] ml.dispute_type`을 그대로
    포함하고 `CATEGORY_MATCH_BOOST`도 그 category로 주는 바람에 검색 자체가 엉뚱한
    카테고리 조항으로 편향됐고, (2) `buildPrompt()`가 `LEGAL_KNOWLEDGE[ml.category]`로
    `legal_basis`/`estimated_refund`의 고정 참고자료를 조회해 "이어폰 환불" 질문에
    "학원비 환급" 규정이 그대로 노출되는 문제가 실사용 중 재발했다. 두 곳 모두
    신뢰도<35%일 때는 category/dispute_type을 배제하고 사용자 원문 텍스트만 쓰도록
    고쳤다(`legalSearch.ts`의 `isCategoryReliable`, `buildPrompt()`의 `knowledge` 조회를
    `DEFAULT_LEGAL_KNOWLEDGE`로 강제).
  - **UI**: `ResultReport.tsx`의 게이지에서 "85.0%" 같은 % 숫자 표시를 완전히 제거했다
    (`certainty_level`이 있으면 등급 텍스트만, 없으면(`ml_similarity`) `gaugeColor()`의
    3단계 정성적 라벨("구제 가능성 높음/보통/낮음")만 표시 — 어느 경로든 정밀해 보이는
    숫자를 보여주지 않는다). 링의 채움 정도(시각적 진행률)는 내부적으로 여전히 숫자
    `success_rate`를 쓰지만 텍스트로는 노출하지 않는다.
- 새 카테고리를 추가/변경하면 `lib/legalKnowledge.ts`와 `components/DisputeForm.tsx`의
  `CATEGORY_OPTIONS`도 함께 갱신해야 한다 (세 곳의 카테고리 문자열이 일치해야 함).

## 디자인 톤앤매너
Slate & Deep Blue 테마. `tailwind.config.ts`의 `brand` 팔레트와 `slate-950` 배경을 기준으로
카드형 UI(`rounded-2xl`, `border-white/10`, `shadow-card`)를 유지한다.

## 코드 스타일
- 사소한 확인 질문 없이 명세서에 정의된 컴포넌트/엔드포인트를 즉시 구현
- TypeScript strict 모드, 불필요한 주석/추상화 지양

- Communication: Always explain progress, errors, and summaries in Korean.