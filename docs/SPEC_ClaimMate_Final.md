# SPEC_ClaimMate_Final.md — ClaimMate 프로젝트 최종 개발 보고서

---

## 1. 개요

**ClaimMate**는 소비자의 비정형 상담/피해 서술문을 입력받아, 분쟁 유형을 자동 분류하고
구제 성공 가능성과 법적 근거, 실행 계획, 내용증명 초안까지 한 번에 제공하는 AI 소비자
피해구제 분석 에이전트다. 공정거래위원회·한국소비자원 공공데이터 3종과 소비자분쟁해결기준
법령 지식을 결합해, 통계적 ML 추론과 LLM 기반 법률 추론을 함께 활용하는 하이브리드
파이프라인으로 동작한다.

- **배포**: [claim-mate-five.vercel.app](https://claim-mate-five.vercel.app)
- **저장소**: [github.com/segamja/ClaimMate](https://github.com/segamja/ClaimMate)

관련 세부 문서: [docs/기술스택.md](기술스택.md)(데이터 전처리), [docs/분석.md](분석.md)
(ML 분류 알고리즘 수식), [docs/SPEC_RAG.md](SPEC_RAG.md)(RAG 최초 명세).

---

## 2. 전체 파이프라인 아키텍처

```
[1] 사용자 입력 (DisputeForm.tsx)
    text, amount, date, category(힌트, 분류에는 미반영)
        │
        ▼
[2] /api/analyze — ML 분석 (api/ml_predict.py, Python)
    cleaner.py   : PII 마스킹 → 노이즈 제거 → 파생변수(길이/단어수/키워드플래그) 생성
    predictor.py : TF-IDF(char n-gram) 벡터화
                   → Logistic Regression 카테고리 분류 (+ confidence)
                   → confidence≥35%: 동일 카테고리 내 검색 / <35%: K-Means 군집 전체로 확장
                   → BM25로 Top-3 유사사례 검색(문서 길이 정규화, §4 참고)
                   → success_rate = 0.55×confidence + 0.45×(Top-3 평균 base_success_rate)
        │  결과: category, dispute_type, success_rate, confidence, similar_cases, derived_features
        ▼
[3] /api/agent — RAG 검색 + LLM 리포트 생성 (Node.js)
  lib/questionRules.ts : ML 결과와 상담 원문을 기준으로 카테고리별 필수 정보 누락 여부 검사
              누락 시 next_action=ask_questions와 최대 3개 추가 질문 반환
              (헬스장/학원/미용 이용량, 쇼핑/전자제품 상품 상태,
               통신 해지 사유, 부동산 계약 종료·상대방 답변, 공통 피해 금액 등)
  답변 제출 시 app/page.tsx가 기존 상담 원문과 답변을 합쳐 /api/analyze부터 재실행
  정보가 충분할 때만 lib/legalSearch.ts가 상담 텍스트를 text-embedding-3-small로 쿼리 임베딩
                          → legal_kb_embedded.json(49개 법령 조항 청크)과 코사인 유사도
                          → 카테고리 일치 가중치 + 키워드 매칭 가중치(하이브리드 검색)
                          → confidence<35%면 category/dispute_type을 검색 쿼리·가중치에서
                            배제(§6.3, 저신뢰도 오염 방지)
                          → Top-2 조항 반환
    app/api/agent/route.ts:
      - 필수 정보가 충분하면 검색된 조항을 [REFERENCE LEGAL CONTEXT]로 프롬프트에 주입
      - GPT-4o-mini(Structured JSON Output, temperature 0.2)가
        legal_basis / referenced_clauses / estimated_refund / action_plan /
        proof_documents / notice_letter_template 생성
      - 결제/계약일이 있으면 computeElapsedDays()로 경과일수를 계산해 프롬프트에
        "오늘 기준 N일 경과"로 명시(§6.5)
      - LLM이 먼저 certainty_level(등급) → 그 등급 범위 안에서 legal_success_estimate를
        산정(§6.2), ML 값보다 10%p 이상 높을 때만 최종 success_rate를 대체(rerank)
      - OPENAI_API_KEY 없음/RAG 실패/LLM 실패 시 → lib/fallbackAgent.ts 규칙 기반 폴백
        (이 경우 success_rate_basis는 항상 "ml_similarity", 날짜 기반 완만한 보정만 적용)
        │  결과: AnalysisReport (success_rate, success_rate_basis, certainty_level,
        │        referenced_clauses, notice_letter_template, ...)
        ▼
[4] ResultReport.tsx 등에서 렌더링
    - 구제 성공률 게이지: certainty_level이 있으면 정밀 %를 완전히 숨기고 등급 텍스트
      ("매우 높음"/"높음"/"조정 필요"/"구제 어려움") + "법령 근거 기반 추정" 배지 표시,
      없으면(ml_similarity) 3단계 정성적 라벨("구제 가능성 높음/보통/낮음")만 표시
    - 인용 법령 조항 아코디언(RAG 결과가 있을 때만)
    - 과거 유사 사례 Top-3(클릭 시 SimilarCaseModal로 전체 내용 팝업), 단계별 실행 계획,
      준비할 증빙 자료, 내용증명 모달(NoticeLetterModal)
    - 메인 페이지(StatCharts.tsx)에 공정거래위원회 1372 소비자상담센터 실통계
      (품목별 분쟁 빈도, 처리 결과 분포) 대시보드 표시
    - 정보 부족 시 추가 질문 폼을 표시하고, 모든 답변 제출 후 기존 파이프라인을 재실행
```

---

## 3. 데이터 및 전처리

`api/ml_engine/build_case_bank.py`가 실제 공공데이터 **3종**을 병합해
`case_bank_data.json`(약 **1,399건**)으로 정제한다.

1. **공정거래위원회 "소비자 민원학습데이터 모범상담 사례" CSV** — (사건번호, 상담제목,
   상담내용, 답변내용) 4개 컬럼만 존재, 카테고리 라벨 없음.
2. **한국소비자원 "품목별 피해구제 사례" XML**(Excel 2003 SpreadsheetML) — (일련번호,
   품목, 출처, 제목, 질문, 답변) 6개 컬럼. XML 자체의 "품목" 라벨은 ClaimMate 카테고리와
   1:1로 깨끗하게 매핑되지 않아(예: "금융/보험" 안에 상조회사 폐업·유사투자자문 사기가
   섞임) 신뢰하지 않는다.
3. **한국소비자원 "소비자상담 표준답변" CSV** — (번호, 품목명, 구분, 질문, 답변) 5개
   컬럼. "품목명"은 TV/헬스장/아파트처럼 단일 품목 단위로 깨끗하게 라벨링돼 있어
   `STD_ITEM_CATEGORY_MAP`으로 직접 매핑한다. "구분" 필드(예: "계약해제·해지_위약금
   공제_소비자 귀책_개시일 이전")는 분쟁유형을 이미 구조화된 형태로 담고 있어
   `dispute_type`으로 그대로 사용한다.

1·2번 소스는 카테고리 라벨을 신뢰할 수 없어 `build_case_bank.py`의
`CATEGORY_KEYWORDS`/`_assign_category()`로 제목·본문 키워드 매칭을 통해 카테고리를
배정한다. 원본 파일명 필터링(`data/raw/*.csv` 전체가 아니라 "모범상담" 포함 CSV만 로드)과
중복 제거((제목, 답변 앞 80자) 기준), "안전" 구분(화상·주차사고 등 신체/재산 피해 배상)
제외 등 세부 규칙은 `CLAUDE.md` 참고.

카테고리별 실데이터 분포가 여전히 불균형해, 표본이 특히 희소한 카테고리(체육시설/
헬스장·화장품/미용·부동산/임대차·식품)는 `predictor.py`의 `SYNTHETIC_CASES`(수작성
예시)로 보강해 최종 `CASE_BANK`를 구성한다.

세부 전처리(PII 마스킹, 노이즈 제거, 파생변수 생성 로직)는
[docs/기술스택.md](기술스택.md) 참고.

---

## 4. ML 분류 / 유사사례 검색

- **벡터화**: `TfidfVectorizer(analyzer="char_wb", ngram_range=(2,4))` — 형태소 분석기
  대신 문자 n-gram을 사용해 조사·어미 변화에 강건하면서도 서버리스 환경에 가볍게 동작
- **분류**: `LogisticRegression`으로 14개 카테고리 다중분류, `predict_proba()`의 최댓값을
  confidence로 사용
- **유사사례 검색 범위**: confidence≥35%면 같은 카테고리 내에서, 미만이면 `KMeans` 군집
  전체로 검색 범위를 넓힘
- **유사사례 랭킹 — 코사인 유사도 → BM25 교체**: char n-gram TF-IDF 코사인 유사도는
  문서가 길수록(공공데이터 특유의 격식체+부가 설명) 벡터가 희석되어, 짧고 구어체인
  `SYNTHETIC_CASES` 예시가 사안이 다른데도 실제 공공데이터 사례보다 부당하게 높은
  유사도를 받는 역전 현상이 실사용 중 확인됐다. `_bm25_similarity()`(BM25 k1=1.5,
  b=0.75, 문서 길이를 평균 길이 대비 명시적으로 정규화)로 유사사례 검색만 교체해
  해결했다 — 분류기(`_classifier`)는 여전히 기존 TF-IDF/코사인 기반 K-Means를 그대로
  쓰므로 분류 정확도에는 영향이 없다. BM25용 원시 빈도 행렬은 `_vectorizer`와 동일
  vocabulary를 공유하는 별도 `CountVectorizer`로 만든다.
- **구제 성공확률**: `0.55×confidence + 0.45×(Top-3 base_success_rate 평균)`

수식과 설계 근거의 상세 설명은 [docs/분석.md](분석.md) 참고.

**알려진 한계**: 사용자가 폼에서 직접 고른 카테고리는 `input.category_hint`로 저장만
될 뿐 분류 로직에 실제로 입력되지 않는다. 신뢰도<35% 폴백(K-Means 군집 확장)이 여전히
무관한 카테고리의 사례를 끌어올 수 있는 근본 원인은 남아 있으나, 그 결과가 LLM
리포트를 오염시키는 하위 문제는 §6.3에서 별도로 차단했다.

---

## 5. RAG 법령 지식베이스

`docs/SPEC_RAG.md` 명세에 따라 구축한 경량 RAG 검색 엔진.

### 5.1 지식베이스
`lib/knowledge_base/legal_kb.json` — 14개 업종 + 일반(cross-cutting) 조항, 총 **49개**
청크. 각 청크는 `{id, category, topic, law_name, clause_summary, refund_formula,
keywords}` 스키마를 따른다. 초기 14개 업종 중 7개(의료/병원·보험·자동차·상조/결혼서비스·
부동산/임대차·전자제품·식품)는 업종당 2건뿐이었는데, 실제 상담 데이터에서 반복
확인됐지만 근거가 없던 시나리오(전자제품 렌탈 계약 해지, 성형외과 수술 취소 계약금,
보험금 지급 지연이자, 결혼중개업체 허위정보, 전세사기 피해자 지원, 중고차 사고이력
미고지, 배달앱 오배달)를 업종당 1건씩 추가해 보강했다.

### 5.2 임베딩 빌드
`scripts/build_legal_kb.py`가 각 청크를 `text-embedding-3-small`로 1회 임베딩해
`lib/knowledge_base/legal_kb_embedded.json`(빌드 산출물, 저장소에 커밋됨)으로 저장한다.
법령 개정 등으로 `legal_kb.json`을 수정하면 이 스크립트를 재실행해야 한다. 로컬 1회성
개발 도구라 `api/requirements.txt`(Vercel Python 함수 번들)에는 포함하지 않는다.

### 5.3 검색 (`lib/legalSearch.ts`)
사용자 상담 텍스트를 쿼리 임베딩한 뒤, 사전 임베딩된 49개 청크와 **하이브리드
검색**(코사인 유사도 + 카테고리 일치 가중치 + 키워드 매칭 가중치)으로 Top-2 조항을
반환한다. 순수 코사인 유사도만 쓰면 "천재지변" 같은 명시적 키워드가 있는 조항이
무관한 조항보다 낮은 순위로 밀리는 사례가 실제로 관측되어, 키워드 가중치를 추가로
도입했다.

`ml.confidence < 35%`(저신뢰도)일 때는 `isCategoryReliable = false`로 판정해 검색
쿼리와 유사도 가중치 계산 모두에서 `ml.category`/`ml.dispute_type`을 배제하고 사용자
원문 텍스트만 사용한다(§6.3).

지식베이스 미빌드/`OPENAI_API_KEY` 없음/임베딩 API 실패 시 `null`을 반환해, 호출부가
기존 정적 `legalKnowledge.ts` 기반 동작으로 무중단 폴백한다.

### 5.4 프롬프트 통합
검색된 조항을 `[REFERENCE LEGAL CONTEXT]`로 시스템/유저 프롬프트에 주입하고, LLM이
"제공된 조항만 인용하고 법령을 상상해 만들어내지 말 것"이라는 강한 hallucination 방지
규칙 아래 `referenced_clauses`를 채우도록 한다. 여러 조항이 함께 제공될 때 어떤 조항을
우선 인용할지는 검색 순위(1번=최고 관련도)를 명시해도 완전히 안정적이지 않다는 한계가
확인되었다(§8 참고).

---

## 6. success_rate 재산정 (법령 근거 기반 Rerank)

공공데이터 표본 부족으로 실제로는 환급 사유가 명확한데도 ML의 `success_rate`가 낮게
나오는 문제를 보정하기 위한 기능.

### 6.1 등급(certainty_level) 기반 설계
초기에는 LLM에게 "법이 명확히 유리하면 80~95 같은 값을 넣으라"고만 지시했는데,
"미개봉+7일 이내=100% 환급"처럼 reasoning 자체가 전혀 망설임 없는 사건도 훨씬 애매한
사건과 똑같이 85%가 나오는 문제(단일 예시 범위에 모델이 그대로 앵커링)가 실사용 중
발견됐다. 현재는 LLM이 먼저 `certainty_level`("매우 높음" | "높음" | "조정 필요" |
"구제 어려움")과 `is_legally_clear`(불리언, "매우 높음"일 때만 true)를 판단하고,
`legal_success_estimate`는 그 등급에 대응하는 고정 범위(`CERTAINTY_RANGES`: 매우
높음=95~99, 높음=75~90, 조정 필요=40~65, 구제 어려움=10~30) 안에서만 산정하도록
프롬프트로 강제한다. `resolveSuccessRate()`가 LLM이 준 숫자를 그 범위로 다시 clamp해
등급과 숫자가 항상 일치하도록 보장한다.

`clampedEstimate`가 ML `success_rate`보다 **10%p(`MIN_LEGAL_OVERRIDE_MARGIN`) 이상
높을 때만** 최종 `success_rate`를 그 값으로 대체한다(`basis: "legal_reasoning"`).
조건이 안 맞거나 LLM이 근거 부족으로 ML 값과 비슷한 값을 반환하면 `basis:
"ml_similarity"`를 유지한다(과장 방지, 상향 전용 — 하향 조정은 하지 않음).

`ResultReport.tsx`는 `certainty_level`이 있으면 게이지 중앙에 "85.0%" 같은 정밀해
보이는 숫자 대신 등급 텍스트와 등급별 색상 배지를 보여준다(§2 참고). 어느 경로든
정밀 % 숫자는 화면에 노출하지 않는다 — 게이지 링의 채움 정도(시각적 진행률)만
내부적으로 숫자 `success_rate`를 사용한다.

### 6.2 설계 변경 이력 (경계값 불연속 문제)
최초 구현은 "ML success_rate < 50%일 때만 재산정 시도"라는 고정 임계값을 썼는데,
49.9%와 50.4%처럼 근거 차이가 거의 없는 두 사안의 결과가 임계값 하나를 사이에 두고
완전히 달라지는 문제가 실사용 중 발견되었다. 현재는 ML 값의 높낮이와 무관하게 항상
법령 기반 판단을 시도하고, "법령이 ML보다 뚜렷이 유리할 때만" 위 마진 규칙으로
한 방향(상향)으로만 반영해 이 불연속을 제거했다.

### 6.3 저신뢰도 ML 오염 차단
`ml.confidence < 35%`일 때 `predictor.py`가 K-Means 군집 전체로 검색을 넓히면서, 짧고
일반적인 입력에서 완전히 무관한 `dispute_type`/`similar_cases`가 나올 수 있다(실사례:
"이어폰 환불 가능?"에 "해외구매대행 신발 사이즈" 사례가 매칭). 이 경우 LLM이 잘못된
정보를 사실로 착각해 reasoning을 오염시키는 문제와, `legal_basis`/`estimated_refund`
(고정 참고자료 기반)가 별도로 엉뚱한 카테고리 내용을 노출하는 문제가 각각 발견되어
두 곳 모두 수정했다.

- `buildPrompt()`가 신뢰도<35%면 `[ML 분석 결과]`에 "분류기 신뢰도가 매우 낮습니다"
  경고를 주입해, dispute_type/유사사례를 사실로 취급하지 말고 사용자 입력 원문만
  근거로 삼도록 강제한다. `legal_basis`/`estimated_refund` 참고자료 조회도
  `LEGAL_KNOWLEDGE[ml.category]` 대신 `DEFAULT_LEGAL_KNOWLEDGE`로 강제한다.
- `lib/legalSearch.ts`의 RAG 검색 쿼리·가중치도 동일 기준으로 category/dispute_type을
  배제한다(§5.3).
- 시스템 프롬프트에 legal_basis/estimated_refund와 certainty_level 계열이 서로
  모순되면 안 된다는 정합성 규칙을 별도로 추가했다.

### 6.4 hallucination 방지 핵심 기준
프롬프트 튜닝 중 실사례로 반복 검증해 확정한 규칙:

1. 사용자 입력을 **"① 본인 상황에 대한 사실"**(무엇을 했는지/안 했는지, 언제·왜
   취소했는지 — 사용자가 말한 대로 사실로 인정)과 **"② 사용자 본인이 스스로 주장하는
   법령/정책 내용"**(예: "n일 전 취소하면 무료로 알고 있다" — 조항 원문이 그 구체적
   수치를 직접 뒷받침하지 않는 한 사실로 인정하지 않음)으로 구분해야 한다. 이 구분이
   없으면 (a) 사용자의 미확인 주장을 사실로 오인해 근거 없이 override하거나, (b) 반대로
   지시가 복잡해지면 모델이 과도하게 보수적으로 변해 명백히 정당한 사안까지 재산정을
   거부하는 두 실패 모드가 번갈아 관측되었다.
2. `legal_success_estimate`는 "위약금율/공제율"이 아니라 "소비자 주장이 받아들여질
   확률"이라고 명시하지 않으면, 모델이 조항 속 숫자(예: "위약금 10%")를 그대로
   성공확률 필드에 넣는 오류가 관측되었다.
3. `temperature`를 0.4→0.2로 낮춰 이 재산정의 실행 간 일관성을 확보했다.

### 6.5 결제/계약 일자(date) 반영
`buildPrompt()`가 `computeElapsedDays()`(`lib/dateUtils.ts`)로 계약일로부터 경과일수를
계산해 "오늘 날짜"와 함께 프롬프트에 명시하고, "7일 이내 청약철회" 같은 시간 기준
법령 조건 판단에 쓰도록 지시한다. 같은 상담 문장이라도 날짜만 다르면(예: 3일 전 vs
55일 전) `success_rate` 재산정 여부가 실제로 달라지는 것을 확인했다. ML 분류 자체
(`predictor.py`)에는 여전히 `date`가 전달되지 않는다 — 반영 범위는 LLM 리포트 생성
단계로 한정된다.

`OPENAI_API_KEY`가 없어 규칙 기반 폴백(`fallbackAgent.ts`)이 동작할 때도
`applyDateAdjustment()`가 같은 `computeElapsedDays()`를 공유해 완만한 보정(7일
이내 +10, 90일 초과 -10, ML `success_rate`를 5~97로 clamp)을 적용한다. 카테고리별
구체적인 법령 조건 판단은 LLM만 할 수 있으므로 이 보정은 "최근일수록 대체로
유리하다"는 범용 경향만 반영하는 근사치이며, `success_rate_basis`는 그대로
`ml_similarity`로 유지된다. `estimated_refund`에 "(결제/계약일로부터 N일 경과)"를
덧붙여 폴백 모드에서도 날짜가 화면에 반영된다.

### 6.6 계속거래 환급액 계산 (이미 사용한 만큼 차감)
회원권/이용권처럼 정해진 횟수·기간 중 일부를 이미 사용한 계속거래 계약(헬스장·피부관리실
등)에서, LLM이 `estimated_refund`를 "결제 총액 - 위약금(10%)"으로만 계산해 이미 사용한
부분까지 환급 대상에 포함시키는 오류가 실사용 중 발견됐다(예: "10회권 800,000원 중
3회 사용" 사례에서 720,000원으로 계산 — 실제로는 미사용분 7/10인 560,000원에서
위약금을 공제해야 함). 시스템 프롬프트 지시만으로는 고쳐지지 않았고, **JSON 스키마의
`estimated_refund` 필드 `description`에 같은 규칙을 한 번 더 명시**하고 나서야
반영됐다 — 구조화 출력(Structured Output)에서는 필드 레벨 설명이 시스템 프롬프트
본문의 규칙보다 더 강하게 반영되는 경향이 있다는 뜻이다.

---

## 7. 내용증명 템플릿

`notice_letter_template`(LLM 경로)과 `fallbackAgent.ts`의 규칙 기반 템플릿 모두 다음
6개 항목을 갖추도록 보강했다: ① 제목("내용증명") ② 수신인·수신인 주소 / 발신인·발신인
주소·연락처(플레이스홀더 포함) ③ 건명 ④ 번호 매긴 본문(인사말 → 계약 경위 → 피해
사실 → 법적 근거 → 예상 환급 범위 및 환급계좌 `[은행명] [계좌번호] (예금주: [소비자
성명])` → 회신기한(7일) 및 미이행 시 조치) ⑤ 첨부서류 목록 ⑥ 작성일/서명란. 이전
버전은 본문 위주였고 주소/환급계좌/첨부서류/미이행 시 조치 항목이 빠져 있어 실제
발송 문서로 쓰기엔 부족했다. `components/NoticeLetterModal.tsx`가 이 템플릿을 모달로
렌더링한다.

---

## 8. 추가 질문 Agent

분석에 필요한 사실관계가 부족한 상태에서 ML·RAG·LLM이 임의로 가정하지 않도록,
`lib/questionRules.ts`의 규칙 엔진이 `/api/agent`의 첫 응답을 제어한다. 규칙 엔진은
LLM을 호출하지 않으며, ML 카테고리와 상담 원문에 이미 포함된 표현을 검사해 누락된
정보를 찾는다. 질문은 우선순위에 따라 최대 3개만 반환한다.

| 대상 | 추가 확인 정보 |
|---|---|
| 체육시설/헬스장·화장품/미용·학원/교육서비스 | 총 이용 기간/횟수와 실제 이용량 |
| 온라인쇼핑몰·전자제품·의류/패션잡화 | 개봉·사용·훼손·하자·배송 상태 |
| 통신/인터넷 | 해지 또는 환불의 직접적인 사유 |
| 부동산/임대차 | 계약 만료·해지 요청 시점과 상대방 답변 |
| 공통 | 사업자 대응 내용과 결제·피해 금액 |

정보가 부족하면 `/api/agent`는 최종 `AnalysisReport` 대신 다음 형태의 응답을 반환한다.

```json
{
  "next_action": "ask_questions",
  "questions": [
    {
      "id": "usage",
      "question": "총 이용 기간 또는 총 이용 횟수와, 지금까지 실제로 이용한 기간 또는 횟수는 얼마인가요?",
      "reason": "이미 이용한 부분을 제외해야 환급액을 계산할 수 있습니다."
    }
  ]
}
```

사용자가 질문에 답하면 `app/page.tsx`가 기존 상담 원문에 `[추가 확인 답변]`을 덧붙여
`/api/analyze`부터 전체 파이프라인을 다시 실행한다. 질문 판단은 규칙 기반으로 일관되게
수행하고, 정보가 충분해진 뒤에만 기존 RAG 검색과 GPT-4o mini 리포트 생성을 진행한다.
현재 답변과 보류 상태는 페이지 메모리에 유지되며 영속 사건 저장소는 아직 도입하지 않았다.

---

## 9. UX — 분석 대기 표시

`components/DisputeForm.tsx`는 ML 분석(1/2단계) → LLM 리포트 생성(2/2단계) 두 단계의
진행 상태를 버튼에 표시한다. 초기 구현은 고정된 예상 소요시간(12초)을 카운트다운으로
보여줬으나, 실제 처리 시간이 입력 길이·RAG 검색·OpenAI 응답 지연에 따라 크게
달라져 카운트다운이 끝나도 분석이 계속되는 경우 사용자에게 혼란을 줄 수 있었다.
현재는 각 단계 진입 시각(`stageStartedAt`)을 렌더 중 동기적으로 리셋하고 1초 간격
타이머로 **실제 경과 시간**을 세어 "2/2단계: AI 리포트 생성 중... (N초 경과)"처럼
보여주며, 20초(`LONG_WAIT_THRESHOLD_SECONDS`)를 넘기면 "평소보다 다소 걸리고
있어요" 안내를 추가로 표시한다.

---

## 10. 배포 아키텍처

- **Vercel에 Next.js 프레임워크와 Python 서버리스 함수를 함께 배포.**
  `api/ml_predict.py`(Vercel Python 서버리스 함수, `/api/ml_predict`로 노출)와
  `app/api/analyze/route.ts`(Next.js Node 함수)를 **서로 다른 경로**에 두어야 한다 —
  같은 경로에 두면 빌드 시 두 런타임이 같은 출력 슬롯을 두고 충돌해 인접 함수(예:
  `/api/agent`)까지 잘못된 런타임으로 대체되는 문제가 실제로 발생했었다(초기 배포 시
  발견·수정됨).
- `app/api/analyze/route.ts`는 로컬 개발에서는 `api/ml_predict.py`를 subprocess로,
  Vercel 배포 환경에서는 같은 배포 내의 `/api/ml_predict` 함수를 내부 HTTP로 호출한다.
- **자동 버전 갱신**: `app/api/version/route.ts` + `components/VersionWatcher.tsx`가
  60초 주기/탭 재활성화 시 배포 버전(Git 커밋 해시)을 확인해, 새 배포가 감지되면
  브라우저 캐시를 지우고 자동 새로고침한다.
- **1372 실통계 대시보드**: `scripts/fetch_1372_api.py`가 공정거래위원회 1372
  소비자상담센터 공공데이터로 `public/consumer_stats.json`을 생성하고,
  `components/StatCharts.tsx`(Recharts)가 이를 메인 페이지(`app/page.tsx`)에서
  품목별 분쟁 빈도 막대그래프와 처리 결과 분포 원그래프로 렌더링한다.

---

## 11. 알려진 한계 및 향후 과제

| 항목 | 현재 상태 | 개선 방향 |
|---|---|---|
| 카테고리 힌트 미반영 | 사용자가 고른 카테고리가 ML 분류에 입력되지 않음 | 힌트를 피처 또는 사전 필터로 반영 |
| 저신뢰도 유사사례 검색 근본 원인 | confidence<35%일 때 K-Means 군집 확장 자체는 여전히 무관한 카테고리를 끌어올 수 있음(다만 그 결과가 LLM 리포트를 오염시키는 하위 문제는 §6.3에서 차단됨) | 법령 키워드 매칭으로 카테고리 재판단하는 보조 로직(제안했으나 미구현) |
| 다중 조항 인용 우선순위 | 여러 법령 조항이 함께 검색되면, 검색 순위상 더 관련성 높은 조항이 있어도 LLM이 더 일반적인 조항을 인용하는 경향이 반복 관측됨(프롬프트로 완전히 해결되지 않음) | Top-1만 전달하거나 검색 단계에서 특정 조항의 가중치를 더 강하게 부여 |
| 법령 지식베이스 정확도 | `legal_kb.json`은 원문 대조 없이 일반 지식으로 작성한 참고용 요약(기존 `legalKnowledge.ts`와 동일 수준) | 국가법령정보센터 Open API 등으로 원문 대조·주기적 갱신 |
| `_assign_category()`의 복합 사례 오배정 | 카테고리당 키워드 최고점 하나에만 배정하는 단순 방식이라, 여러 카테고리 키워드가 동시에 등장하는 복합 사례(예: "무료 노트북 증정 상조서비스"가 "노트북" 키워드로 전자제품에 오분류)에서 여전히 오배정 가능. 대용량 위해위험 CSV(13만 건)는 이 문제가 훨씬 심해 `data/staging/`에서 별도 검토 중이며 아직 미반영 | 다중 카테고리 점수 비교, 짧은 한글 키워드의 단어 경계 처리 개선 |
| `success_rate` LLM 변동성 | temperature 0.2로 낮췄으나 LLM 기반 판단이라 완전한 결정론은 아님 | 필요 시 더 낮은 temperature 또는 다수결(self-consistency) 검토 |
| 추가 질문 판별 범위 | 현재 카테고리별 정규식 규칙으로 누락 정보를 판단하며, 자연어의 모호한 표현을 완전히 해석하지 못함 | LLM 사실 추출기를 보조적으로 도입하고, 사용자 답변을 구조화된 사실 필드로 저장 |
| 질문 상태 지속성 | 질문과 답변이 현재 브라우저 페이지 상태에만 존재해 새로고침 시 사라짐 | Supabase 등 영속 저장소와 사건별 대화 이력 도입 |

---

## 12. 기술 스택 요약

| 영역 | 기술 |
|---|---|
| 프론트엔드 | Next.js 14 (App Router), React, TypeScript, Tailwind CSS, Recharts, Framer Motion |
| 백엔드 API | Next.js Route Handlers (Node.js runtime) |
| ML 파이프라인 | Python, Pandas, NumPy, Scikit-learn(TF-IDF, Logistic Regression, KMeans), BM25(자체 구현), SciPy, joblib |
| RAG 검색 | OpenAI `text-embedding-3-small`, 로컬 JSON + 코사인 유사도(브루트포스) + 키워드 가중치 |
| LLM Agent | OpenAI `gpt-4o-mini`, Structured JSON Output, temperature 0.2 |
| 배포 | Vercel (Next.js 프레임워크 + Python 서버리스 함수 동시 배포) |
| 자동화 | Git 커밋 해시 기반 버전 감지·자동 새로고침(VersionWatcher) |

---

## 13. 개발 이력 요약

1. 초기 파이프라인 구축(Pandas 전처리, Scikit-learn 분류, OpenAI 리포트 생성) 및 Vercel 배포
2. 배포 버전 표시 및 자동 업데이트 감지 기능 추가
3. `/api/agent` 500 오류 근본 원인 수정(Python/Next.js 함수 경로 충돌) — 자동 버전 갱신과 함께 커밋
4. 데이터 전처리·ML 알고리즘 문서화(`기술스택.md`, `분석.md`), README 작성
5. `docs/SPEC_RAG.md` 명세에 따라 RAG 법령 검색 엔진 도입, 하이브리드(임베딩+키워드) 검색으로 개선
6. `success_rate`를 법령 근거로 재산정(Rerank/Override)하는 기능 추가, 경계값 불연속 수정 및 콘도미니엄업 조항 보강
7. 결제/계약 일자를 실제 ML/LLM 분석에 반영, `success_rate`를 등급(certainty_level) 기반으로 재설계
8. 저신뢰도 ML 카테고리 오염을 RAG/정적지식까지 근본 차단, 게이지 % 표시 완전 제거(등급/정성적 라벨만 표시)
9. 유사 사례 카드 클릭 시 전체 내용 팝업(SimilarCaseModal), 게이지 라벨 줄바꿈 등 UI 개선
10. 실제 공공데이터 3종(모범상담 CSV + 품목별 피해구제 XML + 표준답변 CSV) 통합으로 케이스뱅크 확대(425건 → 1,399건), 1372 API 실통계 대시보드(StatCharts) 추가
11. OpenAI 키 없는 규칙 기반 폴백에도 결제/계약일 경과일수 반영
12. 유사 사례 검색을 코사인 유사도에서 BM25로 교체해 문서 길이에 따른 왜곡 제거
13. 분석 대기 표시를 고정 카운트다운에서 단계별 실제 경과 시간으로 교체, 내용증명 템플릿에 주소/환급계좌/첨부서류/미이행 조치 항목 보강
14. 계속거래 환급액 계산에서 이미 사용한 만큼 차감하는 로직 누락 수정
15. 본 최종 보고서 갱신
16. 카테고리별 규칙 기반 추가 질문 Agent와 답변 후 전체 재분석 흐름 추가

---

## 14. 면책 조항

본 서비스의 분석 결과는 참고용 정보이며 법적 효력을 갖지 않는다. 법령 지식베이스는
원문 대조 없이 작성된 참고용 요약이므로, 실제 대응 전 반드시 1372 소비자상담센터 또는
관련 법률 전문가의 확인을 거쳐야 한다.
