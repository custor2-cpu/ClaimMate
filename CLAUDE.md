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
`data/raw/`에 공정거래위원회 "소비자 민원학습데이터 모범상담 사례" 원본 CSV가 있다. 이 CSV는
(사건번호, 상담제목, 상담내용, 답변내용) 4개 컬럼만 있고 카테고리 라벨이 없으므로,
`python api/ml_engine/build_case_bank.py`가 제목/본문 키워드 매칭으로 카테고리를 부여해
`api/ml_engine/case_bank_data.json`(약 425건)으로 정제한다. `predictor.py`는 이 JSON을
`SYNTHETIC_CASES`(카테고리별 최소 표본 확보용 수작성 예시, 특히 데이터가 희소한
체육시설/헬스장·화장품/미용·부동산/임대차·식품)와 합쳐 `CASE_BANK`를 구성한다.
- 원본 CSV를 교체하거나 `CATEGORY_KEYWORDS`를 수정하면 `build_case_bank.py`를 다시 실행해
  `case_bank_data.json`을 재생성해야 한다.
- `predictor.py`는 매 요청(subprocess)마다 재학습하면 느리므로(~7초) 학습된 모델을
  `api/ml_engine/_model_cache.joblib`에 캐시한다. 케이스뱅크가 바뀌면(파일 mtime/건수 변경)
  캐시 키가 달라져 자동으로 재학습된다. 강제로 지우려면 `_model_cache.joblib`를 삭제한다.

## RAG 법령 지식베이스 (docs/SPEC_RAG.md)
`lib/knowledge_base/legal_kb.json`에 14개 업종 + 일반 법령 조항 청크(약 40건, `law_name`/
`clause_summary`/`refund_formula`/`keywords` 스키마)가 정의되어 있다. 이를
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
- RAG 검색은 순수 코사인 유사도만 쓰면 같은 카테고리 내에서 무관한 조항이 더 높은
  점수를 받는 경우가 있어(예: "천재지변" 사례가 "폐업" 조항보다 낮은 순위로 밀림),
  청크의 `keywords`가 분쟁유형/상담 텍스트에 실제로 등장하면 가중치를 더하는
  하이브리드(임베딩+키워드) 방식을 쓴다.

## success_rate 재산정 (ML 수치가 낮을 때 법령 근거로 Rerank/Override)
공공데이터 표본 부족(특히 체육시설/헬스장·화장품/미용·부동산/임대차·식품·여행/숙박처럼
실 사례가 적은 카테고리)으로, 실제로는 환급 사유가 명확한데도 ML의 `success_rate`
(분류 신뢰도 + 유사사례 평균 가중치)가 낮게 나오는 경우가 있다. `app/api/agent/route.ts`는
이를 보정하기 위해, **ML의 `success_rate`가 낮고(<50%, `route.ts`의
`LOW_SUCCESS_RATE_THRESHOLD`) RAG로 법령 조항이 검색된 경우**, LLM이
`[REFERENCE LEGAL CONTEXT]`의 조항을 사용자가 명시한 사실관계와 하나하나 대조해 법적
타당성을 직접 판단하고, 그 결과(`legal_success_estimate`)가 ML 값과 의미 있게 다를 때만
`success_rate`를 그 값으로 대체한다(`resolveSuccessRate()`). 조건이 안 맞거나 LLM이 근거
부족으로 ML 값과 동일한 값을 반환하면 항상 ML 산출값을 그대로 쓴다(과장 방지). 최종
값의 출처는 `success_rate_basis`(`"ml_similarity"` | `"legal_reasoning"`) 필드로 표시되며,
`ResultReport.tsx`가 `legal_reasoning`일 때 게이지 아래에 "법령 근거 기반 추정" 배지를
보여준다.
- 규칙 기반 폴백(`fallbackAgent.ts`)은 OpenAI를 호출하지 않으므로 이 재산정을 수행하지
  않고 항상 `success_rate_basis: "ml_similarity"`다.
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
- 새 카테고리를 추가/변경하면 `lib/legalKnowledge.ts`와 `components/DisputeForm.tsx`의
  `CATEGORY_OPTIONS`도 함께 갱신해야 한다 (세 곳의 카테고리 문자열이 일치해야 함).

## 디자인 톤앤매너
Slate & Deep Blue 테마. `tailwind.config.ts`의 `brand` 팔레트와 `slate-950` 배경을 기준으로
카드형 UI(`rounded-2xl`, `border-white/10`, `shadow-card`)를 유지한다.

## 코드 스타일
- 사소한 확인 질문 없이 명세서에 정의된 컴포넌트/엔드포인트를 즉시 구현
- TypeScript strict 모드, 불필요한 주석/추상화 지양

- Communication: Always explain progress, errors, and summaries in Korean.