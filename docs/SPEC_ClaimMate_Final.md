# SPEC_ClaimMate_Final.md — ClaimMate 프로젝트 최종 개발 보고서

---

## 1. 개요

**ClaimMate**는 소비자의 비정형 상담/피해 서술문을 입력받아, 분쟁 유형을 자동 분류하고
구제 성공 가능성과 법적 근거, 실행 계획, 내용증명 초안까지 한 번에 제공하는 AI 소비자
피해구제 분석 에이전트다. 공정거래위원회 공공데이터(소비자 민원학습데이터)와
소비자분쟁해결기준 법령 지식을 결합해, 통계적 ML 추론과 LLM 기반 법률 추론을 함께
활용하는 하이브리드 파이프라인으로 동작한다.

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
                   → Cosine Similarity로 Top-3 유사사례 검색
                   → success_rate = 0.55×confidence + 0.45×(Top-3 평균 base_success_rate)
        │  결과: category, dispute_type, success_rate, confidence, similar_cases, derived_features
        ▼
[3] /api/agent — RAG 검색 + LLM 리포트 생성 (Node.js)
    lib/legalSearch.ts : 상담 텍스트를 text-embedding-3-small로 쿼리 임베딩
                          → legal_kb_embedded.json(42개 법령 조항 청크)과 코사인 유사도
                          → 카테고리 일치 가중치 + 키워드 매칭 가중치(하이브리드 검색)
                          → Top-2 조항 반환
    app/api/agent/route.ts:
      - 검색된 조항을 [REFERENCE LEGAL CONTEXT]로 프롬프트에 주입
      - GPT-4o-mini(Structured JSON Output, temperature 0.2)가
        legal_basis / referenced_clauses / estimated_refund / action_plan /
        proof_documents / notice_letter_template 생성
      - legal_success_estimate: 검색된 조항 + 사용자가 명시한 "본인 상황 사실"만 근거로
        독립 산정한 구제 성공률. ML 값보다 10%p 이상 높을 때만 최종 success_rate를
        그 값으로 대체(rerank) — 하향 조정은 하지 않는 상향 전용 보정
      - OPENAI_API_KEY 없음/RAG 실패/LLM 실패 시 → lib/fallbackAgent.ts 규칙 기반 폴백
        (이 경우 success_rate_basis는 항상 "ml_similarity")
        │  결과: AnalysisReport (success_rate, success_rate_basis, referenced_clauses, ...)
        ▼
[4] ResultReport.tsx 등에서 렌더링
    - 구제 성공률 게이지 (success_rate_basis === "legal_reasoning"이면
      "법령 근거 기반 추정" 배지 표시)
    - 인용 법령 조항 아코디언(RAG 결과가 있을 때만)
    - 과거 유사 사례 Top-3, 단계별 실행 계획, 준비할 증빙 자료, 내용증명 모달
```

---

## 3. 데이터 및 전처리

공정거래위원회 "소비자 민원학습데이터 모범상담 사례" 원본 CSV(565건, 사건번호/제목/
내용/답변 4개 컬럼만 존재, 카테고리 라벨 없음)를 `api/ml_engine/build_case_bank.py`가
제목·본문 키워드 매칭으로 14개 카테고리에 배정해 `case_bank_data.json`(425건)으로
정제한다. 답변 어조의 긍정/부정 신호 단어 빈도로 `base_success_rate`(처리결과 근사치)를
함께 산출한다.

카테고리별 실데이터 분포가 불균형(의료/병원 153건 vs 체육시설/헬스장 4건)해, 표본이
희소한 카테고리는 `predictor.py`의 `SYNTHETIC_CASES`(수작성 94건)로 보강한다.

세부 전처리(PII 마스킹, 노이즈 제거, 파생변수 생성 로직)는
[docs/기술스택.md](기술스택.md) 참고.

---

## 4. ML 분류 / 유사사례 검색

- **벡터화**: `TfidfVectorizer(analyzer="char_wb", ngram_range=(2,4))` — 형태소 분석기
  대신 문자 n-gram을 사용해 조사·어미 변화에 강건하면서도 서버리스 환경에 가볍게 동작
- **분류**: `LogisticRegression`으로 14개 카테고리 다중분류, `predict_proba()`의 최댓값을
  confidence로 사용
- **유사사례 검색**: confidence≥35%면 같은 카테고리 내에서, 미만이면 `KMeans` 군집
  전체로 검색 범위를 넓혀 Top-3 코사인 유사도 매칭
- **구제 성공확률**: `0.55×confidence + 0.45×(Top-3 base_success_rate 평균)`

수식과 설계 근거의 상세 설명은 [docs/분석.md](분석.md) 참고.

**알려진 한계**: 사용자가 폼에서 직접 고른 카테고리는 `input.category_hint`로 저장만
될 뿐 분류 로직에 실제로 입력되지 않는다. 또한 신뢰도<35% 폴백(K-Means 군집 확장)은
때때로 무관한 카테고리의 사례를 끌어와 `dispute_type`이나 `success_rate`를 왜곡시킬 수
있다(개발 중 "콘도 취소" 사례에서 실제로 관측·기록됨).

---

## 5. RAG 법령 지식베이스

`docs/SPEC_RAG.md` 명세에 따라 구축한 경량 RAG 검색 엔진.

### 5.1 지식베이스
`lib/knowledge_base/legal_kb.json` — 14개 업종 + 일반(cross-cutting) 조항, 총 42개 청크.
각 청크는 `{id, category, topic, law_name, clause_summary, refund_formula, keywords}`
스키마를 따른다. 기존 `lib/legalKnowledge.ts`(카테고리당 요약 1건)를 세분화·확장한
것으로, 소비자분쟁해결기준의 주요 시나리오(중도해지, 청약철회, 불가항력, 폐업 등)를
업종별로 나눠 담았다.

### 5.2 임베딩 빌드
`scripts/build_legal_kb.py`가 각 청크를 `text-embedding-3-small`로 1회 임베딩해
`lib/knowledge_base/legal_kb_embedded.json`(빌드 산출물, 저장소에 커밋됨)으로 저장한다.
법령 개정 등으로 `legal_kb.json`을 수정하면 이 스크립트를 재실행해야 한다. 로컬 1회성
개발 도구라 `api/requirements.txt`(Vercel Python 함수 번들)에는 포함하지 않는다.

### 5.3 검색 (`lib/legalSearch.ts`)
사용자 상담 텍스트를 쿼리 임베딩한 뒤, 사전 임베딩된 42개 청크와 **하이브리드
검색**(코사인 유사도 + 카테고리 일치 가중치 + 키워드 매칭 가중치)으로 Top-2 조항을
반환한다. 순수 코사인 유사도만 쓰면 "천재지변" 같은 명시적 키워드가 있는 조항이
무관한 조항보다 낮은 순위로 밀리는 사례가 실제로 관측되어, 키워드 가중치를 추가로
도입했다.

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

- LLM은 RAG로 검색된 조항이 있으면 **항상** `legal_success_estimate`(법령 조항 +
  사용자가 명시한 "본인 상황 사실"만 근거로 독립 산정한 구제 성공률)를 함께 산출한다.
- `legal_success_estimate`가 ML `success_rate`보다 **10%p(`MIN_LEGAL_OVERRIDE_MARGIN`)
  이상 높을 때만** 최종 `success_rate`를 그 값으로 대체한다. 낮게 나와도 하향 조정은
  하지 않는다(상향 전용 보정).
- 최종 값의 출처는 `success_rate_basis`(`"ml_similarity"` | `"legal_reasoning"`)로
  표시되고, `legal_reasoning`일 때 UI 게이지 아래 "법령 근거 기반 추정" 배지가 뜬다.

### 6.1 설계 변경 이력 (경계값 불연속 문제)
최초 구현은 "ML success_rate < 50%일 때만 재산정 시도"라는 고정 임계값을 썼는데,
49.9%와 50.4%처럼 근거 차이가 거의 없는 두 사안의 결과가 임계값 하나를 사이에 두고
완전히 달라지는 문제가 실사용 중 발견되었다. 현재는 ML 값의 높낮이와 무관하게 항상
법령 기반 판단을 시도하고, "법령이 ML보다 뚜렷이 유리할 때만" 위 마진 규칙으로
한 방향(상향)으로만 반영해 이 불연속을 제거했다.

### 6.2 hallucination 방지 핵심 기준
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
3. `temperature`를 0.4→0.2로 낮춰 이 재산정의 실행 간 일관성을 확보했다(동일 입력에
   대해 5회 연속 동일 결과 확인).

---

## 7. 배포 아키텍처

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

---

## 8. 알려진 한계 및 향후 과제

| 항목 | 현재 상태 | 개선 방향 |
|---|---|---|
| 카테고리 힌트 미반영 | 사용자가 고른 카테고리가 ML 분류에 입력되지 않음 | 힌트를 피처 또는 사전 필터로 반영 |
| 저신뢰도 유사사례 검색 | confidence<35%일 때 K-Means 군집 확장이 무관한 카테고리를 끌어올 수 있음 | 법령 키워드 매칭으로 카테고리 재판단하는 보조 로직(제안했으나 미구현) |
| 다중 조항 인용 우선순위 | 여러 법령 조항이 함께 검색되면, 검색 순위상 더 관련성 높은 조항이 있어도 LLM이 더 일반적인 조항을 인용하는 경향이 반복 관측됨(프롬프트로 완전히 해결되지 않음) | Top-1만 전달하거나 검색 단계에서 특정 조항의 가중치를 더 강하게 부여 |
| 법령 지식베이스 정확도 | `legal_kb.json`은 원문 대조 없이 일반 지식으로 작성한 참고용 요약(기존 `legalKnowledge.ts`와 동일 수준) | 국가법령정보센터 Open API 등으로 원문 대조·주기적 갱신 |
| 사례 유사도 검색 | 케이스뱅크 유사사례 검색은 여전히 TF-IDF 기반(임베딩 미적용) | RAG처럼 임베딩 기반으로 고도화 (제안했으나 범위상 미착수) |
| `success_rate` LLM 변동성 | temperature 0.2로 낮췄으나 LLM 기반 판단이라 완전한 결정론은 아님 | 필요 시 더 낮은 temperature 또는 다수결(self-consistency) 검토 |

---

## 9. 기술 스택 요약

| 영역 | 기술 |
|---|---|
| 프론트엔드 | Next.js 14 (App Router), React, TypeScript, Tailwind CSS, Recharts, Framer Motion |
| 백엔드 API | Next.js Route Handlers (Node.js runtime) |
| ML 파이프라인 | Python, Pandas, NumPy, Scikit-learn(TF-IDF, Logistic Regression, KMeans), SciPy, joblib |
| RAG 검색 | OpenAI `text-embedding-3-small`, 로컬 JSON + 코사인 유사도(브루트포스) |
| LLM Agent | OpenAI `gpt-4o-mini`, Structured JSON Output, temperature 0.2 |
| 배포 | Vercel (Next.js 프레임워크 + Python 서버리스 함수 동시 배포) |
| 자동화 | Git 커밋 해시 기반 버전 감지·자동 새로고침(VersionWatcher) |

---

## 10. 개발 이력 요약

1. 초기 파이프라인 구축(Pandas 전처리, Scikit-learn 분류, OpenAI 리포트 생성) 및 Vercel 배포
2. 배포 버전 표시 및 자동 업데이트 감지 기능 추가
3. `/api/agent` 500 오류 근본 원인 수정(Python/Next.js 함수 경로 충돌) — 자동 버전 갱신과 함께 커밋
4. 데이터 전처리·ML 알고리즘 문서화(`기술스택.md`, `분석.md`)
5. README 작성
6. `docs/SPEC_RAG.md` 명세에 따라 RAG 법령 검색 엔진 도입, 하이브리드(임베딩+키워드) 검색으로 개선
7. `success_rate`를 법령 근거로 재산정(Rerank/Override)하는 기능 추가, 경계값 불연속 수정 및 콘도미니엄업 조항 보강
8. 본 최종 보고서 작성

---

## 11. 면책 조항

본 서비스의 분석 결과는 참고용 정보이며 법적 효력을 갖지 않는다. 법령 지식베이스는
원문 대조 없이 작성된 참고용 요약이므로, 실제 대응 전 반드시 1372 소비자상담센터 또는
관련 법률 전문가의 확인을 거쳐야 한다.
