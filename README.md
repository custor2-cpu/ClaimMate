# ClaimMate

소비자의 비정형 상담/피해 글을 입력받아 **Pandas/NumPy 정제 → Scikit-learn 분류/예측 →
RAG 법령 검색 → OpenAI LLM Agent 인사이트 리포트** 파이프라인을 수행하는 풀스택 AI
웹 애플리케이션입니다.

**🔗 배포**: [claim-mate-five.vercel.app](https://claim-mate-five.vercel.app)
**📄 최종 개발 보고서**: [docs/SPEC_ClaimMate_Final.md](docs/SPEC_ClaimMate_Final.md)

## 주요 기능

- 자유 서술형 소비자 피해 상담 내용을 입력하면 분쟁 유형을 자동 분류
- 공정거래위원회·한국소비자원 공공데이터 3종(모범상담 CSV, 품목별 피해구제 XML,
  표준답변 CSV) 기반 케이스뱅크(실제 약 1,399건)에서 BM25로 유사 사례 검색
  (문서 길이에 따른 코사인 유사도 왜곡 문제를 해결하기 위해 BM25로 교체)
- 구제 성공 가능성을 등급(certainty_level: 매우 높음/높음/조정 필요/구제 어려움)으로
  제시 — RAG로 검색된 소비자분쟁해결기준 법령 조항과 사용자 사실관계를 대조해, 공공데이터
  표본 부족으로 ML 수치가 부당하게 낮게 나온 경우 법령 근거로 상향 보정(Rerank).
  정밀 % 숫자는 노출하지 않고 등급/정성적 라벨만 표시
- ML 분류 신뢰도가 낮을 때(<35%) 무관한 카테고리 정보가 리포트를 오염시키지 않도록
  RAG 검색·법적 근거·프롬프트 전 구간에서 원문 텍스트만 근거로 사용하도록 차단
- 49개 법령 조항 지식베이스(임베딩 기반 하이브리드 검색)에서 관련 조항을 찾아 인용 —
  법령명·조항 원문·산정식을 UI에서 직접 확인 가능
- 법적 근거, 예상 환급 범위(계속거래 계약의 기사용분 차감 반영), 단계별 실행 계획,
  맞춤형 내용증명 초안(주소/환급계좌/첨부서류/미이행 시 조치 포함)을 GPT-4o-mini로 생성
  (`OPENAI_API_KEY` 미설정 시 규칙 기반 폴백으로 자동 대체, 이 경우도 결제/계약일
  경과일수를 반영해 완만하게 보정)
- 정보가 부족한 경우 카테고리별 규칙 엔진이 환급 계산·법령 판단에 필요한 추가 질문을
  최대 3개까지 반환하고, 사용자의 답변을 원문에 합쳐 ML 분석부터 다시 실행
- 공정거래위원회 1372 소비자상담센터 실통계(품목별 분쟁 빈도, 처리 결과 분포) 대시보드
- 배포마다 자동 갱신되는 버전 정보 표시 및 새 배포 감지 시 자동 새로고침

## 아키텍처

```
사용자 입력 (DisputeForm.tsx)
   │  text, amount, date, category
   ▼
/api/analyze (Next.js route.ts)
   │  로컬: subprocess로 api/ml_predict.py 실행
   │  Vercel 배포: 내부적으로 /api/ml_predict(Python 서버리스 함수) 호출
   ▼
api/ml_predict.py → ml_engine/
   │  cleaner.py   : PII 마스킹, 노이즈 제거, 파생변수 생성
   │  predictor.py : TF-IDF(char n-gram) + Logistic Regression 분류
   │                 + KMeans/BM25 유사사례 검색(코사인 유사도 대신 BM25로
   │                   문서 길이 왜곡 제거)
   │                 + 구제 성공확률 가중 산출
   ▼
ML 분석 결과 (category, dispute_type, success_rate, confidence, similar_cases ...)
   ▼
/api/agent (Next.js route.ts)
  │  lib/questionRules.ts: ML 결과와 상담 원문에서 카테고리별 필수 정보 누락 여부를 검사
  │  누락 있음 → next_action=ask_questions와 최대 3개 질문 반환
  │  답변 제출 시 상담 원문에 답변을 합쳐 /api/analyze부터 재실행
  │  정보 충분 → lib/legalSearch.ts가 상담 텍스트를 임베딩해 legal_kb_embedded.json(49개 법령
   │  조항)과 하이브리드(코사인 유사도+카테고리+키워드) 검색으로 Top-2 조항 검색
   │  (ML confidence<35%면 category/dispute_type을 검색에서 배제)
   │  OPENAI_API_KEY 있음 → 검색된 조항 + 결제일 경과일수를 프롬프트에 주입해
   │  GPT-4o-mini(Structured JSON Output)로 리포트 생성. LLM이 먼저 certainty_level
   │  등급을 정하고 그 범위 안에서 legal_success_estimate 산정 → ML success_rate보다
   │  뚜렷이 유리하면 그 값으로 재산정(상향 전용, resolveSuccessRate())
   │  없음/실패 → lib/fallbackAgent.ts 규칙 기반 폴백(날짜 기반 완만한 보정만 적용)
   ▼
ResultReport.tsx 등에서 최종 리포트 렌더링 (등급/정성적 라벨 기반 게이지,
인용 법령 조항 아코디언, 유사 사례 팝업, 내용증명 모달 등)
```

분석 알고리즘의 수식/근거는 [docs/분석.md](docs/분석.md), 데이터 전처리 과정은
[docs/기술스택.md](docs/기술스택.md), RAG·success_rate 재산정·저신뢰도 오염 차단 등
전체 개발 내역은 [docs/SPEC_ClaimMate_Final.md](docs/SPEC_ClaimMate_Final.md)에
정리되어 있습니다.

## 기술 스택

| 영역 | 기술 |
|---|---|
| 프론트엔드 | Next.js 14 (App Router), React, TypeScript, Tailwind CSS, Recharts, Framer Motion |
| 백엔드 API | Next.js Route Handlers (Node.js runtime) |
| ML 파이프라인 | Python, Pandas, NumPy, Scikit-learn (TF-IDF, Logistic Regression, KMeans), BM25(자체 구현), SciPy, joblib |
| RAG 검색 | OpenAI `text-embedding-3-small`, 로컬 JSON + 코사인 유사도(브루트포스) + 키워드 가중치 |
| LLM Agent | OpenAI API (`gpt-4o-mini`, Structured JSON Output, temperature 0.2) |
| 배포 | Vercel (Next.js 프레임워크 + Python 서버리스 함수 동시 배포) |

## 프로젝트 구조

```
app/
  page.tsx                 # 메인 페이지 (1372 실통계 대시보드 포함)
  layout.tsx                # 루트 레이아웃
  api/analyze/route.ts     # ML 분석 라우트 (로컬 subprocess / 배포 시 내부 fetch 분기)
  api/agent/route.ts       # LLM 리포트 생성 라우트
  api/version/route.ts     # 배포 버전 조회 (자동 업데이트 감지용)
components/
  DisputeForm.tsx           # 상담 입력 폼 (단계별 분석 대기 표시 포함)
  ResultReport.tsx          # 최종 리포트 렌더링 (등급 게이지, 조항 아코디언)
  SimilarCaseModal.tsx      # 유사 사례 전체 내용 팝업
  NoticeLetterModal.tsx     # 내용증명 초안 모달
  ActionChecklist.tsx       # 실행 계획/증빙 자료 체크리스트
  StatCharts.tsx             # 1372 실통계 대시보드 차트
  Header.tsx / VersionWatcher.tsx  # 헤더, 배포 버전 자동 감지·새로고침
lib/
  questionRules.ts           # 카테고리별 추가 질문 및 누락 정보 판별 규칙
  legalSearch.ts           # RAG 검색 (쿼리 임베딩 + 하이브리드 유사도, 저신뢰도 배제)
  legalKnowledge.ts        # 정적 법적 근거 지식(폴백/저신뢰도용)
  fallbackAgent.ts          # OPENAI_API_KEY 없을 때 규칙 기반 리포트 생성
  dateUtils.ts               # 결제/계약일 경과일수 계산 (route.ts/fallbackAgent.ts 공유)
  types.ts / openai.ts / utils.ts
  knowledge_base/
    legal_kb.json          # 법령 조항 청크 원본 (49개)
    legal_kb_embedded.json # 임베딩 빌드 산출물 (scripts/build_legal_kb.py 결과)
scripts/
  build_legal_kb.py        # legal_kb.json -> legal_kb_embedded.json 임베딩 빌드
  fetch_1372_api.py         # 1372 공공데이터 -> public/consumer_stats.json 생성
api/
  ml_predict.py             # Vercel Python 서버리스 함수 진입점 (/api/ml_predict)
  ml_engine/
    cleaner.py             # 텍스트 전처리 (Pandas/NumPy)
    predictor.py           # 분류/유사사례 검색 (Scikit-learn + BM25)
    build_case_bank.py     # 공공데이터 3종 → case_bank_data.json 빌드 스크립트
    case_bank_data.json    # 정제된 케이스뱅크 (약 1,399건)
data/raw/                  # 공정거래위원회·한국소비자원 원본 데이터
docs/                      # 명세서, 기술 문서 (SPEC_ClaimMate_Final.md가 종합 보고서)
```

## 로컬 개발

```bash
npm install
python -m pip install -r api/requirements.txt
cp .env.example .env.local   # OPENAI_API_KEY 입력 (선택, 없으면 폴백 리포트 사용)
npm run dev
```

- `PYTHON_BIN` 환경 변수로 Python 인터프리터 경로를 지정할 수 있습니다 (기본:
  `python` → `python3` 순 탐색).
- `python api/ml_predict.py`는 stdin으로 JSON을 받아 stdout으로 JSON을 반환하는
  CLI로도 동작하며, Vercel Python Runtime의 `handler` 클래스로도 동작합니다
  (배포 시 `/api/ml_predict` 경로로 노출).
- `npm run dev`와 `npm run build`를 동시에 실행하지 마세요. 둘 다 `.next/`에
  동시에 쓰기 때문에 캐시가 깨질 수 있습니다.

### ML 학습 데이터 재생성

`data/raw/`의 원본 데이터(모범상담 CSV, 품목별 피해구제 XML, 표준답변 CSV)는
일부 카테고리 라벨이 없거나 신뢰할 수 없으므로, 다음 스크립트가 제목/본문 키워드
매칭(또는 표준답변의 경우 품목명 직접 매핑)으로 카테고리를 부여해
`api/ml_engine/case_bank_data.json`을 만듭니다.

```bash
python api/ml_engine/build_case_bank.py
```

원본 데이터를 교체하거나 카테고리 키워드를 수정하면 위 스크립트를 다시 실행해야
합니다. 케이스뱅크가 바뀌면 `predictor.py`의 학습 모델 캐시(`_model_cache.joblib`)가
자동으로 재생성됩니다.

### RAG 법령 지식베이스 재생성

`lib/knowledge_base/legal_kb.json`(법령 조항 원본)을 수정했다면 임베딩을 다시 빌드해야
합니다. `OPENAI_API_KEY`가 필요하며, 이 스크립트는 로컬 1회성 도구라
`api/requirements.txt`에는 포함하지 않습니다(`pip install openai` 별도 필요).

```bash
pip install openai
python scripts/build_legal_kb.py
```

## 배포 (Vercel)

이 프로젝트는 **Next.js 프레임워크와 Python 서버리스 함수를 같은 배포에 함께 올립니다.**
`vercel.json`에 `"framework": "nextjs"`를 명시해야 하며, Python 진입점
(`api/ml_predict.py`)은 Next.js API 라우트와 **다른 경로**에 두어야 합니다 — 같은
경로를 쓰면 빌드 시 두 런타임이 같은 출력 슬롯을 두고 충돌해 인접 함수까지
오작동할 수 있습니다.

```bash
vercel link --project <project-name>
vercel deploy --prod
```

## 면책 조항

본 서비스의 분석 결과는 참고용 정보이며 법적 효력을 갖지 않습니다. 법령 지식베이스
(`legal_kb.json`)는 원문 대조 없이 작성된 참고용 요약이므로, 정확한 상담은 1372
소비자상담센터 또는 관련 법률 전문가를 통해 확인해 주세요.
