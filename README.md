# ClaimMate

소비자의 비정형 상담/피해 글을 입력받아 **Pandas/NumPy 정제 → Scikit-learn 분류/예측 →
OpenAI LLM Agent 인사이트 리포트** 파이프라인을 수행하는 풀스택 AI 웹 애플리케이션입니다.

**🔗 배포**: [claim-mate-five.vercel.app](https://claim-mate-five.vercel.app)

## 주요 기능

- 자유 서술형 소비자 피해 상담 내용을 입력하면 분쟁 유형을 자동 분류
- 공정거래위원회 소비자 민원학습데이터(실제 425건 + 수작성 보강 사례) 기반 유사 사례 검색
- 구제 성공 확률 산출
- 법적 근거, 예상 환급 범위, 단계별 실행 계획, 맞춤형 내용증명 초안을 GPT-4o-mini로 생성
  (`OPENAI_API_KEY` 미설정 시 규칙 기반 폴백으로 자동 대체)
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
   │                 + KMeans/Cosine Similarity 유사사례 검색
   │                 + 구제 성공확률 가중 산출
   ▼
ML 분석 결과 (category, dispute_type, success_rate, similar_cases ...)
   ▼
/api/agent (Next.js route.ts)
   │  OPENAI_API_KEY 있음 → GPT-4o-mini(Structured JSON Output)로 리포트 생성
   │  없음/실패 → lib/fallbackAgent.ts 규칙 기반 폴백
   ▼
ResultReport.tsx 등에서 최종 리포트 렌더링
```

분석 알고리즘의 수식/근거는 [docs/분석.md](docs/분석.md), 데이터 전처리 과정은
[docs/기술스택.md](docs/기술스택.md)에 상세히 정리되어 있습니다.

## 기술 스택

| 영역 | 기술 |
|---|---|
| 프론트엔드 | Next.js 14 (App Router), React, TypeScript, Tailwind CSS, Recharts, Framer Motion |
| 백엔드 API | Next.js Route Handlers (Node.js runtime) |
| ML 파이프라인 | Python, Pandas, NumPy, Scikit-learn (TF-IDF, Logistic Regression, KMeans), SciPy, joblib |
| LLM Agent | OpenAI API (`gpt-4o-mini`, Structured JSON Output) |
| 배포 | Vercel (Next.js 프레임워크 + Python 서버리스 함수 동시 배포) |

## 프로젝트 구조

```
app/
  page.tsx                 # 메인 페이지
  api/analyze/route.ts     # ML 분석 라우트 (로컬 subprocess / 배포 시 내부 fetch 분기)
  api/agent/route.ts       # LLM 리포트 생성 라우트
  api/version/route.ts     # 배포 버전 조회 (자동 업데이트 감지용)
components/                # UI 컴포넌트 (DisputeForm, ResultReport, VersionWatcher 등)
lib/                       # 타입, 법적 근거 지식베이스, 폴백 리포트 생성기
api/
  ml_predict.py            # Vercel Python 서버리스 함수 진입점 (/api/ml_predict)
  ml_engine/
    cleaner.py             # 텍스트 전처리 (Pandas/NumPy)
    predictor.py           # 분류/유사사례 검색 (Scikit-learn)
    build_case_bank.py     # 원본 공공데이터 CSV → case_bank_data.json 빌드 스크립트
data/raw/                  # 공정거래위원회 원본 CSV
docs/                      # 명세서, 기술 문서
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

`data/raw/`의 원본 CSV는 카테고리 라벨이 없으므로, 다음 스크립트가 제목/본문
키워드 매칭으로 카테고리를 부여해 `api/ml_engine/case_bank_data.json`을 만듭니다.

```bash
python api/ml_engine/build_case_bank.py
```

원본 CSV를 교체하거나 카테고리 키워드를 수정하면 위 스크립트를 다시 실행해야 합니다.

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

본 서비스의 분석 결과는 참고용 정보이며 법적 효력을 갖지 않습니다. 정확한 상담은
1372 소비자상담센터를 이용해 주세요.
