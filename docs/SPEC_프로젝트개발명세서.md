Markdown
# [CLAUDE.md] AI 데이터 분석 Agent 개발 프로젝트 명세서 (ClaimMate)

---

## 1. 프로젝트 개요
* **프로젝트명**: ClaimMate (클레임메이트)
* **목적**: 소비자의 비정형 상담/피해 글을 입력받아 **Pandas/NumPy 정제 ➔ Scikit-learn 머신러닝 분류/예측 ➔ LLM Agent 실행형 인사이트 리포트 도출** 파이프라인을 온전히 수행하는 풀스택 AI 웹 애플리케이션 구축[cite: 1]
* **데이터셋**: 한국소비자원 소비자상담 표준답변 및 피해구제 정보 공공데이터[cite: 1]
* **개발 환경 및 배포**: Next.js 14 App Router + Vercel Serverless (TypeScript & Python)[cite: 1]
* **개발 방식**: Claude Code 기반 자율 에이전트 빌드

---

## 2. 기술 스택 및 역할 분담

| 레이어 | 기술 스택 | 담당 역할 |
| :--- | :--- | :--- |
| **Frontend** | Next.js 14, React, TypeScript, Tailwind CSS, Lucide React, Recharts, Framer Motion | 반응형 상담 입력 폼, 구제 확률 게이지, 인터랙티브 통계 차트, 원클릭 내용증명 복사 뷰어[cite: 1] |
| **Data Engine** | Python 3.12, Pandas, NumPy[cite: 1] | 비정형 문자열 결측치/공백 필터링, 정규표현식 노이즈 제거, 텍스트 길이 및 불만 키워드 파생변수 생성[cite: 1] |
| **ML Engine** | Scikit-learn (TF-IDF, Logistic Regression / Random Forest, K-Means)[cite: 1] | 비정형 텍스트 수치 벡터화, 분쟁 유형 자동 분류, 과거 유사 사례 Top-3 매칭, 구제 성공 확률 스코어링[cite: 1] |
| **LLM Agent** | OpenAI API (`gpt-4o-mini` Structured JSON Output)[cite: 1] | ML 모델 추론값 종합, 소비자분쟁해결기준 매핑, 맞춤형 내용증명 템플릿 및 단계별 조치 체크리스트 생성[cite: 1] |
| **Database** | (선택 확장) Supabase PostgreSQL | 상담 분석 이력 저장 및 통계 적재 |

---

## 3. 엔드투엔드 파이프라인 아키텍처

```text
[1. 사용자 입력]
 ├─ 비정형 피해 상담 내역 (예: "헬스장 1년권 결제 후 2주 만에 환불 요청했으나 위약금 50% 요구")
 └─ 피해 금액, 결제 일자, 품목 카테고리
        │
        ▼
[2. Pandas / NumPy 전처리 파이프라인] (/api/ml_engine/cleaner.py)
 ├─ df.dropna() 및 빈 공백/노이즈 텍스트 필터링[cite: 1]
 ├─ 정규표현식 기반 불필요한 특수문자/개인정보 마스킹[cite: 1]
 ├─ 텍스트 길이, 단어 수, '환불/위약금/하자' 키워드 플래그 파생변수 생성[cite: 1]
        │
        ▼
[3. Scikit-learn 머신러닝 모델 추론] (/api/ml_engine/predictor.py)
 ├─ 사전 학습된 TF-IDF Vectorizer 및 Classifier 구동[cite: 1]
 ├─ 분쟁 유형(계약해지/위약금, 품질불량, 배송지연 등) 분류[cite: 1]
 ├─ 구제 성공 확률 스코어 산출 (예: 84.5%)[cite: 1]
        │
        ▼
[4. OpenAI LLM Agent 종합 인사이트 도출] (/api/agent/route.ts)
 ├─ 공정거래위원회 소비자분쟁해결기준 법적 적정 환급금 계산[cite: 1]
 ├─ 사업자 발송용 표준 답변 및 법적 대응 문구(내용증명 양식) 자동 작성
 ├─ 증빙 자료 준비 체크리스트(영수증, 대화 캡처 등) 도출
        │
        ▼
[5. Next.js 프론트엔드 대시보드 렌더링]
 └─ 분석 결과 카드, 구제 확률 차트, 내용증명 원클릭 복사 모달 표출
4. 세부 기능 및 입출력 명세
4.1. LLM 구조화된 JSON 스키마
JSON
{
  "category": "체육시설/헬스장",
  "dispute_type": "중도 해지 위약금 과다 청구",
  "success_rate": 84.5,
  "legal_basis": "소비자분쟁해결기준(체육시설업): 총 결제금액의 10% 위약금 공제 후 이용일수 일할 계산 잔여액 환급",
  "estimated_refund": "결제금액의 약 75~80%",
  "action_plan": [
    "1단계: 표준약관 위반 사실 및 적정 환급액 산정표를 사업자에게 서면 통보",
    "2단계: 환불 거부 지속 시 1372 소비자상담센터 피해구제 접수",
    "3단계: 신용카드 결제 시 카드사에 할부항변권/철회권 서면 행사"
  ],
  "proof_documents": ["이용 계약서/회원권 약관", "결제 영수증", "환불 요청 문자/카카오톡 대화 캡처"],
  "notice_letter_template": "수신: [사업자 상호]\n발신: [소비자 성명]\n\n제목: 헬스장 이용계약 해지에 따른 적정 잔여 대금 환급 요청의 건\n\n1. 귀 사의 무궁한 발전을 기원합니다.\n2. 본인은 [계약일자] 체결한 이용계약과 관련하여 공정거래위원회 고시 소비자분쟁해결기준에 의거하여 다음과 같이 정당한 환급을 요청합니다..."
}
4.2. UI/UX 구성
상단 Hero & 빠른 테스트 프리셋: "헬스장 중도 환불", "인터넷 해지 위약금", "의류 배송 지연" 등 원클릭 예시 버튼 제공.

입력 폼: 비정형 텍스트 에어리어 + 결제 금액/일자 보조 입력 필드.

진단 결과 대시보드:

구제 성공 확률 프로그레스 바 / 게이지.

법적 기준 요약 및 예상 환급 범위 카드.

단계별 조치 가이드 아코디언 / 체크리스트.

맞춤형 내용증명 뷰어 및 '클립보드 복사' 버튼.

통계 대시보드: Recharts를 이용한 소비자원 품목별 분쟁 빈도 및 처리 결과 도넛/막대 차트.

5. 디렉토리 구조
Plaintext
claim-mate/
├── app/
│   ├── layout.tsx              # 루트 레이아웃 (Inter 폰트, 메타데이터)
│   ├── page.tsx                # 메인 에이전트 대시보드
│   └── globals.css             # Tailwind CSS 설정
├── components/
│   ├── Header.tsx              # 서비스 헤더 & 네비게이션
│   ├── DisputeForm.tsx         # 비정형 상담 입력 폼
│   ├── ResultReport.tsx        # 분석 결과 종합 리포트
│   ├── ActionChecklist.tsx     # 단계별 실행 체크리스트
│   ├── NoticeLetterModal.tsx   # 내용증명 복사 모달
│   └── StatCharts.tsx          # Recharts 기반 통계 차트
├── api/
│   ├── analyze.py              # Vercel Serverless Python 전처리 & ML 추론[cite: 1]
│   └── requirements.txt        # pandas, numpy, scikit-learn, joblib[cite: 1]
├── lib/
│   ├── openai.ts               # OpenAI Client 설정
│   └── types.ts                # TypeScript 데이터 인터페이스 정의
├── public/
│   └── sample_stats.json       # 차트 시각화용 공공데이터 통계
├── CLAUDE.md                   # 프로젝트 개발 가이드라인
├── package.json
└── tailwind.config.ts
6. Claude Code 작업 및 실행 지침
자율적 구현 원칙:

사소한 확인 질문은 생략하고 명세서에 정의된 컴포넌트, API 엔드포인트, 스타일링을 즉시 코딩할 것.

필요한 npm 패키지(lucide-react, recharts, clsx, tailwind-merge, framer-motion) 및 Python 의존성을 스스로 확인하여 설치 및 구성할 것.

에러 자동 복구:

실행 또는 빌드 중 발생하는 TypeScript 타입 오류, 모듈 경로 에러 등은 터미널 로그를 확인하여 자체 수정할 것.

디자인 톤앤매너:

전문적이고 신뢰감을 주는 Slate & Deep Blue 테마 기반의 모던 카드 UI 적용.

모바일 및 데스크톱 환경 모두 완벽 지원하는 반응형 레이아웃 구성.