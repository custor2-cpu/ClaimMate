Markdown
# [SPEC_RAG.md] ClaimMate 법률/고시 지식베이스 기반 RAG 고도화 명세서

---

## 1. 개요 및 목적
* **기능명**: 소비자분쟁해결기준 기반 경량 RAG (Retrieval-Augmented Generation) 엔진[cite: 3]
* **목적**: 
  * 기존 Scikit-learn 머신러닝의 유사 사례 검색에 더해, 공정거래위원회 고시 **'소비자분쟁해결기준' 및 관련 법률(전자상거래법, 약관규제법 등) 조항 원문을 실시간 검색/주입**하여 LLM의 법적 근거 신뢰도 대폭 향상[cite: 3]
  * 외부 법령 Open API 연동 없이 **로컬 사전 구축 JSON 벡터 지식베이스(Offline Pre-indexed KB)**를 활용하여 0.01초 이내 초고속 응답 및 서버리스 환경 최적화 달성[cite: 5]

---

## 2. RAG 파이프라인 아키텍처

```text
[1. 1회성 빌드 단계 (오프라인/사전 인덱싱)]
공정위 고시 / 소비자원 품목별 해결기준 텍스트
        │
        ▼ (scripts/build_legal_kb.py 실행)
 ├─ 14개 주요 카테고리별 핵심 조항(환불/위약금 기준 등) 50~100개 청킹[cite: 3]
 ├─ OpenAI text-embedding-3-small 기반 임베딩 벡터 생성
 └─ `lib/knowledge_base/legal_kb_embedded.json` 파일로 프로젝트에 저장
        │
        ▼
[2. 서비스 런타임 단계 (사용자 요청 시)]
사용자 입력 (상담 텍스트, 피해 금액, 일자)
        │
        ├── ① ML 분류 및 성공확률 추론 (api/ml_predict.py)
        │
        └── ② 경량 RAG 검색 (lib/legalSearch.ts)
             ├─ 사용자 상담글 임베딩 생성 (text-embedding-3-small)
             ├─ ML 예측 카테고리 기반 필터링 + 코사인 유사도 검색 (Top-2 추출)[cite: 4]
             └─ 관련 법률 조항 원문 및 적정 위약금 산정식 도출[cite: 3]
                    │
                    ▼
[3. LLM Agent 인사이트 생성 (/api/agent/route.ts)]
 ├─ Context: [ML 예측 결과 + RAG 검색 법적 조항 원문 + 유사 판례 사례]
 └─ 최종 JSON 출력: 인용 조항 번호(`referenced_clauses`), 맞춤형 내용증명, 실행 계획[cite: 3]
        │
        ▼
[4. UI 렌더링 (ResultReport.tsx)]
 └─ 인용된 실제 법적 근거 조항 원문 아코디언 및 법적 배지 표출
3. 지식베이스 데이터 스키마 (lib/knowledge_base/legal_kb.json)
JSON
[
  {
    "id": "KB_FITNESS_01",
    "category": "체육시설/헬스장",
    "topic": "중도 해지 위약금 및 환급 기준",
    "law_name": "소비자분쟁해결기준 별표 II (체육시설업)",
    "clause_summary": "소비자 귀책사유로 인한 중도 해지 시 개시일 이전에는 총 이용금액의 10% 공제 후 환급, 개시일 이후에는 취소일까지의 이용일수 해당 금액과 총 이용금액의 10% 위약금 공제 후 잔여액 환급",
    "refund_formula": "총결제액 - (일일이용료 × 사용일수) - (총결제액 × 10%)",
    "keywords": ["헬스장", "피트니스", "중도해지", "위약금", "환불"]
  },
  {
    "id": "KB_ECOMMERCE_01",
    "category": "전자상거래/통신판매",
    "topic": "단순 변심에 따른 청약철회 기간 및 반품 비용",
    "law_name": "전자상거래 등에서의 소비자보호에 관한 법률 제17조 제1항",
    "clause_summary": "소비자는 계약내용에 관한 서면을 교부받은 날 또는 재화 등을 공급받은 날부터 7일 이내에 청약철회를 할 수 있다. 반품 비용은 소비자가 부담하되 위약금이나 손해배상을 청구할 수 없다.",
    "refund_formula": "전액 환급 (반품 배송비 실비 공제)",
    "keywords": ["온라인쇼핑", "전자상거래", "단순변심", "7일", "청약철회", "반품"]
  }
]
4. 확장된 LLM 구조화 JSON 응답 스키마
/api/agent/route.ts의 OpenAI 응답 포맷에 referenced_clauses를 추가 정의합니다.

JSON
{
  "category": "체육시설/헬스장",
  "dispute_type": "중도 해지 위약금 과다 청구",
  "success_rate": 84.5,
  "legal_basis": "소비자분쟁해결기준(체육시설업): 총 결제금액의 10% 위약금 공제 후 이용일수 일할 계산 잔여액 환급",
  "referenced_clauses": [
    {
      "law_name": "소비자분쟁해결기준 별표 II (체육시설업)",
      "clause_content": "소비자 귀책사유로 인한 중도 해지 시, 취소일까지의 이용일수 해당 금액과 총 이용금액의 10% 위약금 공제 후 잔여액 환급",
      "formula": "총결제액 - (일일이용료 × 사용일수) - (총결제액 × 10%)"
    }
  ],
  "estimated_refund": "결제금액의 약 75~80%",
  "action_plan": [
    "1단계: 표준약관 위반 사실 및 법정 위약금(10%) 산정표를 사업자에게 서면 통보",
    "2단계: 환불 거부 지속 시 1372 소비자상담센터 피해구제 접수",
    "3단계: 신용카드 결제 시 카드사에 할부항변권/철회권 서면 행사"
  ],
  "proof_documents": ["이용 계약서/회원권 약관", "결제 영수증", "환불 요청 문자/카카오톡 대화 캡처"],
  "notice_letter_template": "수신: [사업자 상호]\n발신: [소비자 성명]\n\n제목: 헬스장 이용계약 해지에 따른 적정 잔여 대금 환급 요청의 건\n\n1. 귀 사의 무궁한 발전을 기원합니다.\n2. 본인은 [계약일자] 체결한 이용계약과 관련하여 공정거래위원회 고시 소비자분쟁해결기준에 의거하여 다음과 같이 정당한 환급을 요청합니다..."
}
5. 세부 구현 요구사항
5.1. 지식베이스 사전 임베딩 스크립트 (scripts/build_legal_kb.py)
lib/knowledge_base/legal_kb.json에 정의된 14개 카테고리별 법률 조항 텍스트를 text-embedding-3-small 모델을 통해 벡터화.

결과를 lib/knowledge_base/legal_kb_embedded.json 파일로 1회 빌드 저장.

5.2. 경량 검색 유틸리티 (lib/legalSearch.ts)
사용자 입력 텍스트를 text-embedding-3-small로 임베딩.

legal_kb_embedded.json의 조항 벡터들과 코사인 유사도(cosine similarity) 계산[cite: 4].

ML 분류 카테고리 힌트를 우선 가중치로 부여하여 유사도 상위 Top-2 조항을 반환.

5.3. Agent 프롬프트 주입 (app/api/agent/route.ts)
검색된 법률 조항 원문 스니펫을 System/User Prompt의 [REFERENCE LEGAL CONTEXT] 섹션에 주입.

모델이 상상(Hallucination)하지 않고 검색된 조항의 명칭과 공식을 엄밀하게 인용하도록 System Prompt 강제.

5.4. 프론트엔드 UI 컴포넌트 갱신 (components/ResultReport.tsx)
법적 근거 카드: 검색된 law_name 태그 및 공식 환급 산정식 표시.

조항 원문 아코디언: 사용자가 클릭하여 실제 고시 조항 내용을 펼쳐볼 수 있는 접이식 UI 제공.

6. Claude Code 실행 지침
자율 작업 범위:

lib/knowledge_base/ 디렉토리에 14개 주요 업종별(체육시설, 전자상거래, 여행/숙박, 미용, 이동통신 등) 법률 조항 데이터 생성.

scripts/build_legal_kb.py 및 lib/legalSearch.ts 검색 엔진 구현.

/api/agent/route.ts 및 프론트엔드 리포트 컴포넌트 연동 완료.

무중단성 보장:

OpenAI API 키가 없거나 임베딩 실패 시 기존 fallbackAgent.ts 기반 규칙 엔진으로 안전하게 전환되도록 예외 처리 구현.