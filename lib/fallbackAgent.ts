import type { AgentReport, MLAnalysisResult } from "@/lib/types";
import { DEFAULT_LEGAL_KNOWLEDGE, LEGAL_KNOWLEDGE } from "@/lib/legalKnowledge";

/**
 * OPENAI_API_KEY가 설정되지 않았거나 OpenAI 호출이 실패했을 때 사용하는
 * 규칙 기반 폴백 리포트 생성기. LLM 없이도 데모/로컬 개발이 즉시 동작하도록 한다.
 */
export function buildFallbackReport(ml: MLAnalysisResult): AgentReport {
  const knowledge = LEGAL_KNOWLEDGE[ml.category] ?? DEFAULT_LEGAL_KNOWLEDGE;
  const amountText = ml.input.amount
    ? `${ml.input.amount.toLocaleString("ko-KR")}원`
    : "[결제금액]";
  const contractDate = ml.input.date ?? "[계약일자]";

  const action_plan = [
    `1단계: ${knowledge.legal_basis.split(":")[0]} 근거 및 적정 환급액 산정 내역을 정리하여 사업자에게 서면(내용증명)으로 통보`,
    "2단계: 사업자가 환불 요청을 거부하거나 응답이 없을 경우 1372 소비자상담센터(국번없이 1372) 또는 소비자24에 피해구제를 접수",
    "3단계: 신용카드로 결제한 경우 카드사에 할부항변권 또는 청약철회권을 서면으로 행사하여 결제 취소를 요청",
    "4단계: 위 절차로도 해결되지 않으면 한국소비자원 소비자분쟁조정위원회에 조정을 신청",
  ];

  const notice_letter_template = `수신: [사업자 상호]
발신: [소비자 성명] (연락처: [연락처])

제목: ${ml.category} 관련 ${ml.dispute_type}에 따른 정당한 환급 요청의 건

1. 귀 사의 무궁한 발전을 기원합니다.
2. 본인은 ${contractDate}경 귀 사와 체결한 계약과 관련하여 아래와 같은 사유로 정당한 환급을 요청드립니다.

   - 상담 내용: ${ml.input.text}
   - 결제 금액: ${amountText}
   - 분쟁 유형: ${ml.dispute_type}

3. 이는 공정거래위원회 고시 「소비자분쟁해결기준」 및 관련 법령에 근거한 정당한 권리 행사입니다.
   (근거: ${knowledge.legal_basis})
4. 이에 따른 예상 환급 범위는 ${knowledge.estimated_refund} 수준으로 판단됩니다.
5. 본 통지 수령일로부터 7일 이내에 성실한 답변 및 환급 조치를 요청드리며, 미이행 시 소비자원 피해구제 절차 등 법적 조치를 진행할 수 있음을 알려드립니다.

${new Date().toISOString().slice(0, 10)}
발신인: [소비자 성명] (서명 또는 인)`;

  return {
    category: ml.category,
    dispute_type: ml.dispute_type,
    success_rate: ml.success_rate,
    legal_basis: knowledge.legal_basis,
    // 규칙 기반 폴백은 RAG 검색을 수행하지 않으므로(OpenAI 호출 자체가 없음) 빈 배열로 둔다.
    referenced_clauses: [],
    estimated_refund: knowledge.estimated_refund,
    action_plan,
    proof_documents: knowledge.proof_documents,
    notice_letter_template,
  };
}
