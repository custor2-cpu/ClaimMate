import { NextRequest, NextResponse } from "next/server";
import { getOpenAIClient, OPENAI_MODEL } from "@/lib/openai";
import { buildFallbackReport } from "@/lib/fallbackAgent";
import { DEFAULT_LEGAL_KNOWLEDGE, LEGAL_KNOWLEDGE } from "@/lib/legalKnowledge";
import { retrieveLegalClauses } from "@/lib/legalSearch";
import type { AgentReport, AnalysisReport, MLAnalysisResult, ReferencedClause } from "@/lib/types";

export const runtime = "nodejs";

/**
 * 공공데이터 표본 부족(특히 체육시설/헬스장·화장품/미용·부동산/임대차·식품·여행/숙박처럼
 * 실 사례가 적은 카테고리)으로 ML의 success_rate(분류 신뢰도 + 유사사례 평균 가중치)가
 * 낮게 나오는 경우가 있다. 이 값 미만이면서 RAG로 관련 법령 조항이 검색된 경우, 그
 * 법령 근거에 명확한 환급 기준이 있는지 LLM이 직접 대조·판단한 legal_success_estimate로
 * success_rate를 대체(rerank)한다 — ML 통계보다 법적 근거를 우선한다.
 */
const LOW_SUCCESS_RATE_THRESHOLD = 50;

const AGENT_JSON_SCHEMA = {
  name: "claim_mate_report",
  strict: true,
  schema: {
    type: "object",
    additionalProperties: false,
    properties: {
      category: { type: "string", description: "분쟁 품목/업종 카테고리" },
      dispute_type: { type: "string", description: "분쟁 유형" },
      success_rate: { type: "number", description: "구제 성공 확률(0~100)" },
      legal_success_estimate: {
        type: "number",
        description:
          "[REFERENCE LEGAL CONTEXT]와 [사용자 입력]에 명시된 사실관계만 근거로 독립적으로 " +
          "추정한 '소비자 주장이 받아들여져 구제받을 확률'(0~100). 위약금율/공제율/환급율 같은 " +
          "조항 속 계산 숫자를 그대로 넣지 말 것 — 법이 소비자에게 명확히 유리하면 이 값은 " +
          "높아야(80~95) 한다. 근거가 부족하면 success_rate와 동일한 값을 넣는다.",
      },
      legal_success_reasoning: {
        type: "string",
        description:
          "legal_success_estimate를 그렇게 산정한 근거를 1~2문장으로. 제공되지 않은 사실을 " +
          "가정하지 말고, 근거 부족 시 '법령 조항만으로 독립 추정하기에 근거 부족'이라고 쓴다.",
      },
      legal_basis: {
        type: "string",
        description: "공정거래위원회 소비자분쟁해결기준 등 법적 근거 요약",
      },
      referenced_clauses: {
        type: "array",
        description:
          "RAG로 검색된 법령 조항 인용 목록. [REFERENCE LEGAL CONTEXT]에 제공된 조항만 " +
          "그대로 인용하고, 제공된 근거가 없으면 빈 배열로 둔다.",
        items: {
          type: "object",
          additionalProperties: false,
          properties: {
            law_name: { type: "string", description: "인용 법령/고시명" },
            clause_content: { type: "string", description: "조항 원문 요약" },
            formula: { type: "string", description: "환급/위약금 산정식 (없으면 빈 문자열)" },
          },
          required: ["law_name", "clause_content", "formula"],
        },
      },
      estimated_refund: { type: "string", description: "예상 환급 범위" },
      action_plan: {
        type: "array",
        items: { type: "string" },
        description: "단계별 실행 계획 (번호 매김 문자열 배열)",
      },
      proof_documents: {
        type: "array",
        items: { type: "string" },
        description: "준비해야 할 증빙 자료 목록",
      },
      notice_letter_template: {
        type: "string",
        description: "사업자 발송용 맞춤형 내용증명 전문(수신/발신/제목/본문 포함)",
      },
    },
    required: [
      "category",
      "dispute_type",
      "success_rate",
      "legal_success_estimate",
      "legal_success_reasoning",
      "legal_basis",
      "referenced_clauses",
      "estimated_refund",
      "action_plan",
      "proof_documents",
      "notice_letter_template",
    ],
  },
} as const;

function buildPrompt(ml: MLAnalysisResult, retrievedClauses: ReferencedClause[] | null) {
  const knowledge = LEGAL_KNOWLEDGE[ml.category] ?? DEFAULT_LEGAL_KNOWLEDGE;

  const referenceContext =
    retrievedClauses && retrievedClauses.length > 0
      ? `[REFERENCE LEGAL CONTEXT] (RAG 검색 결과 — 아래 조항만 인용하고, 여기 없는 법령명이나
산정식을 임의로 만들어내지 마세요)
${retrievedClauses
  .map(
    (c, i) =>
      `${i + 1}. [${c.law_name}] ${c.clause_content}${
        c.formula && !c.formula.startsWith("해당 없음") ? ` (산정식: ${c.formula})` : ""
      }`
  )
  .join("\n")}`
      : `[REFERENCE LEGAL CONTEXT] 없음 — referenced_clauses는 빈 배열로 두세요.`;

  const isLowSuccessRate = ml.success_rate < LOW_SUCCESS_RATE_THRESHOLD;

  const system = `당신은 한국소비자원 소비자상담 데이터를 기반으로 학습된 소비자분쟁 해결 전문 AI 에이전트 "ClaimMate"입니다.
Pandas/Scikit-learn 파이프라인이 산출한 ML 분석 결과를 종합하여, 공정거래위원회 고시 「소비자분쟁해결기준」에 근거한
실행 가능한 인사이트 리포트를 JSON으로 작성합니다.

규칙:
- success_rate는 ML 모델이 산출한 값(${ml.success_rate})을 그대로 사용하세요.
- legal_success_estimate/legal_success_reasoning: ML의 success_rate(${ml.success_rate}%)가
  ${LOW_SUCCESS_RATE_THRESHOLD}% 미만${isLowSuccessRate ? "이라 이번 사안이 여기 해당합니다" : "이 아니라 이번 사안은 해당하지 않습니다"}일 때,
  공공데이터 표본 부족으로 ML 수치가 실제 법적 타당성보다 낮게 나왔을 가능성이 있습니다.
  [REFERENCE LEGAL CONTEXT] 조항의 적용 조건이 사용자 "본인의 상황"에 대한 진술(무엇을
  했는지/안 했는지, 언제·왜 취소했는지 등)로 충족되면, legal_success_estimate를 그 법적
  판단에 따라 재산정해 ML 수치를 대체(rerank)하세요. 이 값은 "위약금 비율"이 아니라 "소비자
  주장이 받아들여질 확률"이므로, 법이 소비자에게 명확히 유리하면 80~95 같은 높은 값을 넣으세요
  (예: "가입 후 이틀, 한 번도 이용 안 함" + 조항 "이용 개시 이전엔 위약금 10% 이내만 공제" →
  사업자가 요구한 50%는 부당하므로 85~95).
  반대로, 조건 충족이 오직 사용자 "본인이 법령/정책을 이렇게 알고 있다"는 추정(예: "n일 전
  취소하면 무료라고 알고 있다")에만 의존하고 [REFERENCE LEGAL CONTEXT] 본문에 그 구체적 수치가
  없다면, 그 추정은 검증되지 않은 것이므로 legal_success_estimate에는 success_rate와 동일한
  값을 넣고 legal_success_reasoning에 "법령 조항만으로 독립 추정하기에 근거 부족"이라고 쓰세요.
  [REFERENCE LEGAL CONTEXT]가 없거나 success_rate가 이미 충분히 높을 때도 마찬가지입니다.
- legal_basis와 estimated_refund는 아래 제공된 참고 법적 근거를 기반으로 사안에 맞게 구체화하세요.
- referenced_clauses는 반드시 [REFERENCE LEGAL CONTEXT]에 제공된 조항만 그대로(법령명/산정식 왜곡 없이)
  인용하세요. 제공된 근거가 없다고 명시된 경우 referenced_clauses는 빈 배열([])로 두세요. 이 필드에서
  법령을 상상해서 만들어내는 것(hallucination)은 절대 금지입니다.
- action_plan은 "1단계:", "2단계:" 형식으로 실제로 실행 가능한 절차를 3~5단계로 작성하세요.
- proof_documents는 참고 자료를 바탕으로 사안에 맞게 3~5개 항목으로 작성하세요.
- notice_letter_template은 실제 발송 가능한 내용증명 형식(수신/발신/제목/본문/날짜)으로, 소비자가 [사업자 상호], [소비자 성명] 등
  괄호 안 플레이스홀더만 채우면 되도록 작성하세요. 사용자가 입력한 상담 내용, 결제금액, 계약일자를 본문에 자연스럽게 반영하세요.
- 모든 문장은 한국어 존댓말로, 전문적이고 신뢰감 있는 톤으로 작성하세요.
- 참고 법적 근거는 정보 제공 목적이며 실제 법률 자문이 아님을 감안하여 과장 없이 작성하세요.`;

  const user = `[ML 분석 결과]
- 분쟁 카테고리: ${ml.category}
- 분쟁 유형: ${ml.dispute_type}
- 구제 성공 확률: ${ml.success_rate}% (분류기 신뢰도 ${ml.confidence}%)
- 매칭된 불만 키워드: ${ml.derived_features.matched_keywords.join(", ") || "없음"}
- 유사 사례 Top-3: ${ml.similar_cases
    .map((c) => `[${c.category}/${c.dispute_type}, 유사도 ${c.similarity}%, 처리결과: ${c.outcome}]`)
    .join(" / ")}

[참고 법적 근거]
- ${knowledge.legal_basis}
- 예상 환급 범위 참고: ${knowledge.estimated_refund}
- 참고 증빙 자료: ${knowledge.proof_documents.join(", ")}

${referenceContext}

[사용자 입력]
- 상담 내용: ${ml.input.text}
- 피해 금액: ${ml.input.amount ? `${ml.input.amount.toLocaleString("ko-KR")}원` : "미입력"}
- 결제/계약 일자: ${ml.input.date ?? "미입력"}

위 정보를 종합하여 ClaimMate 리포트 JSON을 생성하세요.`;

  return { system, user };
}

/** OpenAI 구조화 응답 원본 — legal_success_estimate/reasoning은 success_rate 재산정에만 쓰이는 중간 값 */
type RawAgentResponse = AgentReport & {
  legal_success_estimate: number;
  legal_success_reasoning: string;
};

async function generateWithOpenAI(
  ml: MLAnalysisResult,
  retrievedClauses: ReferencedClause[] | null
): Promise<RawAgentResponse> {
  const client = getOpenAIClient();
  const { system, user } = buildPrompt(ml, retrievedClauses);

  const completion = await client.chat.completions.create({
    model: OPENAI_MODEL,
    messages: [
      { role: "system", content: system },
      { role: "user", content: user },
    ],
    response_format: { type: "json_schema", json_schema: AGENT_JSON_SCHEMA },
    // legal_success_estimate가 법령 조항 대조라는 사실기반 판단이라 실행마다 결과가
    // 흔들리면 신뢰도가 떨어진다. 창의성보다 일관성이 중요해 기존 0.4보다 낮춘다.
    temperature: 0.2,
  });

  const content = completion.choices[0]?.message?.content;
  if (!content) {
    throw new Error("OpenAI 응답이 비어 있습니다.");
  }

  return JSON.parse(content) as RawAgentResponse;
}

/**
 * success_rate를 최종 확정한다. ML의 success_rate가 낮고(<${LOW_SUCCESS_RATE_THRESHOLD}%)
 * RAG로 법령 조항이 검색되었으며, LLM이 그 조항을 사용자 사실관계와 대조해 실제로 다른
 * (의미 있게 차이 나는) 값을 산정한 경우에만 법령 기반 값으로 대체(rerank)한다. 그 외에는
 * 항상 ML 산출값을 그대로 쓴다(재현 가능성/신뢰도 보존, 애매할 때 과장 방지).
 */
function resolveSuccessRate(
  ml: MLAnalysisResult,
  retrievedClauses: ReferencedClause[] | null,
  raw: RawAgentResponse
): { rate: number; basis: AgentReport["success_rate_basis"]; reasoning: string } {
  const isLowSuccessRate = ml.success_rate < LOW_SUCCESS_RATE_THRESHOLD;
  const hasLegalGrounding = Boolean(retrievedClauses && retrievedClauses.length > 0);
  const hasLegalEstimate = typeof raw.legal_success_estimate === "number";
  const estimateDiffers = hasLegalEstimate && Math.abs(raw.legal_success_estimate - ml.success_rate) >= 0.1;

  if (isLowSuccessRate && hasLegalGrounding && estimateDiffers) {
    return {
      rate: raw.legal_success_estimate,
      basis: "legal_reasoning",
      reasoning: raw.legal_success_reasoning,
    };
  }
  return {
    rate: ml.success_rate,
    basis: "ml_similarity",
    reasoning: raw.legal_success_reasoning ?? "",
  };
}

export async function POST(req: NextRequest) {
  try {
    const ml = (await req.json()) as MLAnalysisResult;

    if (!ml || !ml.category || !ml.dispute_type) {
      return NextResponse.json(
        { error: "ML 분석 결과가 유효하지 않습니다." },
        { status: 400 }
      );
    }

    let report: AgentReport;
    let usedFallback = false;

    try {
      if (!process.env.OPENAI_API_KEY) {
        throw new Error("OPENAI_API_KEY 미설정");
      }
      const retrievedClauses = await retrieveLegalClauses(ml);
      const raw = await generateWithOpenAI(ml, retrievedClauses);
      const { rate, basis, reasoning } = resolveSuccessRate(ml, retrievedClauses, raw);
      report = { ...raw, success_rate: rate, success_rate_basis: basis, legal_success_reasoning: reasoning };
    } catch (err) {
      console.warn("[/api/agent] OpenAI 호출 실패, 규칙 기반 폴백으로 대체합니다:", err);
      report = buildFallbackReport(ml);
      usedFallback = true;
    }

    const analysisReport: AnalysisReport = {
      ...report,
      similar_cases: ml.similar_cases,
      derived_features: ml.derived_features,
      used_fallback: usedFallback,
    };

    return NextResponse.json(analysisReport);
  } catch (err) {
    console.error("[/api/agent] 리포트 생성 실패:", err);
    return NextResponse.json(
      { error: err instanceof Error ? err.message : "리포트 생성 중 오류가 발생했습니다." },
      { status: 500 }
    );
  }
}
