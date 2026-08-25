import { NextRequest, NextResponse } from "next/server";
import { getOpenAIClient, OPENAI_MODEL } from "@/lib/openai";
import { buildFallbackReport } from "@/lib/fallbackAgent";
import { DEFAULT_LEGAL_KNOWLEDGE, LEGAL_KNOWLEDGE } from "@/lib/legalKnowledge";
import { retrieveLegalClauses } from "@/lib/legalSearch";
import type { AgentReport, AnalysisReport, MLAnalysisResult, ReferencedClause } from "@/lib/types";

export const runtime = "nodejs";

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

  const system = `당신은 한국소비자원 소비자상담 데이터를 기반으로 학습된 소비자분쟁 해결 전문 AI 에이전트 "ClaimMate"입니다.
Pandas/Scikit-learn 파이프라인이 산출한 ML 분석 결과를 종합하여, 공정거래위원회 고시 「소비자분쟁해결기준」에 근거한
실행 가능한 인사이트 리포트를 JSON으로 작성합니다.

규칙:
- success_rate는 ML 모델이 산출한 값(${ml.success_rate})을 그대로 사용하세요.
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

async function generateWithOpenAI(
  ml: MLAnalysisResult,
  retrievedClauses: ReferencedClause[] | null
): Promise<AgentReport> {
  const client = getOpenAIClient();
  const { system, user } = buildPrompt(ml, retrievedClauses);

  const completion = await client.chat.completions.create({
    model: OPENAI_MODEL,
    messages: [
      { role: "system", content: system },
      { role: "user", content: user },
    ],
    response_format: { type: "json_schema", json_schema: AGENT_JSON_SCHEMA },
    temperature: 0.4,
  });

  const content = completion.choices[0]?.message?.content;
  if (!content) {
    throw new Error("OpenAI 응답이 비어 있습니다.");
  }

  const parsed = JSON.parse(content) as AgentReport;
  // 게이지/차트와의 수치 일관성을 위해 ML 산출값으로 고정
  parsed.success_rate = ml.success_rate;
  return parsed;
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
      report = await generateWithOpenAI(ml, retrievedClauses);
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
