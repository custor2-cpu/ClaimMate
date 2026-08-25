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
 * 부당하게 낮게 나오는 경우가 있다. RAG로 관련 법령 조항이 검색되면, 그 법령 근거에
 * 명확한 환급 기준이 있는지 LLM이 항상 직접 대조·판단하고(legal_success_estimate),
 * 그 값이 ML 값보다 이 마진 이상 높을 때만 success_rate를 대체(rerank)한다.
 *
 * 이전에는 "ML success_rate < 50%"라는 고정 임계값으로 재산정 여부를 결정했는데, 이
 * 방식은 49.9%와 50.4%처럼 근거 차이가 거의 없는 두 사안의 결과가 임계값 하나를
 * 사이에 두고 완전히 달라지는 경계값 불연속 문제가 있었다. 지금은 ML 값과 무관하게
 * 항상 법령 기반 판단을 시도하고, "법령이 ML보다 뚜렷이 유리하게 나올 때만" 한
 * 방향으로만(위로만) 보정하므로 이런 불연속이 없다 — 법령 판단이 ML보다 낮게 나와도
 * 절대 하향 조정하지 않는다.
 */
const MIN_LEGAL_OVERRIDE_MARGIN = 10;

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

/**
 * ml.input.date(결제/계약 일자)로부터 오늘까지 경과일수를 계산한다. "7일 이내 청약철회",
 * "이용 개시 이전(=계약 후 상당 기간 미이용)" 같은 시간 기준 법령 조건은 이 경과일수 없이는
 * LLM이 판단할 근거가 없다 — 이전에는 오늘 날짜 자체가 프롬프트에 없어 date 필드가 있어도
 * 사실상 장식적인 값이었다.
 */
function computeElapsedDays(dateStr: string | null): number | null {
  if (!dateStr) return null;
  const contractDate = new Date(dateStr);
  if (Number.isNaN(contractDate.getTime())) return null;
  const msPerDay = 1000 * 60 * 60 * 24;
  return Math.floor((Date.now() - contractDate.getTime()) / msPerDay);
}

function buildPrompt(ml: MLAnalysisResult, retrievedClauses: ReferencedClause[] | null) {
  const knowledge = LEGAL_KNOWLEDGE[ml.category] ?? DEFAULT_LEGAL_KNOWLEDGE;
  const todayStr = new Date().toISOString().slice(0, 10);
  const elapsedDays = computeElapsedDays(ml.input.date);

  const referenceContext =
    retrievedClauses && retrievedClauses.length > 0
      ? `[REFERENCE LEGAL CONTEXT] (RAG 검색 결과 — 아래 조항만 인용하고, 여기 없는 법령명이나
산정식을 임의로 만들어내지 마세요. 번호는 검색 엔진이 이 사안과의 관련성 순으로 매긴
것입니다 — 1번이 가장 관련성 높은 조항이니 특별한 사정이 없는 한 1번을 주된 근거로 삼고,
legal_basis/referenced_clauses/legal_success_estimate 판단에서 1번을 우선하세요. 1번 조항
본문에 "다른 기준을 따른다/확인이 필요하다"처럼 불확실성이 명시되어 있다면, 그 불확실성 때문에
legal_success_estimate를 과도하게 높이지 말고 신중하게 판단하세요.)
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
- [사용자 입력]의 "결제/계약 일자(오늘 기준 N일 경과)"는 서버가 계산한 객관적 사실이므로,
  "n일 이내", "이용 개시 이전" 같은 시간 기준 법령 조건을 판단할 때 그대로 신뢰해 사용하세요
  (사용자의 주장/추측이 아닙니다). 다만 상담 내용에 "취소 요청일"이 계약일과 다르게 별도로
  언급되어 있으면(예: "한 달 전에 가입했고 어제 취소 요청했다") 상담 내용에 명시된 시점을
  우선하세요.
- legal_success_estimate/legal_success_reasoning: 공공데이터 표본 부족으로 ML의
  success_rate(${ml.success_rate}%)가 실제 법적 타당성보다 낮게 나왔을 수 있습니다. ML 값의
  높고 낮음과 무관하게 항상 아래 판단을 하세요.
  [REFERENCE LEGAL CONTEXT] 조항의 적용 조건이 사용자 "본인의 상황"에 대한 진술(무엇을
  했는지/안 했는지, 언제·왜 취소했는지 등, 위 경과일수 포함)로 충족되면, legal_success_estimate를 그 법적
  판단에 따라 독립적으로 산정하세요. 이 값은 "위약금 비율"이 아니라 "소비자 주장이 받아들여질
  확률"이므로, 법이 소비자에게 명확히 유리하면 80~95 같은 높은 값을 넣으세요 (예: "가입 후
  이틀, 한 번도 이용 안 함" + 조항 "이용 개시 이전엔 위약금 10% 이내만 공제" → 사업자가 요구한
  50%는 부당하므로 85~95).
  반대로, 조건 충족이 오직 사용자 "본인이 법령/정책을 이렇게 알고 있다"는 추정(예: "n일 전
  취소하면 무료라고 알고 있다")에만 의존하고 [REFERENCE LEGAL CONTEXT] 본문에 그 구체적 수치가
  없다면, 그 추정은 검증되지 않은 것이므로 legal_success_estimate에는 success_rate와 동일한
  값을 넣고 legal_success_reasoning에 "법령 조항만으로 독립 추정하기에 근거 부족"이라고 쓰세요.
  [REFERENCE LEGAL CONTEXT]가 없을 때도 마찬가지입니다. legal_success_estimate가
  success_rate보다 낮아야 한다고 판단되더라도, 이 시스템은 상향 보정에만 쓰이므로 낮추지 말고
  success_rate와 동일한 값을 넣으세요.
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
- 오늘 날짜: ${todayStr}
- 상담 내용: ${ml.input.text}
- 피해 금액: ${ml.input.amount ? `${ml.input.amount.toLocaleString("ko-KR")}원` : "미입력"}
- 결제/계약 일자: ${ml.input.date ?? "미입력"}${
    elapsedDays !== null ? ` (오늘 기준 ${elapsedDays}일 경과)` : ""
  }

위 정보를 종합하여 ClaimMate 리포트 JSON을 생성하세요. notice_letter_template의 발신 날짜는
"오늘 날짜"를 사용하세요.`;

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
 * success_rate를 최종 확정한다. RAG로 법령 조항이 검색되었고, LLM이 그 조항을 사용자
 * 사실관계와 대조해 산정한 legal_success_estimate가 ML 값보다 MIN_LEGAL_OVERRIDE_MARGIN
 * 이상 높을 때만 그 값으로 대체(rerank)한다. ML success_rate 자체의 높낮이로 판단 여부를
 * 가르지 않으므로(고정 임계값 없음) 경계값 근처에서 결과가 불연속으로 뒤집히지 않고, 법령
 * 판단이 ML보다 낮게 나와도 하향 조정은 하지 않는다(상향 보정 전용, 애매할 때 과장 방지).
 */
function resolveSuccessRate(
  ml: MLAnalysisResult,
  retrievedClauses: ReferencedClause[] | null,
  raw: RawAgentResponse
): { rate: number; basis: AgentReport["success_rate_basis"]; reasoning: string } {
  const hasLegalGrounding = Boolean(retrievedClauses && retrievedClauses.length > 0);
  const hasLegalEstimate = typeof raw.legal_success_estimate === "number";
  const isMeaningfullyHigher =
    hasLegalEstimate && raw.legal_success_estimate - ml.success_rate >= MIN_LEGAL_OVERRIDE_MARGIN;

  if (hasLegalGrounding && isMeaningfullyHigher) {
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
