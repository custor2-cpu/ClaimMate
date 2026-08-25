export interface DisputeFormInput {
  text: string;
  amount: number | null;
  date: string | null;
  category: string | null;
}

export interface SimilarCase {
  category: string;
  dispute_type: string;
  similarity: number;
  outcome: string;
}

export interface DerivedFeatures {
  text_length: number;
  word_count: number;
  matched_keywords: string[];
}

/** /api/analyze (Pandas/NumPy 전처리 + Scikit-learn 추론) 응답 */
export interface MLAnalysisResult {
  category: string;
  dispute_type: string;
  success_rate: number;
  confidence: number;
  similar_cases: SimilarCase[];
  derived_features: DerivedFeatures;
  input: {
    text: string;
    amount: number | null;
    date: string | null;
    category_hint: string | null;
  };
  error?: string;
}

/** RAG로 검색된 법령 조항 인용 (docs/SPEC_RAG.md 4.) */
export interface ReferencedClause {
  law_name: string;
  clause_content: string;
  formula: string;
}

/**
 * success_rate가 어떻게 산출되었는지 표시.
 * - ml_similarity: 기본값. ML 분류 신뢰도 + 유사사례 평균(predictor.py)에서 산출
 * - legal_reasoning: RAG 법령 조항이 검색되고, 그 조항과 사용자가 명시한 사실관계만
 *   근거로 LLM이 독립 판단한 값이 ML 값보다 뚜렷이 높을 때 대체됨(상향 전용)
 */
export type SuccessRateBasis = "ml_similarity" | "legal_reasoning";

/**
 * 법령 근거의 명확성 등급. 단일 %(예: 85%)가 "정밀한 확률"처럼 보여 오해를 사고
 * LLM이 임의의 숫자에 앵커링되는 문제가 있어, 등급을 1차 신호로 쓰고 %는 등급에 종속된
 * 범위 안에서만 참고용으로 산정한다. success_rate_basis가 "legal_reasoning"일 때만 채워지고,
 * "ml_similarity"일 때는 null이다.
 */
export type CertaintyLevel = "매우 높음" | "높음" | "조정 필요" | "구제 어려움";

/** /api/agent (OpenAI LLM Agent) 최종 구조화 응답 — 명세서 4.1 JSON 스키마 */
export interface AgentReport {
  category: string;
  dispute_type: string;
  success_rate: number;
  success_rate_basis: SuccessRateBasis;
  certainty_level: CertaintyLevel | null;
  is_legally_clear: boolean;
  legal_success_reasoning: string;
  legal_basis: string;
  referenced_clauses: ReferencedClause[];
  estimated_refund: string;
  action_plan: string[];
  proof_documents: string[];
  notice_letter_template: string;
}

/** 최종 프론트엔드 렌더링용 통합 결과 */
export interface AnalysisReport extends AgentReport {
  similar_cases: SimilarCase[];
  derived_features: DerivedFeatures;
  /** true면 OpenAI 미설정/실패로 규칙 기반 폴백 리포트가 사용된 것 */
  used_fallback?: boolean;
}

export interface StatCategoryDatum {
  category: string;
  count: number;
}

export interface StatOutcomeDatum {
  name: string;
  value: number;
}

export interface StatsData {
  category_frequency: StatCategoryDatum[];
  resolution_outcome: StatOutcomeDatum[];
}

export const QUICK_PRESETS: { label: string; input: DisputeFormInput }[] = [
  {
    label: "헬스장 중도 환불",
    input: {
      text: "헬스장 1년권을 결제하고 2주 만에 개인 사정으로 환불을 요청했는데 위약금 50%를 요구했습니다. 표준약관에 어긋나는 것 같은데 어떻게 대응해야 할까요?",
      amount: 600000,
      date: null,
      category: "체육시설/헬스장",
    },
  },
  {
    label: "인터넷 해지 위약금",
    input: {
      text: "인터넷 3년 약정을 했는데 서비스가 되지 않는 지역으로 이사를 가게 되어 해지했더니 위약금 30만원을 청구했습니다. 부당한 것 같습니다.",
      amount: 300000,
      date: null,
      category: "통신/인터넷",
    },
  },
  {
    label: "의류 배송 지연",
    input: {
      text: "온라인 쇼핑몰에서 옷을 주문했는데 3주가 지나도 배송이 오지 않고, 판매자에게 문의해도 답변이 없습니다. 환불을 요청하고 싶습니다.",
      amount: 89000,
      date: null,
      category: "의류/패션잡화",
    },
  },
];
