import { existsSync, readFileSync } from "node:fs";
import path from "node:path";
import { getOpenAIClient } from "@/lib/openai";
import type { MLAnalysisResult, ReferencedClause } from "@/lib/types";

/**
 * docs/SPEC_RAG.md 기반 경량 RAG 검색 엔진.
 * scripts/build_legal_kb.py가 생성한 사전 임베딩 지식베이스(legal_kb_embedded.json)를
 * 읽어 코사인 유사도로 관련 법령 조항 Top-2를 검색한다. 외부 벡터 DB 없이 로컬 JSON +
 * 브루트포스 코사인 유사도만 사용한다 (지식베이스가 수십 건 규모라 충분히 빠르다).
 */

interface LegalKbChunk {
  id: string;
  category: string;
  topic: string;
  law_name: string;
  clause_summary: string;
  refund_formula: string;
  keywords: string[];
}

interface EmbeddedChunk extends LegalKbChunk {
  embedding: number[];
}

const EMBEDDED_KB_PATH = path.join(
  process.cwd(),
  "lib",
  "knowledge_base",
  "legal_kb_embedded.json"
);
const EMBEDDING_MODEL = "text-embedding-3-small";
const TOP_K = 2;
/** ML이 예측한 카테고리와 일치하는 청크에 부여하는 가중치 (명세서 5.2) */
const CATEGORY_MATCH_BOOST = 0.08;
/**
 * 청크의 keywords가 사용자 입력(분쟁유형/상담 텍스트)에 등장할 때마다 부여하는
 * 가중치. 순수 임베딩 코사인 유사도만으로는 같은 카테고리 내 비슷한 길이/어조의
 * 조항끼리 순위가 뒤바뀌는 경우가 있어(예: "천재지변" 키워드가 명시된 조항이
 * 무관한 "폐업" 조항보다 낮은 점수를 받는 사례 확인), 키워드 일치라는 명시적
 * 어휘 신호를 더해 하이브리드 검색으로 보정한다.
 */
const KEYWORD_MATCH_BOOST = 0.035;
/**
 * api/ml_engine/predictor.py의 _LOW_CONFIDENCE_THRESHOLD(0.35), app/api/agent/route.ts의
 * ML_LOW_CONFIDENCE_WARNING_THRESHOLD와 같은 기준(%). 이 미만이면 ml.category/dispute_type
 * 자체가 무관한 K-Means 군집에서 나온 오분류일 수 있다(실사례: "이어폰 환불 가능?"이
 * "학원/교육서비스"로 분류됨). 이 경우 검색 쿼리와 카테고리 가중치에 그 오분류를 그대로
 * 쓰면 완전히 무관한 법령 조항(예: 학원 환급 규정)이 검색되므로, 신뢰도가 낮을 때는
 * category/dispute_type을 배제하고 사용자 원문 텍스트만으로 검색한다.
 */
const LOW_CONFIDENCE_THRESHOLD = 35;

let cachedChunks: EmbeddedChunk[] | null | undefined;

function loadEmbeddedChunks(): EmbeddedChunk[] | null {
  if (cachedChunks !== undefined) return cachedChunks;

  if (!existsSync(EMBEDDED_KB_PATH)) {
    console.warn(
      "[legalSearch] legal_kb_embedded.json이 없습니다. " +
        "`python scripts/build_legal_kb.py`로 지식베이스를 먼저 빌드해 주세요. " +
        "빌드 전까지는 정적 legalKnowledge.ts로 폴백합니다."
    );
    cachedChunks = null;
    return null;
  }

  try {
    cachedChunks = JSON.parse(readFileSync(EMBEDDED_KB_PATH, "utf-8")) as EmbeddedChunk[];
  } catch (err) {
    console.warn("[legalSearch] legal_kb_embedded.json 파싱 실패:", err);
    cachedChunks = null;
  }
  return cachedChunks;
}

function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  if (normA === 0 || normB === 0) return 0;
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

/**
 * ML 분석 결과(상담 텍스트 + 예측 카테고리)를 바탕으로 사전 임베딩된 법령
 * 지식베이스에서 관련도 상위 Top-2 조항을 검색한다.
 *
 * 지식베이스 미빌드(legal_kb_embedded.json 없음), OPENAI_API_KEY 미설정, 임베딩
 * API 호출 실패 등 어떤 이유로든 검색이 불가능하면 null을 반환해 호출부가 기존
 * 정적 legalKnowledge.ts 기반 폴백을 쓰도록 한다 (명세서 6. 무중단성 보장).
 */
export async function retrieveLegalClauses(
  ml: MLAnalysisResult
): Promise<ReferencedClause[] | null> {
  const chunks = loadEmbeddedChunks();
  if (!chunks || chunks.length === 0) return null;
  if (!process.env.OPENAI_API_KEY) return null;

  try {
    const client = getOpenAIClient();
    const isCategoryReliable = ml.confidence >= LOW_CONFIDENCE_THRESHOLD;
    const query = isCategoryReliable
      ? `[${ml.category}] ${ml.dispute_type}\n${ml.input.text}`
      : ml.input.text;
    const res = await client.embeddings.create({ model: EMBEDDING_MODEL, input: query });
    const queryVector = res.data[0]?.embedding;
    if (!queryVector) return null;

    const haystack = isCategoryReliable ? `${ml.dispute_type} ${ml.input.text}` : ml.input.text;
    const scored = chunks.map((chunk) => {
      const similarity = cosineSimilarity(queryVector, chunk.embedding);
      const categoryBoost =
        isCategoryReliable && chunk.category === ml.category ? CATEGORY_MATCH_BOOST : 0;
      const keywordMatches = chunk.keywords.filter((kw) => haystack.includes(kw)).length;
      const keywordBoost = keywordMatches * KEYWORD_MATCH_BOOST;
      return { chunk, score: similarity + categoryBoost + keywordBoost };
    });

    scored.sort((a, b) => b.score - a.score);

    return scored.slice(0, TOP_K).map(({ chunk }) => ({
      law_name: chunk.law_name,
      clause_content: chunk.clause_summary,
      formula: chunk.refund_formula,
    }));
  } catch (err) {
    console.warn("[legalSearch] RAG 검색 실패, 정적 지식베이스로 폴백합니다:", err);
    return null;
  }
}
