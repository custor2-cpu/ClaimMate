import type { AgentQuestion, MLAnalysisResult } from "@/lib/types";

const MAX_QUESTIONS = 3;

const CATEGORY_SIGNALS = [
  /헬스장|피트니스|필라테스|요가|수영장|체육관/,
  /통신|인터넷|휴대폰|알뜰폰|정액제|와이파이/,
  /옷|의류|신발|가방|패션|쇼핑몰|온라인몰/,
  /전자제품|노트북|컴퓨터|휴대폰|이어폰|TV|냉장고|세탁기|정수기/,
  /여행|숙박|호텔|펜션|항공|항공권|콘도/,
  /학원|교육|과외|수강|강의|학습지/,
  /상조|결혼중개|결혼식|웨딩/,
  /병원|의원|치과|성형|수술|진료/,
  /보험|보험금|보험료/,
  /자동차|중고차|렌터카|차량|정비|수리/,
  /화장품|피부관리|미용실|네일|마사지/,
  /부동산|임대차|전세|월세|아파트|주택|집|공인중개사/,
  /식품|음식|배달|도시락|건강식품/,
];

function hasAny(text: string, patterns: RegExp[]): boolean {
  return patterns.some((pattern) => pattern.test(text));
}

function addQuestion(
  questions: AgentQuestion[],
  id: string,
  question: string,
  reason: string,
  text: string,
  patterns: RegExp[]
) {
  if (!hasAny(text, patterns)) questions.push({ id, question, reason });
}

export function getMissingQuestions(ml: MLAnalysisResult): AgentQuestion[] {
  const text = ml.input.text;
  const questions: AgentQuestion[] = [];
  const category = ml.category;

  if (!ml.input.category_hint && !hasAny(text, CATEGORY_SIGNALS)) {
    questions.push({
      id: "category",
      question: "환불을 요청한 품목이나 서비스는 무엇인가요?",
      reason: "품목이나 서비스에 따라 적용되는 환불 기준이 달라 정확한 분석에 필요합니다.",
    });
  }

  if (["체육시설/헬스장", "화장품/미용", "학원/교육서비스"].includes(category)) {
    addQuestion(
      questions,
      "usage",
      "총 이용 기간 또는 총 이용 횟수와, 지금까지 실제로 이용한 기간 또는 횟수는 얼마인가요?",
      "이미 이용한 부분을 제외해야 환급액을 계산할 수 있습니다.",
      text,
      [/\d+\s*(회|개월|일|주)/, /이용\s*(하지|안|못)/, /사용\s*(하지|안|못)/, /전혀/]
    );
  }

  if (["온라인쇼핑몰", "전자제품", "의류/패션잡화"].includes(category)) {
    addQuestion(
      questions,
      "product_condition",
      "상품을 개봉하거나 사용했나요? 상품의 하자 또는 배송 상태도 함께 알려주세요.",
      "청약철회 가능 여부와 환불 책임을 판단하는 데 필요합니다.",
      text,
      [/개봉/, /사용/, /훼손/, /하자/, /불량/, /배송\s*(완료|지연|안|못|중)/, /받지\s*못/]
    );
  }

  if (category === "통신/인터넷") {
    addQuestion(
      questions,
      "termination_reason",
      "해지 또는 환불을 요청한 직접적인 이유는 무엇인가요? 서비스 장애, 이사, 단순 변심 중 어떤 사유인가요?",
      "계약 해지 위약금의 적정성을 판단하는 핵심 사실입니다.",
      text,
      [/이사/, /장애/, /불통/, /서비스가?\s*(안|되지|끊)/, /단순\s*변심/, /변심/, /해지\s*사유/]
    );
  }

  if (category === "부동산/임대차") {
    addQuestion(
      questions,
      "contract_end",
      "계약 만료일 또는 해지 요청일은 언제이며, 상대방은 어떤 답변을 했나요?",
      "보증금 반환 또는 계약 해지 의무의 시점을 판단하는 데 필요합니다.",
      text,
      [/만료/, /종료/, /해지\s*(요청|통보)/, /반환\s*(예정|거부|지연)/, /답변/, /연락/]
    );
  }

  addQuestion(
    questions,
    "business_response",
    "사업자에게 환불이나 해결을 요청했나요? 요청했다면 사업자는 어떤 이유로 답변했나요?",
    "현재 분쟁 단계와 다음 대응 방법을 정하는 데 필요합니다.",
    text,
    [/사업자.*(답변|거부|거절|환불|해결)/, /업체.*(답변|거부|거절|환불|해결)/, /판매자.*(답변|거부|거절|환불)/, /요청했/, /거부했/, /환불.*(안|못|거부)/]
  );

  if (!ml.input.amount) {
    addQuestion(
      questions,
      "amount",
      "결제했거나 피해를 입은 금액은 얼마인가요?",
      "예상 환급액을 계산하는 데 필요합니다.",
      text,
      [/\d[\d,]*\s*(원|만원|천원)/, /금액/, /결제액/, /가격/]
    );
  }

  return questions.slice(0, MAX_QUESTIONS);
}
