import OpenAI from "openai";

let client: OpenAI | null = null;

/** OpenAI 클라이언트를 지연 초기화한다 (빌드 타임에 API 키가 없어도 실패하지 않도록). */
export function getOpenAIClient(): OpenAI {
  if (!client) {
    const apiKey = process.env.OPENAI_API_KEY;
    if (!apiKey) {
      throw new Error(
        "OPENAI_API_KEY가 설정되지 않았습니다. .env.local 파일에 키를 추가해 주세요."
      );
    }
    client = new OpenAI({ apiKey });
  }
  return client;
}

export const OPENAI_MODEL = "gpt-4o-mini";
