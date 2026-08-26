/**
 * 결제/계약 일자로부터 오늘까지 경과일수를 계산한다. "7일 이내 청약철회" 같은
 * 시간 기준 법령 조건은 이 경과일수 없이는 판단할 근거가 없다.
 * app/api/agent/route.ts(LLM 프롬프트)와 lib/fallbackAgent.ts(규칙 기반 폴백)가
 * 공유해서 쓴다.
 */
export function computeElapsedDays(dateStr: string | null): number | null {
  if (!dateStr) return null;
  const contractDate = new Date(dateStr);
  if (Number.isNaN(contractDate.getTime())) return null;
  const msPerDay = 1000 * 60 * 60 * 24;
  return Math.floor((Date.now() - contractDate.getTime()) / msPerDay);
}
