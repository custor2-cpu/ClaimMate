import { NextResponse } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

/**
 * 배포마다 갱신되는 버전 정보를 캐시 없이 반환한다.
 * 클라이언트가 이 값을 자신이 로드된 시점의 빌드 정보와 비교해
 * 새 배포가 감지되면 캐시를 지우고 자동으로 새로고침하는 데 사용한다.
 */
export async function GET() {
  return NextResponse.json(
    {
      version: process.env.NEXT_PUBLIC_APP_VERSION ?? "0.0.0",
      gitSha: process.env.NEXT_PUBLIC_GIT_SHA ?? "dev",
      buildTime: process.env.NEXT_PUBLIC_BUILD_TIME ?? "",
    },
    { headers: { "Cache-Control": "no-store, must-revalidate" } }
  );
}
