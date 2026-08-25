"use client";

import { useEffect } from "react";

const CURRENT_GIT_SHA = process.env.NEXT_PUBLIC_GIT_SHA ?? "dev";
const CHECK_INTERVAL_MS = 60_000;

async function clearCachesAndReload() {
  try {
    if ("caches" in window) {
      const keys = await caches.keys();
      await Promise.all(keys.map((key) => caches.delete(key)));
    }
  } catch {
    // 캐시 API를 사용할 수 없어도 새로고침은 계속 진행한다.
  }
  window.location.reload();
}

/**
 * 새 배포가 감지되면(서버의 Git 커밋 해시가 이 탭이 로드된 시점의 해시와 달라지면)
 * 브라우저 캐시를 지우고 자동으로 새로고침해 최신 버전을 반영한다.
 * "dev" 빌드(로컬 개발)에서는 매 요청마다 값이 달라질 수 있어 비교를 건너뛴다.
 */
export default function VersionWatcher() {
  useEffect(() => {
    if (CURRENT_GIT_SHA === "dev") return;

    let cancelled = false;

    const checkVersion = async () => {
      try {
        const res = await fetch("/api/version", { cache: "no-store" });
        if (!res.ok) return;
        const data = (await res.json()) as { gitSha?: string };
        if (!cancelled && data.gitSha && data.gitSha !== CURRENT_GIT_SHA) {
          await clearCachesAndReload();
        }
      } catch {
        // 네트워크 오류 시 다음 주기에 재시도한다.
      }
    };

    const intervalId = window.setInterval(checkVersion, CHECK_INTERVAL_MS);

    const onVisibilityChange = () => {
      if (document.visibilityState === "visible") checkVersion();
    };
    document.addEventListener("visibilitychange", onVisibilityChange);

    return () => {
      cancelled = true;
      window.clearInterval(intervalId);
      document.removeEventListener("visibilitychange", onVisibilityChange);
    };
  }, []);

  return null;
}
