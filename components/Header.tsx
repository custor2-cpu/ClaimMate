"use client";

import { Scale, ShieldCheck } from "lucide-react";

const APP_VERSION = process.env.NEXT_PUBLIC_APP_VERSION ?? "0.0.0";
const GIT_SHA = process.env.NEXT_PUBLIC_GIT_SHA ?? "dev";
const BUILD_TIME = process.env.NEXT_PUBLIC_BUILD_TIME ?? "";

function formatBuildTime(iso: string) {
  if (!iso) return "";
  return new Date(iso).toLocaleString("ko-KR", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export default function Header() {
  return (
    <header className="sticky top-0 z-40 border-b border-white/10 bg-slate-950/80 backdrop-blur-md">
      <div className="mx-auto flex max-w-6xl items-center justify-between px-4 py-4 sm:px-6">
        <div className="flex items-center gap-2.5">
          <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-gradient-to-br from-brand-400 to-brand-700 shadow-glow">
            <Scale className="h-5 w-5 text-white" strokeWidth={2.25} />
          </div>
          <div>
            <p className="text-base font-bold leading-none tracking-tight text-white">
              ClaimMate
            </p>
            <p className="mt-1 text-[11px] font-medium leading-none text-slate-400">
              AI 소비자 피해구제 분석 에이전트
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <div className="hidden items-center gap-2 rounded-full border border-emerald-400/20 bg-emerald-400/10 px-3 py-1.5 text-xs font-medium text-emerald-300 sm:flex">
            <ShieldCheck className="h-3.5 w-3.5" />
            한국소비자원 공공데이터 기반
          </div>
          <span
            className="rounded-full border border-white/10 bg-white/5 px-2.5 py-1 font-mono text-[11px] text-slate-400"
            title={BUILD_TIME ? `빌드: ${formatBuildTime(BUILD_TIME)}` : undefined}
          >
            v{APP_VERSION} · {GIT_SHA}
          </span>
        </div>
      </div>
    </header>
  );
}
