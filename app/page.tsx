"use client";

import { useState } from "react";
import { AlertCircle, BarChart3 } from "lucide-react";
import Header from "@/components/Header";
import DisputeForm from "@/components/DisputeForm";
import ResultReport from "@/components/ResultReport";
import StatCharts from "@/components/StatCharts";
import type { AnalysisReport, DisputeFormInput, MLAnalysisResult } from "@/lib/types";

export default function Home() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [report, setReport] = useState<AnalysisReport | null>(null);

  const handleSubmit = async (input: DisputeFormInput) => {
    setLoading(true);
    setError(null);
    setReport(null);

    try {
      const analyzeRes = await fetch("/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(input),
      });
      const mlResult = (await analyzeRes.json()) as MLAnalysisResult & { error?: string };
      if (!analyzeRes.ok || mlResult.error) {
        throw new Error(mlResult.error ?? "ML 분석에 실패했습니다.");
      }

      const agentRes = await fetch("/api/agent", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(mlResult),
      });
      const agentReport = (await agentRes.json()) as AnalysisReport & { error?: string };
      if (!agentRes.ok || agentReport.error) {
        throw new Error(agentReport.error ?? "리포트 생성에 실패했습니다.");
      }

      setReport(agentReport);
    } catch (err) {
      setError(err instanceof Error ? err.message : "알 수 없는 오류가 발생했습니다.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="min-h-screen">
      <Header />

      <section className="mx-auto max-w-6xl px-4 pb-8 pt-12 sm:px-6 sm:pt-16">
        <div className="mx-auto max-w-2xl text-center">
          <span className="inline-flex items-center gap-1.5 rounded-full border border-brand-400/30 bg-brand-500/10 px-3 py-1 text-xs font-medium text-brand-200">
            Pandas · Scikit-learn · GPT-4o mini 파이프라인
          </span>
          <h1 className="mt-4 text-3xl font-bold tracking-tight text-white sm:text-4xl">
            내 소비자 피해, <span className="text-brand-400">구제받을 수 있을까요?</span>
          </h1>
          <p className="mt-3 text-sm leading-relaxed text-slate-400 sm:text-base">
            상담 내용을 입력하면 AI가 한국소비자원 공공데이터를 기반으로 분쟁 유형을 분류하고,
            구제 성공 확률과 법적 근거, 맞춤형 내용증명까지 한 번에 제공합니다.
          </p>
        </div>
      </section>

      <section className="mx-auto max-w-6xl px-4 sm:px-6">
        <DisputeForm onSubmit={handleSubmit} loading={loading} />
      </section>

      {error && (
        <section className="mx-auto max-w-6xl px-4 pt-4 sm:px-6">
          <div className="flex items-center gap-2 rounded-xl border border-rose-400/20 bg-rose-400/10 px-4 py-3 text-sm text-rose-200">
            <AlertCircle className="h-4 w-4 shrink-0" />
            {error}
          </div>
        </section>
      )}

      {report && (
        <section className="mx-auto max-w-6xl px-4 pt-8 sm:px-6">
          <ResultReport report={report} />
        </section>
      )}

      <section className="mx-auto max-w-6xl px-4 py-12 sm:px-6">
        <div className="mb-1 flex items-center gap-2 text-sm font-semibold text-slate-200">
          <BarChart3 className="h-4 w-4 text-brand-400" />
          한국소비자원 공공데이터 통계 대시보드
        </div>
        <p className="mb-4 text-xs text-slate-500">
          공정거래위원회 1372 소비자상담 상담상세현황 공공데이터(data.go.kr) 2026년 1월 접수분 표본 3,000건 기준
        </p>
        <StatCharts />
      </section>

      <footer className="border-t border-white/5 py-8 text-center text-xs text-slate-600">
        본 서비스의 분석 결과는 참고용 정보이며 법적 효력을 갖지 않습니다. 정확한 상담은 1372 소비자상담센터를
        이용해 주세요. © 2026 ClaimMate
      </footer>
    </main>
  );
}
