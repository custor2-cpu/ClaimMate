"use client";

import { useState } from "react";
import { AlertCircle, BarChart3, MessageCircle, Send } from "lucide-react";
import Header from "@/components/Header";
import DisputeForm from "@/components/DisputeForm";
import ResultReport from "@/components/ResultReport";
import StatCharts from "@/components/StatCharts";
import type {
  AgentQuestion,
  AgentQuestionResponse,
  AnalysisReport,
  AnalysisStage,
  DisputeFormInput,
  MLAnalysisResult,
} from "@/lib/types";

export default function Home() {
  const [stage, setStage] = useState<AnalysisStage>("idle");
  const [error, setError] = useState<string | null>(null);
  const [report, setReport] = useState<AnalysisReport | null>(null);
  const [questions, setQuestions] = useState<AgentQuestion[]>([]);
  const [questionAnswers, setQuestionAnswers] = useState<Record<string, string>>({});
  const [pendingInput, setPendingInput] = useState<DisputeFormInput | null>(null);

  const handleSubmit = async (input: DisputeFormInput) => {
    setStage("analyzing");
    setError(null);
    setReport(null);
    setQuestions([]);

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

      // ML 분석(보통 수 초 이내)과 달리 LLM 리포트 생성은 OpenAI 응답 속도에 따라
      // 걸리는 시간이 크게 달라진다 — 단계를 나눠 표시해야 사용자가 지금 어느 단계에
      // 있는지, 왜 오래 걸릴 수 있는지 알 수 있다.
      setStage("generating");
      const agentRes = await fetch("/api/agent", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(mlResult),
      });
      const agentResponse = (await agentRes.json()) as
        | (AnalysisReport & { error?: string })
        | (AgentQuestionResponse & { error?: string });
      if (!agentRes.ok || "error" in agentResponse && agentResponse.error) {
        throw new Error(
          ("error" in agentResponse && agentResponse.error) || "리포트 생성에 실패했습니다."
        );
      }

      if ("next_action" in agentResponse && agentResponse.next_action === "ask_questions") {
        setQuestions(agentResponse.questions);
        setQuestionAnswers({});
        setPendingInput(input);
        return;
      }

      setReport(agentResponse as AnalysisReport);
    } catch (err) {
      setError(err instanceof Error ? err.message : "알 수 없는 오류가 발생했습니다.");
    } finally {
      setStage("idle");
    }
  };

  const handleQuestionSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!pendingInput || questions.some((question) => !questionAnswers[question.id]?.trim())) return;

    const answers = questions
      .map((question) => `${question.question}\n답변: ${questionAnswers[question.id].trim()}`)
      .join("\n\n");
    await handleSubmit({ ...pendingInput, text: `${pendingInput.text}\n\n[추가 확인 답변]\n${answers}` });
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
        <DisputeForm onSubmit={handleSubmit} stage={stage} />
      </section>

      {error && (
        <section className="mx-auto max-w-6xl px-4 pt-4 sm:px-6">
          <div className="flex items-center gap-2 rounded-xl border border-rose-400/20 bg-rose-400/10 px-4 py-3 text-sm text-rose-200">
            <AlertCircle className="h-4 w-4 shrink-0" />
            {error}
          </div>
        </section>
      )}

      {questions.length > 0 && (
        <section className="mx-auto max-w-6xl px-4 pt-8 sm:px-6">
          <form
            onSubmit={handleQuestionSubmit}
            className="rounded-2xl border border-brand-400/20 bg-slate-900/70 p-5 shadow-card sm:p-6"
          >
            <div className="mb-5 flex items-start gap-3">
              <MessageCircle className="mt-0.5 h-5 w-5 shrink-0 text-brand-400" />
              <div>
                <h2 className="text-base font-semibold text-slate-100">정확한 분석을 위해 확인이 필요합니다</h2>
                <p className="mt-1 text-sm leading-relaxed text-slate-400">
                  아래 질문에 답하면 환급액과 적용 기준을 다시 계산합니다.
                </p>
              </div>
            </div>
            <div className="space-y-4">
              {questions.map((question) => (
                <label key={question.id} className="block">
                  <span className="mb-1.5 block text-sm font-medium text-slate-200">{question.question}</span>
                  <span className="mb-2 block text-xs text-slate-500">{question.reason}</span>
                  <textarea
                    rows={2}
                    value={questionAnswers[question.id] ?? ""}
                    onChange={(event) =>
                      setQuestionAnswers((previous) => ({
                        ...previous,
                        [question.id]: event.target.value,
                      }))
                    }
                    className="w-full resize-y rounded-xl border border-white/10 bg-slate-950/60 px-4 py-3 text-sm text-slate-100 outline-none transition placeholder:text-slate-500 focus:border-brand-400/60 focus:ring-2 focus:ring-brand-400/20"
                    placeholder="답변을 입력해 주세요"
                  />
                </label>
              ))}
            </div>
            <button
              type="submit"
              disabled={stage !== "idle" || questions.some((question) => !questionAnswers[question.id]?.trim())}
              className="mt-5 flex w-full items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-brand-500 to-brand-700 py-3 text-sm font-semibold text-white shadow-glow transition hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-50"
            >
              <Send className="h-4 w-4" />
              답변하고 다시 분석하기
            </button>
          </form>
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
