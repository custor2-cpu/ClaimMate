"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import {
  AlertTriangle,
  BookOpen,
  ChevronDown,
  Gavel,
  Layers,
  Percent,
  Scale,
  ScanSearch,
  Sparkles,
  Tag,
} from "lucide-react";
import type { AnalysisReport, CertaintyLevel, SimilarCase } from "@/lib/types";
import ActionChecklist from "@/components/ActionChecklist";
import NoticeLetterModal from "@/components/NoticeLetterModal";
import SimilarCaseModal from "@/components/SimilarCaseModal";

interface ResultReportProps {
  report: AnalysisReport;
}

function gaugeColor(rate: number) {
  if (rate >= 70) return { stroke: "#34d399", text: "text-emerald-400", label: "구제 가능성 높음" };
  if (rate >= 40) return { stroke: "#fbbf24", text: "text-amber-400", label: "구제 가능성 보통" };
  return { stroke: "#f87171", text: "text-rose-400", label: "구제 가능성 낮음" };
}

/**
 * 법령 근거로 재산정된(success_rate_basis === "legal_reasoning") 경우, "85.0%" 같은
 * 정밀해 보이는 단일 숫자 대신 등급 텍스트 자체를 1차 정보로 보여준다(% 표시 없음).
 */
const CERTAINTY_STYLES: Record<CertaintyLevel, { stroke: string; text: string; badgeClass: string }> = {
  "매우 높음": {
    stroke: "#34d399",
    text: "text-emerald-400",
    badgeClass: "border-emerald-400/30 bg-emerald-500/10 text-emerald-300",
  },
  높음: {
    stroke: "#34d399",
    text: "text-emerald-400",
    badgeClass: "border-emerald-400/30 bg-emerald-500/10 text-emerald-300",
  },
  "조정 필요": {
    stroke: "#fbbf24",
    text: "text-amber-400",
    badgeClass: "border-amber-400/30 bg-amber-500/10 text-amber-300",
  },
  "구제 어려움": {
    stroke: "#f87171",
    text: "text-rose-400",
    badgeClass: "border-rose-400/30 bg-rose-500/10 text-rose-300",
  },
};

function SuccessGauge({
  rate,
  basis,
  certaintyLevel,
  reasoning,
}: {
  rate: number;
  basis?: AnalysisReport["success_rate_basis"];
  certaintyLevel?: CertaintyLevel | null;
  reasoning?: string;
}) {
  const radius = 76;
  const circumference = 2 * Math.PI * radius;
  const clamped = Math.min(100, Math.max(0, rate));
  const offset = circumference * (1 - clamped / 100);

  const isLegalGrade = basis === "legal_reasoning" && Boolean(certaintyLevel);
  const grade = certaintyLevel ? CERTAINTY_STYLES[certaintyLevel] : null;
  const mlGauge = gaugeColor(clamped);
  const stroke = grade?.stroke ?? mlGauge.stroke;
  const text = grade?.text ?? mlGauge.text;
  const label = mlGauge.label;

  return (
    <div className="flex flex-col items-center justify-center">
      <div className="relative h-44 w-44">
        <svg className="h-full w-full -rotate-90" viewBox="0 0 176 176">
          <circle
            cx="88"
            cy="88"
            r={radius}
            fill="none"
            stroke="rgba(255,255,255,0.08)"
            strokeWidth="14"
          />
          <motion.circle
            cx="88"
            cy="88"
            r={radius}
            fill="none"
            stroke={stroke}
            strokeWidth="14"
            strokeLinecap="round"
            strokeDasharray={circumference}
            initial={{ strokeDashoffset: circumference }}
            animate={{ strokeDashoffset: offset }}
            transition={{ duration: 1, ease: "easeOut" }}
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center px-4 text-center">
          {isLegalGrade && grade ? (
            <>
              <span className={`break-keep text-2xl font-bold leading-tight ${text}`}>{certaintyLevel}</span>
              <span className="mt-1 text-[11px] font-medium text-slate-400">법령 근거 기반 추정</span>
            </>
          ) : (
            <>
              <span className={`break-keep text-xl font-bold leading-tight ${text}`}>{label}</span>
              <span className="mt-1 text-[11px] font-medium text-slate-400">과거 유사사례 통계 기반</span>
            </>
          )}
        </div>
      </div>
      {isLegalGrade && grade && (
        <span
          className={`mt-3 flex items-center gap-1.5 rounded-full border px-3 py-1 text-[11px] font-medium ${grade.badgeClass}`}
          title={reasoning || undefined}
        >
          <Scale className="h-3 w-3" />
          법령 근거 기반 추정 (유사사례 신뢰도 낮음)
        </span>
      )}
    </div>
  );
}

export default function ResultReport({ report }: ResultReportProps) {
  const [modalOpen, setModalOpen] = useState(false);
  const [selectedCase, setSelectedCase] = useState<SimilarCase | null>(null);

  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
      className="space-y-4"
    >
      {report.used_fallback && (
        <div className="flex items-center gap-2 rounded-xl border border-amber-400/20 bg-amber-400/10 px-4 py-3 text-xs text-amber-200">
          <AlertTriangle className="h-4 w-4 shrink-0" />
          OPENAI_API_KEY가 설정되지 않아 규칙 기반 폴백 리포트로 생성되었습니다. .env.local에 키를 등록하면 GPT-4o mini
          에이전트가 결과를 생성합니다.
        </div>
      )}

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-5">
        <div className="rounded-2xl border border-white/10 bg-slate-900/60 p-6 shadow-card lg:col-span-2">
          <SuccessGauge
            rate={report.success_rate}
            basis={report.success_rate_basis}
            certaintyLevel={report.certainty_level}
            reasoning={report.legal_success_reasoning}
          />
          <div className="mt-5 flex flex-wrap justify-center gap-2">
            <span className="flex items-center gap-1.5 rounded-full border border-brand-400/30 bg-brand-500/10 px-3 py-1.5 text-xs font-medium text-brand-200">
              <Tag className="h-3.5 w-3.5" />
              {report.category}
            </span>
            <span className="flex items-center gap-1.5 rounded-full border border-white/10 bg-white/5 px-3 py-1.5 text-xs font-medium text-slate-300">
              <ScanSearch className="h-3.5 w-3.5" />
              {report.dispute_type}
            </span>
          </div>
        </div>

        <div className="space-y-4 lg:col-span-3">
          <div className="rounded-2xl border border-white/10 bg-slate-900/60 p-5 shadow-card sm:p-6">
            <div className="mb-2 flex items-center gap-2 text-sm font-semibold text-slate-100">
              <Gavel className="h-4 w-4 text-brand-400" />
              법적 기준 요약
            </div>
            <p className="text-sm leading-relaxed text-slate-300">{report.legal_basis}</p>
          </div>

          <div className="rounded-2xl border border-white/10 bg-slate-900/60 p-5 shadow-card sm:p-6">
            <div className="mb-2 flex items-center gap-2 text-sm font-semibold text-slate-100">
              <Percent className="h-4 w-4 text-brand-400" />
              예상 환급 범위
            </div>
            <p className="text-lg font-bold text-emerald-300">{report.estimated_refund}</p>
          </div>
        </div>
      </div>

      {report.referenced_clauses && report.referenced_clauses.length > 0 && (
        <div className="rounded-2xl border border-white/10 bg-slate-900/60 p-5 shadow-card sm:p-6">
          <div className="mb-3 flex items-center gap-2 text-sm font-semibold text-slate-100">
            <BookOpen className="h-4 w-4 text-brand-400" />
            인용 법령 조항 (RAG 검색 결과)
          </div>
          <div className="space-y-2">
            {report.referenced_clauses.map((c, idx) => (
              <details
                key={idx}
                className="group rounded-xl border border-white/10 bg-white/[0.02] p-3 text-sm open:bg-white/[0.04]"
              >
                <summary className="flex cursor-pointer list-none items-center justify-between gap-2 text-slate-200">
                  <span className="rounded-full bg-brand-500/15 px-2.5 py-1 text-[11px] font-medium text-brand-300">
                    {c.law_name}
                  </span>
                  <ChevronDown className="h-4 w-4 shrink-0 text-slate-500 transition group-open:rotate-180" />
                </summary>
                <p className="mt-2.5 leading-relaxed text-slate-400">{c.clause_content}</p>
                {c.formula && !c.formula.startsWith("해당 없음") && (
                  <p className="mt-2 rounded-lg bg-slate-950/60 px-2.5 py-1.5 font-mono text-xs text-emerald-300">
                    산정식: {c.formula}
                  </p>
                )}
              </details>
            ))}
          </div>
        </div>
      )}

      <div className="rounded-2xl border border-white/10 bg-slate-900/60 p-5 shadow-card sm:p-6">
        <div className="mb-3 flex items-center gap-2 text-sm font-semibold text-slate-100">
          <Layers className="h-4 w-4 text-brand-400" />
          과거 유사 사례 Top-3
        </div>
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
          {report.similar_cases.map((c, idx) => (
            <button
              key={idx}
              type="button"
              onClick={() => setSelectedCase(c)}
              className="rounded-xl border border-white/10 bg-white/[0.02] p-4 text-left text-xs transition hover:border-brand-400/30 hover:bg-white/[0.04]"
            >
              <div className="mb-2 flex items-center justify-between">
                <span className="font-semibold text-slate-200">{c.category}</span>
                <span className="rounded-full bg-brand-500/15 px-2 py-0.5 font-medium text-brand-300">
                  유사도 {c.similarity}%
                </span>
              </div>
              <p className="mb-1.5 text-slate-400">{c.dispute_type}</p>
              <p className="line-clamp-3 text-slate-500">처리결과: {c.outcome}</p>
            </button>
          ))}
        </div>
      </div>

      <ActionChecklist actionPlan={report.action_plan} proofDocuments={report.proof_documents} />

      <button
        onClick={() => setModalOpen(true)}
        className="flex w-full items-center justify-center gap-2 rounded-xl border border-brand-400/30 bg-brand-500/10 py-3.5 text-sm font-semibold text-brand-200 transition hover:bg-brand-500/20"
      >
        <Sparkles className="h-4 w-4" />
        맞춤형 내용증명 확인 및 복사하기
      </button>

      <NoticeLetterModal
        open={modalOpen}
        onClose={() => setModalOpen(false)}
        template={report.notice_letter_template}
      />

      <SimilarCaseModal caseData={selectedCase} onClose={() => setSelectedCase(null)} />
    </motion.div>
  );
}
