"use client";

import { AnimatePresence, motion } from "framer-motion";
import { Layers, X } from "lucide-react";
import type { SimilarCase } from "@/lib/types";

interface SimilarCaseModalProps {
  caseData: SimilarCase | null;
  onClose: () => void;
}

export default function SimilarCaseModal({ caseData, onClose }: SimilarCaseModalProps) {
  return (
    <AnimatePresence>
      {caseData && (
        <motion.div
          className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/70 p-4 backdrop-blur-sm"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          onClick={onClose}
        >
          <motion.div
            className="flex max-h-[85vh] w-full max-w-lg flex-col overflow-hidden rounded-2xl border border-white/10 bg-slate-900 shadow-2xl"
            initial={{ opacity: 0, y: 16, scale: 0.98 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 16, scale: 0.98 }}
            transition={{ duration: 0.2 }}
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between border-b border-white/10 px-5 py-4">
              <div className="flex items-center gap-2 text-sm font-semibold text-slate-100">
                <Layers className="h-4 w-4 text-brand-400" />
                유사 사례 상세
              </div>
              <button
                onClick={onClose}
                className="rounded-lg p-1.5 text-slate-400 transition hover:bg-white/5 hover:text-white"
                aria-label="닫기"
              >
                <X className="h-4 w-4" />
              </button>
            </div>

            <div className="overflow-y-auto px-5 py-4">
              <div className="mb-3 flex items-center justify-between">
                <span className="font-semibold text-slate-200">{caseData.category}</span>
                <span className="rounded-full bg-brand-500/15 px-2.5 py-1 text-xs font-medium text-brand-300">
                  유사도 {caseData.similarity}%
                </span>
              </div>
              <p className="mb-4 text-sm font-medium text-slate-300">{caseData.dispute_type}</p>
              <div className="rounded-xl border border-white/10 bg-slate-950/60 p-4">
                <p className="mb-1.5 text-xs font-medium text-slate-500">처리결과</p>
                <p className="whitespace-pre-wrap break-words text-sm leading-relaxed text-slate-300">
                  {caseData.outcome}
                </p>
              </div>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
