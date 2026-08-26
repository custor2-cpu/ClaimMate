"use client";

import { useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { Check, Copy, FileSignature, X } from "lucide-react";

interface NoticeLetterModalProps {
  open: boolean;
  onClose: () => void;
  template: string;
}

export default function NoticeLetterModal({ open, onClose, template }: NoticeLetterModalProps) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(template);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      setCopied(false);
    }
  };

  return (
    <AnimatePresence>
      {open && (
        <motion.div
          className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/70 p-4 backdrop-blur-sm"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          onClick={onClose}
        >
          <motion.div
            className="flex max-h-[85vh] w-full max-w-2xl flex-col overflow-hidden rounded-2xl border border-white/10 bg-slate-900 shadow-2xl"
            initial={{ opacity: 0, y: 16, scale: 0.98 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 16, scale: 0.98 }}
            transition={{ duration: 0.2 }}
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between border-b border-white/10 px-5 py-4">
              <div className="flex items-center gap-2 text-sm font-semibold text-slate-100">
                <FileSignature className="h-4 w-4 text-brand-400" />
                맞춤형 내용증명 템플릿
              </div>
              <button
                onClick={onClose}
                className="rounded-lg p-1.5 text-slate-400 transition hover:bg-white/5 hover:text-white"
                aria-label="닫기"
              >
                <X className="h-4 w-4" />
              </button>
            </div>

            <div className="min-h-0 overflow-y-auto px-5 py-4">
              <pre className="whitespace-pre-wrap break-words rounded-xl border border-white/10 bg-slate-950/60 p-4 font-mono text-[13px] leading-relaxed text-slate-200">
                {template}
              </pre>
            </div>

            <div className="flex items-center justify-between gap-3 border-t border-white/10 bg-slate-900/80 px-5 py-4">
              <p className="text-xs text-slate-500">
                본 템플릿은 참고용이며 법적 효력을 보장하지 않습니다. 발송 전 내용을 검토해 주세요.
              </p>
              <button
                onClick={handleCopy}
                className="flex shrink-0 items-center gap-2 rounded-xl bg-gradient-to-r from-brand-500 to-brand-700 px-4 py-2.5 text-sm font-semibold text-white shadow-glow transition hover:brightness-110"
              >
                {copied ? (
                  <>
                    <Check className="h-4 w-4" />
                    복사됨
                  </>
                ) : (
                  <>
                    <Copy className="h-4 w-4" />
                    클립보드 복사
                  </>
                )}
              </button>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
