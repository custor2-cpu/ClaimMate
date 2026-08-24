"use client";

import { useState } from "react";
import { CheckCircle2, Circle, ClipboardList, FileText } from "lucide-react";
import { cn } from "@/lib/utils";

interface ActionChecklistProps {
  actionPlan: string[];
  proofDocuments: string[];
}

function ChecklistGroup({
  title,
  icon,
  items,
}: {
  title: string;
  icon: React.ReactNode;
  items: string[];
}) {
  const [checked, setChecked] = useState<boolean[]>(() => items.map(() => false));

  const toggle = (idx: number) => {
    setChecked((prev) => prev.map((v, i) => (i === idx ? !v : v)));
  };

  const doneCount = checked.filter(Boolean).length;

  return (
    <div className="rounded-2xl border border-white/10 bg-slate-900/60 p-5 shadow-card sm:p-6">
      <div className="mb-4 flex items-center justify-between">
        <div className="flex items-center gap-2 text-sm font-semibold text-slate-100">
          {icon}
          {title}
        </div>
        <span className="rounded-full bg-white/5 px-2.5 py-1 text-xs font-medium text-slate-400">
          {doneCount}/{items.length} 완료
        </span>
      </div>

      <ul className="space-y-2">
        {items.map((item, idx) => (
          <li key={idx}>
            <button
              type="button"
              onClick={() => toggle(idx)}
              className={cn(
                "flex w-full items-start gap-3 rounded-xl border border-white/5 bg-white/[0.02] px-3.5 py-3 text-left text-sm transition hover:border-brand-400/30 hover:bg-brand-500/5",
                checked[idx] && "border-emerald-400/20 bg-emerald-400/5"
              )}
            >
              {checked[idx] ? (
                <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-emerald-400" />
              ) : (
                <Circle className="mt-0.5 h-4 w-4 shrink-0 text-slate-500" />
              )}
              <span
                className={cn(
                  "leading-relaxed text-slate-200",
                  checked[idx] && "text-slate-400 line-through decoration-slate-600"
                )}
              >
                {item}
              </span>
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}

export default function ActionChecklist({ actionPlan, proofDocuments }: ActionChecklistProps) {
  return (
    <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
      <ChecklistGroup
        title="단계별 실행 계획"
        icon={<ClipboardList className="h-4 w-4 text-brand-400" />}
        items={actionPlan}
      />
      <ChecklistGroup
        title="준비할 증빙 자료"
        icon={<FileText className="h-4 w-4 text-brand-400" />}
        items={proofDocuments}
      />
    </div>
  );
}
