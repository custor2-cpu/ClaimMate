"use client";

import { useState } from "react";
import { Loader2, Search, Sparkles } from "lucide-react";
import { cn } from "@/lib/utils";
import { QUICK_PRESETS, type DisputeFormInput } from "@/lib/types";

const CATEGORY_OPTIONS = [
  "자동 감지",
  "체육시설/헬스장",
  "통신/인터넷",
  "의류/패션잡화",
  "전자제품",
  "여행/숙박",
  "학원/교육서비스",
  "상조/결혼서비스",
  "온라인쇼핑몰",
  "의료/병원",
  "보험",
  "자동차",
  "화장품/미용",
  "부동산/임대차",
  "식품",
];

interface DisputeFormProps {
  onSubmit: (input: DisputeFormInput) => void;
  loading: boolean;
}

export default function DisputeForm({ onSubmit, loading }: DisputeFormProps) {
  const [text, setText] = useState("");
  const [amount, setAmount] = useState("");
  const [date, setDate] = useState("");
  const [category, setCategory] = useState(CATEGORY_OPTIONS[0]);

  const applyPreset = (preset: DisputeFormInput) => {
    setText(preset.text);
    setAmount(preset.amount ? String(preset.amount) : "");
    setDate(preset.date ?? "");
    setCategory(preset.category ?? CATEGORY_OPTIONS[0]);
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!text.trim() || loading) return;
    onSubmit({
      text: text.trim(),
      amount: amount ? Number(amount) : null,
      date: date || null,
      category: category === "자동 감지" ? null : category,
    });
  };

  return (
    <div className="rounded-2xl border border-white/10 bg-slate-900/60 p-5 shadow-card sm:p-7">
      <div className="mb-5 flex flex-wrap items-center gap-2">
        <span className="flex items-center gap-1.5 text-xs font-semibold text-slate-400">
          <Sparkles className="h-3.5 w-3.5 text-brand-400" />
          빠른 테스트
        </span>
        {QUICK_PRESETS.map((preset) => (
          <button
            key={preset.label}
            type="button"
            onClick={() => applyPreset(preset.input)}
            className="rounded-full border border-white/10 bg-white/5 px-3 py-1.5 text-xs font-medium text-slate-200 transition hover:border-brand-400/50 hover:bg-brand-500/10 hover:text-brand-200"
          >
            {preset.label}
          </button>
        ))}
      </div>

      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label className="mb-1.5 block text-sm font-medium text-slate-300">
            상담/피해 내용을 자유롭게 입력해 주세요
          </label>
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="예) 헬스장 1년권 결제 후 2주 만에 환불 요청했으나 위약금 50%를 요구했습니다."
            rows={5}
            className="w-full resize-none rounded-xl border border-white/10 bg-slate-950/60 px-4 py-3 text-sm text-slate-100 placeholder:text-slate-500 outline-none transition focus:border-brand-400/60 focus:ring-2 focus:ring-brand-400/20"
          />
        </div>

        <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
          <div>
            <label className="mb-1.5 block text-sm font-medium text-slate-300">
              피해 금액 (원)
            </label>
            <input
              type="number"
              min={0}
              value={amount}
              onChange={(e) => setAmount(e.target.value)}
              placeholder="600000"
              className="w-full rounded-xl border border-white/10 bg-slate-950/60 px-4 py-2.5 text-sm text-slate-100 placeholder:text-slate-500 outline-none transition focus:border-brand-400/60 focus:ring-2 focus:ring-brand-400/20"
            />
          </div>
          <div>
            <label className="mb-1.5 block text-sm font-medium text-slate-300">
              결제/계약 일자
            </label>
            <input
              type="date"
              value={date}
              onChange={(e) => setDate(e.target.value)}
              className="w-full rounded-xl border border-white/10 bg-slate-950/60 px-4 py-2.5 text-sm text-slate-100 outline-none transition focus:border-brand-400/60 focus:ring-2 focus:ring-brand-400/20 [color-scheme:dark]"
            />
          </div>
          <div>
            <label className="mb-1.5 block text-sm font-medium text-slate-300">
              품목 카테고리
            </label>
            <select
              value={category}
              onChange={(e) => setCategory(e.target.value)}
              className="w-full rounded-xl border border-white/10 bg-slate-950/60 px-4 py-2.5 text-sm text-slate-100 outline-none transition focus:border-brand-400/60 focus:ring-2 focus:ring-brand-400/20"
            >
              {CATEGORY_OPTIONS.map((opt) => (
                <option key={opt} value={opt}>
                  {opt}
                </option>
              ))}
            </select>
          </div>
        </div>

        <button
          type="submit"
          disabled={!text.trim() || loading}
          className={cn(
            "flex w-full items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-brand-500 to-brand-700 py-3.5 text-sm font-semibold text-white shadow-glow transition hover:brightness-110",
            (!text.trim() || loading) && "cursor-not-allowed opacity-50 hover:brightness-100"
          )}
        >
          {loading ? (
            <>
              <Loader2 className="h-4 w-4 animate-spin" />
              AI가 분석 중입니다...
            </>
          ) : (
            <>
              <Search className="h-4 w-4" />
              구제 가능성 분석하기
            </>
          )}
        </button>
      </form>
    </div>
  );
}
