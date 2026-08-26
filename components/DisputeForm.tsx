"use client";

import { useEffect, useState } from "react";
import { Loader2, Search, Sparkles } from "lucide-react";
import { cn } from "@/lib/utils";
import { QUICK_PRESETS, type AnalysisStage, type DisputeFormInput } from "@/lib/types";

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
  stage: AnalysisStage;
}

// LLM 리포트 생성 단계는 OpenAI 응답 속도(+ Vercel 서버리스 콜드 스타트)에 따라 걸리는
// 시간이 매번 크게 달라 정확히 예측할 수 없다. 고정된 예상치를 카운트다운으로 보여주면
// 실제로는 아직 진행 중인데 "1초 남음"에 멈춘 채 계속 표시돼 마치 멈춘 것처럼 보이는
// 문제가 있었다 — 대신 실제 경과 시간을 그대로 세어 보여주고, 오래 걸리면 안내 문구만
// 바꾼다(숫자를 거짓으로 줄이지 않음).
const LONG_WAIT_THRESHOLD_SECONDS = 20;

export default function DisputeForm({ onSubmit, stage }: DisputeFormProps) {
  const [text, setText] = useState("");
  const [amount, setAmount] = useState("");
  const [date, setDate] = useState("");
  const [category, setCategory] = useState(CATEGORY_OPTIONS[0]);
  const loading = stage !== "idle";

  // stage가 바뀌는 것과 "같은" 렌더에서 시작 시각을 리셋해야 한다. useEffect로 리셋하면
  // 그 사이 한 틱 동안 이전 단계에서 누적된 경과 초가 잠깐 그대로 보였다가 다음 렌더에서
  // 뚝 떨어지는 것처럼 보이는 문제가 있었다(예: "5초" -> "1초"). 렌더 중 상태를 갱신하는
  // React의 공식 패턴(경고 없이 리렌더만 한 번 더 유발)으로 이 지연을 없앤다.
  const [prevStage, setPrevStage] = useState(stage);
  const [stageStartedAt, setStageStartedAt] = useState<number | null>(null);
  const [, forceTick] = useState(0);
  if (stage !== prevStage) {
    setPrevStage(stage);
    setStageStartedAt(stage === "idle" ? null : Date.now());
  }

  useEffect(() => {
    if (stageStartedAt === null) return;
    const interval = setInterval(() => forceTick((n) => n + 1), 1000);
    return () => clearInterval(interval);
  }, [stageStartedAt]);

  const elapsedSeconds = stageStartedAt === null ? 0 : Math.floor((Date.now() - stageStartedAt) / 1000);

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
          {stage === "analyzing" ? (
            <>
              <Loader2 className="h-4 w-4 animate-spin" />
              1/2단계: 데이터 분석 중...
            </>
          ) : stage === "generating" ? (
            <>
              <Loader2 className="h-4 w-4 animate-spin" />
              {elapsedSeconds < LONG_WAIT_THRESHOLD_SECONDS
                ? `2/2단계: AI 리포트 생성 중... (${elapsedSeconds}초 경과)`
                : `2/2단계: AI 리포트 생성 중입니다. 평소보다 다소 걸리고 있어요... (${elapsedSeconds}초 경과)`}
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
