"use client";

import { useEffect, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { BarChart3, PieChart as PieChartIcon } from "lucide-react";
import type { StatsData } from "@/lib/types";

const DONUT_COLORS = ["#4d7fff", "#2557f2", "#1a41cc", "#80abff", "#64748b"];

function ChartCard({
  title,
  icon,
  children,
}: {
  title: string;
  icon: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <div className="rounded-2xl border border-white/10 bg-slate-900/60 p-5 shadow-card sm:p-6">
      <div className="mb-4 flex items-center gap-2 text-sm font-semibold text-slate-100">
        {icon}
        {title}
      </div>
      {children}
    </div>
  );
}

export default function StatCharts() {
  const [stats, setStats] = useState<StatsData | null>(null);

  useEffect(() => {
    fetch("/consumer_stats.json")
      .then((res) => res.json())
      .then((data: StatsData) => setStats(data))
      .catch(() => setStats(null));
  }, []);

  if (!stats) {
    return (
      <div className="rounded-2xl border border-white/10 bg-slate-900/60 p-10 text-center text-sm text-slate-500 shadow-card">
        통계 데이터를 불러오는 중입니다...
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 gap-4 lg:grid-cols-5">
      <div className="lg:col-span-3">
        <ChartCard
          title="품목별 소비자 분쟁 빈도"
          icon={<BarChart3 className="h-4 w-4 text-brand-400" />}
        >
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={stats.category_frequency} margin={{ left: -20, right: 12 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" vertical={false} />
              <XAxis
                dataKey="category"
                tick={{ fill: "#94a3b8", fontSize: 11 }}
                interval={0}
                angle={-25}
                textAnchor="end"
                height={70}
                axisLine={{ stroke: "rgba(255,255,255,0.1)" }}
                tickLine={false}
              />
              <YAxis
                tick={{ fill: "#94a3b8", fontSize: 11 }}
                axisLine={false}
                tickLine={false}
              />
              <Tooltip
                cursor={{ fill: "rgba(77,127,255,0.08)" }}
                contentStyle={{
                  background: "#0f1f52",
                  border: "1px solid rgba(255,255,255,0.1)",
                  borderRadius: 12,
                  fontSize: 12,
                  color: "#e2e8f0",
                }}
              />
              <Bar dataKey="count" name="상담 건수" fill="#4d7fff" radius={[6, 6, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>
      </div>

      <div className="lg:col-span-2">
        <ChartCard
          title="분쟁 처리 결과 분포"
          icon={<PieChartIcon className="h-4 w-4 text-brand-400" />}
        >
          <ResponsiveContainer width="100%" height={280}>
            <PieChart>
              <Pie
                data={stats.resolution_outcome}
                dataKey="value"
                nameKey="name"
                innerRadius={60}
                outerRadius={95}
                paddingAngle={2}
              >
                {stats.resolution_outcome.map((_, idx) => (
                  <Cell key={idx} fill={DONUT_COLORS[idx % DONUT_COLORS.length]} stroke="none" />
                ))}
              </Pie>
              <Tooltip
                contentStyle={{
                  background: "#0f1f52",
                  border: "1px solid rgba(255,255,255,0.1)",
                  borderRadius: 12,
                  fontSize: 12,
                  color: "#e2e8f0",
                }}
              />
              <Legend
                wrapperStyle={{ fontSize: 11, color: "#94a3b8" }}
                iconType="circle"
                iconSize={8}
              />
            </PieChart>
          </ResponsiveContainer>
        </ChartCard>
      </div>
    </div>
  );
}
