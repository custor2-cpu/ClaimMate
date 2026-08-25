import type { Metadata } from "next";
import { Inter } from "next/font/google";
import VersionWatcher from "@/components/VersionWatcher";
import "./globals.css";

const inter = Inter({ subsets: ["latin"], variable: "--font-inter" });

export const metadata: Metadata = {
  title: "ClaimMate | AI 소비자 피해구제 분석 에이전트",
  description:
    "비정형 소비자 상담/피해 내역을 Pandas·Scikit-learn·LLM 파이프라인으로 분석해 구제 성공 확률, 법적 근거, 맞춤형 내용증명을 제공하는 AI 에이전트",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="ko" className={inter.variable}>
      <body className="min-h-screen font-sans text-slate-100 antialiased">
        <VersionWatcher />
        {children}
      </body>
    </html>
  );
}
