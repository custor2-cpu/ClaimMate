import { NextRequest, NextResponse } from "next/server";
import { spawn } from "node:child_process";
import path from "node:path";
import type { DisputeFormInput, MLAnalysisResult } from "@/lib/types";

export const runtime = "nodejs";

const PYTHON_CANDIDATES = [process.env.PYTHON_BIN, "python", "python3"].filter(
  (bin): bin is string => Boolean(bin)
);

/**
 * app/api 아래의 이 라우트는 로컬 개발(next dev) 및 Node 기반 배포 환경에서
 * /api/analyze.py (Pandas/NumPy 전처리 + Scikit-learn 추론 파이프라인)를
 * subprocess로 실행해 결과를 중계한다.
 */
function runPythonAnalyzer(payload: DisputeFormInput): Promise<MLAnalysisResult> {
  const scriptPath = path.join(process.cwd(), "api", "analyze.py");

  const tryRun = (index: number): Promise<MLAnalysisResult> => {
    if (index >= PYTHON_CANDIDATES.length) {
      return Promise.reject(
        new Error("Python 인터프리터를 찾을 수 없습니다. PYTHON_BIN 환경 변수를 설정해 주세요.")
      );
    }

    return new Promise((resolve, reject) => {
      const child = spawn(PYTHON_CANDIDATES[index], [scriptPath], {
        cwd: process.cwd(),
      });

      let stdout = "";
      let stderr = "";

      child.stdout.on("data", (chunk) => (stdout += chunk.toString("utf-8")));
      child.stderr.on("data", (chunk) => (stderr += chunk.toString("utf-8")));

      child.on("error", (err) => {
        if ((err as NodeJS.ErrnoException).code === "ENOENT") {
          tryRun(index + 1).then(resolve, reject);
        } else {
          reject(err);
        }
      });

      child.on("close", (code) => {
        if (code !== 0) {
          reject(new Error(`ML 추론 스크립트 실행 실패 (exit ${code}): ${stderr}`));
          return;
        }
        try {
          resolve(JSON.parse(stdout) as MLAnalysisResult);
        } catch {
          reject(new Error(`ML 추론 결과 파싱 실패: ${stdout || stderr}`));
        }
      });

      child.stdin.write(JSON.stringify(payload));
      child.stdin.end();
    });
  };

  return tryRun(0);
}

export async function POST(req: NextRequest) {
  try {
    const body = (await req.json()) as DisputeFormInput;

    if (!body.text || !body.text.trim()) {
      return NextResponse.json({ error: "상담 내용을 입력해 주세요." }, { status: 400 });
    }

    const result = await runPythonAnalyzer(body);

    if (result.error) {
      return NextResponse.json({ error: result.error }, { status: 422 });
    }

    return NextResponse.json(result);
  } catch (err) {
    console.error("[/api/analyze] ML 분석 실패:", err);
    return NextResponse.json(
      { error: err instanceof Error ? err.message : "ML 분석 중 오류가 발생했습니다." },
      { status: 500 }
    );
  }
}
