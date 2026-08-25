import { NextRequest, NextResponse } from "next/server";
import { spawn } from "node:child_process";
import path from "node:path";
import type { DisputeFormInput, MLAnalysisResult } from "@/lib/types";

export const runtime = "nodejs";

const PYTHON_CANDIDATES = [process.env.PYTHON_BIN, "python", "python3"].filter(
  (bin): bin is string => Boolean(bin)
);

/**
 * api/ml_predict.py와 app/api/analyze/route.ts를 같은 "/api/analyze" 경로에 두면
 * Vercel 빌드 시 Python 함수와 Next 함수가 같은 출력 슬롯을 두고 충돌해 인접한
 * 서버리스 함수(/api/agent)까지 잘못된 런타임으로 오염되는 문제가 있었다.
 * 그래서 Python 스크립트는 별도 경로(api/ml_predict.py -> /api/ml_predict)로 분리했고,
 * 이 라우트는 로컬 개발(next dev)에서는 subprocess로, Vercel 배포 환경에서는
 * 같은 배포 내의 /api/ml_predict 서버리스 함수를 내부 HTTP 호출로 중계한다.
 */
function runPythonAnalyzer(payload: DisputeFormInput): Promise<MLAnalysisResult> {
  const scriptPath = path.join(process.cwd(), "api", "ml_predict.py");

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

/**
 * Vercel 배포 환경에서는 python 인터프리터를 subprocess로 실행할 수 없으므로,
 * 같은 배포 내에 별도로 배포된 /api/ml_predict (Vercel Python Runtime) 함수를
 * 내부적으로 호출해 결과를 중계한다.
 */
async function callDeployedMlFunction(
  req: NextRequest,
  payload: DisputeFormInput
): Promise<MLAnalysisResult> {
  const url = new URL("/api/ml_predict", req.url);
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  const text = await res.text();
  try {
    return JSON.parse(text) as MLAnalysisResult;
  } catch {
    throw new Error(`ML 추론 결과 파싱 실패: ${text}`);
  }
}

export async function POST(req: NextRequest) {
  try {
    const body = (await req.json()) as DisputeFormInput;

    if (!body.text || !body.text.trim()) {
      return NextResponse.json({ error: "상담 내용을 입력해 주세요." }, { status: 400 });
    }

    const result = process.env.VERCEL
      ? await callDeployedMlFunction(req, body)
      : await runPythonAnalyzer(body);

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
