"""
Vercel Serverless Python 함수 진입점 (배포 시 /api/ml_predict 경로로 노출).

Pandas/NumPy 전처리(ml_engine.cleaner) -> Scikit-learn 추론(ml_engine.predictor)
파이프라인을 실행하여 분쟁 유형/구제 성공 확률/유사 사례를 반환한다.

두 가지 방식으로 호출 가능하다.
1) Vercel Python Runtime: `handler` 클래스가 HTTP POST 요청을 처리한다.
2) 로컬 개발(Next.js dev 서버): Node.js가 이 스크립트를 subprocess로 실행하고
   stdin으로 JSON을 전달하면 stdout으로 JSON 결과를 반환한다.

주의: 이 파일을 app/api/analyze/route.ts와 같은 "/api/analyze" 경로로 옮기면 안 된다.
Vercel 빌드 시 Python 함수와 Next 함수가 같은 출력 슬롯을 두고 충돌하면서
인접한 서버리스 함수(/api/agent 등)까지 이 Python 함수로 잘못 대체되는 문제가 있었다.
"""

from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ml_engine import predictor  # noqa: E402


def run_analysis(payload: dict) -> dict:
    text = (payload.get("text") or "").strip()
    amount = payload.get("amount")

    if not text:
        return {"error": "상담 내용을 입력해 주세요."}

    try:
        result = predictor.predict(text, amount)
    except ValueError as exc:
        return {"error": str(exc)}

    result["input"] = {
        "text": text,
        "amount": amount,
        "date": payload.get("date"),
        "category_hint": payload.get("category"),
    }
    return result


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        raw_body = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw_body or b"{}")
        except json.JSONDecodeError:
            payload = {}

        result = run_analysis(payload)
        body = json.dumps(result, ensure_ascii=False).encode("utf-8")

        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()


def _main() -> None:
    # sys.stdin/stdout.reconfigure()는 Node.js child_process로 파이프 연결된 스트림에서
    # Windows 기본 코드페이지(cp949 등)를 우선 적용해버려 한글이 깨지는 경우가 있어,
    # 텍스트 모드 대신 raw 바이트 버퍼를 직접 UTF-8로 인코딩/디코딩한다.
    raw_bytes = sys.stdin.buffer.read()
    raw = raw_bytes.decode("utf-8")
    try:
        payload = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError:
        payload = {}

    result = run_analysis(payload)
    sys.stdout.buffer.write(json.dumps(result, ensure_ascii=False).encode("utf-8"))


if __name__ == "__main__":
    _main()
