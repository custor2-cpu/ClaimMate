"""
lib/knowledge_base/legal_kb.json (14개 업종 + 일반 법령 조항 청크)을
OpenAI text-embedding-3-small로 임베딩하여 lib/knowledge_base/legal_kb_embedded.json으로
저장하는 1회성 빌드 스크립트.

RAG 런타임 검색(lib/legalSearch.ts)은 이 산출물만 읽으며, OpenAI를 직접 호출하지
않는다(쿼리 임베딩은 런타임에 별도로 호출). legal_kb.json을 수정(법령 개정 반영 등)한
뒤에는 반드시 이 스크립트를 다시 실행해 임베딩을 갱신해야 한다.

사용법:
  1) OPENAI_API_KEY를 환경변수로 export 하거나 .env.local에 넣어둔다.
  2) pip install openai   (api/requirements.txt에는 포함하지 않음 — 이 스크립트는
     로컬에서 1회만 실행하는 개발 도구이며 Vercel Python 함수 번들에는 포함되지 않는다.)
  3) python scripts/build_legal_kb.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KB_PATH = ROOT / "lib" / "knowledge_base" / "legal_kb.json"
OUTPUT_PATH = ROOT / "lib" / "knowledge_base" / "legal_kb_embedded.json"
ENV_LOCAL_PATH = ROOT / ".env.local"
EMBEDDING_MODEL = "text-embedding-3-small"


def _load_api_key_from_env_local() -> None:
    """OPENAI_API_KEY가 환경변수에 없으면 .env.local에서 읽어 os.environ에 채운다."""
    if os.environ.get("OPENAI_API_KEY") or not ENV_LOCAL_PATH.exists():
        return
    for line in ENV_LOCAL_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        if key.strip() == "OPENAI_API_KEY":
            os.environ["OPENAI_API_KEY"] = value.strip().strip('"').strip("'")
            return


def _embedding_input(chunk: dict) -> str:
    keywords = ", ".join(chunk.get("keywords", []))
    return (
        f"[{chunk['category']}] {chunk['topic']}\n"
        f"근거 법령: {chunk['law_name']}\n"
        f"{chunk['clause_summary']}\n"
        f"키워드: {keywords}"
    )


def build() -> None:
    _load_api_key_from_env_local()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit(
            "OPENAI_API_KEY가 설정되어 있지 않습니다. 환경변수로 export 하거나 "
            ".env.local에 추가한 뒤 다시 실행해 주세요."
        )

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit(
            "openai 패키지가 설치되어 있지 않습니다. `pip install openai`로 설치 후 "
            "다시 실행해 주세요. (이 패키지는 api/requirements.txt에는 포함하지 않습니다.)"
        ) from exc

    chunks: list[dict] = json.loads(KB_PATH.read_text(encoding="utf-8"))
    if not chunks:
        raise SystemExit(f"{KB_PATH}에 청크가 없습니다.")

    client = OpenAI(api_key=api_key)
    inputs = [_embedding_input(c) for c in chunks]

    print(f"{len(chunks)}개 청크를 {EMBEDDING_MODEL}로 임베딩합니다...")
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=inputs)

    embedded = [
        {**chunk, "embedding": item.embedding}
        for chunk, item in zip(chunks, response.data)
    ]

    OUTPUT_PATH.write_text(json.dumps(embedded, ensure_ascii=False), encoding="utf-8")
    print(f"임베딩 {len(embedded)}건을 {OUTPUT_PATH}에 저장했습니다.")


if __name__ == "__main__":
    build()
