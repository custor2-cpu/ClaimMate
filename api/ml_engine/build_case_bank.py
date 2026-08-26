"""
공정거래위원회 소비자 민원학습데이터(모범상담 사례) 원본 CSV(data/raw/*.csv)를
ML 케이스뱅크로 변환하는 1회성 빌드 스크립트.

원본은 (사건번호, 상담제목, 상담내용, 답변내용) 4개 컬럼만 있고 카테고리 라벨이 없으므로,
제목+본문 키워드 매칭으로 품목 카테고리를 부여한 뒤 api/ml_engine/case_bank_data.json으로
저장한다. predictor.py는 런타임에 이 정제된 JSON만 로드해 원본 CSV의 인코딩/파싱 이슈와
무관하게 동작한다.

사용법: python api/ml_engine/build_case_bank.py
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_GLOB = str(ROOT / "data" / "raw" / "*.csv")
OUTPUT_PATH = Path(__file__).resolve().parent / "case_bank_data.json"

# 카테고리별 매칭 키워드. 우선순위(딕셔너리 순서)대로 검사하여
# 가장 많이 매칭된 카테고리에 배정한다.
CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "의료/병원": ["병원", "의료", "진료", "수술", "치료", "의사", "약물", "환자", "입원", "시술"],
    "보험": ["보험", "보험사", "보험금", "보험료"],
    "자동차": ["자동차", "차량", "정비", "견인", "타이어", "엔진", "중고차"],
    "통신/인터넷": ["통신사", "인터넷", "휴대폰", "핸드폰", "이동전화", "와이파이", "이동통신"],
    "의류/패션잡화": ["의류", "패딩", "니트", "코트", "신발", "가방", "원단", "봉제", "세탁", "드라이클리닝"],
    "화장품/미용": ["화장품", "피부마사지", "성형", "미용실", "에스테틱"],
    "온라인쇼핑몰": ["온라인쇼핑", "쇼핑몰", "오픈마켓", "택배", "배송지연", "전자상거래"],
    "여행/숙박": ["여행", "숙박", "호텔", "펜션", "항공", "리조트"],
    "전자제품": ["가전", "전자제품", "냉장고", "세탁기", "에어컨", "노트북", "텔레비전"],
    "식품": ["식품", "음식점", "배달음식", "천일염", "농산물"],
    "학원/교육서비스": ["학원", "교습소", "과외", "수강"],
    "부동산/임대차": ["임대차", "부동산", "전세", "월세", "중개"],
    "상조/결혼서비스": ["상조", "장례", "웨딩", "결혼식"],
    "체육시설/헬스장": ["헬스장", "체육시설", "피트니스", "필라테스", "요가", "크로스핏"],
}

MAX_TEXT_LEN = 400
MIN_TEXT_LEN = 15

# 원본 데이터에는 "환급 성공/실패" 라벨이 없으므로, 답변(ANS_CONTENT)의 어조에서
# 유리/불리 신호를 추출해 구제 성공 확률의 기준값(base_success_rate)을 근사한다.
_POSITIVE_SIGNALS = ["가능합니다", "받으실 수 있", "받을 수 있", "청구할 수 있", "요구하실 수 있", "인정됩니다", "환급 가능"]
_NEGATIVE_SIGNALS = ["어렵습니다", "곤란합니다", "받기 어려운", "책임이 없습니다", "인정되지 않", "거부할 수 있", "보상받기 어려운", "판단됩니다"]


def _estimate_base_success_rate(answer: str) -> float:
    pos = sum(1 for sig in _POSITIVE_SIGNALS if sig in answer)
    neg = sum(1 for sig in _NEGATIVE_SIGNALS if sig in answer)
    score = pos - neg
    rate = 65.0 + score * 8.0
    return float(min(90.0, max(30.0, rate)))


def _load_raw_df() -> pd.DataFrame:
    matches = glob.glob(RAW_GLOB)
    if not matches:
        raise FileNotFoundError(f"원본 CSV를 찾을 수 없습니다: {RAW_GLOB}")

    df = pd.read_csv(matches[0], encoding="utf-8-sig", engine="python", on_bad_lines="skip")
    df.columns = ["no", "title", "content", "answer"]
    return df.dropna(subset=["title", "content", "answer"])


def _assign_category(title: str, content: str) -> str | None:
    haystack = f"{title} {content}"
    best_category = None
    best_score = 0
    for category, keywords in CATEGORY_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in haystack)
        if score > best_score:
            best_score = score
            best_category = category
    return best_category if best_score > 0 else None


def build() -> list[dict]:
    df = _load_raw_df()
    records: list[dict] = []

    for _, row in df.iterrows():
        title = str(row["title"]).strip()
        content = str(row["content"]).strip()
        answer = str(row["answer"]).strip()

        if len(content) < MIN_TEXT_LEN:
            continue

        category = _assign_category(title, content)
        if category is None:
            continue

        records.append(
            {
                "text": content[:MAX_TEXT_LEN],
                "category": category,
                "dispute_type": title[:80],
                "outcome": answer,
                "base_success_rate": _estimate_base_success_rate(answer),
            }
        )

    return records


if __name__ == "__main__":
    records = build()
    OUTPUT_PATH.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(records)}건의 케이스를 {OUTPUT_PATH}에 저장했습니다.")
