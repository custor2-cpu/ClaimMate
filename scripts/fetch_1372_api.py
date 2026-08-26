"""
공정거래위원회_1372 소비자상담 상담상세현황 서비스(data.go.kr)로 공공데이터 통계 대시보드
(public/consumer_stats.json, components/StatCharts.tsx가 렌더링)를 갱신하는 로컬 1회성 스크립트.

중요: 이 API는 개별 상담 "건" 단위 응답이지만 필드가 전부 코드/코드명(품목분류, 분쟁유형,
처리결과, 성별, 연령대, 지역 등)뿐이고 상담 원문/답변 같은 자유텍스트 필드는 전혀 제공하지
않는다(직접 호출해서 확인함 — cnslTxt/answrTxt 같은 필드는 존재하지 않음). 그래서 ML
케이스뱅크(api/ml_engine/case_bank_data.json)를 보강하는 용도로는 쓸 수 없고, 대신 이
스크립트처럼 "품목별 소비자 분쟁 빈도"/"분쟁 처리 결과 분포" 같은 집계 통계용으로만 쓴다.

End Point와 인증키는 data.go.kr 마이페이지의 활용신청 상세 페이지에서 확인한 값이다.
서비스 자체 URL(오퍼레이션명 없는 베이스 경로)로 호출하면 NO_OPENAPI_SERVICE_ERROR가
나므로 반드시 오퍼레이션명(getDscsnDetailSttus_2)까지 붙여야 한다.

사용법:
  1) DATA_GO_KR_SERVICE_KEY를 환경변수로 export 하거나 .env.local에 넣어둔다
     (포털이 제공하는 URL 인코딩된 키 그대로 넣으면 된다).
  2) python scripts/fetch_1372_api.py [--rcpt-ym YYYYMM] [--pages N]
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path
from urllib.parse import unquote

import requests

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "public" / "consumer_stats.json"
ENV_LOCAL_PATH = ROOT / ".env.local"
ENV_KEY_NAME = "DATA_GO_KR_SERVICE_KEY"

URL = "https://apis.data.go.kr/1130000/CcnDscsnDetailSttus_2Service/getDscsnDetailSttus_2"
NUM_OF_ROWS = 100
REQUEST_INTERVAL_SEC = 0.15

# 품목대분류(prdlstLclasNm) -> ClaimMate 카테고리. 이 API의 대분류 체계는 ClaimMate의
# 14개 카테고리와 다른 축이라(업종이 아니라 순수 품목 기준) 깨끗하게 매핑되는 것만 넣는다.
# 온라인쇼핑몰/상조·결혼서비스/체육시설·헬스장은 이 대분류 체계에 대응하는 항목이 없어서
# 제외한다(억지로 끼워 맞추면 그래프가 왜곡된다).
LCLAS_TO_CATEGORY: dict[str, str] = {
    "정보통신서비스": "통신/인터넷",
    "정보통신기기": "전자제품",
    "의류·섬유신변용품": "의류/패션잡화",
    "차량·승용물": "자동차",
    "식료품·기호품": "식품",
    "보건·위생용품": "화장품/미용",
    "보건·위생서비스": "화장품/미용",
    "토지·건물·설비": "부동산/임대차",
    "의료서비스": "의료/병원",
    "교육서비스": "학원/교육서비스",
    "보험": "보험",
    "세탁서비스": "의류/패션잡화",
    "식생활기기": "전자제품",
}
# 여행/숙박은 대분류가 아니라 중분류(prdlstMlsfcNm)에 있다.
TRAVEL_MLSFC_VALUES = {"숙박시설", "여행"}

# 처리결과(dscsnPrcsRsltCdNm) 원본 값이 20종 가까이 되어 도넛 차트로는 너무 잘게 쪼개지므로
# 의미 단위로 묶는다. 매핑에 없는 값은 "기타"로 들어간다.
OUTCOME_BUCKETS: dict[str, str] = {
    "환급": "환급/배상",
    "계약이행": "환급/배상",
    "계약해제.해지": "환급/배상",
    "기타정보제공": "정보제공/안내",
    "법.제도설명": "정보제공/안내",
    "분쟁해결기준설명": "정보제공/안내",
    "시장정보제공": "정보제공/안내",
    "상품정보제공": "정보제공/안내",
    "추가서류안내": "정보제공/안내",
    "피해구제접수안내": "정보제공/안내",
    "사업자자율상담": "정보제공/안내",
    "처리중": "처리중",
    "합의불성립": "조정 불성립/반려",
    "반려(권익위 등)": "조정 불성립/반려",
}


def _load_service_key() -> str:
    """DATA_GO_KR_SERVICE_KEY가 환경변수에 없으면 .env.local에서 읽어 os.environ에 채운다."""
    if not os.environ.get(ENV_KEY_NAME) and ENV_LOCAL_PATH.exists():
        for line in ENV_LOCAL_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            if key.strip() == ENV_KEY_NAME:
                os.environ[ENV_KEY_NAME] = value.strip().strip('"').strip("'")
                break

    encoded_key = os.environ.get(ENV_KEY_NAME)
    if not encoded_key:
        raise SystemExit(
            f"{ENV_KEY_NAME}가 설정되어 있지 않습니다. 환경변수로 export 하거나 "
            ".env.local에 추가한 뒤 다시 실행해 주세요."
        )
    # 포털이 제공하는 "인코딩된" 키를 그대로 쓰면 requests가 재인코딩해 이중 인코딩 오류가
    # 나므로, unquote()로 디코딩한 뒤 params에 넣어 requests가 한 번만 인코딩하게 한다.
    return unquote(encoded_key)


def fetch_items(rcpt_ym: str, pages: int) -> list[dict]:
    service_key = _load_service_key()
    items: list[dict] = []
    for page in range(1, pages + 1):
        params = {
            "serviceKey": service_key,
            "pageNo": page,
            "numOfRows": NUM_OF_ROWS,
            "resultType": "json",
            "rcptYm": rcpt_ym,
        }
        res = requests.get(URL, params=params, timeout=20)
        try:
            data = res.json()
        except ValueError:
            print(f"[{page}페이지] JSON 파싱 실패, 건너뜀: {res.text[:200]}")
            continue

        if data.get("resultCode") != "00":
            print(f"[{page}페이지] API 오류, 중단: {data}")
            break

        page_items = data.get("items", [])
        if not page_items:
            break
        items.extend(page_items)
        time.sleep(REQUEST_INTERVAL_SEC)

    return items


def build_stats(items: list[dict]) -> dict:
    category_counter: Counter[str] = Counter()
    for item in items:
        mlsfc = item.get("prdlstMlsfcNm")
        lclas = item.get("prdlstLclasNm")
        if mlsfc in TRAVEL_MLSFC_VALUES:
            category_counter["여행/숙박"] += 1
        elif lclas in LCLAS_TO_CATEGORY:
            category_counter[LCLAS_TO_CATEGORY[lclas]] += 1

    outcome_counter: Counter[str] = Counter()
    for item in items:
        value = item.get("dscsnPrcsRsltCdNm")
        if value is None:
            continue
        outcome_counter[OUTCOME_BUCKETS.get(value, "기타")] += 1

    return {
        "category_frequency": [
            {"category": cat, "count": count} for cat, count in category_counter.most_common()
        ],
        "resolution_outcome": [
            {"name": name, "value": count} for name, count in outcome_counter.most_common()
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rcpt-ym", default="202601", help="접수년월(YYYYMM), 기본값: 202601")
    parser.add_argument("--pages", type=int, default=30, help="가져올 페이지 수(100건/페이지), 기본값: 30")
    args = parser.parse_args()

    items = fetch_items(args.rcpt_ym, args.pages)
    if not items:
        raise SystemExit("수집된 데이터가 없습니다. rcpt_ym/키/네트워크 상태를 확인해 주세요.")

    stats = build_stats(items)
    OUTPUT_PATH.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    mapped = sum(d["count"] for d in stats["category_frequency"])
    print(f"{len(items)}건 수집, {mapped}건 카테고리 매핑 완료 -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
