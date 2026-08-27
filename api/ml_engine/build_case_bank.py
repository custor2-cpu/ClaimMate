"""
공정거래위원회 소비자 민원학습데이터(모범상담 사례) CSV, 한국소비자원 품목별 피해구제
사례 XML, 한국소비자원 소비자상담 표준답변 CSV(전부 data/raw/)를 ML 케이스뱅크로
변환하는 1회성 빌드 스크립트.

앞의 두 원본은 (제목, 상담내용/질문, 답변내용) 형태의 자유 텍스트만 있고 카테고리
라벨이 없거나(모범상담 CSV) 믿을 수 없어서(XML의 "품목" 라벨엔 예를 들어 "금융/보험"에
상조회사 폐업·유사투자자문 사기 등 무관한 사례가 섞여 있음), 제목+본문 키워드 매칭
(_assign_category)으로 카테고리를 부여한다. 반면 표준답변 CSV는 "품목명"이 TV/헬스장처럼
단일 품목 단위로 깨끗하게 라벨링돼 있어 STD_ITEM_CATEGORY_MAP으로 직접 매핑한다(키워드
추측 불필요, 자세한 근거는 아래 주석 참고). 세 소스 모두 api/ml_engine/case_bank_data.json
으로 합쳐 저장한다. predictor.py는 런타임에 이 정제된 JSON만 로드해 원본 파일의
인코딩/파싱 이슈와 무관하게 동작한다.

사용법: python api/ml_engine/build_case_bank.py
"""

from __future__ import annotations

import glob
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
# data/raw/에는 검토 전 대용량 CSV(위해위험 데이터, 별도 스테이징 검토 대상)도 함께 있을 수 있어,
# glob("*.csv")로 아무 파일이나 집으면 안 되고 반드시 이 원본 하나만 골라야 한다.
RAW_CSV_NAME_HINT = "모범상담"
XML_PATH = RAW_DIR / "한국소비자원_품목별 피해구제 사례_20220331 (2).xml"
STD_ANSWER_PATH = RAW_DIR / "한국소비자원_소비자상담 표준답변_20250725.csv"
OUTPUT_PATH = Path(__file__).resolve().parent / "case_bank_data.json"

# 소비자상담 표준답변 CSV의 "품목명"은 XML의 "품목"과 달리 TV/냉장고/헬스장처럼
# 단일 품목·서비스 단위로 깨끗하게 라벨링돼 있어, 키워드 추측 없이 직접 매핑해도 안전하다고
# 판단했다(실제로 몇백 건을 열어본 결과 품목명과 실제 상담 내용이 어긋나는 사례를 못 찾음).
# 다만 ClaimMate 14개 카테고리 어디에도 깨끗이 안 맞는 품목(유사투자자문, 포장이사·택배
# 운송서비스, 각종공연관람, 신유형상품권 등 실물 결제 상품이 아니거나 카테고리가 없는 것)과
# 품목이 아니라 법률명 자체인 행(방문판매법, 전자상거래법 등)은 매핑하지 않고 제외한다.
STD_ITEM_CATEGORY_MAP: dict[str, str] = {
    "TV": "전자제품",
    "냉장고": "전자제품",
    "노트북컴퓨터": "전자제품",
    "룸에어컨": "전자제품",
    "스마트폰": "전자제품",
    "정수기대여": "전자제품",
    "건강(연금)보험": "보험",
    "실손(상해)보험": "보험",
    "대형승용자동차": "자동차",
    "중형승용자동차": "자동차",
    "자동차대여(렌트)": "자동차",
    "자동차수리": "자동차",
    "이동전화서비스": "통신/인터넷",
    "초고속인터넷": "통신/인터넷",
    "기타가방": "의류/패션잡화",
    "기타간편복": "의류/패션잡화",
    "기타세탁서비스": "의류/패션잡화",
    "기타신발·용품": "의류/패션잡화",
    "기타의류·섬유": "의류/패션잡화",
    "양복세탁": "의류/패션잡화",
    "운동화": "의류/패션잡화",
    "원피스": "의류/패션잡화",
    "점퍼, 재킷류": "의류/패션잡화",
    "캐주얼바지": "의류/패션잡화",
    "티셔츠": "의류/패션잡화",
    "핸드백": "의류/패션잡화",
    "기타일반화장품": "화장품/미용",
    "미용서비스": "화장품/미용",
    "화장품세트": "화장품/미용",
    "기타숙박시설": "여행/숙박",
    "항공여객운송서비스": "여행/숙박",
    "기타건강식품": "식품",
    "식료품·기호품(봉지면)": "식품",
    "외식": "식품",
    "인터넷교육서비스": "학원/교육서비스",
    "아파트": "부동산/임대차",
    "국내결혼중개": "상조/결혼서비스",
    "필라테스": "체육시설/헬스장",
    "헬스장": "체육시설/헬스장",
}
# 화상·사고 등 신체/재산 피해 배상 성격이라 ClaimMate가 다루는 환불/해지 분쟁과
# 결이 달라 제외하는 "구분" 접두어(위해위험 CSV에서 겪은 것과 같은 문제).
STD_EXCLUDED_TYPE_PREFIXES = ("안전",)

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
    "부동산/임대차": ["임대차", "부동산", "전세", "월세", "공인중개사"],
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
    matches = [p for p in glob.glob(str(RAW_DIR / "*.csv")) if RAW_CSV_NAME_HINT in p]
    if not matches:
        raise FileNotFoundError(f"원본 CSV를 찾을 수 없습니다: {RAW_DIR}/*{RAW_CSV_NAME_HINT}*.csv")

    df = pd.read_csv(matches[0], encoding="utf-8-sig", engine="python", on_bad_lines="skip")
    df.columns = ["no", "title", "content", "answer"]
    return df.dropna(subset=["title", "content", "answer"])


def _load_xml_records() -> list[tuple[str, str, str]]:
    """
    한국소비자원_품목별 피해구제 사례(Excel 2003 XML/SpreadsheetML) 원본에서
    (title, content, answer)를 뽑아낸다. XML 자체의 "품목" 라벨(보건/의료, 금융/보험 등)은
    쓰지 않는다 — 실제로 열어보니 예를 들어 "금융/보험" 안에 상조회사 폐업, 유사투자자문 사기,
    리볼빙 수수료처럼 보험과 무관한 사례가 다수 섞여 있어 그대로 매핑하면 오분류가 난다.
    대신 CSV와 동일하게 _assign_category() 키워드 매칭을 거치게 해 일관성과 안전성을 확보한다.
    """
    if not XML_PATH.exists():
        return []

    ns = {"ss": "urn:schemas-microsoft-com:office:spreadsheet"}
    tree = ET.parse(XML_PATH)
    worksheet = tree.getroot().find(".//ss:Worksheet", ns)
    table = worksheet.find("ss:Table", ns)

    records: list[tuple[str, str, str]] = []
    for row in table.findall("ss:Row", ns)[1:]:  # 첫 행은 헤더
        cells = row.findall("ss:Cell", ns)
        values = [(c.find("ss:Data", ns).text or "") if c.find("ss:Data", ns) is not None else "" for c in cells]
        if len(values) < 6:
            continue
        _, _item, _source, title, question, answer = values[:6]
        records.append((title.strip(), question.strip(), answer.strip()))
    return records


def _load_std_answer_records() -> list[tuple[str, str, str, str]]:
    """
    한국소비자원 소비자상담 표준답변 CSV에서 (category, dispute_type, question, answer)를
    뽑아낸다. "품목명"이 TV/헬스장처럼 깨끗한 단일 라벨이라 STD_ITEM_CATEGORY_MAP으로
    직접 매핑하고, "구분"(예: "청약철회_단순변심")을 dispute_type으로 그대로 쓴다.
    """
    if not STD_ANSWER_PATH.exists():
        return []

    df = pd.read_csv(STD_ANSWER_PATH, encoding="cp949", engine="c", on_bad_lines="skip")
    df.columns = ["no", "item", "type", "question", "answer"]
    df = df.dropna(subset=["item", "question", "answer"])

    records: list[tuple[str, str, str, str]] = []
    for _, row in df.iterrows():
        dispute_type = str(row["type"]).strip()
        if dispute_type.startswith(STD_EXCLUDED_TYPE_PREFIXES):
            continue
        category = STD_ITEM_CATEGORY_MAP.get(str(row["item"]).strip())
        if category is None:
            continue
        records.append((category, dispute_type, str(row["question"]).strip(), str(row["answer"]).strip()))
    return records


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


def _to_record(title: str, content: str, answer: str, category: str | None = None) -> dict | None:
    if len(content) < MIN_TEXT_LEN:
        return None

    if category is None:
        category = _assign_category(title, content)
    if category is None:
        return None

    return {
        "text": content[:MAX_TEXT_LEN],
        "category": category,
        "dispute_type": title[:80],
        "outcome": answer,
        "base_success_rate": _estimate_base_success_rate(answer),
    }


def build() -> list[dict]:
    df = _load_raw_df()
    records: list[dict] = []
    seen: set[tuple[str, str]] = set()

    def _append(record: dict | None) -> None:
        if record is None:
            return
        # 한국소비자원 XML 원본 자체에 같은 상담이 소수점/쉼표 표기만 다르게 중복
        # 수록된 경우가 있어(예: "990000원" vs "990,000원"), (제목, 답변 앞부분) 기준으로
        # 중복을 제거한다.
        dedup_key = (record["dispute_type"], record["outcome"][:80])
        if dedup_key in seen:
            return
        seen.add(dedup_key)
        records.append(record)

    for _, row in df.iterrows():
        _append(_to_record(str(row["title"]).strip(), str(row["content"]).strip(), str(row["answer"]).strip()))

    for title, content, answer in _load_xml_records():
        _append(_to_record(title, content, answer))

    for category, dispute_type, question, answer in _load_std_answer_records():
        _append(_to_record(dispute_type, question, answer, category=category))

    return records


if __name__ == "__main__":
    records = build()
    OUTPUT_PATH.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(records)}건의 케이스를 {OUTPUT_PATH}에 저장했습니다.")
