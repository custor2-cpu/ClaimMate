"""
Scikit-learn 기반 분쟁 유형 분류 및 구제 성공 확률 추론 엔진.

- TF-IDF(char n-gram) 벡터화 + Logistic Regression: 비정형 텍스트 -> 분쟁 유형 분류
- K-Means: 과거 상담 사례를 군집화하여 유사 사례 후보군 축소
- Cosine Similarity: 군집 내에서 최종 Top-3 유사 사례 매칭
- 구제 성공 확률: 분류기 신뢰도(confidence)와 유사 사례의 실측 처리결과(base_success_rate)를
  가중 결합하여 산출한다.

형태소 분석기(KoNLPy 등)는 별도 JVM 의존성이 필요해 서버리스 환경에 부적합하므로,
한국어 텍스트에도 안정적으로 동작하는 char_wb n-gram TF-IDF를 사용한다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

from ml_engine import cleaner

SYNTHETIC_CASES: list[dict] = [
    # 체육시설 / 헬스장
    {"text": "헬스장 1년 회원권을 결제하고 2주 만에 개인 사정으로 환불을 요청했는데 위약금 50%를 요구했어요", "category": "체육시설/헬스장", "dispute_type": "중도 해지 위약금 과다 청구", "base_success_rate": 83.0, "outcome": "위약금 10% 공제 후 일할 환불 처리"},
    {"text": "필라테스 3개월 등록했는데 다리를 다쳐서 못 나가게 됐는데 환불이 절대 안된다고 합니다", "category": "체육시설/헬스장", "dispute_type": "중도 해지 환불 거부", "base_success_rate": 80.0, "outcome": "진단서 제출 후 잔여 회차 환불"},
    {"text": "PT 20회 끊었는데 트레이너가 그만둬서 다른 사람으로 바뀌었고 나머지 환불해달라니 안된대요", "category": "체육시설/헬스장", "dispute_type": "중도 해지 위약금 과다 청구", "base_success_rate": 81.0, "outcome": "잔여 회차 환급 권고"},
    {"text": "헬스장이 갑자기 폐업했는데 남은 이용기간에 대한 환불을 안해줍니다 연락도 안됩니다", "category": "체육시설/헬스장", "dispute_type": "사업자 폐업 환불 거부", "base_success_rate": 74.0, "outcome": "소비자분쟁해결기준에 따른 환급 권고"},
    {"text": "수영장 3개월권 결제했는데 한달만 다니고 그만두려니 위약금이 너무 비싸요 절반도 안돌려줘요", "category": "체육시설/헬스장", "dispute_type": "중도 해지 위약금 과다 청구", "base_success_rate": 82.0, "outcome": "위약금 10% 공제 후 환급"},
    {"text": "요가원 등록하고 첫 수업 듣자마자 마음에 안들어서 환불요청했는데 노환불 정책이라며 거부당했습니다", "category": "체육시설/헬스장", "dispute_type": "청약철회 거부", "base_success_rate": 78.0, "outcome": "7일 이내 청약철회 인정"},
    {"text": "크로스핏 6개월 결제 후 이사를 가게 되어 환불 요청했더니 규정에 없다며 거절당했어요", "category": "체육시설/헬스장", "dispute_type": "중도 해지 환불 거부", "base_success_rate": 79.0, "outcome": "잔여기간 일할 환급"},
    {"text": "헬스장 회원권 환불을 요청했더니 카드결제 취소는 안되고 현금으로만 준다고 하는데 이게 맞나요", "category": "체육시설/헬스장", "dispute_type": "결제수단 환불 거부", "base_success_rate": 76.0, "outcome": "결제 수단과 동일하게 환급 권고"},
    # 통신 / 인터넷
    {"text": "인터넷 3년 약정했는데 이사가면서 서비스 안되는 지역이라 해지했더니 위약금 30만원을 청구했어요", "category": "통신/인터넷", "dispute_type": "약정 중도 해지 위약금 과다 청구", "base_success_rate": 70.0, "outcome": "서비스 불가 지역 이전은 위약금 면제 대상"},
    {"text": "휴대폰 개통할때 설명 못들은 부가서비스가 요금에 계속 청구되고 있어서 해지하려니 위약금이 크대요", "category": "통신/인터넷", "dispute_type": "부가서비스 미고지 위약금 분쟁", "base_success_rate": 66.0, "outcome": "미고지 부가서비스 요금 환급 및 위약금 조정"},
    {"text": "인터넷 회선 속도가 계약한거랑 너무 다르게 느려서 해지하겠다니까 위약금을 내라고 합니다", "category": "통신/인터넷", "dispute_type": "품질 미달 계약 해지 위약금 분쟁", "base_success_rate": 68.0, "outcome": "품질 미달 입증 시 위약금 감면"},
    {"text": "약정기간 끝나기 한달 전에 해지했는데도 위약금 전액을 내라고 통신사에서 연락왔어요", "category": "통신/인터넷", "dispute_type": "약정 중도 해지 위약금 과다 청구", "base_success_rate": 69.0, "outcome": "잔여 약정기간 비례 위약금 재산정"},
    {"text": "결합상품 가입했는데 한쪽만 해지하려니까 다른 서비스까지 위약금이 붙는다고 합니다", "category": "통신/인터넷", "dispute_type": "결합상품 해지 위약금 분쟁", "base_success_rate": 65.0, "outcome": "개별 해지 위약금 분리 산정 권고"},
    {"text": "인터넷 설치기사가 안와서 한달을 그냥 날렸는데 그 기간 요금도 다 청구됐어요 해지도 위약금 내래요", "category": "통신/인터넷", "dispute_type": "서비스 미제공 요금 및 위약금 분쟁", "base_success_rate": 72.0, "outcome": "미제공 기간 요금 환급 및 위약금 면제"},
    {"text": "IPTV 셋톱박스가 고장나서 AS 요청했는데 계속 안고쳐줘서 해지했더니 위약금을 물라고 하네요", "category": "통신/인터넷", "dispute_type": "품질 미달 계약 해지 위약금 분쟁", "base_success_rate": 71.0, "outcome": "사업자 귀책 해지로 위약금 면제"},
    # 의류 / 패션잡화
    {"text": "온라인으로 옷을 주문했는데 2주가 지나도 배송이 안오고 문의해도 답이 없어요", "category": "의류/패션잡화", "dispute_type": "배송 지연", "base_success_rate": 75.0, "outcome": "전액 환불 및 지연 배상"},
    {"text": "신발을 주문했는데 다른 색상이 와서 교환 요청했더니 배송비를 소비자가 다 내라고 합니다", "category": "의류/패션잡화", "dispute_type": "오배송 교환 비용 분쟁", "base_success_rate": 78.0, "outcome": "판매자 귀책 오배송은 왕복 배송비 부담"},
    {"text": "구매한 원피스에 실밥이 다 풀려있고 얼룩도 있는데 판매자가 사용한 흔적이라며 환불을 거부해요", "category": "의류/패션잡화", "dispute_type": "제품 하자 환불 거부", "base_success_rate": 76.0, "outcome": "하자 입증 시 전액 환불"},
    {"text": "해외구매대행 옷을 시켰는데 한달이 지나도 오지 않고 셀러가 잠수를 탔어요", "category": "의류/패션잡화", "dispute_type": "미배송 환불 거부", "base_success_rate": 70.0, "outcome": "결제대행사 통한 환불 처리"},
    {"text": "가방을 샀는데 사진과 실물이 너무 달라서 반품하려니 단순변심이라 안된다고 합니다", "category": "의류/패션잡화", "dispute_type": "청약철회(표시광고 상이) 거부", "base_success_rate": 73.0, "outcome": "상세페이지 상이 시 청약철회 인정"},
    {"text": "택배가 파손된 채로 도착했는데 판매자는 택배사 잘못이라며 책임을 떠넘깁니다", "category": "의류/패션잡화", "dispute_type": "배송 파손 환불 거부", "base_success_rate": 74.0, "outcome": "판매자 우선 환불 후 택배사 구상 처리"},
    # 전자제품
    {"text": "노트북을 산지 한달만에 화면이 안켜지는데 산 곳에서는 제조사 AS센터로만 가라고 합니다", "category": "전자제품", "dispute_type": "품질 불량 및 무상 A/S 거부", "base_success_rate": 80.0, "outcome": "구입 1개월 이내 불량은 교환/환불 우선"},
    {"text": "세탁기가 배송온지 일주일만에 물이 새서 교환 요청했는데 유상수리만 해준다고 합니다", "category": "전자제품", "dispute_type": "품질 불량 무상 교환 거부", "base_success_rate": 82.0, "outcome": "초기불량 무상 교환 대상"},
    {"text": "TV를 샀는데 화면에 줄이 가서 수리 맡겼더니 부품이 없다고 두달째 방치중입니다", "category": "전자제품", "dispute_type": "수리 지연 및 A/S 지연", "base_success_rate": 77.0, "outcome": "수리 지연 시 동종 제품 교환/환불"},
    {"text": "냉장고 압축기에서 계속 소음이 나서 AS 불렀는데 정상이라고 하고 그냥 갔어요 소음이 여전해요", "category": "전자제품", "dispute_type": "품질 불량 A/S 거부", "base_success_rate": 75.0, "outcome": "품질보증기간 내 재점검 및 교환 권고"},
    {"text": "에어컨을 설치했는데 냉방이 전혀 안돼서 환불을 요청했더니 설치비는 못돌려준다고 합니다", "category": "전자제품", "dispute_type": "품질 불량 환불 거부", "base_success_rate": 79.0, "outcome": "설치비 포함 전액 환불 대상"},
    # 여행 / 숙박
    {"text": "해외여행 패키지를 예약했는데 개인사정으로 취소하려니 위약금이 여행비의 90%나 됩니다", "category": "여행/숙박", "dispute_type": "여행계약 취소 위약금 과다 청구", "base_success_rate": 63.0, "outcome": "여행 표준약관 기준 위약금 조정"},
    {"text": "펜션을 예약하고 갔는데 사진이랑 완전 다른 시설이라 환불해달라니 예약금은 안된대요", "category": "여행/숙박", "dispute_type": "숙박시설 상이 환불 거부", "base_success_rate": 68.0, "outcome": "표시광고 상이 시 환불 인정"},
    {"text": "호텔 예약을 했는데 사업자가 갑자기 예약을 취소시켜놓고 환불도 늦게 해준다고 합니다", "category": "여행/숙박", "dispute_type": "사업자 귀책 예약 취소 환불 지연", "base_success_rate": 72.0, "outcome": "즉시 전액 환불 및 손해배상"},
    {"text": "여행사가 폐업해서 결제한 여행비를 돌려받을 방법이 없다고 하는데 어떻게 해야하나요", "category": "여행/숙박", "dispute_type": "사업자 폐업 환불 거부", "base_success_rate": 60.0, "outcome": "여행자보험/공제 통한 환급 절차 안내"},
    {"text": "리조트를 예약했는데 태풍으로 못가게 되어 취소했더니 환불이 하나도 안된다고 합니다", "category": "여행/숙박", "dispute_type": "천재지변 취소 환불 거부", "base_success_rate": 65.0, "outcome": "불가항력 사유 위약금 감면"},
    # 학원 / 교육서비스
    {"text": "어학원 6개월 등록했는데 한달 다니고 그만두려니 남은 학원비를 하나도 안돌려준대요", "category": "학원/교육서비스", "dispute_type": "중도 해지 환불 거부", "base_success_rate": 79.0, "outcome": "잔여 수강료 일할 환급 대상"},
    {"text": "온라인 강의를 결제했는데 강사가 바뀌어서 환불요청했더니 이미 결제된건 환불 불가라고 합니다", "category": "학원/교육서비스", "dispute_type": "청약철회 거부", "base_success_rate": 74.0, "outcome": "수강 개시 전 환불 인정"},
    {"text": "코딩 부트캠프 등록금을 냈는데 커리큘럼이 계약서랑 완전히 달라서 환불을 요청했어요", "category": "학원/교육서비스", "dispute_type": "계약 내용 상이 환불 거부", "base_success_rate": 76.0, "outcome": "계약 불일치 시 전액 환불"},
    {"text": "피아노 학원 3개월 선결제했는데 다음 달부터 못다니게 됐고 환불 문의하니 연락을 안받아요", "category": "학원/교육서비스", "dispute_type": "중도 해지 환불 거부", "base_success_rate": 78.0, "outcome": "미사용 기간 환급 및 지연이자"},
    # 상조 / 결혼서비스
    {"text": "상조회사에 5년간 납입했는데 해지하려니 납입금의 절반도 안되는 돈만 돌려준다고 합니다", "category": "상조/결혼서비스", "dispute_type": "해약환급금 과소 지급", "base_success_rate": 62.0, "outcome": "표준약관 해약환급금 기준 재산정"},
    {"text": "웨딩홀 계약금을 걸었는데 다른 날짜로 변경이 안된다며 계약금을 전액 몰취하겠다고 합니다", "category": "상조/결혼서비스", "dispute_type": "계약 해지 위약금 과다 청구", "base_success_rate": 64.0, "outcome": "예식일 전 해지 위약금 기준 조정"},
    {"text": "상조상품을 해지하려는데 설계사가 계속 연락을 피하고 해약환급금 안내를 안해줘요", "category": "상조/결혼서비스", "dispute_type": "해지 절차 지연", "base_success_rate": 60.0, "outcome": "선불식 할부거래법상 해약환급금 지급 의무"},
    # 온라인쇼핑몰
    {"text": "단순 변심으로 반품하려는데 이미 포장을 뜯었다고 반품이 절대 안된다고 못박습니다", "category": "온라인쇼핑몰", "dispute_type": "청약철회(단순 변심) 거부", "base_success_rate": 58.0, "outcome": "포장 훼손이 없으면 7일 이내 청약철회 가능"},
    {"text": "쇼핑몰에서 결제했는데 갑자기 품절이라며 취소시키고 환불은 일주일 넘게 안해줍니다", "category": "온라인쇼핑몰", "dispute_type": "환불 지연", "base_success_rate": 68.0, "outcome": "결제일로부터 3영업일 이내 환불 의무"},
    {"text": "정기구독 서비스를 해지했는데 다음달에도 계속 요금이 빠져나가고 있어요", "category": "온라인쇼핑몰", "dispute_type": "정기결제 해지 후 과오납", "base_success_rate": 71.0, "outcome": "과오납 금액 전액 환급"},
    # 화장품 / 미용
    {"text": "피부관리실에서 10회 결제했는데 관리사가 바뀌고 효과도 없어서 환불을 요청했더니 이미 3회 사용해서 안된다고 합니다", "category": "화장품/미용", "dispute_type": "중도 해지 환불 거부", "base_success_rate": 72.0, "outcome": "잔여 회차 위약금 공제 후 환급"},
    {"text": "무료 체험 이벤트인줄 알고 갔다가 고가의 화장품 세트를 강매당했는데 환불이 안된다고 합니다", "category": "화장품/미용", "dispute_type": "강매 계약 청약철회 거부", "base_success_rate": 75.0, "outcome": "방문판매법상 14일 이내 청약철회 가능"},
    {"text": "네일샵 정기권을 끊었는데 샵이 폐업해서 환불을 받을 방법이 없어요", "category": "화장품/미용", "dispute_type": "사업자 폐업 환불 거부", "base_success_rate": 60.0, "outcome": "소비자분쟁해결기준에 따른 환급 절차 안내"},
    {"text": "쓰던 화장품에서 이물질이 나와서 환불 요청했는데 개봉했다는 이유로 거부당했습니다", "category": "화장품/미용", "dispute_type": "제품 하자 환불 거부", "base_success_rate": 74.0, "outcome": "품질 하자 입증 시 전액 환불"},
    {"text": "피부시술 부작용이 생겨서 병원에 항의했더니 원래 그런 반응이 있을 수 있다며 책임을 회피합니다", "category": "화장품/미용", "dispute_type": "시술 부작용 보상 거부", "base_success_rate": 65.0, "outcome": "의료분쟁조정 신청 및 진료기록 감정 권고"},
    # 부동산 / 임대차
    {"text": "전세 계약 만료로 이사가려는데 집주인이 새 세입자를 못구했다며 보증금을 돌려주지 않습니다", "category": "부동산/임대차", "dispute_type": "임대차 보증금 반환 지연", "base_success_rate": 78.0, "outcome": "임차권등기명령 및 보증금반환청구소송 안내"},
    {"text": "부동산 중개인이 하자를 고지하지 않아서 계약 후 누수를 발견했는데 책임이 없다고 합니다", "category": "부동산/임대차", "dispute_type": "중개 고지의무 위반 분쟁", "base_success_rate": 68.0, "outcome": "중개대상물 확인설명 의무 위반 시 손해배상"},
    {"text": "월세 계약을 중도 해지하려는데 중개수수료와 위약금을 모두 내라고 합니다", "category": "부동산/임대차", "dispute_type": "중도 해지 위약금 과다 청구", "base_success_rate": 62.0, "outcome": "계약서상 특약 확인 후 위약금 조정"},
    # 식품
    {"text": "배달음식에서 이물질이 나와서 환불을 요청했는데 업체에서 증거가 없다며 거부합니다", "category": "식품", "dispute_type": "이물질 혼입 환불 거부", "base_success_rate": 76.0, "outcome": "사진 증빙 시 전액 환불 및 재발방지 요청"},
    {"text": "구매한 식품의 유통기한이 지났는데 판매자가 단순 변심으로 취급해 환불을 거부합니다", "category": "식품", "dispute_type": "유통기한 경과 제품 환불 거부", "base_success_rate": 80.0, "outcome": "식품위생법 위반으로 전액 환불 및 신고 대상"},
    {"text": "정기배송 식품을 해지했는데 다음 회차 결제가 이미 진행되어 환불을 요청한 상태입니다", "category": "식품", "dispute_type": "정기배송 해지 후 과오납", "base_success_rate": 71.0, "outcome": "과오납 금액 전액 환급"},
]

_REAL_CASE_BANK_PATH = Path(__file__).resolve().parent / "case_bank_data.json"


def _load_real_cases() -> list[dict]:
    """공정거래위원회 소비자 민원학습데이터(build_case_bank.py 산출물)를 로드한다.

    빌드 산출물이 없는 환경(최초 클론 직후 등)에서도 앱이 동작하도록,
    파일이 없으면 조용히 빈 리스트를 반환하고 SYNTHETIC_CASES만으로 학습한다.
    """
    if not _REAL_CASE_BANK_PATH.exists():
        return []
    return json.loads(_REAL_CASE_BANK_PATH.read_text(encoding="utf-8"))


CASE_BANK: list[dict] = _load_real_cases() + SYNTHETIC_CASES


@dataclass
class SimilarCase:
    category: str
    dispute_type: str
    similarity: float
    outcome: str


_vectorizer: TfidfVectorizer | None = None
_classifier: LogisticRegression | None = None
_kmeans: KMeans | None = None
_tfidf_matrix = None
_cluster_labels = None
_bank_df = None

# 이 스크립트는 API 요청마다 새 프로세스로 실행되므로(subprocess), 매번 처음부터
# 학습하면 요청당 수 초가 소요된다. 학습된 모델을 디스크에 캐시해 재사용하고,
# 케이스뱅크 데이터가 바뀐 경우에만(캐시 키 불일치) 재학습한다.
_MODEL_CACHE_PATH = Path(__file__).resolve().parent / "_model_cache.joblib"

# 분류기 신뢰도가 이 값 미만이면(애매한 입력) 카테고리 단일 필터 대신
# K-Means 군집 전체를 유사 사례 후보로 넓혀 검색한다.
_LOW_CONFIDENCE_THRESHOLD = 0.35


def _get_bank_df():
    return cleaner.build_dataframe(CASE_BANK)


def _cache_key() -> str:
    real_mtime = _REAL_CASE_BANK_PATH.stat().st_mtime if _REAL_CASE_BANK_PATH.exists() else 0
    return f"{real_mtime}:{len(CASE_BANK)}"


def _load_from_cache(cache_key: str) -> bool:
    global _vectorizer, _classifier, _kmeans, _tfidf_matrix, _cluster_labels, _bank_df

    if not _MODEL_CACHE_PATH.exists():
        return False
    try:
        cached = joblib.load(_MODEL_CACHE_PATH)
    except Exception:
        return False
    if cached.get("cache_key") != cache_key:
        return False

    _vectorizer = cached["vectorizer"]
    _classifier = cached["classifier"]
    _kmeans = cached["kmeans"]
    _tfidf_matrix = cached["tfidf_matrix"]
    _cluster_labels = cached["cluster_labels"]
    _bank_df = cached["bank_df"]
    return True


def _train():
    global _vectorizer, _classifier, _kmeans, _tfidf_matrix, _cluster_labels, _bank_df

    cache_key = _cache_key()
    if _load_from_cache(cache_key):
        return

    bank_df = _get_bank_df()
    _bank_df = bank_df
    clean_texts = bank_df["clean_text"].tolist()

    _vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1)
    _tfidf_matrix = _vectorizer.fit_transform(clean_texts)

    numeric_features = cleaner.numeric_feature_matrix(bank_df)
    x_train = hstack([_tfidf_matrix, csr_matrix(numeric_features)])

    # 카테고리(대분류) 단위로 분류: 세부 dispute_type은 표본이 너무 적어
    # 분류기 대신 유사 사례 검색(Top-1)에서 가져온다.
    categories = bank_df["category"].tolist()
    _classifier = LogisticRegression(max_iter=3000)
    _classifier.fit(x_train, categories)

    n_clusters = min(len(set(categories)), len(bank_df) - 1)
    _kmeans = KMeans(n_clusters=max(n_clusters, 2), n_init=10, random_state=42)
    _cluster_labels = _kmeans.fit_predict(_tfidf_matrix)

    try:
        joblib.dump(
            {
                "cache_key": cache_key,
                "vectorizer": _vectorizer,
                "classifier": _classifier,
                "kmeans": _kmeans,
                "tfidf_matrix": _tfidf_matrix,
                "cluster_labels": _cluster_labels,
                "bank_df": _bank_df,
            },
            _MODEL_CACHE_PATH,
        )
    except Exception:
        pass  # 캐시 저장에 실패해도 이번 요청 결과에는 영향 없음


def _ensure_trained():
    if _classifier is None:
        _train()


def predict(text: str, amount: float | None = None) -> dict:
    _ensure_trained()
    bank_df = _bank_df

    single_df = cleaner.preprocess_single(text, amount)
    clean = single_df.loc[0, "clean_text"]

    vec = _vectorizer.transform([clean])
    numeric = cleaner.numeric_feature_matrix(single_df)
    x_input = hstack([vec, csr_matrix(numeric)])

    proba = _classifier.predict_proba(x_input)[0]
    classes = _classifier.classes_
    top_idx = int(np.argmax(proba))
    confidence = float(proba[top_idx])
    category = classes[top_idx]

    if confidence >= _LOW_CONFIDENCE_THRESHOLD:
        candidate_idx = np.where(bank_df["category"].to_numpy() == category)[0]
    else:
        input_cluster = int(_kmeans.predict(vec)[0])
        candidate_idx = np.where(_cluster_labels == input_cluster)[0]

    if len(candidate_idx) < 3:
        candidate_idx = np.arange(_tfidf_matrix.shape[0])

    sims = cosine_similarity(vec, _tfidf_matrix[candidate_idx])[0]
    order = np.argsort(sims)[::-1][:3]
    ranked = candidate_idx[order]
    ranked_sims = sims[order]
    dispute_type = bank_df.iloc[int(ranked[0])]["dispute_type"]

    similar_cases: list[SimilarCase] = []
    for idx, sim in zip(ranked, ranked_sims):
        row = bank_df.iloc[int(idx)]
        similar_cases.append(
            SimilarCase(
                category=row["category"],
                dispute_type=row["dispute_type"],
                similarity=round(float(sim) * 100, 1),
                outcome=row["outcome"],
            )
        )

    base_rates = [bank_df.iloc[int(idx)]["base_success_rate"] for idx in ranked]
    avg_base_rate = float(np.mean(base_rates)) if base_rates else 60.0

    raw_score = 0.55 * confidence * 100 + 0.45 * avg_base_rate
    success_rate = float(np.clip(raw_score, 5.0, 97.0))

    matched_keywords = [
        kw for kw in cleaner.COMPLAINT_KEYWORDS if single_df.loc[0, f"kw_{kw}"] == 1
    ]

    return {
        "category": category,
        "dispute_type": dispute_type,
        "success_rate": round(success_rate, 1),
        "confidence": round(confidence * 100, 1),
        "similar_cases": [sc.__dict__ for sc in similar_cases],
        "derived_features": {
            "text_length": int(single_df.loc[0, "text_length"]),
            "word_count": int(single_df.loc[0, "word_count"]),
            "matched_keywords": matched_keywords,
        },
    }
