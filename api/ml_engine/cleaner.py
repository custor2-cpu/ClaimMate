"""
Pandas / NumPy 기반 비정형 소비자 상담 텍스트 전처리 파이프라인.

한국소비자원 상담 데이터는 개인정보(전화번호, 카드번호 등)와 특수문자 노이즈가
섞인 자유 서술형 텍스트이므로, ML 추론 전에 마스킹 -> 정제 -> 파생변수 생성
순으로 표준화한다.
"""

from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import pandas as pd

# 소비자 분쟁 상담에서 반복적으로 등장하는 핵심 불만 키워드.
# 정규표현식 노이즈 제거 이후 파생변수(키워드 플래그)로 변환되어 ML 피처에 사용된다.
COMPLAINT_KEYWORDS: list[str] = [
    "환불",
    "위약금",
    "하자",
    "불량",
    "지연",
    "해지",
    "취소",
    "연체",
    "배송",
    "미배송",
    "파손",
    "거부",
    "계약",
    "청약철회",
    "환급",
]

_PHONE_RE = re.compile(r"(01[016789]|0[2-6]\d)-?\d{3,4}-?\d{4}")
_RRN_RE = re.compile(r"\d{6}-?[1-4]\d{6}")
_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
_CARD_RE = re.compile(r"(?:\d[ -]?){13,16}")
_NOISE_RE = re.compile(r"[^가-힣a-zA-Z0-9\s.,!?%~()]")
_MULTISPACE_RE = re.compile(r"\s+")


def _mask_pii(text: str) -> str:
    """전화번호/주민등록번호/이메일/카드번호를 마스킹 처리한다."""
    text = _PHONE_RE.sub("[전화번호]", text)
    text = _RRN_RE.sub("[주민등록번호]", text)
    text = _EMAIL_RE.sub("[이메일]", text)
    text = _CARD_RE.sub("[카드번호]", text)
    return text


def _strip_noise(text: str) -> str:
    """한글/영문/숫자/기본 문장부호 외 특수문자 노이즈를 제거하고 공백을 정규화한다."""
    text = _NOISE_RE.sub(" ", text)
    text = _MULTISPACE_RE.sub(" ", text).strip()
    return text


def clean_text(raw_text: str) -> str:
    if not isinstance(raw_text, str):
        return ""
    masked = _mask_pii(raw_text)
    return _strip_noise(masked)


def _keyword_flags(clean: str) -> dict[str, int]:
    return {f"kw_{kw}": int(kw in clean) for kw in COMPLAINT_KEYWORDS}


def build_dataframe(records: Iterable[dict]) -> pd.DataFrame:
    """
    원본 상담 레코드({"text": ..., "amount": ..., "date": ..., "category": ...})의
    리스트를 받아 결측치/공백 제거 및 정제된 DataFrame을 반환한다.
    """
    df = pd.DataFrame(list(records))
    if "text" not in df.columns:
        df["text"] = ""

    # 결측치 및 빈 문자열/공백만 있는 상담 내역 제거
    df["text"] = df["text"].astype(str)
    df = df.dropna(subset=["text"])
    df = df[df["text"].str.strip().str.len() > 0].copy()

    if df.empty:
        raise ValueError("유효한 상담 텍스트가 없습니다.")

    df["clean_text"] = df["text"].map(clean_text)
    df = df[df["clean_text"].str.len() > 0].copy()

    df["text_length"] = df["clean_text"].str.len()
    df["word_count"] = df["clean_text"].str.split().map(len)

    keyword_df = pd.DataFrame(df["clean_text"].map(_keyword_flags).tolist(), index=df.index)
    df = pd.concat([df, keyword_df], axis=1)
    df["keyword_score"] = keyword_df.sum(axis=1)

    if "amount" in df.columns:
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0).astype(np.int64)
    else:
        df["amount"] = 0

    return df.reset_index(drop=True)


def preprocess_single(text: str, amount: float | None = None, category: str | None = None) -> pd.DataFrame:
    """단건 사용자 입력을 동일한 파이프라인으로 정제하여 1행 DataFrame으로 반환한다."""
    record = {"text": text, "amount": amount or 0, "category": category or ""}
    return build_dataframe([record])


def numeric_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """분류기에 투입할 정규화된 파생 수치 피처 행렬을 생성한다."""
    keyword_cols = [f"kw_{kw}" for kw in COMPLAINT_KEYWORDS]
    features = np.column_stack(
        [
            (df["text_length"].to_numpy() / 200.0).clip(0, 3),
            (df["word_count"].to_numpy() / 40.0).clip(0, 3),
            (df["keyword_score"].to_numpy() / len(COMPLAINT_KEYWORDS)),
            df[keyword_cols].to_numpy(dtype=float),
        ]
    )
    return features
