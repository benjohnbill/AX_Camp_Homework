"""
narrative_logic.py
앱의 핵심 로직: 데이터 저장, 검색, AI 생성
업그레이드: Hybrid Search + Temporal Awareness
"""

import os
import json
from datetime import datetime
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from openai import OpenAI

# ============================================================
# API Key 로드 (하이브리드 방식: Streamlit Cloud → 로컬 .env)
# ============================================================
try:
    api_key = st.secrets["OPENAI_API_KEY"]
except Exception:
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)

# ============================================================
# 상수 정의
# ============================================================
DATA_FILE = "data/user_logs.json"
SIMILARITY_THRESHOLD = 0.6
SELF_SIMILARITY_THRESHOLD = 0.99
TAG_MATCH_BONUS = 0.15  # 태그 일치 시 가산점


def load_logs() -> list:
    """저장된 로그를 불러온다. 파일이 없으면 빈 리스트 반환."""
    if not os.path.exists(DATA_FILE):
        return []
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_logs(logs: list) -> None:
    """로그 리스트를 JSON 파일에 저장한다."""
    os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)


def get_embedding(text: str) -> list:
    """OpenAI 임베딩 API를 사용하여 텍스트의 벡터를 반환한다."""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding


# ============================================================
# 핵심 함수: save_log
# ============================================================
def save_log(text: str, tags: list) -> dict:
    """입력된 텍스트를 임베딩하고 JSON에 저장한다."""
    embedding = get_embedding(text)
    
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "text": text,
        "tags": tags,
        "embedding": embedding
    }
    
    logs = load_logs()
    logs.append(log_entry)
    save_logs(logs)
    
    return log_entry


# ============================================================
# Hybrid Search: 태그 가중치 계산
# ============================================================
def calculate_tag_bonus(current_tags: list, past_tags: list) -> float:
    """
    현재 입력 태그와 과거 로그 태그 비교 후 가산점 반환.
    하나라도 일치하면 TAG_MATCH_BONUS 반환.
    """
    if not current_tags or not past_tags:
        return 0.0
    
    current_set = set(tag.lower() for tag in current_tags)
    past_set = set(tag.lower() for tag in past_tags)
    
    if current_set & past_set:  # 교집합이 있으면
        return TAG_MATCH_BONUS
    return 0.0


# ============================================================
# Temporal Awareness: 시간 경과 계산
# ============================================================
def calculate_days_diff(past_timestamp: str) -> int:
    """과거 로그의 timestamp와 현재 시간 차이를 일(days) 단위로 반환."""
    past_dt = datetime.fromisoformat(past_timestamp)
    now_dt = datetime.now()
    diff = now_dt - past_dt
    return diff.days


def get_temporal_context(days_diff: int) -> str:
    """시간 경과에 따른 AI 지침 반환."""
    if days_diff <= 7:
        return "이 고민은 최근(7일 이내)의 것이다. 그 감정이 여전히 지속되고 있는지, 아니면 이미 변화가 시작되었는지 물어라."
    elif days_diff <= 30:
        return "이 고민은 약 한 달 이내의 것이다. 그때의 결심이 지금까지 유지되고 있는지 확인하라."
    else:
        return "이것은 오래된 기록이다. 이 고민이 당신의 삶에서 반복되는 '영원회귀'의 패턴인지, 아니면 그때와 지금은 어떤 점에서 근본적으로 달라졌는지 질문하라."


# ============================================================
# 페르소나 함수 (Temporal Awareness 적용)
# ============================================================
def run_mirroring_mode(current: str, past_log: dict, current_tags: list) -> str:
    """
    서사적 거울 모드: 과거 기록을 인용하여 현재와의 모순/변화를 질문한다.
    시간 경과를 인지하여 맥락에 맞는 질문을 던진다.
    """
    days_diff = calculate_days_diff(past_log['timestamp'])
    temporal_context = get_temporal_context(days_diff)
    
    system_prompt = f"""너는 서사적 거울이다. 과거의 기록을 인용하여 현재와의 모순이나 변화를 질문하라. 위로하지 마라.

[시간 경과: {days_diff}일 전]
[시간 맥락 지침]: {temporal_context}

[강력한 제약조건]
1. 답변은 총 3문장 이내로 제한한다.
2. 질문은 단 하나(One single question)만 던진다. 여러 질문 나열 절대 금지.
3. 짧고, 묵직하고, 날카롭게 찌르는 톤을 유지한다.
4. 시간의 흐름을 반드시 언급하라."""
    
    past_date = past_log['timestamp'][:10]
    user_prompt = f"""과거 기록 ({past_date}, {days_diff}일 전):
"{past_log['text']}"

현재 고민:
"{current}"

시간의 흐름을 고려하여 단 하나의 날카로운 질문을 던져라."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.7,
        max_tokens=200
    )
    
    return response.choices[0].message.content


def run_nietzsche_fallback(current: str) -> str:
    """
    니체 모드: 관련 데이터가 없을 때 본질을 찌르는 철학적 질문을 던진다.
    """
    system_prompt = """너는 니체다. 관련된 과거 데이터가 없다. 현재 고민의 본질을 찌르는 철학적 질문을 던져라.

[강력한 제약조건]
1. 답변은 총 3문장 이내로 제한한다.
2. 질문은 단 하나(One single question)만 던진다. 여러 질문 나열 절대 금지.
3. 짧고, 묵직하고, 날카롭게 찌르는 톤을 유지한다."""
    
    user_prompt = f"""현재 고민:
"{current}"

이 고민의 본질을 파고드는 단 하나의 철학적 질문을 던져라."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.8,
        max_tokens=150
    )
    
    return response.choices[0].message.content


# ============================================================
# 핵심 함수: generate_echo (Hybrid Search + 자기 복제 방지)
# ============================================================
def generate_echo(current_input: str, current_tags: list = None) -> tuple[str, str, dict | None]:
    """
    현재 입력과 과거 로그를 비교하여 적절한 응답을 생성한다.
    Hybrid Search: 코사인 유사도 + 태그 가중치
    
    Returns:
        (응답 텍스트, 모드 이름, 참조된 과거 로그 또는 None)
    """
    if current_tags is None:
        current_tags = []
    
    logs = load_logs()
    
    # 저장된 로그가 없거나 1개만 있는 경우 → 니체 모드
    if len(logs) <= 1:
        response = run_nietzsche_fallback(current_input)
        return response, "nietzsche", None
    
    # 현재 입력의 임베딩 계산
    current_embedding = np.array(get_embedding(current_input)).reshape(1, -1)
    
    # Hybrid Search: 코사인 유사도 + 태그 가산점
    similarity_scores = []
    for log in logs:
        log_embedding = np.array(log["embedding"]).reshape(1, -1)
        cosine_score = cosine_similarity(current_embedding, log_embedding)[0][0]
        
        # 태그 가산점 계산
        tag_bonus = calculate_tag_bonus(current_tags, log.get("tags", []))
        
        # 최종 점수 (최대 1.0)
        final_score = min(cosine_score + tag_bonus, 1.0)
        
        similarity_scores.append((final_score, cosine_score, log))
    
    # 최종 점수 내림차순 정렬
    similarity_scores.sort(key=lambda x: x[0], reverse=True)
    
    # 자기 자신(코사인 유사도 0.99 이상) 제외하고 가장 유사한 항목 찾기
    best_log = None
    best_score = 0.0
    
    for final_score, cosine_score, log in similarity_scores:
        if cosine_score < SELF_SIMILARITY_THRESHOLD:  # 자기 자신 제외 (원본 코사인 점수 기준)
            best_score = final_score
            best_log = log
            break
    
    # 차선책이 없거나 임계값 미만이면 니체 모드
    if best_log is None or best_score < SIMILARITY_THRESHOLD:
        response = run_nietzsche_fallback(current_input)
        return response, "nietzsche", None
    
    # 유사한 과거 로그 발견 → 거울 모드
    response = run_mirroring_mode(current_input, best_log, current_tags)
    return response, "mirroring", best_log
