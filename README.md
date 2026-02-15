# Antigravity

**AI-Powered Self-Reflection & Narrative Tracking System**

개인의 생각, 감정, 서사를 기록하고 AI가 과거의 기록과 연결하여 자아 성찰을 돕는 Streamlit 기반 애플리케이션입니다.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-green.svg)

---

## 주요 기능

### [Waves] Stream Mode (채팅)
- 실시간으로 생각을 기록하는 채팅 인터페이스
- AI "문학적 천문학자"가 과거 기록을 인용하여 응답
- 첫 입력은 조용히 저장 (Meteor Effect)
- 이후 입력부터 AI가 관련 기록을 찾아 연결

### [Orbit] Universe Mode (시각화)
- 모든 기록을 별자리처럼 시각화
- 3-Body Hierarchy: [Star] Constitution (헌법) → [Activity] Apology (사과) → [Sparkles] Fragment (단편)
- PyVis 기반 인터랙티브 그래프
- 연속 기록(Streak)에 따른 줌 레벨 변화

### [Book-Open] Desk Mode (에세이)
- Fragment 카드를 선택하여 에세이 작성
- 단편적 기록들을 하나의 서사로 연결

### [Droplet] Red Protocol (위반 시스템)
- 헌법(Constitution) 위반 시 "Red Mode" 활성화
- 최소 30자 해명 + 내일의 약속 필요 (권장 50자 이상)
- 해명 완료 시 "Catharsis" 효과와 함께 정상 모드 복귀

### [Moon] Dream Report (Gatekeeper)
- 앱 시작 시 과거 기록에서 모순 탐지
- 어제의 약속 이행 여부 확인
- 벡터 유사도 기반 지능형 분석

---

## 설치 방법

### 1. 저장소 클론
```bash
git clone https://github.com/yourusername/Narrative_Loop.git
cd Narrative_Loop
```

### 2. 가상환경 생성 (권장)
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. 의존성 설치
```bash
pip install -r requirements.txt
```

### 4. API 키 설정

**방법 1: Streamlit Secrets (권장)**
```toml
# .streamlit/secrets.toml
OPENAI_API_KEY = "sk-proj-..."
```

**방법 2: 환경변수**
```bash
export OPENAI_API_KEY="sk-proj-..."
```

**방법 3: 앱 내 입력**
- 사이드바에서 직접 API 키 입력 가능

### 5. 실행
```bash
streamlit run app.py
```

---

## 프로젝트 구조

```
Narrative_Loop/
├── app.py                 # 메인 Streamlit 애플리케이션
├── narrative_logic.py     # AI 로직, 임베딩, 응답 생성
├── db_manager.py          # SQLite 데이터베이스 관리
├── check_similarity.py    # 유사도 검사 유틸리티
├── requirements.txt       # Python 의존성
├── .streamlit/
│   ├── config.toml        # Streamlit 설정
│   └── secrets.toml       # API 키 (gitignore)
└── data/
    └── narrative.db       # SQLite 데이터베이스
```

---

## 기술 스택

| 분류 | 기술 |
|------|------|
| **Frontend** | Streamlit |
| **Backend** | Python 3.9+ |
| **AI/ML** | OpenAI GPT-4o-mini, text-embedding-3-small |
| **Database** | SQLite |
| **Visualization** | PyVis, NetworkX |
| **Vector Search** | scikit-learn (Cosine Similarity) |

---

## 데이터 구조

### 3-Body Hierarchy

| Type | 설명 | 시각화 |
|------|------|--------|
| **Constitution** | 핵심 가치/원칙 | [Star] 황금색 별 (고정) |
| **Apology** | 위반에 대한 해명 | [Activity] 녹색 사각형 |
| **Fragment** | 일상적 기록/생각 | [Sparkles] 흰색 점 |

---

## 환경 설정

### Streamlit 설정 (.streamlit/config.toml)
```toml
[theme]
base = "dark"
primaryColor = "#E94560"

[server]
headless = true
```

---

## 배포

### Streamlit Cloud
1. GitHub에 코드 Push
2. [Streamlit Cloud](https://share.streamlit.io) 접속
3. 저장소 연결 및 배포
4. Secrets에 `OPENAI_API_KEY` 설정

---

## 보안 주의사항

- `.streamlit/secrets.toml`은 절대 Git에 커밋하지 마세요
- `.gitignore`에 민감한 파일들이 포함되어 있는지 확인하세요
- 배포 시 환경변수 또는 Streamlit Secrets 사용

---

## 라이선스

이 프로젝트는 개인 학습 및 자기 성찰 목적으로 제작되었습니다.

---

## 기여

버그 리포트, 기능 제안, PR 모두 환영합니다!

---

<div align="center">
  <sub>Built with Love and Gravity for self-reflection</sub>
</div>

---

## Postgres/Supabase 전환 (DATASTORE 라우터)

### 1) 환경변수
```bash
export DATASTORE=postgres          # 또는 sqlite
export DATABASE_URL='postgresql://USER:PASSWORD@HOST:5432/postgres'
```

### 2) Supabase 스키마 생성
```bash
psql "$DATABASE_URL" -f sql/1_create_logs_table.sql
```

### 3) SQLite → Supabase 마이그레이션
```bash
python tools/migrate_sqlite_to_supabase.py
```

### 4) 테스트
```bash
pytest -q
```

### 한국어 FTS 참고
Postgres 기본 `to_tsvector('english', ...)`는 영어 중심입니다. 한국어 품질이 필요하면 Mecab 등 형태소 분석으로 사전 토큰화한 텍스트를 별도 컬럼(tsvector)에 저장하는 방식을 권장합니다.
