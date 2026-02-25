# 401/403 Error Scenarios & UI Copy Design

This document outlines the UI scenarios and copy for the Streamlit (`app.py`) interface when handling 401/403 errors from the Ingest/Auth Gateway, adhering to the "Design North Star".

## Scenario 1: Initial Gateway Entry without Token (401 Missing Token)

- **Trigger:** User accesses `/?embed=universe_3d` or the main app without a valid session cookie or Bearer token.
- **Empty/Error State Layout:**
  - Centered, clean container (frosted glass/blur background if applicable).
  - Icon: `shield-alert` (or similar security icon).
  - No raw JSON errors visible by default.
- **Narrative Copy:**
  - **Headline:** "연결이 필요합니다." (Connection Required)
  - **Body:** "우주에 입장하기 위해서는 모바일 디바이스 또는 인증된 게이트웨이를 통한 접근이 필요합니다." (Access to the universe requires entry via a mobile device or authenticated gateway.)
  - **Action (Button/Link):** "안드로이드 앱에서 열기" (Open in Android App) - _If technically feasible to scheme-link, otherwise provide clear instruction._
- **Why (Philosophy):** Avoids developer jargon like "Token Missing". Frames the error as a necessary step for entry, maintaining the "invitation" tone.

## Scenario 2: Session Expired during Active Use (401 Expired Token)

- **Trigger:** User is using the app (e.g., in Desk or Stream mode), and the session token expires during an action (e.g., saving an essay, sending a stream log).
- **Empty/Error State Layout:**
  - Toast notification for non-blocking actions, OR a modal dialog if the action was a major save preventing data loss.
  - Icon: `clock` or `shield-alert`.
- **Narrative Copy:**
  - **Headline:** "시간이 오래 지났습니다." (Time has passed.)
  - **Body:** "안전을 위해 연결이 일시적으로 해제되었습니다. 작성 중인 기록은 로컬에 안전하게 보관됩니다." (For safety, the connection was temporarily closed. Your current draft is safely cached locally.) _Requires frontend caching logic (e.g., session_state)._
  - **Action:** "다시 연결하기" (Reconnect) -> _Triggers auth flow or instructs user to refresh via gateway._ -> "다시 로그인하여 기록 이어가기"
- **Why (Philosophy):** Reassures the user that their "decisions" (text) are not lost, reducing anxiety and focusing on the simple action of reconnecting.

## Scenario 3: Invalid Audience or Scope (403 Forbidden Audience)

- **Trigger:** A token is provided, but it was issued for a different service or lacks the necessary audience claim for Universe/Streamlit access.
- **Empty/Error State Layout:**
  - Full-page or section-specific error overlay.
  - Icon: `lock`.
- **Narrative Copy:**
  - **Headline:** "접근 권한이 없습니다." (Access Denied / No Permission)
  - **Body:** "현재 사용하신 열쇠로는 이 공간의 문을 열 수 없습니다." (The key you are currently using cannot open the door to this space.)
  - **Action:** "올바른 접근 경로 확인하기" (Check valid access route) -> _Provides a help link or instruction to use the official Android app._
- **Why (Philosophy):** Uses a metaphor ("key/door") rather than HTTP status codes to explain the permission issue.

## Scenario 4: Admin-Only Route Accessed by Standard User (403 Unauthorized Admin)

- **Trigger:** Attempting to access or perform an action reserved for the Control Tower / Admin using a standard user token.
- **Empty/Error State Layout:**
  - Subtle inline error message near the attempted action.
  - Icon: `eye-off` or `lock`.
- **Narrative Copy:**
  - **Headline:** "관측자 전용 구역입니다." (Observer Only Area)
  - **Body:** "이 영역을 조율하기 위해서는 관측자 권한이 필요합니다." (Observer privileges are required to tune this area.)
  - **Action:** _No primary action needed besides acknowledging the error. Hide the feature if possible._
- **Why (Philosophy):** Clearly marks boundaries without technical chastising. Frames admin features as "tuning" rather than system administration.

## Scenario 5: External API Rejection Context (401/403 bubbling up from Retrieval/OCR)

- **Trigger:** The frontend makes a request to a backend service (like OCR ingest or Retrieval) that rejects the request due to auth issues.
- **Empty/Error State Layout:**
  - Inline error block within the specific component (e.g., below the OCR input area or search bar).
  - Icon: `zap-off` or `alert-triangle`.
- **Narrative Copy:**
  - **Headline:** "기억의 흐름이 끊겼습니다." (The flow of memory was interrupted.)
  - **Body:** "외부 감각 기관(OCR 등)과의 연결을 인증할 수 없습니다. 시스템 상태를 확인해주세요." (Authentication with external sensory organs could not be verified. Please check system status.)
  - **Action:** "연결 상태 재확인" (Recheck Connection)
- **Why (Philosophy):** Maintains the internal narrative of the system acting as an extension of thought (memory, sensory organs), making errors feel like part of the world-building rather than raw software bugs.

---

## Technical Note for Implementation in `app.py`

When implementing these in `app.py`, we should expand the existing `_render_universe_auth_error` function (and create similar handlers for other modes if they fetch data asynchronously in the future).

The current `_render_universe_auth_error` (Lines 98-119) is a good start but uses slightly generic text:

> "Your session has expired or is invalid. Please log in again to access the Universe."

**Suggested Refactoring for `_render_universe_auth_error` in `app.py`:**

```python
def _render_universe_auth_error(auth_result: universe_auth.AuthResult) -> None:
    payload = dict(auth_result.payload or {})
    payload.setdefault("status", auth_result.status)
    payload.setdefault("route", "universe_3d_embed")

    st.markdown("### 🌌 The Universe Space")

    if auth_result.status == 401:
        st.error(f"{icons.get_icon_text('shield-alert')} **연결이 필요합니다.**")
        st.markdown("우주에 입장하기 위해서는 모바일 디바이스 또는 인증된 게이트웨이를 통한 안전한 접근이 필요합니다.")
    elif auth_result.status == 403:
        st.error(f"{icons.get_icon_text('lock')} **접근 권한이 없습니다.**")
        st.markdown("현재 경로로는 이 공간의 문을 열 수 없습니다.")
    else:
        st.error(f"{icons.get_icon_text('shield-alert')} **우주적 미아 상태입니다.**")
        st.markdown("접근 요청을 확인할 수 없습니다. 연결 상태를 점검해주세요.")

    # Technical details remain in the expander for CT/Debugging
    with st.expander("Technical Support (For Debugging)"):
        st.code(json.dumps(payload, ensure_ascii=False, indent=2), language="json")
```
