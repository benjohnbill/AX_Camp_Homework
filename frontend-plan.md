# Frontend Implementation Plan (Cycle03 Minimal UI Rescue)

**Role:** Narrative_Loop Frontend Stabilization Lead
**Goal:** Achieve "Working MVP + Minimal UI Rescue" in cycle03 without migrating frameworks.
**Deadline:** 2026-02-25

## 1. Scope Lock

**What we WILL do:**

- Retain Streamlit (`app.py`) as the frontend framework.
- Remove the aggressive custom CSS (heavy rounded corners, Apple/Toss mimicry) that clashes with Streamlit's internal rendering.
- Apply a minimal, native-friendly Streamlit theme via `.streamlit/config.toml` (or minimal CSS if config.toml is insufficient) to achieve a calm, trustworthy UI tone.
- Ensure the primary user journey (write -> save -> re-open -> re-query -> universe) is functionally robust and visually prioritized.
- Emphasize "Story-first, Universe-second" (Universe remains a supplementary expanded experience).

**What we will NOT do:**

- **NO FRAMEWORK MIGRATION** (No Next.js or React in this cycle).
- No complex custom CSS injections that attempt to override Streamlit's native component structure (e.g., trying to force shadcn/ui exact pixel matches).
- No changes to backend API contracts or security logic.
- No alteration to the existing 401/403 friendly UX narrative copy.

## 2. Minimal UI Rescue List (Max 2 Hours)

| Item                           | Description                                                                                                                                         | Est. Time | Priority                     |
| :----------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------- | :-------- | :--------------------------- |
| **1. Strip Custom CSS**        | Remove the heavy `apply_atmosphere()` CSS overload from `app.py`.                                                                                   | 10m       | P1 (Must Do)                 |
| **2. Apply Native Theme**      | Create/Update `.streamlit/config.toml` for a clean, light/minimalist color palette (Primary Color: Calm Blue/Slate, Background: White/Off-white).   | 15m       | P1 (Must Do)                 |
| **3. Layout Cleanup**          | Simplify Streamlit columns and expanders in `app.py` to ensure one clear Primary CTA per screen (e.g., prominently displaying the input chat area). | 25m       | P1 (Must Do)                 |
| **4. UX Flow Verification**    | Manually click through the write -> save -> re-open -> universe flow locally to confirm visual hierarchy makes sense natively.                      | 30m       | P2 (Should Do)               |
| **5. (Overflow) Custom Fonts** | If time permits, inject _only_ a font-family override (e.g., Inter, Pretendard) via minimal CSS without touching layout properties.                 | 15m       | P3 (Cycle4 Deferred if > 2h) |

## 3. File-level Change Plan

- **`app.py`**:
  - Delete or severely strip down the `apply_atmosphere()` function.
  - Review `st.columns`, `st.button`, and layout containers to ensure the "Story-first" narrative is visually dominant.
- **`.streamlit/config.toml` (NEW/MODIFY)**:
  - Add native Streamlit theming variables:
    ```toml
    [theme]
    primaryColor = "#3b82f6"  # Calm Blue
    backgroundColor = "#ffffff"
    secondaryBackgroundColor = "#f9fafb"
    textColor = "#111827"
    font = "sans serif"
    ```

## 4. Step-by-step Execution Order

1. **(0m - 10m)** Erase aggressive CSS from `app.py`. Ensure app loads without errors in default Streamlit styling.
2. **(10m - 25m)** Implement `.streamlit/config.toml` to set the baseline minimal theme. Restart Streamlit to verify theme application.
3. **(25m - 50m)** Refactor `app.py` layout code to emphasize the Primary CTA (e.g., the chat input for narrative writing) and demote the Universe link to a secondary action.
4. **(50m - 80m)** Run end-to-end local validation of the primary user journey.
5. **(80m - 100m)** Final code review and evidence generation.

## 5. Validation Checklist

- [ ] Narrative Write/Save flow is visually clear and functionally intact.
- [ ] Re-open/Re-query successfully retrieves data without UI distortion.
- [ ] Universe entry point exists but is clearly secondary to the narrative core.
- [ ] 401/403 UX scenarios ("연결이 필요합니다." etc.) render cleanly within the new minimal theme.
- [ ] Total UI intervention feel is "calm and trustworthy", not "broken custom CSS".

## 6. Risks and Fallback

- **Risk:** `.streamlit/config.toml` changes might not apply instantly or cache aggressively in the local browser.
- **Fallback:** If native theming acts unpredictably and takes more than 30 minutes to debug, immediately abort the `.toml` approach. Rely solely on default Streamlit Light Mode and focus remaining time exclusively on layout restructuring (Item 3) to achieve clarity. All Overflow items are instantly deferred to Cycle4.
