# `app.py` Rerun Loop Review & Optimization

This review focuses on identifying potential infinite or unnecessary rerun loops in `app.py` prior to integrating heavy retrieval logic. Streamlit's architecture makes it susceptible to these loops if `st.rerun()` is used carelessly alongside state mutations.

## 1. Known State Initialization (Lines 184-235)

- `init_session_state()` relies on `if 'key' not in st.session_state` checks.
- The `st.session_state['db_bootstrap_done']`, `diagnostics_run`, and `streak_updated` are correctly guarded.
- **Risk Level: Low.** Startup logic correctly performs the DB/Logic calls only once per session.

## 2. Rerun inside the Entropy Gap Workflow (Lines 688-712)

- **Snippet:**
  ```python
  if st.form_submit_button(f"{icons.get_icon_text('zap')} 엔트로피 해소 (Repay Debt)"):
      # ... logic ...
      logic.process_gap(reason, core_id, plan, tags=[f_tag] if f_tag else [])
      st.success("엔트로피 감소 확인. 시스템 정상화."); time.sleep(1.5); st.rerun()
  ```
- **Analysis:** This is an explicit, user-triggered action via a form submission. The `st.rerun()` happens only after the submit button is clicked and the logical processing finishes.
- **Risk Level: Low.** It correctly triggers a single rerun to clear the entropy warning state by resetting the conditions that caused `is_entropy_mode()` to be true.

## 3. The `violation_pending` State in Stream Mode (Lines 405-426)

- **Snippet:**
  ```python
  if st.session_state.get('violation_pending'):
      # ... UI elements ...
      if c1.button("Stop & Analyze Gap"):
          db.increment_debt(1)
          del st.session_state['violation_pending']
          st.rerun()
      if c2.button("Force Merge"):
          db.increment_debt(1)
          logic.save_log(v['text'])
          del st.session_state['violation_pending']
          st.rerun()
      if st.button("Discard"):
          del st.session_state['violation_pending']
          st.rerun()
      return  # EARLY EXIT
  ```
- **Analysis:** Similar to the entropy form, these button clicks correctly clear the `violation_pending` blocking state and execute a single `st.rerun()`. The early `return` correctly blocks the mode rendering from overlapping.
- **Risk Level: Low.**

## 4. `process_stream_input` Chat Flow (Lines 441-462)

- **Snippet:**
  ```python
  def process_stream_input(user_input: str):
      # ... violation check ...
      related = logic.find_related_logs(user_input)
      resp = logic.generate_response(user_input, related)
      st.session_state.messages.append({"role": "assistant", "content": resp})
      # ...
      st.session_state['current_echo'] = logic.get_current_echo(reference_text=user_input)
      st.rerun()
  ```
- **Analysis:** `st.chat_input` triggering this function acts functionally similar to a button press. The code executes a long-running retrieval/generation, appends to the session state, and cleanly forces a rerun to render the new state.
- **Risk Level: Medium (Future Proofing).** When "heavy retrieval logic" or async contexts are added (as noted in `integration_status.md`), holding the UI thread here via synchronous `find_related_logs` and `generate_response` before the `st.rerun()` will cause the UI to freeze without a loading indicator.
- **Recommendation:** When upgrading retrieval, we must replace this synchronous block with an optimistic UI update (e.g., adding a temporary "Thinking..." bubble to `messages`, running `st.rerun()` immediately, and processing the heavy logic in a subsequent phase or thread). However, for purely _runaway loops_, this is safe.

## 5. Potential Hidden Render Loops: "Universe" Universe Render (Lines 163-181, 567-583)

- **Snippet (`_run_universe_embed_route`):**
  ```python
  try:
      logs = logic.load_logs()
      cores = db.get_cores()
      render_3d_universe(logs, cores)
  except Exception as exc: # ...
  ```
  And in `render_universe_mode` (Lines 567-583):
  ```python
   with t4:
      st.markdown(f"### {icons.get_icon_text('orbit')} 1st Person Explorer")
      try:
          logs = logic.load_logs()
          cores = db.get_cores()
          render_3d_universe(logs, cores)
  ```
- **Analysis:** `render_3d_universe` embeds an iframe pointing to an external HTML file or inline components. If that embedded component performs JavaScript `window.parent.postMessage` that Streamlit reacts to by altering session state, it could loop.
- **Risk Level: Unknown/Watch.** Ensure `universe_3d.py` does not aggressively ping the Streamlit backend causing reactivity cycles. Pure rendering is safe.

## 6. API Key Setup (Lines 339-372)

- **Snippet:**
  ```python
  if c1.button("Apply Key", use_container_width=True):
      if entered and entered.strip():
          logic.set_api_key(entered.strip())
          st.rerun()
  ```
- **Analysis:** Explicit button trigger. Safe.

## Conclusion and Action Items

The current `app.py` is structurally sound regarding infinite `st.rerun()` loops. It strictly relies on user-initiated button/input actions to trigger state mutations followed by an explicit rerun.

**Key Optimization for Heavy Retrieval:**
The `process_stream_input` function is fully synchronous. To ensure Streamlit UX stability when the new "Async Worker/Heavy Retrieval" logic arrives, the frontend must move towards an optimistic rendering model.

1.  **Acknowledge Input:** Append `{role: user, content: input}` to session state.
2.  **Display Loading:** Append `{role: assistant, content: "Retrieving context...", is_loading: True}` to session state.
3.  **Rerun (`st.rerun()`):** To show the loading state instantly.
4.  **Process (in a Fragment or next pass):** The backend processes the heavy RRF/Context logic.
5.  **Final Update:** Replace the loading message with the actual logic response and rerun again.
