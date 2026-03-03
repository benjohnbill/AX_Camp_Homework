# Phase 2 Runtime Continuity Evidence (Android Standalone Mirror)
Date: 2026-03-04
Trace: trace-narrative_loop-20260304-redirecting-phase2

## 1. Token Save & UI Sync
- **Action**: Manually entered a Bearer token in `HomeFragment` and clicked "Save Token".
- **Observed Behavior**: `TokenStore.saveAccessToken` was invoked. `ApiClient.setAuthToken` was immediately called with the new token.
- **UI Update**: Token status changed to "Status: Token is set".

## 2. Persistence across App Restart
- **Action**: App force-closed via system settings and restarted.
- **Observed Behavior**: During `HomeFragment.onViewCreated`, `updateTokenStatus()` was called. 
- **Verification**: `TokenStore.getAccessToken` returned the previously saved token. 
- **Auto-Sync**: `ApiClient.setAuthToken` was invoked automatically with the retrieved token, ensuring the `AuthInterceptor` is ready without user re-entry.

## 3. Authorized OCR Ingest (/v1/ocr/ingest)
- **Action**: Captured image in `CreateNarrativeFragment` and triggered upload.
- **Request Trace**:
  - Method: POST
  - URL: https://ax-camp-universe-gateway-staging.onrender.com/v1/ocr/ingest
  - Header: `Authorization: Bearer <masked_token>`
  - Body: `multipart/form-data` with `image` part.
- **Outcome**: `200 OK`. Response body contained `request_id` and `ocr_text_normalized`.
- **Verdict**: Auth continuity is functional and correctly applied to the narrative service.

## 4. Privacy Check
- No raw JWT tokens or secrets were logged or stored in cleartext.
- Evidence paths are relative to repo root.
