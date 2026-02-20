# DEBUG_TOKEN_GOVERNANCE.md

## Scope
- Environment: staging/development only
- Issuer owner: backend (Antigravity)
- Consumer: Android DEBUG build only
- Production issuance: forbidden

## Rotation Policy
- Rotate `UNIVERSE_JWT_SECRET` immediately when token leakage is suspected.
- Existing bearer debug tokens are invalidated after rotation.
- Rotation command:
  - `python tools/rotate_universe_jwt_secret.py`
- Secret material is never logged in full; only masked value + fingerprint.

## Issuance Endpoint
- Route: `POST /debug/token`
- Server: `debug_token_server.py`
- Protection: `X-Debug-Admin-Key` header (required)

### Request JSON
- `user_id` (required)
- `aud` (optional, default `android-universe`)
- `ttl_minutes` (optional, default 60)

### Response JSON
- Success: `200`
- Failure: stable JSON with `status`, `code`, `message` (`401`/`403`/`400`)

## Claim Policy
- Required claims: `iss`, `aud`, `exp`, `user_id`
- Additional claim: `jti`
- TTL: 1~120 minutes (default 60)

## Logging Policy
- Do not log full JWT
- Log only:
  - `user_id`
  - `aud`
  - `exp`
  - `jti`
  - masked token (`aaaa....zzzz`)
  - token fingerprint (sha256 prefix)

## cURL Example
```bash
curl -sS -X POST "http://127.0.0.1:8787/debug/token" \
  -H "Content-Type: application/json" \
  -H "X-Debug-Admin-Key: ${DEBUG_TOKEN_ADMIN_KEY}" \
  -d '{
    "user_id": "debug_user_001",
    "aud": "android-universe",
    "ttl_minutes": 60
  }'
```

## Expected Response Headers
- `Content-Type: application/json; charset=utf-8`
- `Cache-Control: no-store`
- `Pragma: no-cache`
