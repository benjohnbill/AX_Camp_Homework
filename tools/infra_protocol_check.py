import httpx
import universe_auth as ua
import os
import json
from datetime import datetime

GATEWAY_URL = "http://127.0.0.1:8790"
STREAMLIT_URL = "http://127.0.0.1:8501"
JWT_SECRET = "dummy-secret-for-test"

def log_event(msg):
    ts = datetime.now().isoformat()
    print(f"[{ts}] {msg}")

def run_checks():
    user_id = "infra-check-user"
    issuer = ua.DEFAULT_ISSUER
    audience = ua.DEFAULT_AUDIENCE
    token = ua.issue_debug_token(user_id, JWT_SECRET, issuer, audience)
    
    with httpx.Client() as client:
        # 1. Re-verify /gateway/universe_3d with valid bearer
        log_event("Checking /gateway/universe_3d with valid Bearer...")
        r1 = client.get(f"{GATEWAY_URL}/gateway/universe_3d", headers={"Authorization": f"Bearer {token}"}, follow_redirects=False)
        log_event(f"Status: {r1.status_code}, Protocol: {r1.http_version}, Redirect to: {r1.headers.get('location')}")
        
        # 2. Check session fallback (no auth)
        log_event("Checking /gateway/universe_3d with NO auth...")
        r2 = client.get(f"{GATEWAY_URL}/gateway/universe_3d", follow_redirects=False)
        log_event(f"Status: {r2.status_code}, Protocol: {r2.http_version}")

        # 3. Check embed route stability
        log_event("Checking Streamlit embed route...")
        r3 = client.get(f"{STREAMLIT_URL}/?embed=universe_3d")
        log_event(f"Status: {r3.status_code}, Protocol: {r3.http_version}, Body Length: {len(r3.text)}")

if __name__ == "__main__":
    run_checks()
