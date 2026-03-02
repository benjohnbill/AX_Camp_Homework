import httpx
import universe_auth as ua
import os
from datetime import datetime

GATEWAY_URL = "http://127.0.0.1:8790"
JWT_SECRET = "dummy-secret-for-test"

def test_session_fallback():
    user_id = "session-test-user"
    token = ua.issue_debug_token(user_id, JWT_SECRET, ua.DEFAULT_ISSUER, ua.DEFAULT_AUDIENCE)
    
    with httpx.Client() as client:
        print(f"[{datetime.now().isoformat()}] Step 1: Initial call with Bearer")
        r1 = client.get(
            f"{GATEWAY_URL}/gateway/universe_3d", 
            headers={"Authorization": f"Bearer {token}"}, 
            follow_redirects=False
        )
        
        # Manually extract cookie from Jar
        cookie_val = client.cookies.get(ua.DEFAULT_COOKIE_NAME)
        print(f"Status: {r1.status_code}, Cookie Value found: {cookie_val is not None}")
        
        if cookie_val:
            print(f"[{datetime.now().isoformat()}] Step 2: Follow-up call with Session Cookie manually injected")
            # Create a NEW client or just use headers to be absolutely sure
            headers = {"Cookie": f"{ua.DEFAULT_COOKIE_NAME}={cookie_val}"}
            r2 = client.get(f"{GATEWAY_URL}/gateway/universe_3d", headers=headers, follow_redirects=False)
            print(f"Status: {r2.status_code}, Protocol: {r2.http_version}")
            if r2.status_code == 307:
                print("SUCCESS: Session fallback (cookie-based) verified.")
            else:
                print(f"FAIL: Session fallback failed. Body: {r2.text}")
        else:
            print("FAIL: No cookie to test.")

if __name__ == "__main__":
    test_session_fallback()
