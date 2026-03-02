import httpx
import universe_auth as ua
import os
from datetime import datetime

GATEWAY_URL = "http://127.0.0.1:8790"
JWT_SECRET = "dummy-secret-for-test"

def debug_cookies():
    user_id = "debug-user"
    token = ua.issue_debug_token(user_id, JWT_SECRET, ua.DEFAULT_ISSUER, ua.DEFAULT_AUDIENCE)
    
    with httpx.Client() as client:
        print("--- Call 1 (Bearer) ---")
        r1 = client.get(f"{GATEWAY_URL}/gateway/universe_3d", headers={"Authorization": f"Bearer {token}"}, follow_redirects=False)
        print(f"Status: {r1.status_code}")
        print(f"Response Cookies: {client.cookies}")
        
        print("--- Call 2 (Cookie) ---")
        r2 = client.get(f"{GATEWAY_URL}/gateway/universe_3d", follow_redirects=False)
        # Manually check if cookie header was sent in the request object
        cookie_header = r2.request.headers.get("cookie")
        print(f"Request Cookie Header in r2: {cookie_header}")
        print(f"Status: {r2.status_code}")
        print(f"Body: {r2.text}")

if __name__ == "__main__":
    debug_cookies()
