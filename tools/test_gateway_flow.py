import universe_auth as ua
import requests
import os
import json

GATEWAY_URL = "http://127.0.0.1:8790"
JWT_SECRET = "dummy-secret-for-test"

def test_full_flow():
    user_id = "test-user-123"
    issuer = ua.DEFAULT_ISSUER
    audience = ua.DEFAULT_AUDIENCE
    
    # We need to set the environment variable so the gateway uses the same secret
    os.environ["UNIVERSE_JWT_SECRET"] = JWT_SECRET
    
    token = ua.issue_debug_token(user_id, JWT_SECRET, issuer, audience)
    print(f"Issued debug token: {token[:20]}...")
    
    headers = {"Authorization": f"Bearer {token}"}
    
    print("Testing /gateway/universe_3d with valid bearer")
    r = requests.get(f"{GATEWAY_URL}/gateway/universe_3d", headers=headers, allow_redirects=False)
    print(f"Status: {r.status_code}")
    
    # Note: If the gateway was already started without the secret, this might still fail 401.
    # But since gateway_fastapi.py defaults to empty secret, and we are using dummy-secret,
    # it's better to restart gateway with the secret or keep it consistent.
    
    return r.status_code == 307

if __name__ == "__main__":
    ok = test_full_flow()
    print(f"Overall result: {'PASS' if ok else 'FAIL'}")
