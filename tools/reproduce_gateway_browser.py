import requests
import json
import os

GATEWAY_URL = "http://127.0.0.1:8790"
STREAMLIT_URL = "http://127.0.0.1:8501"

def test_gateway_universe_3d_no_auth():
    print("Testing /gateway/universe_3d without auth")
    r = requests.get(f"{GATEWAY_URL}/gateway/universe_3d", allow_redirects=False)
    print(f"Status: {r.status_code}")
    return r.status_code == 401

def test_gateway_universe_3d_with_fake_bearer():
    print("Testing /gateway/universe_3d with invalid bearer")
    headers = {"Authorization": "Bearer invalid-token"}
    r = requests.get(f"{GATEWAY_URL}/gateway/universe_3d", headers=headers, allow_redirects=False)
    print(f"Status: {r.status_code}")
    return r.status_code == 401

def test_embed_route_no_auth():
    print("Testing Streamlit embed route without auth")
    r = requests.get(f"{STREAMLIT_URL}/?embed=universe_3d")
    print(f"Status: {r.status_code}")
    contains_fallback = "연결이 필요합니다." in r.text
    print(f"Contains fallback text: {contains_fallback}")
    return contains_fallback

if __name__ == "__main__":
    s1 = test_gateway_universe_3d_no_auth()
    s2 = test_gateway_universe_3d_with_fake_bearer()
    s3 = test_embed_route_no_auth()
    
    results = {"gateway_no_auth": s1, "gateway_invalid_bearer": s2, "embed_fallback": s3}
    with open("data/evidence/20260302_gateway_repro_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("Results saved.")
