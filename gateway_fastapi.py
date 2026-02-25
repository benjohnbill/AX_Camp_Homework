import os
from typing import Any, Mapping

import uvicorn
from fastapi import FastAPI, Request, Response, Body, File, UploadFile
from fastapi.responses import JSONResponse, RedirectResponse

import universe_auth as ua
import narrative_logic as logic

DEFAULT_UPSTREAM_EMBED_URL = "https://benjohnbill-ax-camp-homework.streamlit.app/?embed=universe_3d"


DEFAULT_UPSTREAM_EMBED_URL = "https://benjohnbill-ax-camp-homework.streamlit.app/?embed=universe_3d"


def _safe_int(raw: Any, default: int) -> int:
    try:
        return int(raw)
    except Exception:
        return default


def _runtime_secret(environ: Mapping[str, Any], key: str, default: str = "") -> str:
    value = environ.get(key)
    return str(value).strip() if value else default


def _cache_headers() -> dict[str, str]:
    return {"Cache-Control": "no-store", "Pragma": "no-cache"}


def _auth_payload(status: int, code: str, message: str) -> dict[str, Any]:
    return {"status": int(status), "code": code, "message": message, "route": "gateway_universe_3d"}


def create_app(environ: Mapping[str, Any] | None = None) -> FastAPI:
    env = dict(environ or os.environ)
    app = FastAPI(title="Universe Strict Auth Gateway", version="1.0.0")

    jwt_secret = _runtime_secret(env, "UNIVERSE_JWT_SECRET", "")
    session_secret = _runtime_secret(env, "UNIVERSE_SESSION_SECRET", "") or jwt_secret
    issuer = _runtime_secret(env, "UNIVERSE_AUTH_ISSUER", ua.DEFAULT_ISSUER)
    audience = _runtime_secret(env, "UNIVERSE_AUTH_AUDIENCE", ua.DEFAULT_AUDIENCE)
    cookie_name = _runtime_secret(env, "UNIVERSE_SESSION_COOKIE", ua.DEFAULT_COOKIE_NAME) or ua.DEFAULT_COOKIE_NAME
    session_ttl = _safe_int(
        _runtime_secret(env, "UNIVERSE_SESSION_TTL_SECONDS", str(ua.DEFAULT_SESSION_TTL_SECONDS)),
        ua.DEFAULT_SESSION_TTL_SECONDS,
    )
    upstream_embed_url = _runtime_secret(env, "UNIVERSE_UPSTREAM_EMBED_URL", DEFAULT_UPSTREAM_EMBED_URL)

    def _authenticate(request: Request) -> ua.AuthResult:
        return ua.authenticate_request(
            headers=dict(request.headers),
            cookies=request.cookies,
            jwt_secret=jwt_secret,
            session_secret=session_secret,
            issuer=issuer,
            audience=audience,
            cookie_name=cookie_name,
            session_ttl_seconds=session_ttl,
        )

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        return {
            "status": "ok",
            "issuer": issuer,
            "audience": audience,
            "cookie_name": cookie_name,
            "upstream_embed_url": upstream_embed_url,
        }

    @app.get("/gateway/session")
    async def gateway_session(request: Request) -> JSONResponse:
        auth = _authenticate(request)
        if not auth.ok:
            payload = dict(auth.payload or _auth_payload(auth.status, "unauthorized", "Authentication failed."))
            payload.setdefault("route", "gateway_session")
            return JSONResponse(payload, status_code=auth.status, headers=_cache_headers())

        payload = {
            "status": 200,
            "code": "ok",
            "source": str((auth.payload or {}).get("source") or "unknown"),
            "user_id": auth.user_id or "",
            "route": "gateway_session",
        }
        response = JSONResponse(payload, status_code=200, headers=_cache_headers())
        if auth.session_cookie_name and auth.session_cookie_value:
            response.set_cookie(
                key=auth.session_cookie_name,
                value=auth.session_cookie_value,
                max_age=int(auth.session_cookie_max_age or session_ttl),
                httponly=True,
                secure=True,
                samesite="none",
                path="/",
            )
        return response

    @app.get("/gateway/universe_3d")
    async def gateway_universe_3d(request: Request) -> Response:
        auth = _authenticate(request)
        if not auth.ok:
            payload = dict(auth.payload or _auth_payload(auth.status, "unauthorized", "Authentication failed."))
            payload.setdefault("route", "gateway_universe_3d")
            return JSONResponse(payload, status_code=auth.status, headers=_cache_headers())

        response = RedirectResponse(upstream_embed_url, status_code=307, headers=_cache_headers())
        response.headers["X-Auth-Source"] = str((auth.payload or {}).get("source") or "unknown")
        response.headers["X-Auth-User"] = auth.user_id or ""
        if auth.session_cookie_name and auth.session_cookie_value:
            response.set_cookie(
                key=auth.session_cookie_name,
                value=auth.session_cookie_value,
                max_age=int(auth.session_cookie_max_age or session_ttl),
                httponly=True,
                secure=True,
                samesite="none",
                path="/",
            )
        return response

    @app.post("/v1/narrative/refine")
    async def refine_narrative(request: Request, body: dict = Body(...)) -> JSONResponse:
        # Simple open endpoint for MVP
        text = body.get("text", "")
        if not text:
            return JSONResponse({"error": "Empty text"}, status_code=400)
        
        refined = logic.refine_narrative_with_ai(text)
        return JSONResponse({"refined_text": refined})

    @app.post("/v1/narrative/vision")
    async def vision_narrative(image: UploadFile = File(...)) -> JSONResponse:
        """
        Receives an image, processes it with Vision AI, and returns the narrative.
        """
        try:
            content = await image.read()
            refined = logic.refine_image_to_narrative_with_ai(content)
            return JSONResponse({"refined_text": refined})
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    @app.post("/v1/narrative")
    async def save_narrative(request: Request, body: dict = Body(...)) -> JSONResponse:
        # Simple open save for MVP (In production, should be auth-gated)
        text = body.get("text", "")
        if not text:
            return JSONResponse({"error": "Empty text"}, status_code=400)
        
        log = logic.save_log(text)
        return JSONResponse({"status": "ok", "log_id": log.get("id")})

    return app


app = create_app()


if __name__ == "__main__":
    uvicorn.run(
        "gateway_fastapi:app",
        host=os.getenv("GATEWAY_HOST", "0.0.0.0"),
        port=_safe_int(os.getenv("GATEWAY_PORT"), 8790),
        log_level=os.getenv("GATEWAY_LOG_LEVEL", "info"),
    )
