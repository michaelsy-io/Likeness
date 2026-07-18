"""Likeness: Vercel-compatible FastAPI evidence-analysis service."""
from __future__ import annotations

import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

APP_DIR = Path(__file__).resolve().parent.parent
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_TYPES = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}

app = FastAPI(title="Likeness", version="1.0.0")


def iso_now() -> str:
    return datetime.now(timezone.utc).strftime("%b %d, %Y · %H:%M UTC")


def domain_name(url: str) -> str:
    return urlparse(url).netloc.removeprefix("www.") or "Unknown source"


def classify_platform(url: str) -> str:
    host = domain_name(url).lower()
    if any(value in host for value in ("amazon", "ebay", "etsy", "shop", "store")):
        return "Marketplace"
    if any(value in host for value in ("instagram", "facebook", "tiktok", "x.com", "twitter")):
        return "Social platform"
    return "Web publisher"


def clean_lens_matches(payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw = payload.get("visual_matches", []) + payload.get("products", [])
    seen: set[str] = set()
    matches: list[dict[str, Any]] = []
    for item in raw[:12]:
        url = item.get("link") or item.get("product_link") or item.get("source") or ""
        title = item.get("title") or item.get("product_title") or "Untitled visual match"
        if not url or url in seen:
            continue
        seen.add(url)
        matches.append({
            "title": title,
            "url": url,
            "domain": domain_name(url),
            "price": item.get("price") or item.get("extracted_price") or "Not listed",
            "thumbnail": item.get("thumbnail") or item.get("image"),
            "platform": classify_platform(url),
        })
    return matches


async def upload_public_blob(contents: bytes, filename: str, content_type: str) -> str:
    """Upload a short-lived evidence image to public Vercel Blob for Lens retrieval."""
    if not (os.getenv("BLOB_READ_WRITE_TOKEN") or os.getenv("VERCEL_OIDC_TOKEN")):
        raise RuntimeError("Vercel Blob is not connected to this project.")
    from vercel.blob import AsyncBlobClient

    extension = Path(filename).suffix.lower() or ALLOWED_TYPES[content_type]
    client = AsyncBlobClient()
    blob = await client.put(
        f"likeness-intake/{uuid.uuid4().hex}{extension}",
        contents,
        access="public",
        content_type=content_type,
        add_random_suffix=True,
        cache_control_max_age=3600,
    )
    return str(blob.url)


async def search_google_lens(image_url: str) -> list[dict[str, Any]]:
    api_key = os.getenv("SERPAPI_API_KEY")
    if not api_key:
        return []
    async with httpx.AsyncClient(timeout=25) as client:
        response = await client.get(
            "https://serpapi.com/search.json",
            params={"engine": "google_lens", "url": image_url, "api_key": api_key},
        )
        response.raise_for_status()
    return clean_lens_matches(response.json())


async def analyze_with_openai(asset_context: dict[str, str], matches: list[dict[str, Any]], route: str) -> dict[str, Any]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {}
    prompt = (
        "You are Likeness, an evidence-oriented IP and privacy risk analyst. "
        "Analyze visual-search metadata only; do not claim a legal conclusion. "
        "Return strict JSON with overall_confidence (integer 0-100), summary (one short sentence), "
        "and matches (same order, each with confidence integer 0-100, threat_level Critical/High/Moderate/Low, "
        "and rationale max 18 words). "
        f"Route: {route}. Asset: {json.dumps(asset_context)}. Matches: {json.dumps(matches)}"
    )
    body = {"model": "gpt-4o", "messages": [{"role": "user", "content": prompt}], "response_format": {"type": "json_object"}, "temperature": 0.2}
    async with httpx.AsyncClient(timeout=25) as client:
        response = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json=body,
        )
        response.raise_for_status()
    return json.loads(response.json()["choices"][0]["message"]["content"])


def demo_matches(route: str) -> list[dict[str, Any]]:
    label = "unauthorized product listing" if route == "commercial" else "identity impersonation"
    return [
        {"title": f"Potential {label} — marketplace listing", "url": "https://example-marketplace.test/listing/8432", "domain": "example-marketplace.test", "price": "$89.00", "thumbnail": None, "platform": "Marketplace", "confidence": 91, "threat_level": "High", "rationale": "Visual composition and subject placement show a strong similarity signal."},
        {"title": f"Potential {label} — social repost", "url": "https://example-social.test/p/9981", "domain": "example-social.test", "price": "Not listed", "thumbnail": None, "platform": "Social platform", "confidence": 76, "threat_level": "Moderate", "rationale": "Potential reuse detected; review the original post and account ownership."},
    ]


def document_text(route: str, asset_name: str, matches: list[dict[str, Any]]) -> dict[str, str]:
    destinations = "\n".join(f"• {match['title']} — {match['url']}" for match in matches)
    if route == "commercial":
        return {"title": "IP / COPYRIGHT CEASE & DESIST NOTICE", "body": f"Re: Unauthorized use of protected asset — {asset_name}\n\nTo whom it may concern:\n\nThis notice concerns the apparent unauthorized display, reproduction, or commercial use of the above referenced asset at the following locations:\n{destinations}\n\nYou are directed to immediately cease the disputed use, remove all copies under your control, preserve relevant records, and confirm compliance in writing. This automated notice is a draft for rights-holder review and does not constitute legal advice."}
    return {"title": "COMPUTER-RELATED IDENTITY THEFT COMPLAINT-AFFIDAVIT", "body": f"Subject asset: {asset_name}\n\nI report suspected unauthorized use of my likeness or identifying image in the following locations:\n{destinations}\n\nI request preservation of relevant account, publication, and access records pending review under applicable cybercrime and data-privacy laws. This generated draft must be reviewed, completed with jurisdictional facts, and executed before filing."}


@app.get("/")
async def api_home() -> FileResponse:
    return FileResponse(APP_DIR / "index.html", media_type="text/html")


async def create_case_impl(
    image: UploadFile = File(...),
    route: Literal["commercial", "personal"] = Form("commercial"),
) -> JSONResponse:
    if image.content_type not in ALLOWED_TYPES:
        raise HTTPException(415, "Upload a JPG, PNG, or WebP image.")
    contents = await image.read(MAX_UPLOAD_BYTES + 1)
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Images must be 10 MB or smaller.")
    filename = re.sub(r"[^A-Za-z0-9._-]", "_", image.filename or "asset")
    live_ready = bool(os.getenv("SERPAPI_API_KEY") and os.getenv("OPENAI_API_KEY"))
    image_url = ""
    try:
        if live_ready:
            image_url = await upload_public_blob(contents, filename, image.content_type)
            matches = await search_google_lens(image_url)
            if not matches:
                raise HTTPException(404, "Google Lens returned no visual matches for this image.")
            analysis, mode = await analyze_with_openai({"filename": filename, "image_url": image_url}, matches, route), "live"
        else:
            matches, analysis, mode = demo_matches(route), {}, "demo"
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(502, f"Secure analysis provider error: {str(exc)[:160]}") from exc

    insights = analysis.get("matches", [])
    for index, match in enumerate(matches):
        insight = insights[index] if index < len(insights) else {}
        match["confidence"] = int(insight.get("confidence", match.get("confidence", 68)))
        match["threat_level"] = insight.get("threat_level", match.get("threat_level", "Moderate"))
        match["rationale"] = insight.get("rationale", match.get("rationale", "Similarity requires manual evidence review."))
        match["timestamp"] = iso_now()

    overall = int(analysis.get("overall_confidence", round(sum(match["confidence"] for match in matches) / len(matches))))
    return JSONResponse({"mode": mode, "matches": matches, "overall_confidence": overall, "summary": analysis.get("summary", "Matches have been triaged and are ready for evidence review."), "document": document_text(route, filename, matches), "source_image_url": image_url or None})


# Vercel routes api/index.py beneath /api. The alias keeps local ASGI testing and
# Vercel's forwarded path behavior consistent.
app.post("/cases")(create_case_impl)
app.post("/api/cases")(create_case_impl)
