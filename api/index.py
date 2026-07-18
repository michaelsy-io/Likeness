"""Likeness: Vercel-compatible evidence and enforcement drafting service."""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from typing import Any, Literal
from urllib.parse import urlparse, urlunparse

import httpx
from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

MAX_CANDIDATES = 10

app = FastAPI(title="Likeness", version="2.0.0")


def iso_now() -> str:
    return datetime.now(timezone.utc).strftime("%b %d, %Y · %H:%M UTC")


def bounded_text(value: str | None, limit: int = 280) -> str:
    return re.sub(r"\s+", " ", (value or "").strip())[:limit]


def domain_name(url: str) -> str:
    return urlparse(url).netloc.lower().removeprefix("www.") or "Unknown source"


def canonical_url(url: str) -> str:
    parsed = urlparse(url)
    return urlunparse((parsed.scheme.lower(), parsed.netloc.lower(), parsed.path.rstrip("/"), "", "", ""))


def parse_domains(raw_domains: str) -> set[str]:
    domains: set[str] = set()
    for value in re.split(r"[,\n\s]+", raw_domains or ""):
        if not value:
            continue
        candidate = domain_name(value if "://" in value else f"https://{value}")
        if candidate and candidate != "unknown source":
            domains.add(candidate)
    return domains


def is_authorized_domain(domain: str, official_domains: set[str]) -> bool:
    return any(domain == official or domain.endswith(f".{official}") for official in official_domains)


def classify_platform(url: str) -> str:
    host = domain_name(url)
    if any(value in host for value in ("amazon", "ebay", "etsy", "shop", "store", "shopee", "lazada", "alibaba")):
        return "Marketplace"
    if any(value in host for value in ("instagram", "facebook", "tiktok", "x.com", "twitter", "youtube")):
        return "Social platform"
    return "Web publisher"


def clean_lens_matches(payload: dict[str, Any], official_domains: set[str]) -> list[dict[str, Any]]:
    """Normalize provider records and remove known first-party/duplicate results."""
    source_groups = (("visual_matches", "Visual match"), ("products", "Product result"))
    seen: set[str] = set()
    matches: list[dict[str, Any]] = []
    for field, source_type in source_groups:
        for position, item in enumerate(payload.get(field, []), start=1):
            url = item.get("link") or item.get("product_link") or item.get("source") or ""
            title = item.get("title") or item.get("product_title") or "Untitled visual match"
            if not isinstance(url, str) or not url.startswith(("http://", "https://")):
                continue
            canonical = canonical_url(url)
            domain = domain_name(url)
            if canonical in seen or is_authorized_domain(domain, official_domains):
                continue
            seen.add(canonical)
            matches.append(
                {
                    "title": bounded_text(str(title), 220) or "Untitled visual match",
                    "url": url,
                    "domain": domain,
                    "price": item.get("price") or item.get("extracted_price") or "Not listed",
                    "thumbnail": item.get("thumbnail") or item.get("image"),
                    "platform": classify_platform(url),
                    "source_type": source_type,
                    "provider_rank": position,
                }
            )
    return matches[:30]


async def search_google_lens(image_url: str, official_domains: set[str]) -> list[dict[str, Any]]:
    api_key = os.getenv("SERPAPI_API_KEY")
    async with httpx.AsyncClient(timeout=18) as client:
        response = await client.get(
            "https://serpapi.com/search.json",
            params={"engine": "google_lens", "url": image_url, "api_key": api_key},
        )
        response.raise_for_status()
    return clean_lens_matches(response.json(), official_domains)


def openai_model() -> str:
    return os.getenv("OPENAI_MODEL", "gpt-4o")


async def openai_json(prompt: str) -> dict[str, Any]:
    api_key = os.getenv("OPENAI_API_KEY")
    body = {
        "model": openai_model(),
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
        "temperature": 0.2,
    }
    async with httpx.AsyncClient(timeout=24) as client:
        response = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json=body,
        )
        response.raise_for_status()
    return json.loads(response.json()["choices"][0]["message"]["content"])


async def analyze_with_openai(asset_context: dict[str, str], matches: list[dict[str, Any]], route: str) -> dict[str, Any]:
    evidence = [
        {key: match[key] for key in ("title", "url", "domain", "price", "platform", "source_type", "provider_rank")}
        for match in matches
    ]
    prompt = (
        "You are Likeness, an evidence-oriented IP risk triage assistant. "
        "Assess only the supplied search metadata and stated ownership information. "
        "Do not claim that infringement is legally proven and do not invent facts or laws. "
        "Return strict JSON with overall_confidence (integer 0-100), summary (one concise sentence), "
        "and matches in the same order. Each match requires confidence (integer 0-100), "
        "threat_level (Critical, High, Moderate, or Low), rationale (max 24 words), "
        "and evidence_basis (max 18 words). "
        f"Route: {route}. Asset context: {json.dumps(asset_context)}. Candidate records: {json.dumps(evidence)}"
    )
    return await openai_json(prompt)


def missing_live_configuration() -> list[str]:
    """Return safe configuration labels without exposing secret values."""
    missing = [name for name in ("SERPAPI_API_KEY", "OPENAI_API_KEY") if not os.getenv(name)]
    if not (os.getenv("PUBLIC_INTAKE_READ_WRITE_TOKEN") or os.getenv("BLOB_READ_WRITE_TOKEN")):
        missing.append("a read-write token from the connected Public Vercel Blob store")
    return missing


def validate_likeness_blob_url(value: str) -> str:
    """Accept only a public Likeness intake URL, never an arbitrary remote URL."""
    url = bounded_text(value, 2048)
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    if (
        parsed.scheme != "https"
        or not host.endswith(".public.blob.vercel-storage.com")
        or not parsed.path.startswith("/likeness-intake/")
    ):
        raise HTTPException(422, "Upload an image through Likeness before starting analysis.")
    return url


def apply_insights(matches: list[dict[str, Any]], analysis: dict[str, Any]) -> list[dict[str, Any]]:
    insights = analysis.get("matches", [])
    for index, match in enumerate(matches):
        insight = insights[index] if index < len(insights) and isinstance(insights[index], dict) else {}
        try:
            confidence = int(insight.get("confidence", 45))
        except (TypeError, ValueError):
            confidence = 45
        match["confidence"] = max(0, min(confidence, 100))
        threat = str(insight.get("threat_level", "Moderate")).title()
        match["threat_level"] = threat if threat in {"Critical", "High", "Moderate", "Low"} else "Moderate"
        match["rationale"] = bounded_text(str(insight.get("rationale", "Similarity signal requires human review.")), 180)
        match["evidence_basis"] = bounded_text(str(insight.get("evidence_basis", "Search-result metadata.")), 140)
        match["timestamp"] = iso_now()
    return sorted(matches, key=lambda item: (-item["confidence"], item["provider_rank"]))[:MAX_CANDIDATES]


class NoticeTarget(BaseModel):
    title: str = Field(max_length=240)
    url: str = Field(max_length=2048)
    domain: str = Field(max_length=255)
    platform: str = Field(max_length=80)
    price: str = Field(default="Not listed", max_length=80)
    confidence: int = Field(ge=0, le=100)
    threat_level: str = Field(max_length=20)
    rationale: str = Field(max_length=280)
    timestamp: str = Field(max_length=80)


class NoticeRequest(BaseModel):
    route: Literal["commercial", "personal"]
    asset_name: str = Field(max_length=240)
    brand_name: str = Field(default="", max_length=160)
    rights_holder: str = Field(default="", max_length=200)
    rights_basis: str = Field(default="", max_length=400)
    jurisdiction: str = Field(default="", max_length=120)
    target: NoticeTarget


async def draft_notice_with_openai(request: NoticeRequest) -> dict[str, str]:
    target = request.target.model_dump()
    prompt = (
        "Draft a detailed, professional enforcement notice as plain text for human legal review. "
        "It is not legal advice, must not state that infringement is proven, and must not fabricate statutes, "
        "registration numbers, addresses, or platform procedures. Use the supplied jurisdiction only as context. "
        "Address it to the selected website/platform, identify the selected listing URL, request preservation of records, "
        "removal or disabling of the disputed content, a written response, and a reasonable response period expressed "
        "as a placeholder. Include sections for rights-holder details, factual basis, requested action, evidence preservation, "
        "reservation of rights, and signature placeholders. End with a clear legal-review disclaimer. "
        "Return strict JSON with title and body. Body should be 650-1000 words. "
        f"Case route: {request.route}. Asset: {request.asset_name}. Brand: {request.brand_name}. "
        f"Rights holder: {request.rights_holder}. Rights basis: {request.rights_basis}. "
        f"Jurisdiction: {request.jurisdiction}. Selected target: {json.dumps(target)}"
    )
    payload = await openai_json(prompt)
    title = bounded_text(str(payload.get("title", "DRAFT ENFORCEMENT NOTICE")), 160)
    body = str(payload.get("body", "")).strip()
    if not body:
        raise RuntimeError("The drafting service returned an empty notice.")
    return {"title": title or "DRAFT ENFORCEMENT NOTICE", "body": body[:9000]}


@app.get("/")
async def api_home() -> FileResponse:
    raise HTTPException(404, "The Likeness web interface is served from the project root.")


async def health_impl() -> JSONResponse:
    missing = missing_live_configuration()
    return JSONResponse({"live_ready": not missing, "missing": missing})


async def create_case_impl(
    image_url: str = Form(...),
    route: Literal["commercial", "personal"] = Form("commercial"),
    asset_name: str = Form(""),
    brand_name: str = Form(""),
    rights_holder: str = Form(""),
    rights_basis: str = Form(""),
    jurisdiction: str = Form(""),
    official_domains: str = Form(""),
) -> JSONResponse:
    missing_configuration = missing_live_configuration()
    if missing_configuration:
        raise HTTPException(
            503,
            "Live analysis is not configured. Add "
            f"{', '.join(missing_configuration)} in this Vercel project's Production settings, then redeploy.",
        )

    context = {
        "asset_name": bounded_text(asset_name, 240) or "Uploaded asset",
        "brand_name": bounded_text(brand_name, 160),
        "rights_holder": bounded_text(rights_holder, 200),
        "rights_basis": bounded_text(rights_basis, 400),
        "jurisdiction": bounded_text(jurisdiction, 120),
    }
    try:
        image_url = validate_likeness_blob_url(image_url)
        candidates = await search_google_lens(image_url, parse_domains(official_domains))
        if not candidates:
            raise HTTPException(404, "Google Lens found no candidate listings after approved domains were excluded.")
        analysis = await analyze_with_openai(context, candidates, route)
        matches = apply_insights(candidates, analysis)
    except HTTPException:
        raise
    except httpx.HTTPStatusError as exc:
        provider = "SerpApi" if "serpapi.com" in str(exc.request.url) else "OpenAI"
        raise HTTPException(502, f"{provider} rejected the secure analysis request. Check that provider key and quota.") from exc
    except Exception as exc:
        raise HTTPException(502, f"Secure analysis provider error: {str(exc)[:160]}") from exc

    average = round(sum(match["confidence"] for match in matches) / len(matches))
    try:
        overall = int(analysis.get("overall_confidence", average))
    except (TypeError, ValueError):
        overall = average
    return JSONResponse(
        {
            "mode": "live",
            "case": context,
            "matches": matches,
            "overall_confidence": max(0, min(overall, 100)),
            "summary": bounded_text(str(analysis.get("summary", "Candidate listings are ready for rights-holder review.")), 260),
            "source_image_url": image_url,
        }
    )


async def create_notice_impl(request: NoticeRequest) -> JSONResponse:
    missing = [name for name in ("OPENAI_API_KEY",) if not os.getenv(name)]
    if missing:
        raise HTTPException(503, f"Notice drafting requires {', '.join(missing)} in Production settings.")
    try:
        document = await draft_notice_with_openai(request)
    except httpx.HTTPStatusError as exc:
        raise HTTPException(502, "OpenAI rejected the notice drafting request. Check the provider key and quota.") from exc
    except Exception as exc:
        raise HTTPException(502, f"Secure notice drafting error: {str(exc)[:160]}") from exc
    return JSONResponse({"document": document, "target_url": request.target.url})


# Each callable entrypoint imports this app. The aliases accommodate Vercel's
# per-file routing and local FastAPI testing.
app.post("/cases")(create_case_impl)
app.post("/api/cases")(create_case_impl)
app.post("/notices")(create_notice_impl)
app.post("/api/notices")(create_notice_impl)
app.get("/health")(health_impl)
app.get("/api/health")(health_impl)
