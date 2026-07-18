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

# SerpApi candidates are deduplicated and assessed before the UI sees them.
# Twenty is still comfortable in the two-column evidence ledger.
MAX_CANDIDATES = 20

app = FastAPI(title="Likeness", version="2.0.0")


def iso_now() -> str:
    return datetime.now(timezone.utc).strftime("%b %d, %Y · %H:%M UTC")


def bounded_text(value: str | None, limit: int = 280) -> str:
    return re.sub(r"\s+", " ", (value or "").strip())[:limit]


def provider_text(value: Any, limit: int = 280, fallback: str = "") -> str:
    """Turn variable SerpApi display fields into safe, human-readable text."""
    if isinstance(value, str):
        text = value
    elif isinstance(value, (int, float)):
        text = str(value)
    elif isinstance(value, dict):
        text = ""
        for key in ("display_price", "extracted_price", "price", "value", "text", "amount"):
            candidate = value.get(key)
            if isinstance(candidate, (str, int, float)):
                text = str(candidate)
                break
    else:
        text = ""
    return bounded_text(text, limit) or fallback


def notice_legal_framework(jurisdiction: str) -> str:
    """Verified high-level legal context; not a substitute for local legal advice."""
    base = (
        "Copyright protection is territorial and the available remedies depend on the law of the place where protection is claimed. "
        "Under the Berne Convention framework, protection for qualifying works is generally not conditional on registration or another formality. "
        "Do not assert that any particular right, registration, or remedy applies unless it is supported by the supplied facts and reviewed by counsel."
    )
    if "philipp" in jurisdiction.lower():
        return (
            base
            + " For Philippine context, Republic Act No. 8293, the Intellectual Property Code of the Philippines, addresses "
            "copyright and related rights, trademarks and service marks, industrial designs, patents, and other intellectual property categories. "
            "Use this reference only as general context and do not cite a section number or state a legal conclusion without counsel review."
        )
    return base


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
                    "price": provider_text(item.get("price") or item.get("extracted_price"), 80, "Not listed"),
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


async def openai_json(
    prompt: str, image_urls: list[str] | None = None, image_labels: list[str] | None = None
) -> dict[str, Any]:
    api_key = os.getenv("OPENAI_API_KEY")
    content: str | list[dict[str, Any]] = prompt
    if image_urls:
        content = [{"type": "text", "text": prompt}]
        for index, url in enumerate(image_urls):
            if image_labels and index < len(image_labels):
                content.append({"type": "text", "text": image_labels[index]})
            content.append({"type": "image_url", "image_url": {"url": url, "detail": "high"}})
    body = {
        "model": openai_model(),
        "messages": [{"role": "user", "content": content}],
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


async def analyze_with_openai(asset_context: dict[str, str], matches: list[dict[str, Any]], route: str, source_image_url: str) -> dict[str, Any]:
    evidence = [
        {key: match[key] for key in ("title", "url", "domain", "price", "platform", "source_type", "provider_rank")}
        for match in matches
    ]
    vision_urls = [source_image_url]
    image_labels = ["IMAGE 0 — rights-holder reference asset."]
    for index, match in enumerate(matches[:10]):
        thumbnail = match.get("thumbnail")
        if isinstance(thumbnail, str) and thumbnail.startswith("https://"):
            vision_urls.append(thumbnail)
            image_labels.append(f"CANDIDATE {index + 1} — thumbnail for the candidate record at index {index + 1}.")
    prompt = (
        "You are Likeness, an evidence-oriented IP risk triage assistant. "
        "The first image is the rights-holder's uploaded asset. Each later image is explicitly labelled with its candidate-record index; do not assign visual observations to a candidate whose thumbnail was not supplied. "
        "Assess visual and metadata similarity only from the supplied images, search metadata, and stated ownership information. Describe concrete, observable signals such as logos, layout, silhouette, colour placement, packaging, model identifiers, title wording, seller, or price. "
        "Do not claim that infringement is legally proven and do not invent facts or laws. "
        "Confidence is a visual-likeness triage score, not a probability of legal infringement. Calibrate it strictly: 90-100 only for the same image, clearly identical product, or distinctive design; "
        "75-89 for substantial visual/design overlap; 50-74 for general category resemblance or incomplete visual evidence; 0-49 for weak or unrelated similarity. "
        "A listing may be visually identical but not a legal threat if it is authorized; use the stated authorized domains/sellers as context. Never boost a score merely because it appears in the search results. "
        "Return strict JSON with overall_confidence (integer 0-100), summary (2-3 concise sentences), "
        "risk_factors (array of 2-4 concise evidence-based signals), recommended_actions (array of 2-4 practical next steps), "
        "and limitations (array of 1-3 evidence limitations). "
        "and matches in the same order. Each match requires confidence (integer 0-100), "
        "threat_level (Critical, High, Moderate, or Low), visual_similarity (integer 0-100), title_similarity (integer 0-100), "
        "metadata_similarity (integer 0-100), display_decision (prioritize, review, or hide), filter_reason (max 16 words), "
        "match_type (SAME IMAGE, NEAR-IDENTICAL PRODUCT, SHARED DESIGN CUES, CATEGORY-LEVEL RESEMBLANCE, or INSUFFICIENT EVIDENCE), "
        "similarities (array of 2-4 concise concrete observations), differences (array of 0-3 concrete observations), "
        "uncertainty (max 22 words; explain a missing thumbnail, obscured detail, or conflicting evidence, otherwise an empty string), "
        "rationale (max 42 words; explain the score in plain language), and evidence_basis (max 36 words). "
        f"Route: {route}. Asset context: {json.dumps(asset_context)}. Candidate records: {json.dumps(evidence)}"
    )
    try:
        analysis = await openai_json(
            prompt, vision_urls if len(vision_urls) > 1 else None, image_labels if len(vision_urls) > 1 else None
        )
        analysis["comparison_mode"] = "Vision and metadata" if len(vision_urls) > 1 else "Metadata only"
        return analysis
    except httpx.HTTPStatusError:
        # A remote thumbnail may deny third-party fetches. Preserve a useful
        # metadata-only result instead of failing the whole case.
        analysis = await openai_json(prompt)
        analysis["comparison_mode"] = "Metadata only (one or more listing images were unavailable)"
        return analysis


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


def apply_insights(matches: list[dict[str, Any]], analysis: dict[str, Any]) -> tuple[list[dict[str, Any]], int]:
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
        match["match_type"] = bounded_text(str(insight.get("match_type", "REVIEW CANDIDATE")).upper(), 42)
        match["uncertainty"] = bounded_text(str(insight.get("uncertainty", "")), 140)
        for field in ("visual_similarity", "title_similarity", "metadata_similarity"):
            try:
                score = int(insight.get(field, confidence if field == "visual_similarity" else 0))
            except (TypeError, ValueError):
                score = confidence if field == "visual_similarity" else 0
            match[field] = max(0, min(score, 100))
        match["similarities"] = analysis_list(
            insight.get("similarities"), 4, ["Visual and listing metadata require rights-holder review."]
        )
        match["differences"] = analysis_list(insight.get("differences"), 3, [])
        decision = str(insight.get("display_decision", "review")).lower()
        match["display_decision"] = decision if decision in {"prioritize", "review", "hide"} else "review"
        match["filter_reason"] = bounded_text(str(insight.get("filter_reason", "Requires rights-holder review.")), 120)
        match["timestamp"] = iso_now()
    ranked = sorted(matches, key=lambda item: (-item["confidence"], item["provider_rank"]))
    visible = [match for match in ranked if match["display_decision"] != "hide"]
    screened_out_count = len(ranked) - len(visible)
    # Do not silently turn a weak AI response into an empty evidence ledger.
    if not visible:
        visible = ranked[: min(10, len(ranked))]
        screened_out_count = 0
    return visible[:MAX_CANDIDATES], screened_out_count


def analysis_list(value: Any, limit: int, fallback: list[str]) -> list[str]:
    if not isinstance(value, list):
        return fallback
    items = [bounded_text(str(item), 220) for item in value if isinstance(item, (str, int, float))]
    return [item for item in items if item][:limit] or fallback


def analysis_brief(analysis: dict[str, Any]) -> dict[str, Any]:
    return {
        "summary": bounded_text(str(analysis.get("summary", "Candidate listings are ready for rights-holder review.")), 650),
        "comparison_mode": bounded_text(str(analysis.get("comparison_mode", "Metadata only")), 100),
        "risk_factors": analysis_list(analysis.get("risk_factors"), 4, ["Visual-search results require comparison against the rights holder's original asset."]),
        "recommended_actions": analysis_list(analysis.get("recommended_actions"), 4, ["Review the selected listing and preserve a dated copy of the page before taking action."]),
        "limitations": analysis_list(analysis.get("limitations"), 3, ["This triage is based on public search-result metadata and does not determine legal liability."]),
    }


class NoticeTarget(BaseModel):
    title: str = Field(max_length=240)
    url: str = Field(max_length=2048)
    domain: str = Field(max_length=255)
    platform: str = Field(max_length=80)
    # Marketplace providers sometimes send price as a structured object.
    # Normalize it before it reaches the drafting prompt.
    price: Any = "Not listed"
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
    asset_description: str = Field(default="", max_length=600)
    identifiers: str = Field(default="", max_length=300)
    authorized_sellers: str = Field(default="", max_length=300)
    target: NoticeTarget


async def draft_notice_with_openai(request: NoticeRequest) -> dict[str, str]:
    target = request.target.model_dump()
    target["price"] = provider_text(target.get("price"), 80, "Not listed")
    prompt = (
        "Draft a detailed, formal enforcement notice in plain text for human legal review. "
        "It is not legal advice, must not state that infringement is proven, and must not fabricate statutes, "
        "registration numbers, addresses, or platform procedures. Use the supplied jurisdiction only as context. "
        "Address it to the selected website/platform, identify the selected listing URL, request preservation of records, "
        "removal or disabling of the disputed content, a written response, and a reasonable response period expressed "
        "as a placeholder. Include sections for rights-holder details, factual basis, requested action, evidence preservation, "
        "reservation of rights, and signature placeholders. End with a clear legal-review disclaimer. "
        "Use conventional legal-letter prose and clear all-caps section labels where useful. Do not use Markdown, hashtags, bullets, "
        "asterisks, or numbered lists. Return strict JSON with title and body. Body should be 900-1200 words. "
        f"Case route: {request.route}. Asset: {request.asset_name}. Brand: {request.brand_name}. "
        f"Rights holder: {request.rights_holder}. Rights basis: {request.rights_basis}. "
        f"Jurisdiction: {request.jurisdiction}. Distinctive features: {request.asset_description}. "
        f"Product identifiers: {request.identifiers}. Known authorized sellers: {request.authorized_sellers}. "
        f"Selected target: {json.dumps(target)}. Verified general legal framework to incorporate carefully: {notice_legal_framework(request.jurisdiction)}"
    )
    payload = await openai_json(prompt)
    title = bounded_text(str(payload.get("title", "DRAFT ENFORCEMENT NOTICE")), 160)
    body = re.sub(r"(?m)^\s*#{1,6}\s*", "", str(payload.get("body", "")).strip())
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
    asset_description: str = Form(""),
    identifiers: str = Form(""),
    authorized_sellers: str = Form(""),
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
        "asset_description": bounded_text(asset_description, 600),
        "identifiers": bounded_text(identifiers, 300),
        "authorized_sellers": bounded_text(authorized_sellers, 300),
    }
    try:
        image_url = validate_likeness_blob_url(image_url)
        candidates = await search_google_lens(image_url, parse_domains(official_domains))
        if not candidates:
            raise HTTPException(404, "Google Lens found no candidate listings after approved domains were excluded.")
        analysis = await analyze_with_openai(context, candidates, route, image_url)
        matches, screened_out_count = apply_insights(candidates, analysis)
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
    ai_findings = analysis_brief(analysis)
    return JSONResponse(
        {
            "mode": "live",
            "case": context,
            "matches": matches,
            "overall_confidence": max(0, min(overall, 100)),
            "summary": bounded_text(ai_findings["summary"], 260),
            "ai_findings": ai_findings,
            "screened_out_count": screened_out_count,
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
