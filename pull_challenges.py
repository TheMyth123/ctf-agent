#!/usr/bin/env python3
"""Pull challenges from a GZCTF instance to a local directory.

Usage:
    python pull_challenges.py --url https://ctf.example.com --game-id 1 \
        --username myteam --password s3cr3t [--output ./challenges]

    python pull_challenges.py --url https://ctf.example.com --game-id 1 \
        --token your_bearer_token [--output ./challenges]

Directory layout:
    <output>/<challenge-slug>/
        metadata.yml   # name, description, value, tags, connection_info, hints
        distfiles/     # attached files
"""

import argparse
import asyncio
import re
import sys
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import yaml
import aiohttp
from markdownify import markdownify as html2md

USER_AGENT = "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/45.0.2454.85 Safari/537.36"


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def bearer_headers(token: str) -> dict:
    return {"User-Agent": USER_AGENT, "Authorization": f"Bearer {token}"}


async def login_password(session: aiohttp.ClientSession, base_url: str, username: str, password: str) -> bool:
    """Login to GZCTF and populate session cookies. Returns True on success."""
    async with session.post(
        f"{base_url}/api/account/login",
        json={"userName": username, "password": password},
        headers={"User-Agent": USER_AGENT, "Content-Type": "application/json"},
    ) as resp:
        if resp.status not in (200, 204):
            print(f"ERROR: Login failed (HTTP {resp.status}) — bad credentials?", file=sys.stderr)
            return False
        return True


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

async def api_get(session: aiohttp.ClientSession, url: str, extra_headers: Optional[dict] = None) -> Optional[dict]:
    headers = {"User-Agent": USER_AGENT}
    if extra_headers:
        headers.update(extra_headers)
    async with session.get(url, headers=headers, ssl=False) as resp:
        if resp.status != 200:
            return None
        return await resp.json()


async def fetch_bytes(session: aiohttp.ClientSession, url: str, extra_headers: Optional[dict] = None) -> Optional[bytes]:
    headers = {"User-Agent": USER_AGENT}
    if extra_headers:
        headers.update(extra_headers)
    async with session.get(url, headers=headers, ssl=False) as resp:
        if resp.status != 200:
            return None
        return await resp.read()


# ---------------------------------------------------------------------------
# HTML / text helpers
# ---------------------------------------------------------------------------

def html_to_markdown(html: Optional[str]) -> str:
    if not html:
        return ""
    md = html2md(html, heading_style="atx", escape_asterisks=False, escape_underscores=False)
    md = re.sub(r"[^\S\r\n]*!\[[^\]]*\]\([^)]*\)\s*", "", md)
    md = re.sub(r"\n{3,}", "\n\n", md)
    return md.strip()


def slugify(name: str) -> str:
    slug = name.lower().strip()
    slug = re.sub(r'[<>:"/\\|?*.\x00-\x1f]', "", slug)
    slug = re.sub(r"[\s_]+", "-", slug)
    slug = re.sub(r"-+", "-", slug).strip("-")
    return slug or "challenge"


def filename_from_url(url: str) -> str:
    path = urlparse(url).path
    name = path.rstrip("/").rsplit("/", 1)[-1]
    return name or "file"


def make_absolute(url: str, base_url: str) -> str:
    if url.startswith("http"):
        return url
    return f"{base_url.rstrip('/')}/{url.lstrip('/')}"


# ---------------------------------------------------------------------------
# Challenge pulling
# ---------------------------------------------------------------------------

def _flatten_challenges(raw: dict | list) -> list[dict]:
    """GZCTF returns challenges as a dict keyed by category; flatten to list."""
    if isinstance(raw, list):
        return raw
    result: list[dict] = []
    for _cat, items in raw.items():
        if isinstance(items, list):
            result.extend(items)
    return result


async def pull_challenges(
    session: aiohttp.ClientSession,
    base_url: str,
    game_id: int,
    extra_headers: Optional[dict] = None,
):
    """Yield full challenge dicts from the GZCTF API."""
    stubs_raw = await api_get(session, f"{base_url}/api/game/{game_id}/challenges", extra_headers)
    if stubs_raw is None:
        print("ERROR: Could not fetch challenge list. Are you logged in?", file=sys.stderr)
        return

    stubs = _flatten_challenges(stubs_raw)
    for stub in stubs:
        ch_id = stub.get("id")
        if not ch_id:
            continue
        detail = await api_get(session, f"{base_url}/api/game/{game_id}/challenges/{ch_id}", extra_headers)
        if detail is None:
            print(f"  WARN: Could not fetch details for challenge id={ch_id}", file=sys.stderr)
            # Fall back to stub data
            yield {
                **stub,
                "name": stub.get("title", f"challenge-{ch_id}"),
                "category": stub.get("tag", ""),
                "value": stub.get("score", 0),
                "description": "",
                "files": [],
                "hints": [],
            }
            continue
        merged = {**stub, **detail}
        merged["name"] = merged.get("title", f"challenge-{ch_id}")
        merged["category"] = merged.get("tag", "")
        merged["value"] = merged.get("score", 0)
        merged["description"] = merged.get("content", "")
        # Flatten attachment to files list
        att = detail.get("attachment")
        if att and att.get("url"):
            merged["files"] = [att["url"]]
        else:
            merged.setdefault("files", [])
        ctx = detail.get("context") or {}
        merged["connection_info"] = ctx.get("instanceEntry") or ""
        yield merged


# ---------------------------------------------------------------------------
# Writing to disk
# ---------------------------------------------------------------------------

def build_metadata(challenge: dict, hints: list[dict]) -> dict:
    tags_raw = challenge.get("tags") or [challenge.get("tag") or challenge.get("category") or ""]
    tags = [t for t in tags_raw if t]
    description = html_to_markdown(challenge.get("description") or "")

    meta = {
        "version": "beta1",
        "name": challenge.get("name", "Unknown"),
        "category": challenge.get("category", ""),
        "description": description,
        "value": challenge.get("value", 0),
    }

    if challenge.get("solves") is not None or challenge.get("solved") is not None:
        meta["solves"] = challenge.get("solves") or challenge.get("solved") or 0

    if tags:
        meta["tags"] = tags

    if challenge.get("connection_info"):
        meta["connection_info"] = challenge["connection_info"]

    if hints:
        meta["hints"] = []
        for hint in hints:
            if isinstance(hint, str):
                meta["hints"].append({"content": hint})
            else:
                entry: dict = {}
                if hint.get("content"):
                    entry["content"] = html_to_markdown(hint["content"])
                meta["hints"].append(entry)

    return meta


async def save_challenge(
    session: aiohttp.ClientSession,
    base_url: str,
    challenge: dict,
    output_dir: Path,
    extra_headers: Optional[dict] = None,
):
    slug = slugify(challenge.get("name", f"challenge-{challenge.get('id')}"))
    chdir = output_dir / slug
    chdir.mkdir(parents=True, exist_ok=True)

    distfiles_dir = chdir / "distfiles"

    for raw_url in challenge.get("files") or []:
        distfiles_dir.mkdir(exist_ok=True)
        url = make_absolute(raw_url, base_url)
        fname = filename_from_url(raw_url)
        dest = distfiles_dir / fname

        content = await fetch_bytes(session, url, extra_headers)
        if content is None:
            print(f"    WARN: Could not download {url}", file=sys.stderr)
        else:
            dest.write_bytes(content)
            print(f"    Downloaded: {fname}")

    raw_hints = challenge.get("hints") or []
    meta = build_metadata(challenge, raw_hints)
    (chdir / "metadata.yml").write_text(
        yaml.dump(meta, allow_unicode=True, default_flow_style=False, sort_keys=False),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main(
    url: str,
    game_id: int,
    output: str,
    username: Optional[str],
    password: Optional[str],
    token: Optional[str],
):
    base_url = url.rstrip("/")
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    extra_headers: Optional[dict] = None

    connector = aiohttp.TCPConnector(ssl=False)
    async with aiohttp.ClientSession(connector=connector) as session:
        if token:
            extra_headers = bearer_headers(token)
            print(f"Using bearer token for {base_url} (game {game_id}).\n")
        else:
            print(f"Logging in to {base_url} as {username}...")
            if not await login_password(session, base_url, username, password):
                sys.exit(1)
            print("Login successful.\n")

        count = 0
        async for challenge in pull_challenges(session, base_url, game_id, extra_headers):
            cname = challenge.get("name", f"id={challenge.get('id')}")
            ccat = challenge.get("category", "?")
            cval = challenge.get("value", 0)
            print(f"  [{ccat}] {cname} ({cval} pts)")
            await save_challenge(session, base_url, challenge, output_dir, extra_headers)
            count += 1

        print(f"\nDone. Pulled {count} challenge(s) to {output_dir.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pull challenges from a GZCTF instance.")
    parser.add_argument("--url", required=True, help="Base URL of the GZCTF instance")
    parser.add_argument("--game-id", required=True, type=int, help="GZCTF game ID")
    parser.add_argument("--output", default="./challenges", help="Output directory (default: ./challenges)")

    auth = parser.add_mutually_exclusive_group(required=True)
    auth.add_argument("--token", help="Bearer token")
    auth.add_argument("--username", help="Login username (use with --password)")

    parser.add_argument("--password", help="Login password (required with --username)")
    args = parser.parse_args()

    if args.username and not args.password:
        parser.error("--password is required when using --username")

    asyncio.run(main(args.url, args.game_id, args.output, args.username, args.password, args.token))
