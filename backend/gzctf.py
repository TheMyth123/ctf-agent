"""GZCTF client — async httpx, token + session auth."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

import httpx

logger = logging.getLogger(__name__)

USER_AGENT = "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36"

# GZCTF AnswerResult enum values (string or int)
_ACCEPTED = {"Accepted", "accepted", 0}
_ALREADY_SOLVED = {"AlreadySolved", "alreadysolved", 4}
_WRONG = {"WrongAnswer", "wronganswer", 1}
_NOT_FOUND = {"NotFound", "notfound", 2}
_CHEAT = {"CheatDetected", "cheatdetected", 3}


@dataclass
class SubmitResult:
    status: str  # "correct" | "already_solved" | "incorrect" | "unknown"
    message: str
    display: str


@dataclass
class GZCTFClient:
    base_url: str = "http://localhost:8080"
    game_id: int = 1
    token: str = ""
    username: str = "admin"
    password: str = "admin"

    _client: httpx.AsyncClient | None = field(default=None, repr=False)
    _logged_in: bool = False
    _challenge_ids: dict[str, int] = field(default_factory=dict)

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url.rstrip("/"),
                follow_redirects=True,
                verify=False,
                timeout=30.0,
                headers={"User-Agent": USER_AGENT},
            )
        return self._client

    async def _ensure_logged_in(self) -> None:
        if self._logged_in or self.token:
            return
        client = await self._ensure_client()

        resp = await client.post(
            "/api/account/login",
            json={"userName": self.username, "password": self.password},
            headers={"Content-Type": "application/json"},
        )
        if resp.status_code not in (200, 204):
            raise RuntimeError(
                f"GZCTF login failed (HTTP {resp.status_code}) — bad credentials?"
            )
        self._logged_in = True

    def _base_headers(self) -> dict[str, str]:
        h: dict[str, str] = {"Content-Type": "application/json"}
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        return h

    async def _get(self, path: str) -> Any:
        await self._ensure_logged_in()
        client = await self._ensure_client()
        resp = await client.get(path, headers=self._base_headers())
        resp.raise_for_status()
        return resp.json()

    async def _post(self, path: str, body: dict[str, Any]) -> Any:
        await self._ensure_logged_in()
        client = await self._ensure_client()
        resp = await client.post(path, json=body, headers=self._base_headers())
        resp.raise_for_status()
        return resp.json()

    def _flatten_challenges(self, raw: Any) -> list[dict[str, Any]]:
        """Flatten GZCTF challenge response (dict-of-lists keyed by category)."""
        if isinstance(raw, list):
            return raw
        if isinstance(raw, dict):
            result: list[dict[str, Any]] = []
            for _cat, items in raw.items():
                if isinstance(items, list):
                    result.extend(items)
            return result
        return []

    async def fetch_challenge_stubs(self) -> list[dict[str, Any]]:
        """Fetch lightweight challenge list."""
        raw = await self._get(f"/api/game/{self.game_id}/challenges")
        challenges = self._flatten_challenges(raw)
        return [
            {
                "id": ch.get("id"),
                "name": ch.get("title", f"challenge-{ch.get('id')}"),
                "category": ch.get("tag", ""),
                "value": ch.get("score", 0),
                "solves": ch.get("solved", 0),
                "isSolved": ch.get("isSolved", False),
            }
            for ch in challenges
        ]

    async def get_challenge_id(self, name: str) -> int:
        if name in self._challenge_ids:
            return self._challenge_ids[name]

        stubs = await self.fetch_challenge_stubs()
        for ch in stubs:
            self._challenge_ids[ch["name"]] = ch["id"]

        if name not in self._challenge_ids:
            raise RuntimeError(f'Challenge "{name}" not found in GZCTF')
        return self._challenge_ids[name]

    async def submit_flag(self, challenge_name: str, flag: str) -> SubmitResult:
        challenge_id = await self.get_challenge_id(challenge_name)
        try:
            resp = await self._post(
                f"/api/game/{self.game_id}/challenges/{challenge_id}",
                {"flag": flag},
            )
        except httpx.HTTPStatusError as exc:
            return SubmitResult("unknown", str(exc), f"Submit error: {exc}")

        # GZCTF returns {"status": "Accepted"} or {"status": 0} or similar
        status_raw = resp if isinstance(resp, (str, int)) else resp.get("status", "unknown")
        status_key = status_raw.lower().replace(" ", "") if isinstance(status_raw, str) else status_raw

        if status_raw in _ACCEPTED or status_key == "accepted":
            return SubmitResult("correct", "Accepted", f'CORRECT — "{flag}" accepted.')
        if status_raw in _ALREADY_SOLVED or status_key == "alreadysolved":
            return SubmitResult(
                "already_solved", "Already solved",
                f'ALREADY SOLVED — "{flag}" accepted.'
            )
        if status_raw in _WRONG or status_key == "wronganswer":
            return SubmitResult("incorrect", "Wrong answer", f'INCORRECT — "{flag}" rejected.')
        if status_raw in _CHEAT or status_key == "cheatdetected":
            return SubmitResult("incorrect", "Cheat detected", f'CHEAT DETECTED — "{flag}" flagged.')
        return SubmitResult("unknown", str(status_raw), f"Unknown status: {status_raw}")

    async def fetch_all_challenges(self) -> list[dict[str, Any]]:
        """Fetch challenges with full detail (description, hints, files)."""
        stubs = await self.fetch_challenge_stubs()
        challenges: list[dict[str, Any]] = []
        for stub in stubs:
            try:
                detail = await self._get(
                    f"/api/game/{self.game_id}/challenges/{stub['id']}"
                )
                # Merge stub data with detail
                merged = {**stub, **detail}
                # Normalise field names to match the rest of the codebase
                merged.setdefault("name", merged.get("title", stub["name"]))
                merged.setdefault("description", merged.get("content", ""))
                merged.setdefault("category", merged.get("tag", ""))
                merged.setdefault("value", merged.get("score", 0))
                merged.setdefault("connection_info", "")
                # Flatten attachment into a files list
                att = detail.get("attachment")
                if att and att.get("url"):
                    merged["files"] = [att["url"]]
                else:
                    merged.setdefault("files", [])
                # Context may contain instance entry (nc host:port)
                ctx = detail.get("context") or {}
                if ctx.get("instanceEntry") and not merged.get("connection_info"):
                    merged["connection_info"] = ctx["instanceEntry"]
                challenges.append(merged)
            except Exception:
                logger.warning(
                    "Could not fetch detail for challenge %s (id=%s)",
                    stub["name"], stub["id"], exc_info=True,
                )
                challenges.append(stub)
        return challenges

    async def fetch_solved_names(self) -> set[str]:
        """Return names of challenges already solved by the current team."""
        try:
            stubs = await self.fetch_challenge_stubs()
            return {ch["name"] for ch in stubs if ch.get("isSolved")}
        except Exception:
            logger.warning("Could not fetch solved challenges", exc_info=True)
            return set()

    async def pull_challenge(self, challenge: dict[str, Any], output_dir: str) -> str:
        """Download challenge distfiles and write metadata.yml.

        Returns the challenge directory path.
        """
        from pathlib import Path
        from urllib.parse import urlparse

        import yaml

        name = challenge.get("name") or challenge.get("title", f"challenge-{challenge.get('id')}")
        slug = re.sub(r'[<>:"/\\|?*.\x00-\x1f]', "", name.lower().strip())
        slug = re.sub(r"[\s_]+", "-", slug)
        slug = re.sub(r"-+", "-", slug).strip("-") or "challenge"

        ch_dir = Path(output_dir) / slug
        ch_dir.mkdir(parents=True, exist_ok=True)

        await self._ensure_logged_in()
        client = await self._ensure_client()

        for raw_url in challenge.get("files") or []:
            dist_dir = ch_dir / "distfiles"
            dist_dir.mkdir(exist_ok=True)
            url = (
                raw_url
                if raw_url.startswith("http")
                else f"{self.base_url.rstrip('/')}/{raw_url.lstrip('/')}"
            )
            url_path = urlparse(url).path
            fname = url_path.rstrip("/").rsplit("/", 1)[-1] or "file"
            dest = dist_dir / fname
            if not dest.exists():
                try:
                    headers = (
                        self._base_headers()
                        if urlparse(url).hostname == urlparse(self.base_url).hostname
                        else {}
                    )
                    resp = await client.get(
                        url, headers=headers, follow_redirects=True, timeout=60.0
                    )
                    resp.raise_for_status()
                    dest.write_bytes(resp.content)
                    logger.info("Downloaded: %s (%d bytes)", fname, len(resp.content))
                except Exception as e:
                    logger.warning("Failed to download %s: %s", url, e)

        from markdownify import markdownify as html2md

        desc = challenge.get("description") or challenge.get("content") or ""
        try:
            desc = html2md(desc, heading_style="atx", escape_asterisks=False)
        except Exception:
            pass

        tags = challenge.get("tags") or [challenge.get("tag") or challenge.get("category") or ""]
        hints_raw = challenge.get("hints") or []
        hints = [{"content": h} if isinstance(h, str) else h for h in hints_raw]

        meta: dict[str, Any] = {
            "name": name,
            "category": challenge.get("category") or challenge.get("tag") or "",
            "description": desc.strip(),
            "value": challenge.get("value") or challenge.get("score") or 0,
            "connection_info": challenge.get("connection_info") or "",
            "tags": [t for t in tags if t],
            "solves": challenge.get("solves") or challenge.get("solved") or 0,
        }
        if hints:
            meta["hints"] = hints

        (ch_dir / "metadata.yml").write_text(
            yaml.dump(meta, allow_unicode=True, default_flow_style=False, sort_keys=False)
        )
        return str(ch_dir)

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None
