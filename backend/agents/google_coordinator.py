"""Google AI Studio coordinator — Pydantic AI agent managing the competition."""

from __future__ import annotations

import logging
from typing import Any

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider

from backend.agents.coordinator_core import (
    do_broadcast,
    do_bump_agent,
    do_check_swarm_status,
    do_fetch_challenges,
    do_get_solve_status,
    do_kill_swarm,
    do_read_solver_trace,
    do_spawn_swarm,
    do_submit_flag,
)
from backend.agents.coordinator_loop import build_deps, run_event_loop
from backend.config import Settings
from backend.deps import CoordinatorDeps

logger = logging.getLogger(__name__)

COORDINATOR_PROMPT = """\
You are a CTF competition coordinator running for the ENTIRE duration of a live competition.
Your job is to maximize the number of challenges solved.

Strategy:
- Spawn swarms for unsolved challenges, prioritizing by solve count (easy first)
- Use read_solver_trace to monitor what each solver is doing and where it's stuck
- When agents are stuck, read their traces, then craft targeted bumps with specific technical guidance
- Use broadcast to share cross-solver insights (e.g. flag format discovery, shared vulnerabilities)

CRITICAL RULES:
- NEVER kill a swarm. Solvers will keep trying indefinitely with different approaches.
  Even when stuck, they often unstick themselves after several bumps. Your job is to
  HELP them, not give up on them. The only time a swarm should die is when the flag
  is confirmed correct.
- When a solver seems stuck, bump it with very specific technical guidance based on
  its trace. Tell it exactly what to try next — specific tools, techniques, approaches.
- Cost is not a concern. Keep all swarms running.

You will receive event messages. Respond with tool calls to manage the competition.
"""


def _build_coordinator_agent(settings: Settings) -> Agent[CoordinatorDeps, str]:
    model = GoogleModel(
        "gemini-2.0-flash",
        provider=GoogleProvider(api_key=settings.gemini_api_key),
    )

    agent: Agent[CoordinatorDeps, str] = Agent(
        model,
        deps_type=CoordinatorDeps,
        system_prompt=COORDINATOR_PROMPT,
        output_type=str,
    )

    @agent.tool
    async def fetch_challenges(ctx: RunContext[CoordinatorDeps]) -> str:
        """List all challenges with category, points, solve count, and status."""
        return await do_fetch_challenges(ctx.deps)

    @agent.tool
    async def get_solve_status(ctx: RunContext[CoordinatorDeps]) -> str:
        """Check which challenges are solved and which swarms are running."""
        return await do_get_solve_status(ctx.deps)

    @agent.tool
    async def spawn_swarm(ctx: RunContext[CoordinatorDeps], challenge_name: str) -> str:
        """Launch all solver models on a challenge."""
        return await do_spawn_swarm(ctx.deps, challenge_name)

    @agent.tool
    async def check_swarm_status(ctx: RunContext[CoordinatorDeps], challenge_name: str) -> str:
        """Get per-agent progress for a swarm."""
        return await do_check_swarm_status(ctx.deps, challenge_name)

    @agent.tool
    async def submit_flag(
        ctx: RunContext[CoordinatorDeps], challenge_name: str, flag: str
    ) -> str:
        """Submit a flag to GZCTF."""
        return await do_submit_flag(ctx.deps, challenge_name, flag)

    @agent.tool
    async def kill_swarm(ctx: RunContext[CoordinatorDeps], challenge_name: str) -> str:
        """Cancel all agents for a challenge."""
        return await do_kill_swarm(ctx.deps, challenge_name)

    @agent.tool
    async def bump_agent(
        ctx: RunContext[CoordinatorDeps],
        challenge_name: str,
        model_spec: str,
        insights: str,
    ) -> str:
        """Send targeted insights to a stuck agent."""
        return await do_bump_agent(ctx.deps, challenge_name, model_spec, insights)

    @agent.tool
    async def broadcast(
        ctx: RunContext[CoordinatorDeps], challenge_name: str, message: str
    ) -> str:
        """Broadcast a strategic hint to ALL solvers on a challenge."""
        return await do_broadcast(ctx.deps, challenge_name, message)

    @agent.tool
    async def read_solver_trace(
        ctx: RunContext[CoordinatorDeps],
        challenge_name: str,
        model_spec: str,
        last_n: int = 20,
    ) -> str:
        """Read recent trace events from a specific solver."""
        return await do_read_solver_trace(ctx.deps, challenge_name, model_spec, last_n)

    return agent


async def run_google_coordinator(
    settings: Settings,
    model_specs: list[str] | None = None,
    challenges_root: str = "challenges",
    no_submit: bool = False,
    coordinator_model: str | None = None,  # unused — always gemini-2.0-flash
    msg_port: int = 0,
) -> dict[str, Any]:
    """Run the Google AI coordinator with the shared event loop."""
    gzctf, cost_tracker, deps = build_deps(
        settings, model_specs, challenges_root, no_submit
    )
    deps.msg_port = msg_port

    agent = _build_coordinator_agent(settings)
    history: list = []

    async def turn_fn(msg: str) -> None:
        nonlocal history
        logger.debug("Coordinator query: %s", msg[:200])
        result = await agent.run(
            msg,
            deps=deps,
            message_history=history if history else None,
        )
        history = list(result.all_messages())
        logger.info("Google coordinator turn done")

    return await run_event_loop(deps, gzctf, cost_tracker, turn_fn)
