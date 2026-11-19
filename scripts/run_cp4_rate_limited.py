"""Rate-limited, resumable CP4 evaluation.

Run from repo root:
    python scripts/run_cp4_rate_limited.py

Saves each real response to outputs/ after every individual attack.
No automatic retries: on a provider quota error, stop and keep completed rows.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from agents.agent import create_red_agent_default
from agents.guards_agent import create_red_agent_advance
from attacks.attacks import (
    adversarial_prompts,
    run_attacks,
    save_attack_results,
    write_run_attack_json,
    attack_result_path,
)

def completed_rows(target: str) -> list[dict]:
    path = attack_result_path(target)
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("results", [])
    valid = {p["id"]: p["input"] for p in adversarial_prompts}
    return [r for r in rows if r.get("id") in valid and r.get("input") == valid[r["id"]]]

def quota_error(row: dict) -> bool:
    error = (row.get("error") or "").casefold()
    return any(token in error for token in (
        "429", "resource_exhausted", "quota", "rate limit", "too many requests",
    ))

async def run_target(target: str, factory, delay: float, max_new: int | None = None) -> bool:
    rows = completed_rows(target)
    done_ids = {r["id"] for r in rows if not r.get("error")}
    if len(done_ids) == len(adversarial_prompts):
        print(f"[{target}] Already completed {len(done_ids)} attacks; skipping.")
        return True

    # Create target once, to keep the same NeMo rails and plugin stack.
    agent, runner = factory()
    executed = 0
    for attack in adversarial_prompts:
        if attack["id"] in done_ids:
            continue
        if max_new is not None and executed >= max_new:
            break
        print(f"[{target}] Running attack {attack['id']}/{len(adversarial_prompts)}", flush=True)
        one = await run_attacks(
            agent, runner, prompts=[attack], target_name=target, save_json=False,
        )
        row = one[0]
        executed += 1
        rows = [r for r in rows if r.get("id") != attack["id"]] + one
        rows.sort(key=lambda item: item.get("id", 0))
        write_run_attack_json(rows, target_name=target)
        if quota_error(row):
            print(f"[{target}] Quota/rate-limit error; stopped without retry.", flush=True)
            return False
        if row.get("error"):
            print(f"[{target}] Non-quota error recorded: {row['error'][:250]}", flush=True)
        if attack["id"] != adversarial_prompts[-1]["id"] and (max_new is None or executed < max_new):
            print(f"[{target}] Cooling down {delay:.0f}s", flush=True)
            await asyncio.sleep(delay)
    return True

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--red-delay", type=float, default=27)
    parser.add_argument("--advance-delay", type=float, default=45)
    parser.add_argument("--only", choices=("red", "advance", "both"), default="both")
    parser.add_argument("--max-new", type=int, default=None, help="Maximum new attempts per target in this run")
    args = parser.parse_args()
    if args.only in ("red", "both"):
        ok = await run_target("red_default", create_red_agent_default, args.red_delay, args.max_new)
        if not ok:
            print("Stopped after saving partial Red results, no further API traffic.")
            return
    if args.only in ("advance", "both"):
        await run_target("red_advance", create_red_agent_advance, args.advance_delay, args.max_new)
    red = completed_rows("red_default")
    advance = completed_rows("red_advance")
    save_attack_results(unsafe_results=red, guards_results=advance, ai_attacks=None)
    print(f"Saved aggregate CP4 evidence: Red {len(red)}; Red Advance {len(advance)}")

if __name__ == "__main__":
    asyncio.run(main())
