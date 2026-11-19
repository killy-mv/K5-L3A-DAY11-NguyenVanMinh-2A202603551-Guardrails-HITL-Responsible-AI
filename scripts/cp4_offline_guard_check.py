"""Offline Red Advance L1 guardrail verification (no model/API traffic)."""
from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from attacks.attacks import adversarial_prompts
from agents.guards_agent import RedAdvanceGuardPlugin

class NoNetworkNemo:
    async def check_input(self, text):
        raise AssertionError("Offline test must not call NeMo or Gemini")

async def main():
    plugin = RedAdvanceGuardPlugin(NoNetworkNemo(), runtime="adk")
    rows = []
    for attack in adversarial_prompts:
        try:
            response = await plugin.guard_input(attack["input"])
            rows.append({
                "id": attack["id"],
                "category": attack["category"],
                "blocked_by_regex": response is not None,
                "message": response or "",
                "model_called": False,
                "nemo_called": False,
            })
        except AssertionError:
            rows.append({
                "id": attack["id"],
                "category": attack["category"],
                "blocked_by_regex": False,
                "message": "Requires NeMo/LLM evaluation (not run)",
                "model_called": False,
                "nemo_called": False,
            })
    result = {
        "type": "offline_red_advance_regex_precheck",
        "note": "NOT a completed end-to-end Red Advance evaluation",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checks": rows,
    }
    dest = ROOT / "outputs" / "red_advance_offline_precheck.json"
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved:", dest)
    print("Regex blocked:", sum(r["blocked_by_regex"] for r in rows), "/", len(rows))
    print("Gemini/NeMo API requests: 0")

if __name__ == "__main__":
    asyncio.run(main())
