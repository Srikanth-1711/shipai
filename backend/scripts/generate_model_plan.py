#!/usr/bin/env python3
"""End-to-end: detect environment + negotiate models → ~/.shipai/model_plan.json"""
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.install.model_plan import PLAN_PATH, load_model_plan
from app.install.model_negotiator import build_and_save_model_plan


async def main() -> None:
    print("ShipAI model plan (genuine inventory + matrix estimates)\n")
    plan = await build_and_save_model_plan()
    print(f"Written: {PLAN_PATH}\n")
    print(json.dumps(plan.to_dict(), indent=2))
    print("\n--- summary ---")
    print(f"Runtime: {plan.primary_runtime} @ {plan.runtime_base_url or 'n/a'}")
    if plan.embedding:
        print(f"Embeddings: {plan.embedding.model} ({plan.embedding.confidence})")
    for node, asn in plan.node_assignments.items():
        print(
            f"  {node}: {asn.model or '(none)'} "
            f"[{asn.status}, {asn.confidence}, score={asn.quality_score}]"
        )
    if plan.models_to_download:
        print(f"\nSuggested downloads: {', '.join(plan.models_to_download)}")
    if plan.warnings:
        print("\nWarnings:")
        for w in plan.warnings:
            print(f"  ! {w}")


if __name__ == "__main__":
    asyncio.run(main())
