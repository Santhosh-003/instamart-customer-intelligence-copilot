import csv
import json
from pathlib import Path

import part_d

ROOT = Path(__file__).resolve().parent
SCENARIOS_PATH = ROOT / "data" / "raw" / "agent_scenarios.csv"
OUTPUT_PATH = ROOT / "outputs" / "agent_runs.jsonl"


def main():
    if OUTPUT_PATH.exists():
        raise SystemExit(
            f"STOP: {OUTPUT_PATH} already exists. Do not overwrite it."
        )
    state_dir = ROOT / "state"
    planned_runs = [
        *(f"SAFE_{scenario_id}.json" for scenario_id in [
            "S1", "S2", "S3", "S4", "S5", "S6", "S7"
        ]),
        "SAFE_TOKEN_LIMIT_DEMO.json",
        "SAFE_TIMEOUT_DEMO.json",
    ]

    conflicts = [
        name for name in planned_runs
        if (state_dir / name).exists()
    ]

    if conflicts:
        raise SystemExit(
            "STOP: These checkpoint files already exist: "
            + ", ".join(conflicts)
            + ". Review them before running."
        )
    with SCENARIOS_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        scenarios = list(csv.DictReader(f))

    # Prevent this runner from recording new refund actions in the real ledger.
    original_issue = part_d.TOOL_MAP["issue_refund"]
    original_escalate = part_d.TOOL_MAP["escalate_to_human"]

    part_d.TOOL_MAP["issue_refund"] = lambda order_id, refund_inr: {
        "refund_inr": round(float(refund_inr), 2),
        "simulation_only": True,
    }
    part_d.TOOL_MAP["escalate_to_human"] = (
        lambda order_id, refund_inr: {
            "refund_inr": round(float(refund_inr), 2),
            "simulation_only": True,
            "status": "ESCALATED",
        }
    )

    records = []

    try:
        for scenario in scenarios:
            result = part_d.run_agent(
                user_message=scenario["message"],
                scenario_id=scenario["scenario_id"],
                run_id=f"SAFE_{scenario['scenario_id']}",
            )
            records.append({
                "scenario_id": scenario["scenario_id"],
                "expected_outcome": scenario["expected_outcome"],
                "expected_refund_inr": float(
                    scenario["expected_refund_inr"]
                ),
                "status": result["status"],
                "final_answer": result["final_answer"],
                "llm_calls": result["llm_calls"],
                "tokens_used": result["tokens_used"],
                "steps": result["steps"],
                "simulation_only": True,
            })
            print(
                scenario["scenario_id"],
                "| status:", result["status"],
                "| calls:", result["llm_calls"],
                "| tokens:", result["tokens_used"],
            )

        # Required token-budget demonstration.
        token_demo = part_d.run_agent(
            user_message=scenarios[0]["message"],
            scenario_id="TOKEN_LIMIT_DEMO",
            run_id="SAFE_TOKEN_LIMIT_DEMO",
            token_limit=200,
        )
        records.append({
            "scenario_id": "TOKEN_LIMIT_DEMO",
            "status": token_demo["status"],
            "final_answer": token_demo["final_answer"],
            "llm_calls": token_demo["llm_calls"],
            "tokens_used": token_demo["tokens_used"],
            "steps": token_demo["steps"],
            "simulation_only": True,
        })
        print("TOKEN_LIMIT_DEMO | status:", token_demo["status"])

        # Required timeout demonstration. A zero-second limit stops immediately.
        timeout_demo = part_d.run_agent(
            user_message=scenarios[0]["message"],
            scenario_id="TIMEOUT_DEMO",
            run_id="SAFE_TIMEOUT_DEMO",
            time_limit_seconds=0,
        )
        records.append({
            "scenario_id": "TIMEOUT_DEMO",
            "status": timeout_demo["status"],
            "final_answer": timeout_demo["final_answer"],
            "llm_calls": timeout_demo["llm_calls"],
            "tokens_used": timeout_demo["tokens_used"],
            "steps": timeout_demo["steps"],
            "simulation_only": True,
        })
        print("TIMEOUT_DEMO | status:", timeout_demo["status"])

        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with OUTPUT_PATH.open("x", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, default=str) + "\n")

        print(f"\nSaved {len(records)} records to {OUTPUT_PATH}")

    finally:
        part_d.TOOL_MAP["issue_refund"] = original_issue
        part_d.TOOL_MAP["escalate_to_human"] = original_escalate


if __name__ == "__main__":
    main()