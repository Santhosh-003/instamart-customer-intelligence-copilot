import csv
import json
from pathlib import Path

from part_d import run_agent

PROJECT_ROOT = Path(__file__).resolve().parent
SCENARIOS_PATH = PROJECT_ROOT / "data" / "raw" / "agent_scenarios.csv"
LEDGER_PATH = PROJECT_ROOT / "outputs" / "refund_ledger.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "agent_runs.jsonl"


def load_ledger():
    """Read previously recorded simulated refund actions."""
    if not LEDGER_PATH.exists():
        return []

    records = []
    with LEDGER_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))
    return records


def main():
    if OUTPUT_PATH.exists():
        print(f"STOP: {OUTPUT_PATH} already exists.")
        print("Review or back up the existing file before replacing it.")
        return

    with SCENARIOS_PATH.open(
        "r", encoding="utf-8", newline=""
    ) as file:
        scenarios = list(csv.DictReader(file))

    ledger = load_ledger()

    recorded_orders = {
        record["order_id"]
        for record in ledger
        if record.get("tool_name") in {
            "issue_refund",
            "escalate_to_human",
        }
    }

    # Check order IDs before running any scenario.
    # This prevents repeating scenarios for previously recorded orders.
    for scenario in scenarios:
        message = scenario["message"]

        for order_id in recorded_orders:
            if order_id in message:
                print(
                    f"STOP: {scenario['scenario_id']} mentions "
                    f"{order_id}, which already appears in the ledger."
                )
                print("No scenarios were run and no files were changed.")
                return

    print("Pre-check passed. Running scenarios...")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("x", encoding="utf-8") as output_file:
        for scenario in scenarios:
            scenario_id = scenario["scenario_id"]

            result = run_agent(
                user_message=scenario["message"],
                scenario_id=scenario_id,
                run_id=f"SCENARIO_{scenario_id}",
            )

            record = {
                "scenario_id": scenario_id,
                "expected_outcome": scenario["expected_outcome"],
                "expected_refund_inr": float(
                    scenario["expected_refund_inr"]
                ),
                "status": result["status"],
                "final_answer": result["final_answer"],
                "llm_calls": result["llm_calls"],
                "tokens_used": result["tokens_used"],
                "steps": result["steps"],
            }

            output_file.write(json.dumps(record, default=str) + "\n")
            output_file.flush()

            print(
                scenario_id,
                "| status:", result["status"],
                "| calls:", result["llm_calls"],
                "| tokens:", result["tokens_used"],
            )

    print(f"\nSaved scenario logs to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()