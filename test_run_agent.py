import json
from pathlib import Path
from part_d import resume_agent

ledger_path = Path("outputs/refund_ledger.jsonl")

def count_order_entries(order_id):
    if not ledger_path.exists():
        return 0

    count = 0
    for line in ledger_path.read_text(encoding="utf-8").splitlines():
        entry = json.loads(line)
        if entry.get("order_id") == order_id:
            count += 1
    return count

order_id = "ORD-100048"
before = count_order_entries(order_id)

print("Ledger entries before resume:", before)

result = resume_agent("RESUME_DEMO")

print("Status:", result["status"])
print("Final answer:", result["final_answer"])
print("Tools across complete run:", [
    step["tool_name"] for step in result["steps"]
])
print("LLM calls:", result["llm_calls"])
print("Tokens used:", result["tokens_used"])

after = count_order_entries(order_id)
print("Ledger entries after resume:", after)
print("New ledger entries:", after - before)