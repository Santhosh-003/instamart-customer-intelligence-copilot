"""Part C: Support Ticket Triage with Guardrails."""

import json
import random
import re
import time
from pathlib import Path

import mock_llm
import pandas as pd

def est_tokens(text):
    """Estimate the number of input tokens in a ticket."""
    return len(text.split())

MAX_INPUT_TOKENS=400

INPUT_FILE = "data/raw/tickets.csv"


def redact_pii(text):
    """Redact email addresses, 10-digit Indian phone numbers, and order IDs."""
    if pd.isna(text):
        return ""

    text = str(text)

    # Redact email addresses.
    text = re.sub(
        r"[\w.+-]+@[\w-]+\.[\w.]+",
        "[EMAIL]",
        text,
    )

    # Redact 10-digit Indian mobile numbers.
    text = re.sub(
        r"\b[6-9]\d{9}\b",
        "[PHONE]",
        text,
    )

    # Redact order IDs such as ORD-105035.
    text = re.sub(
        r"ORD-\d{6}",
        "[ORDER_ID]",
        text,
    )

    return text


def main():
    tickets = pd.read_csv(INPUT_FILE)

    tickets["text"] = tickets["text"].fillna("").astype(str)
    tickets["redacted_text"] = tickets["text"].apply(redact_pii)

    # Count the original PII patterns before redaction.
    original_text = " ".join(tickets["text"].tolist())

    email_count = len(
        re.findall(r"[\w.+-]+@[\w-]+\.[\w.]+", original_text)
    )
    phone_count = len(
        re.findall(r"\b[6-9]\d{9}\b", original_text)
    )
    order_id_count = len(
        re.findall(r"ORD-\d{6}", original_text)
    )

    # Check that the original patterns no longer remain in redacted text.
    redacted_text = " ".join(tickets["redacted_text"].tolist())

    remaining_emails = len(
        re.findall(r"[\w.+-]+@[\w-]+\.[\w.]+", redacted_text)
    )
    remaining_phones = len(
        re.findall(r"\b[6-9]\d{9}\b", redacted_text)
    )
    remaining_order_ids = len(
        re.findall(r"ORD-\d{6}", redacted_text)
    )

    print("=" * 55)
    print("PART C — PII REDACTION")
    print("=" * 55)
    print(f"Tickets loaded: {len(tickets)}")
    print(f"Email addresses redacted: {email_count}")
    print(f"Phone numbers redacted: {phone_count}")
    print(f"Order IDs redacted: {order_id_count}")
    print(
        "Total PII patterns redacted:",
        email_count + phone_count + order_id_count,
    )
    print()
    print(f"Emails remaining: {remaining_emails}")
    print(f"Phone numbers remaining: {remaining_phones}")
    print(f"Order IDs remaining: {remaining_order_ids}")

    if remaining_emails == 0 and remaining_phones == 0 and remaining_order_ids == 0:
        print("\nPII REDACTION CHECK: PASS")
    else:
        print("\nPII REDACTION CHECK: FAIL")

    print("\nExample before redaction:")
    example = tickets.loc[tickets["text"].str.len().gt(0), "text"].iloc[0]
    print(example)

    print("\nExample after redaction:")
    print(redact_pii(example))

    # Save the redacted dataset.
    from pathlib import Path

    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    tickets.drop(columns=["text"]).assign(
        text=tickets["redacted_text"]
    ).drop(columns=["redacted_text"]).to_csv(
        output_dir / "tickets_redacted.csv",
        index=False,
    )

    print("\nSaved redacted tickets to outputs/tickets_redacted.csv")


VALID_CATEGORIES = {"billing", "delivery", "technical", "fraud"}


ZERO_SHOT_SYSTEM = """
CLASSIFY each customer support ticket into exactly one category:
billing, delivery, technical, or fraud.

Return JSON only.
Use this format: {"category": "billing"}

billing: payments, invoices, charges, or refunds.
delivery: late, missing, or delayed deliveries.
technical: app errors, login problems, or technical failures.
fraud: unauthorized transactions or suspicious activity.
"""


FEW_SHOT_SYSTEM = """
CLASSIFY each customer support ticket into exactly one category:
billing, delivery, technical, or fraud.

Return JSON only.
Use this format: {"category": "billing"}

billing: payments, invoices, charges, or refunds.
delivery: late, missing, or delayed deliveries.
technical: app errors, login problems, or technical failures.
fraud: unauthorized transactions or suspicious activity.

EXAMPLES:
Ticket: My invoice shows the wrong amount.
{"category": "billing"}

Ticket: My delivery has not arrived.
{"category": "delivery"}

Ticket: The app crashes when I log in.
{"category": "technical"}

Ticket: I did not authorize this transaction.
{"category": "fraud"}
"""


def parse_classification(response_text):
    """Extract and validate a category from the model response."""
    try:
        result = json.loads(response_text)
    except (json.JSONDecodeError, TypeError):
        return None

    if not isinstance(result, dict):
        return None

    category = result.get("category")

    if not isinstance(category, str):
        return None

    category = category.strip().lower()

    if category not in VALID_CATEGORIES:
        return None

    return category


def classify_ticket(ticket_text, ticket_id="UNKNOWN", few_shot=False):
    """Classify a ticket with retry handling and safe fallback."""

    system_prompt = FEW_SHOT_SYSTEM if few_shot else ZERO_SHOT_SYSTEM

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": ticket_text},
    ]

    # Verify that the user message contains no unredacted PII.
    assert not re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", ticket_text)
    assert not re.search(r"\b[6-9]\d{9}\b", ticket_text)
    assert not re.search(r"ORD-\d{6}", ticket_text)

    response = call_model_with_retries(
        messages,
        ticket_id,
        max_retries=3,
    )

    response_text = response.choices[0].message.content

    try:
        parsed = json.loads(response_text)
    except (json.JSONDecodeError, TypeError):
        parsed = None

    if parsed is None:
        category = keyword_fallback(response_text)

        if category is not None:
            method = "classified_keyword"
        else:
            method = "flagged_invalid_output"

    elif (
        isinstance(parsed, dict)
        and isinstance(parsed.get("category"), str)
        and parsed["category"].strip().lower() in VALID_CATEGORIES
    ):
        category = parsed["category"].strip().lower()
        method = "classified_json"

    else:
        category = None
        method = "flagged_invalid_output"

    return {
        "category": category,
        "response": response_text,
        "method": method,
        "prompt_type": "few_shot" if few_shot else "zero_shot",
        "total_tokens": response.usage.total_tokens,
    }


def keyword_fallback(response_text):
    """Accept a non-JSON response only when exactly one valid category appears."""

    words = re.findall(r"[a-z]+", response_text.lower())

    allowed_categories = set(VALID_CATEGORIES)
    found_categories = allowed_categories.intersection(words)

    if len(found_categories) == 1:
        return next(iter(found_categories))

    return None

def call_model_with_retries(messages, ticket_id, max_retries=3):
    """Retry temporary mock LLM failures up to three times."""

    for attempt in range(max_retries + 1):
        try:
            return mock_llm.chat(messages)

        except mock_llm.TransientLLMError:
            if attempt == max_retries:
                print(
                    f"Ticket {ticket_id}: failed after "
                    f"{max_retries} retries."
                )
                raise

            wait_seconds = (
                0.05 * (2 ** attempt) * random.uniform(0.9, 1.15)
            )

            print(
                f"Ticket {ticket_id}: retry {attempt + 1}/"
                f"{max_retries}, waiting {wait_seconds:.3f}s"
            )

            time.sleep(wait_seconds)

from functools import lru_cache


@lru_cache(maxsize=1)

def get_premium_customer_ids():
    """Load the IDs of premium customers from the segmentation output."""

    segments = pd.read_csv("outputs/customer_segments.csv")

    premium_mask = (
        segments["is_premium"]
        .astype(str)
        .str.lower()
        .isin(["true", "1", "yes"])
    )

    return set(
        segments.loc[premium_mask, "customer_id"].astype(str)
    )


def route_ticket(category, customer_id, ticket_text):
    """Route tickets safely using category and customer segment."""

    # Empty or excessively long tickets require human review.
    if not ticket_text or not ticket_text.strip():
        return "human_review", "NORMAL"

    if len(ticket_text.split()) > 500:
        return "human_review", "NORMAL"

    # Invalid classifications require human review.
    if category not in VALID_CATEGORIES:
        return "human_review", "NORMAL"

    # Fraud takes priority over customer segment.
    if category == "fraud":
        return "security_team", "HIGH"

    # Premium customers receive priority support.
    premium_ids = get_premium_customer_ids()

    if str(customer_id) in premium_ids:
        return "premium_desk", "HIGH"

    # All other valid tickets go to the normal support queue.
    return "support_queue", "NORMAL"


def evaluate_tickets(few_shot=False, token_budget=None):
    """Classify tickets safely, enforce the token budget, and report metrics."""

    tickets = pd.read_csv(INPUT_FILE)
    tickets["text"] = tickets["text"].fillna("").astype(str)

    # Redact PII before any model call.
    tickets["redacted_text"] = tickets["text"].apply(redact_pii)

    # Reset mock-model behaviour before each independent run.
    mock_llm.reset()

    results = []
    tokens_used = 0

    for _, row in tickets.iterrows():
        ticket_id = row["ticket_id"]
        customer_id = row["customer_id"]
        ticket_text = row["redacted_text"]
        true_category = row["true_category"]

        prompt_type = "few_shot" if few_shot else "zero_shot"

        # Empty tickets must never reach the model.
        if not ticket_text or not ticket_text.strip():
            results.append({
                "ticket_id": ticket_id,
                "customer_id": customer_id,
                "true_category": true_category,
                "predicted_category": None,
                "method": "skipped_empty",
                "prompt_type": prompt_type,
                "route": "human_review",
                "priority": "NORMAL",
                "total_tokens": 0,
            })
            continue

        # Long tickets must never reach the model.
        if est_tokens(ticket_text) > MAX_INPUT_TOKENS:
            results.append({
                "ticket_id": ticket_id,
                "customer_id": customer_id,
                "true_category": true_category,
                "predicted_category": None,
                "method": "flagged_long",
                "prompt_type": prompt_type,
                "route": "human_review",
                "priority": "NORMAL",
                "total_tokens": 0,
            })
            continue

        # Never call the model after the token budget is exhausted.
        if token_budget is not None and tokens_used >= token_budget:
            results.append({
                "ticket_id": ticket_id,
                "customer_id": customer_id,
                "true_category": true_category,
                "predicted_category": None,
                "method": "budget_exhausted",
                "prompt_type": prompt_type,
                "route": "human_review",
                "priority": "NORMAL",
                "total_tokens": 0,
            })
            continue

        try:
            prediction = classify_ticket(
                ticket_text,
                ticket_id=ticket_id,
                few_shot=few_shot,
            )
        except mock_llm.TransientLLMError:
            prediction = {
                "category": None,
                "method": "flagged_invalid_output",
                "prompt_type": prompt_type,
                "total_tokens": 0,
            }

        ticket_tokens = prediction.get("total_tokens", 0)
        tokens_used += ticket_tokens

        category = prediction.get("category")

        # Apply safe routing rules.
        route, priority = route_ticket(
            category,
            customer_id,
            ticket_text,
        )

        results.append({
            "ticket_id": ticket_id,
            "customer_id": customer_id,
            "true_category": true_category,
            "predicted_category": category,
            "method": prediction.get(
                "method", "flagged_invalid_output"
            ),
            "prompt_type": prediction.get("prompt_type", prompt_type),
            "route": route,
            "priority": priority,
            "total_tokens": ticket_tokens,
        })

    result_df = pd.DataFrame(results)

    from pathlib import Path

    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    mode = "few_shot" if few_shot else "zero_shot"

    result_df.to_csv(
        output_dir / f"triage_{mode}_results.csv",
        index=False,
    )

    # Evaluate tickets that received a valid category.
    eligible = result_df[
        result_df["true_category"].isin(VALID_CATEGORIES)
        & result_df["predicted_category"].notna()
    ]

    correct = (
        eligible["true_category"] == eligible["predicted_category"]
    ).sum()

    accuracy = correct / len(eligible) if len(eligible) else 0.0

    classified_count = len(eligible)
    total_classified_tokens = int(eligible["total_tokens"].sum())

    tokens_per_classified_ticket = (
        total_classified_tokens / classified_count
        if classified_count
        else 0.0
    )

    print("\n" + "=" * 55)
    print(f"PART C — {mode.upper()} EVALUATION")
    print("=" * 55)
    print(f"Total tickets: {len(result_df)}")
    print(f"Eligible classified tickets: {classified_count}")
    print(f"Correct predictions: {correct}")
    print(f"Accuracy: {accuracy:.1%}")
    print(f"Total tokens used: {tokens_used}")
    print(f"Tokens per classified ticket: {tokens_per_classified_ticket:.2f}")

    if token_budget is not None:
        print(f"Token budget: {token_budget}")
        print(f"Budget exhausted: {tokens_used >= token_budget}")
        print(
            "Tickets blocked by budget:",
            int((result_df["method"] == "budget_exhausted").sum()),
        )

    print("\nClassification methods:")
    print(result_df["method"].value_counts().to_string())

    print("\nRouting destinations:")
    print(result_df["route"].value_counts().to_string())

    if not eligible.empty:
        print("\nConfusion table (actual rows, predicted columns):")
        print(
            pd.crosstab(
                eligible["true_category"],
                eligible["predicted_category"],
            ).to_string()
        )

    print(f"\nSaved: outputs/triage_{mode}_results.csv")

    return result_df

def run_budget_demo(token_limit=5000):
    """Demonstrate safe ticket triage under a limited token budget."""

    from pathlib import Path

    tickets = pd.read_csv(INPUT_FILE)
    tickets["text"] = tickets["text"].fillna("").astype(str)
    tickets["redacted_text"] = tickets["text"].apply(redact_pii)

    mock_llm.reset()

    results = []
    tokens_used = 0

    for _, row in tickets.iterrows():
        ticket_id = row["ticket_id"]
        customer_id = row["customer_id"]
        ticket_text = row["redacted_text"]
        true_category = row["true_category"]

        # Empty tickets must never reach the model.
        if not ticket_text or not ticket_text.strip():
            results.append({
                "ticket_id": ticket_id,
                "customer_id": customer_id,
                "true_category": true_category,
                "predicted_category": None,
                "method": "skipped_empty",
                "route": "human_review",
                "priority": "NORMAL",
                "total_tokens": 0,
            })
            continue

        # Long tickets must never reach the model.
        if est_tokens(ticket_text) > MAX_INPUT_TOKENS:
            results.append({
                "ticket_id": ticket_id,
                "customer_id": customer_id,
                "true_category": true_category,
                "predicted_category": None,
                "method": "flagged_long",
                "route": "human_review",
                "priority": "NORMAL",
                "total_tokens": 0,
            })
            continue

        # Do not call the model after the budget is exhausted.
        if tokens_used >= token_limit:
            results.append({
                "ticket_id": ticket_id,
                "customer_id": customer_id,
                "true_category": true_category,
                "predicted_category": None,
                "method": "budget_exhausted",
                "route": "human_review",
                "priority": "NORMAL",
                "total_tokens": 0,
            })
            continue

        try:
            prediction = classify_ticket(
                ticket_text,
                ticket_id=ticket_id,
                few_shot=False,
            )

            ticket_tokens = prediction["total_tokens"]

            # Do not accept a response that would exceed the budget.
            if tokens_used + ticket_tokens > token_limit:
                results.append({
                    "ticket_id": ticket_id,
                    "customer_id": customer_id,
                    "true_category": true_category,
                    "predicted_category": None,
                    "method": "budget_exhausted",
                    "route": "human_review",
                    "priority": "NORMAL",
                    "total_tokens": 0,
                })
                continue

            tokens_used += ticket_tokens
            category = prediction["category"]

            route, priority = route_ticket(
                category,
                customer_id,
                ticket_text,
            )

            results.append({
                "ticket_id": ticket_id,
                "customer_id": customer_id,
                "true_category": true_category,
                "predicted_category": category,
                "method": prediction["method"],
                "route": route,
                "priority": priority,
                "total_tokens": ticket_tokens,
            })

        except mock_llm.TransientLLMError:
            results.append({
                "ticket_id": ticket_id,
                "customer_id": customer_id,
                "true_category": None,
                "predicted_category": None,
                "method": "flagged_invalid_output",
                "route": "human_review",
                "priority": "NORMAL",
                "total_tokens": 0,
            })

    result_df = pd.DataFrame(results)

    Path("outputs").mkdir(parents=True, exist_ok=True)
    result_df.to_csv("outputs/triage_budget_demo.csv", index=False)

    exhausted_count = (
        result_df["method"] == "budget_exhausted"
    ).sum()

    print("\n" + "=" * 55)
    print("PART C — TOKEN BUDGET DEMONSTRATION")
    print("=" * 55)
    print(f"Token budget: {token_limit}")
    print(f"Tokens used: {tokens_used}")
    print(f"Tickets blocked by budget: {exhausted_count}")
    print(f"Budget exhausted: {exhausted_count > 0}")
    print(f"Tickets saved: {len(result_df)}")
    print("Saved: outputs/triage_budget_demo.csv")

    return result_df

if __name__ == "__main__":
    main()

    # Run zero-shot evaluation for comparison.
    zero_shot_results = evaluate_tickets(few_shot=False)

    # Run few-shot evaluation for comparison.
    few_shot_results = evaluate_tickets(
    few_shot=True,
    token_budget=100000,
)

    # Build the final assessment CSV using the few-shot results.
    from pathlib import Path

    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load customer segment information.
    segments = pd.read_csv(output_dir / "customer_segments.csv")
    segments["customer_id"] = segments["customer_id"].astype(str)

    segment_lookup = segments.set_index("customer_id")["segment_name"].to_dict()
    premium_lookup = {
        str(row["customer_id"]): str(row["is_premium"]).strip().lower()
        in {"true", "1", "yes"}
        for _, row in segments.iterrows()
    }

    final_rows = []

    for _, row in few_shot_results.iterrows():
        ticket_id = row["ticket_id"]
        customer_id = str(row["customer_id"])
        status = row["method"]
        category = row["predicted_category"]

        # Apply the assessment's routing rules in the required order.
        if status in {
            "skipped_empty",
            "flagged_long",
            "flagged_invalid_output",
            "budget_exhausted",
        }:
            route, priority = "human_review", "NORMAL"
        elif category == "fraud":
            route, priority = "security_team", "HIGH"
        elif premium_lookup.get(customer_id, False):
            route, priority = "premium_desk", "HIGH"
        else:
            route, priority = "support_queue", "NORMAL"

        final_rows.append({
            "ticket_id": ticket_id,
            "customer_id": row["customer_id"],
            "status": status,
            "category": category if pd.notna(category) else "",
            "segment_name": segment_lookup.get(customer_id, ""),
            "priority": priority,
            "route": route,
            "tokens_used": int(row["total_tokens"]),
        })

    final_results = pd.DataFrame(
        final_rows,
        columns=[
            "ticket_id",
            "customer_id",
            "status",
            "category",
            "segment_name",
            "priority",
            "route",
            "tokens_used",
        ],
    )

    final_file = output_dir / "triage_results.csv"
    final_results.to_csv(final_file, index=False)

    print(f"\nFinal results saved to: {final_file}")
    print(f"Total tickets in final file: {len(final_results)}")
    print("Final CSV columns:", ", ".join(final_results.columns))
    print("\nFinal status counts:")
    print(final_results["status"].value_counts().to_string())
    print("\nFinal routing destinations:")
    print(final_results["route"].value_counts().to_string())