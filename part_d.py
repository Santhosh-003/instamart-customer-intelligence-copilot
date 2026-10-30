from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path(__file__).resolve().parent
POLICY_DIR = PROJECT_ROOT / "data" / "policies"


def chunk_policy_file(file_path, chunk_size=40, overlap=10):
    """Split a policy document into overlapping word chunks."""
    text = file_path.read_text(encoding="utf-8")
    words = text.split()
    chunks = []

    if not words:
        return chunks

    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))

        chunks.append({
            "source": file_path.name,
            "text": " ".join(words[start:end]),
        })

        if end == len(words):
            break

        start += chunk_size - overlap

    return chunks


def load_policy_chunks():
    """Load and chunk every .txt policy document."""
    all_chunks = []

    for file_path in sorted(POLICY_DIR.glob("*.txt")):
        all_chunks.extend(chunk_policy_file(file_path))

    return all_chunks

def build_policy_search():
    """Build a TF-IDF search index for all policy chunks."""
    chunks = load_policy_chunks()

    if not chunks:
        raise ValueError("No policy chunks found.")

    vectorizer = TfidfVectorizer()
    chunk_texts = [chunk["text"] for chunk in chunks]
    document_vectors = vectorizer.fit_transform(chunk_texts)

    return chunks, vectorizer, document_vectors


def retrieve(query, k=3, search_index=None):
    """Return the k most relevant policy chunks for a query."""
    if search_index is None:
        search_index = build_policy_search()

    chunks, vectorizer, document_vectors = search_index

    query_vector = vectorizer.transform([query])
    scores = cosine_similarity(query_vector, document_vectors).flatten()

    ranked_indices = scores.argsort()[::-1][:k]

    results = []
    for index in ranked_indices:
        results.append({
            "source": chunks[index]["source"],
            "score": float(scores[index]),
            "text": chunks[index]["text"],
        })

    return results

def search_policy(query):
    """Search the policy documents for relevant information."""

    results = retrieve(query, k=3)

    return {
        "results": results
    }

def answer(question, search_index=None):
    """Answer a policy question using retrieved context and mock_llm."""
    import mock_llm

    if search_index is None:
        search_index = build_policy_search()

    retrieved_chunks = retrieve(
        question,
        k=3,
        search_index=search_index,
    )

    context_parts = []

    for chunk in retrieved_chunks:
        context_parts.append(
            f"[{chunk['source']}]\n{chunk['text']}"
        )

    context = "\n\n".join(context_parts)

    messages = [
        {
            "role": "system",
            "content": (
                "POLICY ASSISTANT: Answer questions using only the "
                "provided policy context. Do not invent policy rules. "
                "If the context does not contain the answer, reply exactly: "
                "'I could not find that information in the policy documents.'"
            ),
        },
        {
            "role": "user",
            "content": (
                f"CONTEXT:\n{context}\n\n"
                f"QUESTION:\n{question}"
            ),
        },
    ]

    response = mock_llm.chat(messages)
    answer_text = response.choices[0].message.content

    return {
        "answer": answer_text,
        "sources": list(dict.fromkeys(
            chunk["source"] for chunk in retrieved_chunks
        )),
        "retrieved_chunks": retrieved_chunks,
    }
def evaluate_policy_assistant(overlap=10):
    """Evaluate retrieval and answer accuracy using policy_eval.csv."""
    import csv

    policy_eval_path = PROJECT_ROOT / "data" / "raw" / "policy_eval.csv"

    # Create policy chunks using the requested overlap.
    all_chunks = []

    for file_path in sorted(POLICY_DIR.glob("*.txt")):
        all_chunks.extend(
            chunk_policy_file(
                file_path,
                chunk_size=40,
                overlap=overlap,
            )
        )

    if not all_chunks:
        raise ValueError("No policy chunks found.")

    # Build the TF-IDF search index.
    vectorizer = TfidfVectorizer()
    chunk_texts = [chunk["text"] for chunk in all_chunks]
    document_vectors = vectorizer.fit_transform(chunk_texts)

    search_index = (
        all_chunks,
        vectorizer,
        document_vectors,
    )

    # Read the evaluation questions.
    with policy_eval_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        evaluation_rows = list(csv.DictReader(file))

    answerable_total = 0
    retrieval_hits = 0
    fact_hits = 0

    unanswerable_total = 0
    fallback_hits = 0

    failures = []

    for row in evaluation_rows:
        question = row["question"]
        expected_doc = row["expected_doc"].strip()
        expected_fact = row["expected_fact"].strip()

        # CASE 1: The question cannot be answered from the policies.
        if not expected_doc or expected_doc.upper() == "NONE":
            unanswerable_total += 1

            result = answer(
                question,
                search_index=search_index,
            )

            if result["answer"].startswith("I could not find"):
                fallback_hits += 1
            else:
                failures.append({
                    "question": question,
                    "reason": "Expected the unanswerable fallback.",
                    "answer": result["answer"],
                })

            # Move to the next question.
            continue

        # CASE 2: The question has an expected policy answer.
        answerable_total += 1

        retrieved = retrieve(
            question,
            k=3,
            search_index=search_index,
        )

        # Check whether the expected document is in the top three.
        retrieval_ok = any(
            chunk["source"] == expected_doc
            for chunk in retrieved
        )

        if retrieval_ok:
            retrieval_hits += 1
        else:
            failures.append({
                "question": question,
                "reason": (
                    f"Expected document not in top 3: {expected_doc}"
                ),
                "answer": "",
            })

        # Check whether the answer contains the exact expected fact.
        result = answer(
            question,
            search_index=search_index,
        )

        fact_ok = expected_fact in result["answer"]

        if fact_ok:
            fact_hits += 1
        else:
            failures.append({
                "question": question,
                "reason": (
                    f"Expected fact not found: {expected_fact}"
                ),
                "answer": result["answer"],
            })

    # Print the evaluation results.
    print(f"\nEvaluation with overlap={overlap}")
    print(f"Total questions: {len(evaluation_rows)}")

    print(
        f"Answerable retrieval hits: "
        f"{retrieval_hits}/{answerable_total}"
    )

    print(
        f"Expected facts found: "
        f"{fact_hits}/{answerable_total}"
    )

    print(
        f"Unanswerable fallbacks correct: "
        f"{fallback_hits}/{unanswerable_total}"
    )

    print("\nEvaluation failures:")

    if failures:
        for failure in failures:
            print(f"- Question: {failure['question']}")
            print(f"  Reason: {failure['reason']}")
            print(f"  Answer: {failure['answer']}")
    else:
        print("None")

    return {
        "overlap": overlap,
        "total_questions": len(evaluation_rows),
        "answerable_total": answerable_total,
        "retrieval_hits": retrieval_hits,
        "fact_hits": fact_hits,
        "unanswerable_total": unanswerable_total,
        "fallback_hits": fallback_hits,
        "failures": failures,
    }
def get_order(order_id):
    """Find an order in the Instamart database."""

    import sqlite3

    database_path = PROJECT_ROOT / "data" / "instamart.db"

    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row

    try:
        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT order_id, status, order_amount, delivery_minutes
            FROM orders
            WHERE order_id = ?
            """,
            (order_id,),
        )

        row = cursor.fetchone()

        if row is None:
            return {"error": "order not found"}

        return {
            "order_id": row["order_id"],
            "status": row["status"],
            "order_amount": row["order_amount"],
            "delivery_minutes": row["delivery_minutes"],
        }

    finally:
        connection.close()
def calculate_refund(order_amount, delivery_minutes):
    """Calculate the refund for a late delivery."""

    order_amount = float(order_amount)
    delivery_minutes = float(delivery_minutes)

    if delivery_minutes > 45:
        refund_amount = order_amount
    else:
        refund_amount = 0.0

    return {
        "refund_inr": round(refund_amount, 2)
    }
def record_refund_action(order_id, refund_inr, tool_name):
    """Record a simulated refund action in the JSONL ledger."""

    import json
    from datetime import datetime, timezone

    ledger_path = PROJECT_ROOT / "outputs" / "refund_ledger.jsonl"

    # Ensure the outputs folder exists.
    ledger_path.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool_name": tool_name,
        "order_id": order_id,
        "refund_inr": round(float(refund_inr), 2),
    }

    # Add one JSON record without overwriting previous records.
    with ledger_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record) + "\n")

    return {"refund_inr": round(float(refund_inr), 2)}


def issue_refund(order_id, refund_inr):
    """Record a simulated refund that has been approved."""

    return record_refund_action(
        order_id,
        refund_inr,
        "issue_refund",
    )


def escalate_to_human(order_id, refund_inr):
    """Record a refund request requiring human approval."""

    return record_refund_action(
        order_id,
        refund_inr,
        "escalate_to_human",
    )
TOOLS_SPEC = [
    {
        "name": "get_order",
        "description": "Look up an order by its order ID.",
        "parameters": {
            "order_id": "string"
        },
    },
    {
        "name": "search_policy",
        "description": "Search the policy documents for relevant information.",
        "parameters": {
            "query": "string"
        },
    },
    {
        "name": "calculate_refund",
        "description": "Calculate the refund amount for a late delivery.",
        "parameters": {
            "order_amount": "number",
            "delivery_minutes": "number"
        },
    },
    {
        "name": "issue_refund",
        "description": "Record an approved simulated refund.",
        "parameters": {
            "order_id": "string",
            "refund_inr": "number"
        },
    },
    {
        "name": "escalate_to_human",
        "description": "Record a refund request that needs human approval.",
        "parameters": {
            "order_id": "string",
            "refund_inr": "number"
        },
    },
]


TOOL_MAP = {
    "get_order": get_order,
    "search_policy": search_policy,
    "calculate_refund": calculate_refund,
    "issue_refund": issue_refund,
    "escalate_to_human": escalate_to_human,
}

def save_agent_checkpoint(
    run_id,
    scenario_id,
    messages,
    tokens_used,
    steps,
    llm_calls,
):
    """Save the agent's progress so the run can be resumed."""

    import json

    state_dir = PROJECT_ROOT / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "run_id": run_id,
        "scenario_id": scenario_id,
        "messages": messages,
        "tokens_used": tokens_used,
        "step_number": len(steps),
        "steps": steps,
        "llm_calls": llm_calls,
        "executed_tools": [
            step["tool_name"] for step in steps
        ],
    }

    checkpoint_path = state_dir / f"{run_id}.json"

    with open(checkpoint_path, "w", encoding="utf-8") as file:
        json.dump(checkpoint, file, indent=2, default=str)

    return checkpoint_path

def run_agent(
    user_message,
    scenario_id="MANUAL_TEST",
    max_iterations=6,
    token_limit=6000,
    time_limit_seconds=10,
    run_id=None,
    crash_after_tools=None,
):

    """Run the refund agent using the supplied offline mock LLM."""

    import json
    import time
    import mock_llm

    run_id = run_id or scenario_id
    start_time = time.monotonic()
    messages = [
        {
            "role": "system",
            "content": (
                "AGENT: You are an Instamart refund agent. "
                "Use the supplied tools to check orders and policies. "
                "Only completed orders may receive refunds. "
                "Refunds are allowed only when delivery takes more "
                "than 45 minutes. Refunds above INR 1000 must be "
                "escalated to a human. Never issue real payments."
            ),
        },
        {
            "role": "user",
            "content": user_message,
        },
    ]

    tokens_used = 0
    steps = []
    llm_calls = 0

    for iteration in range(max_iterations):

        # Stop if the time limit has been reached.
        if time.monotonic() - start_time >= time_limit_seconds:
            return {
                "status": "FALLBACK_TIMEOUT",
                "final_answer": (
                    "FALLBACK_TIMEOUT: Request stopped safely. "
                    "Please hand this request to a human support agent."
                ),
                "steps": steps,
                "llm_calls": llm_calls,
                "tokens_used": tokens_used,
            }

        # Ask the mock LLM what to do next.
        response = mock_llm.chat(messages, tools=TOOLS_SPEC)
        llm_calls += 1

        call_tokens = response.usage.total_tokens
        tokens_used += call_tokens

        # Do not execute a requested tool if the token budget is exceeded.
        if tokens_used > token_limit:
            return {
                "status": "FALLBACK_TOKEN_LIMIT",
                "final_answer": (
                    "FALLBACK_TOKEN_LIMIT: Request stopped safely. "
                    "Please hand this request to a human support agent."
                ),
                "steps": steps,
                "llm_calls": llm_calls,
                "tokens_used": tokens_used,
            }

        assistant_message = response.choices[0].message

        # If the model has finished, return its final answer.
        if not assistant_message.tool_calls:
            final_answer = assistant_message.content or (
                "FALLBACK_EMPTY_RESPONSE: Please contact human support."
            )

            status = final_answer.split(":", 1)[0]

            return {
                "status": status,
                "final_answer": final_answer,
                "steps": steps,
                "llm_calls": llm_calls,
                "tokens_used": tokens_used,
            }

        # Add the assistant's tool request to the conversation.
        tool_calls_for_history = []

        for tool_call in assistant_message.tool_calls:
            arguments = tool_call.arguments

            if isinstance(arguments, str):
                arguments = json.loads(arguments)

            tool_calls_for_history.append(
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.name,
                        "arguments": json.dumps(arguments),
                    },
                }
            )

        messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": tool_calls_for_history,
            }
        )

        # Execute each requested tool.
        for tool_call in assistant_message.tool_calls:
            tool_name = tool_call.name
            arguments = tool_call.arguments

            if isinstance(arguments, str):
                arguments = json.loads(arguments)

            if tool_name not in TOOL_MAP:
                observation = {"error": "unknown tool"}
            else:
                try:
                    observation = TOOL_MAP[tool_name](**arguments)
                except Exception as error:
                    observation = {
                        "error": f"{type(error).__name__}: {error}"
                    }

            steps.append(
                {
                    "tool_name": tool_name,
                    "arguments": arguments,
                    "observation": observation,
                }
            )

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_name,
                    "content": json.dumps(observation, default=str),
                }
            )

            save_agent_checkpoint(
               run_id=run_id,
               scenario_id=scenario_id,
               messages=messages,
               tokens_used=tokens_used,
               steps=steps,
               llm_calls=llm_calls,
            )
            if (
               crash_after_tools is not None
               and len(steps) >= crash_after_tools
           ):
               return {
                   "status": "SIMULATED_CRASH",
                   "final_answer": (
                       "SIMULATED_CRASH: Checkpoint saved."
                   ),
                   "steps": steps,
                   "llm_calls": llm_calls,
                   "tokens_used": tokens_used,
               }
    # The model did not finish within the permitted number of iterations.
    return {
        "status": "FALLBACK_MAX_ITERATIONS",
        "final_answer": (
            "FALLBACK_MAX_ITERATIONS: Request stopped safely. "
            "Please hand this request to a human support agent."
        ),
        "steps": steps,
        "llm_calls": llm_calls,
        "tokens_used": tokens_used,
    }

def resume_agent(run_id, max_iterations=6, token_limit=6000, time_limit_seconds=10):
    """Load a saved checkpoint and continue the agent."""

    import json
    import time
    import mock_llm

    checkpoint_path = PROJECT_ROOT / "state" / f"{run_id}.json"

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"No checkpoint found for run ID: {run_id}"
        )

    with open(checkpoint_path, "r", encoding="utf-8") as file:
        checkpoint = json.load(file)

    messages = checkpoint["messages"]
    scenario_id = checkpoint["scenario_id"]
    tokens_used = checkpoint["tokens_used"]
    steps = checkpoint["steps"]
    llm_calls = checkpoint["llm_calls"]

    start_time = time.monotonic()

    for iteration in range(max_iterations):
        if time.monotonic() - start_time >= time_limit_seconds:
            return {
                "status": "FALLBACK_TIMEOUT",
                "final_answer": (
                    "FALLBACK_TIMEOUT: Request stopped safely. "
                    "Please hand this request to a human support agent."
                ),
                "steps": steps,
                "llm_calls": llm_calls,
                "tokens_used": tokens_used,
            }

        response = mock_llm.chat(messages, tools=TOOLS_SPEC)
        llm_calls += 1
        tokens_used += response.usage.total_tokens

        if tokens_used > token_limit:
            return {
                "status": "FALLBACK_TOKEN_LIMIT",
                "final_answer": (
                    "FALLBACK_TOKEN_LIMIT: Request stopped safely. "
                    "Please hand this request to a human support agent."
                ),
                "steps": steps,
                "llm_calls": llm_calls,
                "tokens_used": tokens_used,
            }

        assistant_message = response.choices[0].message

        if not assistant_message.tool_calls:
            final_answer = assistant_message.content or (
                "FALLBACK_EMPTY_RESPONSE: Please contact human support."
            )

            return {
                "status": final_answer.split(":", 1)[0],
                "final_answer": final_answer,
                "steps": steps,
                "llm_calls": llm_calls,
                "tokens_used": tokens_used,
            }

        tool_calls_for_history = []

        for tool_call in assistant_message.tool_calls:
            arguments = tool_call.arguments

            if isinstance(arguments, str):
                arguments = json.loads(arguments)

            tool_calls_for_history.append({
                "id": tool_call.id,
                "type": "function",
                "function": {
                    "name": tool_call.name,
                    "arguments": json.dumps(arguments),
                },
            })

        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": tool_calls_for_history,
        })

        for tool_call in assistant_message.tool_calls:
            tool_name = tool_call.name
            arguments = tool_call.arguments

            if isinstance(arguments, str):
                arguments = json.loads(arguments)

            if tool_name not in TOOL_MAP:
                observation = {"error": "unknown tool"}
            else:
                try:
                    observation = TOOL_MAP[tool_name](**arguments)
                except Exception as error:
                    observation = {
                        "error": f"{type(error).__name__}: {error}"
                    }

            steps.append({
                "tool_name": tool_name,
                "arguments": arguments,
                "observation": observation,
            })

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_name,
                "content": json.dumps(observation, default=str),
            })

            save_agent_checkpoint(
                run_id=run_id,
                scenario_id=scenario_id,
                messages=messages,
                tokens_used=tokens_used,
                steps=steps,
                llm_calls=llm_calls,
            )

    return {
        "status": "FALLBACK_MAX_ITERATIONS",
        "final_answer": (
            "FALLBACK_MAX_ITERATIONS: Request stopped safely. "
            "Please hand this request to a human support agent."
        ),
        "steps": steps,
        "llm_calls": llm_calls,
        "tokens_used": tokens_used,
    }

if __name__ == "__main__":

    chunks = load_policy_chunks()

    print("\nPolicy search test:")
    search_index = build_policy_search()
    results = retrieve(
        "refund policy late delivery",
        k=3,
        search_index=search_index,
    )

    for result in results:
        print(f"\nSource: {result['source']}")
        print(f"Similarity: {result['score']:.4f}")
        print(result["text"])
    print(f"Policy files found: {len(list(POLICY_DIR.glob('*.txt')))}")
    print(f"Total chunks created: {len(chunks)}")
    print("\nChunks per policy file:")

    counts = {}
    for chunk in chunks:
        counts[chunk["source"]] = counts.get(chunk["source"], 0) + 1

    for filename, count in sorted(counts.items()):
        print(f"{filename}: {count}")

    print("\nFirst chunk preview:")
    if chunks:
        print(f"Source: {chunks[0]['source']}")
        print(chunks[0]["text"])
    print("\nPolicy answer test:")
    result = answer(
        "If my completed order arrives more than 45 minutes late, "
        "what refund will I receive?"
    )

    print("Question: If my completed order arrives more than 45 minutes late, what refund will I receive?")
    print("Answer:", result["answer"])
    print("Sources:", result["sources"])
    print("\n--- POLICY EVALUATION: 10-WORD OVERLAP ---")
    evaluate_policy_assistant(overlap=10)

    print("\n--- POLICY EVALUATION: ZERO OVERLAP ---")
    evaluate_policy_assistant(overlap=0)
