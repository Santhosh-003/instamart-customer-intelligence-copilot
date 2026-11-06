# Instamart Customer Intelligence & Support Copilot

## Overview

This project analyses Instamart customer and order data to understand customer spending, product performance, customer behaviour, and customer support requirements.

It uses Python, Pandas, SQLite, SQL, and machine learning techniques to clean data, analyse customer behaviour, predict customer lapse, classify support tickets, and develop customer intelligence and support automation features.

### Technologies Used

- Python
- Pandas
- NumPy
- SQLite
- SQL
- Scikit-learn
- Joblib
- Regular expressions
- TF-IDF and cosine similarity (planned for Part D)

## Reproduce

The project requires Python 3.11 or newer and SQLite 3.25 or newer.

Activate the virtual environment and install the required libraries before running the scripts.

Run the supplied data-generation script before cleaning the datasets. Then run the cleaning scripts, build the SQLite database, execute the SQL queries, and run the analysis and machine learning scripts.

The support-ticket classification uses the supplied offline `mock_llm.py` implementation. No external model API or API key is required.

## Part A — Data Foundation: Clean, Store, Query

### Data Cleaning

The raw data contains customer, product, and order information.

The data-cleaning process:

- Removes exact duplicate order records.
- Keeps the latest record for each order ID.
- Standardises customer city and gender values.
- Handles missing and invalid customer ages.
- Handles missing and invalid order amounts.
- Removes pending orders from the cleaned order dataset.
- Preserves missing delivery times for cancelled orders.

### Database

A SQLite database was created at `data/instamart.db`.

It contains three tables:

- `customers`
- `products`
- `orders`

### Data Validation

The database validation checks passed successfully.

- Customers: 1,200
- Products: 40
- Cleaned orders: 12,309
- Completed orders: 11,110
- Cancelled orders: 1,199
- Completed-order revenue: ₹2,998,668.80

### SQL Analytics

Seven SQL queries were implemented in `sql/queries.sql`.

They analyse:

1. Customers with high spending on completed orders above ₹500.
2. Revenue by city and product category.
3. Products priced above the average catalogue price.
4. Customers without completed orders.
5. Customers whose spending exceeds average customer spending.
6. Customer spending rankings within each city.
7. Daily revenue and cumulative revenue.

The SQL queries executed successfully, and all basic SQL checkpoints passed.

### Pandas Cross-Checks

Pandas was used to independently verify completed-order revenue and customer counts by city.

Both checks passed.

Completed-order statistics:

- Mean order amount: ₹269.91
- Median order amount: ₹135.00
- Standard deviation: ₹432.07
- First quartile: ₹64.80
- Third quartile: ₹297.00
- Upper outlier fence: ₹645.30
- Orders identified as statistical outliers: 897

The correlation between order amount and discount percentage was -0.095. The correlation between order amount and quantity was 0.087. These are weak linear relationships and do not establish causation.

## Part B — Customer Segments and Lapse Prediction

### RFM Analysis and Customer Segmentation

Calculated recency, frequency, and monetary value (RFM) for 1,175 customers with at least one completed order.

Applied `StandardScaler` and evaluated K-Means clustering for K values from 2 to 8 using inertia and silhouette scores. Selected K=4, which achieved a silhouette score of approximately 0.526.

The four customer segments are:

- Developing Customers
- Premium Customers
- Frequent Loyalists
- At-Risk Customers

The Premium segment contains 138 customers. The first two PCA components explain approximately 85.2% of the variance in the standardised RFM data.

The customer segmentation results are saved in `outputs/customer_segments.csv`.

### Lapse Prediction

Created a lapse-prediction dataset containing 1,156 customers with completed orders before September 1, 2025.

Customers with no completed orders during September 2025 were labelled as lapsed. The dataset contains:

- Lapsed customers: 443
- Non-lapsed customers: 713

Built features using information available before the September 1 cutoff, including recency, frequency, monetary value, average discount, cancellation rate, average delivery time, favourite product category, city, age, gender, and customer tenure.

Used a stratified 80/20 train-test split with `random_state=42`. Preprocessing was fitted on the training data only.

### Model Evaluation

Tuned Logistic Regression and Random Forest using five-fold cross-validation and F1 scoring.

Random Forest achieved a test F1 score of 0.6554 for the lapsed class, exceeding the required minimum of 0.55.

The selected Random Forest achieved:

- Test accuracy: 73.71%
- Precision for lapsed customers: 65.91%
- Recall for lapsed customers: 65.17%
- F1 score for lapsed customers: 0.6554

The fitted preprocessing object and model were saved together in `outputs/lapse_model.joblib`. Reloading the saved model produced exactly the same test predictions.

### Class-Imbalance Experiment

Oversampled lapsed customers in the training set only, balancing the two classes at 570 rows each.

The balanced Logistic Regression model achieved the following results on the original, unchanged test set:

- Precision: 0.6000
- Recall: 0.8090
- F1 score: 0.6890

Compared with the original model, recall improved while precision decreased.

### Interpretability and Retention

Random Forest's most important features were frequency, monetary value, recency, average delivery time, and cancellation rate.

Saved feature-importance scores and Logistic Regression coefficients for further interpretation.

Generated `outputs/retention_list.csv`, containing the 10 highest predicted lapse-risk Premium customers from the test set, with predicted lapse probabilities and actual lapse labels.

### Part B Limitations

The lapse-prediction results depend on the supplied dataset and the chosen September 2025 labelling window. The retention list is illustrative and should not be treated as a guarantee that a customer will lapse.

## Part C — Support Ticket Triage with Guardrails

### Input Validation and Guardrails

Implemented support-ticket preprocessing using `data/raw/tickets.csv`.

Before making any model call, the pipeline checks ticket text:

- Empty or whitespace-only tickets are marked `skipped_empty`.
- Tickets with estimated input length greater than 400 tokens are marked `flagged_long`.
- Both groups are excluded from model calls and routed to human review.

The dataset contained four empty tickets and four long tickets. These guardrails prevent unsuitable inputs from being sent to the model.

### Personally Identifiable Information Redaction

Redacted sensitive information before sending ticket text to the mock language model. The redaction order is:

1. Email addresses are replaced with `[EMAIL]`.
2. Phone numbers are replaced with `[PHONE]`.
3. Order IDs are replaced with `[ORDER_ID]`.

The redaction process detected:

- Email addresses: 55
- Phone numbers: 91
- Order IDs: 132
- Total replacements: 278

Validation confirmed that no matching unredacted values remained in the processed ticket text.

The redacted data is saved to `outputs/tickets_redacted.csv`.

### Zero-Shot and Few-Shot Classification

Implemented zero-shot and few-shot ticket classification using the supplied offline `mock_llm.py`.

The model classifies eligible tickets into four categories:

- Billing
- Delivery
- Technical
- Fraud

The pipeline includes retry handling for transient model errors, JSON validation, and a keyword-based fallback. Responses that cannot be validated safely are marked `flagged_invalid_output` and routed to human review rather than being assigned a guessed category.

Both runs processed 147 eligible tickets. The other 17 tickets consisted of four empty tickets, four long tickets, and nine invalid model outputs.

#### Classification Results

| Metric | Zero-Shot | Few-Shot |
|---|---:|---:|
| Eligible tickets | 147 | 147 |
| Correct classifications | 126 | 147 |
| Accuracy | 85.7% | 100% |
| Total tokens used | 14,236 | 21,724 |
| Tokens per classified ticket | 91.12 | 139.12 |
| JSON classifications | 135 | 135 |
| Keyword-fallback classifications | 12 | 12 |
| Invalid outputs flagged | 9 | 9 |

Accuracy is calculated against the ground-truth category for the 147 classified tickets. Tickets skipped or flagged for human review are excluded from the accuracy calculation.

The zero-shot run incorrectly classified some fraud and technical tickets as billing. The few-shot run achieved 100% accuracy on the classified tickets in this evaluation.

Few-shot classification used approximately 52.6% more tokens than zero-shot. For this simulated dataset, the improvement in classification accuracy makes the additional token cost worthwhile. These results reflect the supplied mock model and do not guarantee the same performance in production.

### Token Budget Demonstration

Implemented a token-budget check to prevent further model calls when the running token usage reaches the configured budget.

Tickets blocked by the budget are marked `budget_exhausted` and routed to human review.

The budget demonstration is saved to `outputs/triage_budget_demo.csv`. With a budget of 5,000 tokens, the run used 4,921 tokens and blocked 102 tickets from making further model calls.

The final few-shot run used a token budget of 100,000 and produced `outputs/triage_results.csv`, containing all 164 original tickets.

### Routing and Outputs

The final output uses the following routing rules:

- `skipped_empty`, `flagged_long`, `flagged_invalid_output`, and `budget_exhausted` tickets go to `human_review` with NORMAL priority.
- Fraud tickets go to `security_team` with HIGH priority.
- Other eligible tickets from Premium customers go to `premium_desk` with HIGH priority.
- Remaining eligible tickets go to `support_queue` with NORMAL priority.

Customer segment information is taken from `outputs/customer_segments.csv`. Customers not found in that file are treated as non-premium.

The final CSV contains these columns:

`ticket_id`, `customer_id`, `status`, `category`, `segment_name`, `priority`, `route`, `tokens_used`.

Generated outputs:

- `outputs/tickets_redacted.csv`
- `outputs/triage_zero_shot_results.csv`
- `outputs/triage_few_shot_results.csv`
- `outputs/triage_budget_demo.csv`
- `outputs/triage_results.csv`

All model calls use the supplied offline mock implementation. No external model API or API key is required.

### Part C Limitations

The classification results are based on a supplied mock language model and a limited synthetic ticket dataset. High evaluation accuracy does not establish production-level reliability.

PII redaction depends on the specified regular expressions and may not detect every possible form of sensitive information. Invalid model outputs and unsuitable inputs are routed to human review to reduce the risk of incorrect automated decisions.

## Part D — Policy RAG Assistant and Refund Agent

### Overview

Part D implements a policy retrieval assistant and a simulated refund-support agent. The agent uses tools to look up orders, retrieve relevant policies, calculate refunds, issue simulated refunds, and escalate cases to a human. The implementation also includes iteration, token, and timeout limits, along with checkpoint/resume support.

The project uses the supplied offline `mock_llm.py` implementation and does not require a live LLM API key for these demonstrations.

### Agent Scenario Evaluation

The safe scenario runner evaluates the seven supplied scenarios. Refund issuance and escalation are simulated in memory during these runs to avoid adding new actions to the actual refund ledger.

| Scenario | Expected outcome | Observed status | LLM calls | Tokens used |
|---|---|---|---:|---:|
| S1 | REFUND | REFUND | 5 | 1181 |
| S2 | REFUND | REFUND | 5 | 1181 |
| S3 | ESCALATE | ESCALATE | 5 | 1170 |
| S4 | DENIED | DENIED | 4 | 822 |
| S5 | DENIED | DENIED | 3 | 494 |
| S6 | NOT_FOUND | NOT_FOUND | 2 | 173 |
| S7 | FALLBACK | FALLBACK_MAX_ITERATIONS | 6 | 3354 |

### Safety Limit Demonstrations

| Demonstration | Configuration | Observed status |
|---|---|---|
| Token limit | `token_limit=200` | `FALLBACK_TOKEN_LIMIT` |
| Timeout | `time_limit_seconds=0` | `FALLBACK_TIMEOUT` |

The token-budget and timeout demonstrations confirm that the agent can stop with an explicit fallback status when a configured execution limit is reached.

### Checkpoint and Resume

The agent supports saving progress to checkpoint files and resuming a run. A separate `RESUME_DEMO` was previously used to demonstrate a simulated interruption after the first two tool executions and subsequent continuation.

The ordered tools across the interrupted and resumed portions were:

1. `get_order`
2. `search_policy`
3. `calculate_refund`
4. `issue_refund`

The resumed run returned `REFUND: INR 128.25 issued.` The demonstration showed continuation from the saved checkpoint rather than repeating the earlier tool executions.

### Output Files

- `outputs/agent_runs.jsonl` — nine records covering the seven scenarios and the token-limit and timeout demonstrations. Each record includes the scenario ID, status, final answer, LLM call count, token usage, and execution steps.
- `outputs/refund_ledger.jsonl` — append-only JSONL records of refund and escalation actions from earlier tool tests and runs. The safe scenario evaluation did not add new ledger entries.
- `state/` — checkpoint files used for agent progress and resume demonstrations.

### Reproducing the Safe Scenario Evaluation

Activate the project's virtual environment, then run:

```powershell
python .\run_agent_scenarios_safe.py
```

The script stops if `outputs/agent_runs.jsonl` already exists. It also uses dedicated checkpoint run IDs and temporarily substitutes simulated refund and escalation functions in memory.

Do not rerun the script over existing outputs or checkpoints without reviewing them first.

### Limitations

- The language model is simulated by the supplied offline `mock_llm.py`; results do not establish production LLM performance.
- Refund issuance and human escalation are simulated in the safe scenario runner.
- The refund ledger appends actions but does not, by itself, guarantee idempotency across separate runs.
- The demonstration uses project data and simulated actions rather than a live Instamart service.
## Limitations and Next Steps

The results depend on the quality and coverage of the supplied dataset. Statistical outliers are not necessarily data errors. Correlation results should not be interpreted as evidence of causation.

Further work includes completing and evaluating policy retrieval, validating the refund agent against all supplied scenarios, improving handling of unexpected tool observations, and assessing the system's reliability before any production use.