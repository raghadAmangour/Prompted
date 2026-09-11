"""
Build the routing/classification ground truth evaluation set.

This script recreates the EXACT train/test split used when training
issue_type_model.joblib and priority_model.joblib (same random_state,
same test_size, same stratify columns), then takes the INTERSECTION of
both test sets. Rows in this intersection were held out from BOTH
model trainings, so they are safe (zero leakage) to use for evaluating:
  - issue type classification accuracy
  - priority classification accuracy
  - queue routing accuracy (agent decision, never used in ML training)
  - response quality (human rubric)
  - end-to-end latency

From that clean held-out pool, we draw a STRATIFIED sample by queue,
guaranteeing minimum coverage of rare queues (e.g. General Inquiry,
Human Resources) while keeping the total sample around ~200 tickets.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# CONFIG (must match the training script exactly)
# ---------------------------------------------------------------------------

SOURCE_FILE = "/mnt/user-data/uploads/ml_ready_support_tickets.csv"
OUTPUT_FILE = "/mnt/user-data/outputs/routing_ground_truth.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.20

MIN_PER_QUEUE = 15     # guarantee at least this many per queue (or all available if fewer)
TARGET_TOTAL = 200     # approximate overall sample size

SAMPLE_RANDOM_STATE = 7  # separate seed just for sampling, for reproducibility


# ---------------------------------------------------------------------------
# 1. LOAD + CLEAN (identical steps to the training script)
# ---------------------------------------------------------------------------

df = pd.read_csv(SOURCE_FILE)

df["subject"] = df["subject"].fillna("")
df["body"] = df["body"].fillna("")

df["type"] = df["type"].astype(str).str.strip().str.lower()
df["priority"] = df["priority"].astype(str).str.strip().str.lower()
df["queue"] = df["queue"].astype(str).str.strip()

df["text"] = (df["subject"].astype(str) + " " + df["body"].astype(str)).str.strip()
df = df[df["text"].str.len() > 0].copy()
df = df.dropna(subset=["type", "priority"]).copy()

print("Usable tickets in source file:", len(df))


# ---------------------------------------------------------------------------
# 2. RECREATE THE EXACT TRAIN/TEST SPLITS USED DURING ML TRAINING
# ---------------------------------------------------------------------------

_, idx_test_type = train_test_split(
    df.index, test_size=TEST_SIZE, stratify=df["type"], random_state=RANDOM_STATE
)

_, idx_test_prio = train_test_split(
    df.index, test_size=TEST_SIZE, stratify=df["priority"], random_state=RANDOM_STATE
)

heldout_idx = sorted(set(idx_test_type) & set(idx_test_prio))
heldout = df.loc[heldout_idx].copy()

print("Rows held out from BOTH type and priority training:", len(heldout))
print("\nQueue distribution in the clean held-out pool:")
print(heldout["queue"].value_counts())


# ---------------------------------------------------------------------------
# 3. STRATIFIED SAMPLE BY QUEUE (guarantee rare-queue coverage)
# ---------------------------------------------------------------------------

queue_counts = heldout["queue"].value_counts()

samples = []
remaining_pool = []

for queue_name, available in queue_counts.items():
    subset = heldout[heldout["queue"] == queue_name]

    take_n = min(available, MIN_PER_QUEUE)
    chosen = subset.sample(n=take_n, random_state=SAMPLE_RANDOM_STATE)
    samples.append(chosen)

    # keep leftover rows from this queue as a pool for topping up to TARGET_TOTAL
    leftover = subset.drop(chosen.index)
    remaining_pool.append(leftover)

base_sample = pd.concat(samples)
remaining_pool = pd.concat(remaining_pool)

# Top up remaining slots proportionally from the larger queues until ~TARGET_TOTAL
slots_left = max(TARGET_TOTAL - len(base_sample), 0)

if slots_left > 0 and len(remaining_pool) > 0:
    top_up = remaining_pool.sample(
        n=min(slots_left, len(remaining_pool)),
        random_state=SAMPLE_RANDOM_STATE,
    )
    final_sample = pd.concat([base_sample, top_up])
else:
    final_sample = base_sample

final_sample = final_sample.sample(frac=1, random_state=SAMPLE_RANDOM_STATE)  # shuffle rows


# ---------------------------------------------------------------------------
# 4. BUILD OUTPUT GROUND TRUTH FILE
# ---------------------------------------------------------------------------

output = final_sample[["subject", "body", "type", "priority", "queue"]].copy()
output = output.rename(columns={
    "type": "true_type",
    "priority": "true_priority",
    "queue": "true_queue",
})

output.insert(0, "ticket_id", [f"GT-{i+1:04d}" for i in range(len(output))])

# Placeholder columns to be filled in DURING/AFTER pipeline evaluation
output["predicted_type"] = ""
output["predicted_priority"] = ""
output["predicted_queue"] = ""
output["escalate_predicted"] = ""
output["response_quality_score"] = ""     # 1-5, filled by human reviewer
output["hallucination_flag"] = ""         # yes/no, filled by human reviewer
output["latency_seconds"] = ""

output.to_csv(OUTPUT_FILE, index=False)

print(f"\nFinal ground truth sample size: {len(output)}")
print("\nFinal queue distribution:")
print(output["true_queue"].value_counts())
print("\nFinal type distribution:")
print(output["true_type"].value_counts())
print("\nFinal priority distribution:")
print(output["true_priority"].value_counts())
print(f"\nSaved to: {OUTPUT_FILE}")
