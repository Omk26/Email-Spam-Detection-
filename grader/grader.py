import os
import json
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
from datetime import datetime, timezone
from glob import glob

GROUP_NAME = os.environ.get("GROUP_NAME", "unknown")
PR_NUMBER  = os.environ.get("PR_NUMBER", "0")

# ── Load ground truth ────────────────────────────────────────────────────────
truth_path = "grader/test_labels.csv"
truth = pd.read_csv(truth_path)

# ── Find submission file ──────────────────────────────────────────────────────
matches = glob("submission/*submission.csv")
if not matches:
    print("❌ No submission file found in submission/ folder.")
    exit(1)

sub_path = matches[0]
sub = pd.read_csv(sub_path)

# ── Validate columns ──────────────────────────────────────────────────────────
required_cols = {"email_id", "label"}
if not required_cols.issubset(sub.columns):
    print(f"❌ submission.csv must have columns: {required_cols}")
    print(f"   Found columns: {list(sub.columns)}")
    exit(1)

# ── Merge on email_id ─────────────────────────────────────────────────────────
merged = truth.merge(sub[["email_id", "label"]], on="email_id", suffixes=("_true", "_pred"))

if len(merged) != len(truth):
    print(f"⚠️  Warning: expected {len(truth)} rows, got {len(merged)} after merge.")

y_true = merged["label_true"]
y_pred = merged["label_pred"]

# ── Compute metrics ───────────────────────────────────────────────────────────
f1  = round(f1_score(y_true, y_pred, average="binary"), 4)
acc = round(accuracy_score(y_true, y_pred) * 100, 2)
correct = int((y_true == y_pred).sum())
total   = len(y_true)

print(f"Group     : {GROUP_NAME}")
print(f"PR        : #{PR_NUMBER}")
print(f"F1 Score  : {f1}")
print(f"Accuracy  : {acc}%")
print(f"Correct   : {correct} / {total}")
print(f"Date      : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")

# ── Save result for leaderboard update ───────────────────────────────────────
os.makedirs("leaderboard_data", exist_ok=True)
result = {
    "group":    GROUP_NAME,
    "f1_score": f1,
    "accuracy": acc,
    "correct":  correct,
    "total":    total,
    "pr":       PR_NUMBER,
    "date":     datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
}
with open("leaderboard_data/result.json", "w") as f:
    json.dump(result, f, indent=2)
