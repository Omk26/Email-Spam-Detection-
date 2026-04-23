# 📧 Email Spam Detection Challenge 2026

[![Leaderboard](https://img.shields.io/badge/Leaderboard-Live-brightgreen)](https://omk26.github.io/Email-Spam-Detection-/)

> A machine learning challenge to classify emails as **Spam** or **Ham**.  
> Build your model, submit predictions, and see your score on the live leaderboard!

---

## 📊 Live Leaderboard

👉 **[View Leaderboard](https://omk26.github.io/Email-Spam-Detection-/)**

---

## 📁 Repository Structure

```
Email-Spam-Detection/
├── .github/
│   └── workflows/
│       └── grade.yml          ← Auto-grades every PR submission
├── grader/
│   └── grader.py              ← Scoring script (test labels are secret)
├── docs/
│   ├── index.html             ← Live leaderboard UI
│   └── leaderboard.json       ← Auto-updated scores
├── data/
│   ├── train.csv              ← Training data (with labels)
│   └── test.csv               ← Test data (no labels — predict these!)
├── submission/
│   └── .gitkeep               ← Upload your submission here
└── README.md
```

---

## 🚀 How to Participate

### Step 1 — Fork this repo
Click the **Fork** button at the top right of this page.

### Step 2 — Train your model
Use `data/train.csv` to train your spam detection model.

### Step 3 — Predict on test set
Run predictions on `data/test.csv` (it has no `label` column — that's what you predict!).

### Step 4 — Create your submission file
Save your predictions as a CSV file named:
```
<YourGitHubUsername>_submission.csv
```
It must have exactly two columns:
```csv
email_id,label
0,1
1,0
2,1
...
```
- `email_id` — matches the `email_id` in `test.csv`
- `label` — your prediction: `1` = Spam, `0` = Ham

### Step 5 — Place it in the `submission/` folder
Put your file at:
```
submission/<YourGitHubUsername>_submission.csv
```

### Step 6 — Open a Pull Request
Create a PR from your fork to this repo's `main` branch.  
The GitHub Action will automatically:
- ✅ Validate your file
- 📊 Calculate your **F1 Score** and **Accuracy**
- 💬 Post your score as a PR comment
- 🏆 Update the live leaderboard

---

## 📐 Evaluation Metric

Submissions are ranked by **F1 Score (binary)** on the hidden test labels.

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

---

## 📦 Dataset

| File | Rows | Description |
|------|------|-------------|
| `data/train.csv` | 8,000 | Training data with labels |
| `data/test.csv` | 2,000 | Test data — predict the `label` column |

**Features include:** subject, email_text, num_words, num_links, has_suspicious_link, sender_reputation_score, contains_money_terms, contains_urgency_terms, and more.

**Target:** `label` → `1` = Spam, `0` = Ham

---

## 💡 Tips

- Start with **TF-IDF + Naive Bayes** — classic and effective for spam detection
- The structured features (num_links, sender_reputation_score, etc.) are very informative
- Combining text features + structured features usually beats either alone
- Try **Logistic Regression**, **Random Forest**, or **Gradient Boosting**

---

*Challenge created for coursework — 2026*
