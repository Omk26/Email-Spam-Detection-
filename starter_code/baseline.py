import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression

# 1. Load data
df = pd.read_csv("./data/train.csv")

FEATURES = [
    'num_words', 'num_characters', 'num_exclamation_marks', 'num_links',
    'has_suspicious_link', 'num_attachments', 'has_attachment',
    'sender_reputation_score', 'email_hour', 'email_day_of_week',
    'is_weekend', 'num_recipients', 'contains_money_terms',
    'contains_urgency_terms'
]

X = df[FEATURES]
y = df["label"]

# 2. Train-validation split
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 3. Models to test
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_split=2,
        random_state=42,
        class_weight="balanced"
    ),
    "Extra Trees": ExtraTreesClassifier(
        n_estimators=700,
        max_depth=None,
        min_samples_split=2,
        random_state=42,
        class_weight="balanced"
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
}

best_model = None
best_name = ""
best_f1 = -1

# 4. Train and compare
for name, model in models.items():
    if name == "Logistic Regression":
        scaler = StandardScaler()
        X_train_used = scaler.fit_transform(X_train)
        X_val_used = scaler.transform(X_val)
    else:
        scaler = None
        X_train_used = X_train
        X_val_used = X_val

    model.fit(X_train_used, y_train)
    preds = model.predict(X_val_used)

    f1 = f1_score(y_val, preds)
    acc = accuracy_score(y_val, preds)

    print("\n==============================")
    print(name)
    print(f"F1 Score : {f1:.4f}")
    print(f"Accuracy : {acc:.4f}")

    if f1 > best_f1:
        best_f1 = f1
        best_model = model
        best_name = name
        best_scaler = scaler

print("\nBEST MODEL:", best_name)
print("BEST F1:", round(best_f1, 4))

# 5. Final detailed report
if best_scaler is not None:
    X_val_final = best_scaler.transform(X_val)
else:
    X_val_final = X_val

final_preds = best_model.predict(X_val_final)

print("\nFinal Classification Report:")
print(classification_report(y_val, final_preds, target_names=["Ham", "Spam"]))

# 6. Save model
os.makedirs("models", exist_ok=True)
joblib.dump(best_model, "models/best_spam_model.pkl")
joblib.dump(best_scaler, "models/scaler.pkl")

# 7. Generate submission
test_df = pd.read_csv("./data/test.csv")
X_test = test_df[FEATURES]

if best_scaler is not None:
    X_test = best_scaler.transform(X_test)

test_preds = best_model.predict(X_test)

os.makedirs("submission", exist_ok=True)
submission = pd.DataFrame({
    "email_id": test_df["email_id"],
    "label": test_preds
})

submission.to_csv("submission/Omk26_submission.csv", index=False)

print("\nSubmission saved to submission/Omk26_submission.csv")
print("Rows:", len(submission))
