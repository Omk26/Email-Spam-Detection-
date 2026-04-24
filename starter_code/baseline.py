import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import StandardScaler
import joblib
import os

# 1. Load your data
df = pd.read_csv('../data/train.csv')

# 2. Basic Preprocessing
FEATURES = [
    'num_words', 'num_characters', 'num_exclamation_marks', 'num_links',
    'has_suspicious_link', 'num_attachments', 'has_attachment',
    'sender_reputation_score', 'email_hour', 'email_day_of_week',
    'is_weekend', 'num_recipients', 'contains_money_terms',
    'contains_urgency_terms'
]

X = df[FEATURES]
y = df['label']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)

# 4. Define Baseline Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 5. Evaluate
predictions = model.predict(X_val)
print(f"Baseline F1-Score: {f1_score(y_val, predictions):.4f}")
print(classification_report(y_val, predictions, target_names=['Ham', 'Spam']))

# 6. Save model
os.makedirs('models', exist_ok=True)
joblib.dump(model,  'models/baseline_spam_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
print("Model saved to models/baseline_spam_model.pkl")

# 7. Generate submission on test set
test_df  = pd.read_csv('../data/test.csv')
X_test   = scaler.transform(test_df[FEATURES])
test_preds = model.predict(X_test)

os.makedirs('submission', exist_ok=True)
submission = pd.DataFrame({'email_id': test_df['email_id'], 'label': test_preds})
submission.to_csv('submission/Omk26_submission.csv', index=False)
print(f"Submission saved to submission/Omk26_submission.csv  ({len(submission)} rows)")
