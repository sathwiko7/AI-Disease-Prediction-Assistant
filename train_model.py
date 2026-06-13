import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# Load dataset
df = pd.read_csv("data/Training.csv")

# Disease column
y = df["Disease"]

# Symptom columns
symptom_cols = [col for col in df.columns if col.startswith("Symptom")]

# Convert symptoms into list
symptoms = df[symptom_cols].fillna("").values.tolist()

# Convert symptoms to binary features
mlb = MultiLabelBinarizer()
X = mlb.fit_transform(symptoms)

# Encode diseases
le = LabelEncoder()
y = le.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Accuracy
pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)

print("Accuracy:", acc)

# Save files
pickle.dump(model, open("models/disease_model.pkl", "wb"))
pickle.dump(le, open("models/label_encoder.pkl", "wb"))
pickle.dump(mlb, open("models/symptom_encoder.pkl", "wb"))

print("Model Saved Successfully!")