import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score


df = pd.read_csv("ObesityDataSet.csv")


label_cols = df.select_dtypes(include='object').columns
label_encoders = {}

for col in label_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le


X = df.drop("NObeyesdad", axis=1)
y = df["NObeyesdad"]


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)


model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
print("📊 Clasificación:")
print(classification_report(y_test, y_pred))
print(f"🎯 Precisión: {accuracy_score(y_test, y_pred) * 100:.2f}%")


importances = model.feature_importances_
importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)
print("\n🔍 Variables más importantes:")
print(importance_df.head(10))
