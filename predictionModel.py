import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

df_life = pd.read_csv("datasets/LifeStyle_dataset.csv")
df_gen = pd.read_csv("datasets/genomic_data.csv")
if "Unnamed: 133" in df_life.columns:
    df_life = df_life.drop(columns=["Unnamed: 133"])
diabetes_cols = ["Diabetes_binary", "Diabetes"]
heart_cols = ["HeartDiseaseorAttack", "HeartDisease"]

def find_col(df, names):
    for c in names:
        if c in df.columns:
            return c
    return None

diab_col = find_col(df_life, diabetes_cols) or find_col(df_gen, diabetes_cols)
heart_col = find_col(df_life, heart_cols) or find_col(df_gen, heart_cols)
if diab_col is None:
    raise ValueError("Diabetes column not found.")
if heart_col is None:
    raise ValueError("Heart disease column not found.")

s_diab = (df_life.get(diab_col, df_gen[diab_col])).astype(int)
s_heart = (df_life.get(heart_col, df_gen[heart_col])).astype(int)

min_len = min(len(s_diab), len(s_heart))
s_diab = s_diab.iloc[:min_len].reset_index(drop=True)
s_heart = s_heart.iloc[:min_len].reset_index(drop=True)

y = np.column_stack([s_diab.values, s_heart.values])

X_life = df_life.select_dtypes(include=[np.number]).iloc[:min_len, :]
X_gen = df_gen.select_dtypes(include=[np.number]).iloc[:min_len, :]

life_scaler = StandardScaler()
gen_scaler = StandardScaler()

life_scaler.fit(X_life.values)
gen_scaler.fit(X_gen.values)

X_life_scaled = life_scaler.transform(X_life.values)
X_gen_scaled  = gen_scaler.transform(X_gen.values)

X = np.hstack((X_life_scaled, X_gen_scaled))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_diabetes = RandomForestClassifier(
    n_estimators=200, max_depth=10, min_samples_leaf=5,
    random_state=42, class_weight="balanced"
)

model_heart = RandomForestClassifier(
    n_estimators=200, max_depth=10, min_samples_leaf=5,
    random_state=42, class_weight="balanced"
)

model_diabetes.fit(X_train, y_train[:, 0])
model_heart.fit(X_train, y_train[:, 1])

def safe_prob(model, X):
    if len(model.classes_) == 1:
        cls = model.classes_[0]
        if cls == 0:
            return np.array([1.0, 0.0])
        else:
            return np.array([0.0, 1.0])
    return model.predict_proba(X)[0]

print("\nModel Performance")
print(f"Diabetes Train Accuracy: {accuracy_score(y_train[:, 0], model_diabetes.predict(X_train))*100:.2f}%")
print(f"Diabetes Test Accuracy : {accuracy_score(y_test[:, 0], model_diabetes.predict(X_test))*100:.2f}%\n")

print(f"Heart Disease Train Accuracy: {accuracy_score(y_train[:, 1], model_heart.predict(X_train))*100:.2f}%")
print(f"Heart Disease Test Accuracy : {accuracy_score(y_test[:, 1], model_heart.predict(X_test))*100:.2f}%\n")

os.makedirs("models", exist_ok=True)

joblib.dump(model_diabetes, "models/diabetes_model.pkl")
joblib.dump(model_heart, "models/heart_model.pkl")
joblib.dump(life_scaler, "models/life_scaler.pkl")
joblib.dump(gen_scaler, "models/gen_scaler.pkl")

def predict_diseases(life_data, gen_data):

    life_scaled = life_scaler.transform(np.array(life_data).reshape(1, -1))
    gen_scaled  = gen_scaler.transform(np.array(gen_data).reshape(1, -1))

    combined = np.hstack((life_scaled, gen_scaled))

    diabetes_prob = safe_prob(model_diabetes, combined)[1]
    heart_prob    = safe_prob(model_heart, combined)[1]

    return {
        "Diabetes Risk": f"{diabetes_prob*100:.2f}%",
        "Heart Disease Risk": f"{heart_prob*100:.2f}%"
    }

if __name__ == "__main__":
    print("Enter Patient Data")

    print(f"Enter {X_life.shape[1]} lifestyle features (comma-separated):")
    life_str = input("Lifestyle Data: ")
    life_data = [float(x.strip()) for x in life_str.split(",")]

    print(f"Enter {X_gen.shape[1]} genomic features (comma-separated):")
    gen_str = input("Genomic Data: ")
    gen_data = [float(x.strip()) for x in gen_str.split(",")]

    result = predict_diseases(life_data, gen_data)

    print("\nPrediction Results")
    print("Chances of Diabetes:", result["Diabetes Risk"])
    print("Chances of Heart Disease:", result["Heart Disease Risk"])
