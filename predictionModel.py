import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
import joblib
import os

df_life = pd.read_csv("datasets/LifeStyle_dataset.csv")
df_gen  = pd.read_csv("datasets/genomic_data.csv")

if "Unnamed: 133" in df_life.columns:
    df_life = df_life.drop(columns=["Unnamed: 133"])

diabetes_cols = ["Diabetes_binary", "Diabetes"]
heart_cols = ["HeartDiseaseorAttack", "HeartDisease"]

def find_col(df, names):
    for c in names:
        if c in df.columns:
            return c
    return None

diab_col  = find_col(df_life, diabetes_cols) or find_col(df_gen, diabetes_cols)
heart_col = find_col(df_life, heart_cols)   or find_col(df_gen, heart_cols)

if diab_col is None:
    raise ValueError("Diabetes column not found.")
if heart_col is None:
    raise ValueError("Heart disease column not found.")

s_diab  = (df_life.get(diab_col,  df_gen[diab_col])).astype(int)
s_heart = (df_life.get(heart_col, df_gen[heart_col])).astype(int)

min_len = min(len(s_diab), len(s_heart))
s_diab  = s_diab.iloc[:min_len].reset_index(drop=True)
s_heart = s_heart.iloc[:min_len].reset_index(drop=True)

y = np.column_stack([s_diab.values, s_heart.values])

leak_cols = set(diabetes_cols + heart_cols)

X_life = df_life.select_dtypes(include=[np.number]).iloc[:min_len, :]
X_gen  = df_gen.select_dtypes(include=[np.number]).iloc[:min_len, :]

X_life = X_life.drop(columns=[c for c in X_life.columns if c in leak_cols], errors="ignore")
X_gen  = X_gen.drop(columns=[c for c in X_gen.columns if c in leak_cols],  errors="ignore")

X_life_train, X_life_test, X_gen_train, X_gen_test, y_train, y_test = train_test_split(
    X_life.values,
    X_gen.values,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y[:, 0]
)

life_scaler = StandardScaler()
gen_scaler  = StandardScaler()

X_life_train_scaled = life_scaler.fit_transform(X_life_train)
X_life_test_scaled  = life_scaler.transform(X_life_test)

X_gen_train_scaled = gen_scaler.fit_transform(X_gen_train)
X_gen_test_scaled  = gen_scaler.transform(X_gen_test)

X_train = np.hstack((X_life_train_scaled, X_gen_train_scaled))
X_test  = np.hstack((X_life_test_scaled,  X_gen_test_scaled))

model_diabetes = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_leaf=5,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42
)

model_heart = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_leaf=5,
    max_features="sqrt",
    class_weight="balanced_subsample",
    random_state=42
)

model_diabetes.fit(X_train, y_train[:, 0])
model_heart.fit(X_train, y_train[:, 1])

def safe_prob(model, X):
    if len(model.classes_) == 1:
        return np.array([1.0, 0.0]) if model.classes_[0] == 0 else np.array([0.0, 1.0])
    return model.predict_proba(X)[0]

print("\nMODEL PERFORMANCE")

print(f"Diabetes Train Accuracy: {accuracy_score(y_train[:,0], model_diabetes.predict(X_train))*100:.2f}%")
print(f"Diabetes Test Accuracy : {accuracy_score(y_test[:,0], model_diabetes.predict(X_test))*100:.2f}%\n")

print(f"Heart Train Accuracy: {accuracy_score(y_train[:,1], model_heart.predict(X_train))*100:.2f}%")
print(f"Heart Test Accuracy : {accuracy_score(y_test[:,1], model_heart.predict(X_test))*100:.2f}%\n")


skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

cv_diab = cross_val_score(
    model_diabetes,
    X_train,
    y_train[:, 0],
    cv=skf,
    scoring="accuracy"
)

cv_heart = cross_val_score(
    model_heart,
    X_train,
    y_train[:, 1],
    cv=skf,
    scoring="accuracy"
)

print(f"Cross-Validation Accuracy (Diabetes): {cv_diab.mean()*100:.2f}%")
print(f"Cross-Validation Accuracy (Heart)   : {cv_heart.mean()*100:.2f}%\n")

y_prob = model_heart.predict_proba(X_test)[:,1]
auc1 = roc_auc_score(y_test[:,1], y_prob)
print(f"Heart Disease ROC-AUC: {auc1*100:.4f}%")

d_prob = model_diabetes.predict_proba(X_test)[:,1]
auc2 = roc_auc_score(y_test[:,0], d_prob)
print(f"Diabetes ROC-AUC: {auc2*100:.2f}%")

os.makedirs("models", exist_ok=True)

joblib.dump(model_diabetes, "models/diabetes_model.pkl")
joblib.dump(model_heart,    "models/heart_model.pkl")
joblib.dump(life_scaler,    "models/life_scaler.pkl")
joblib.dump(gen_scaler,     "models/gen_scaler.pkl")

def predict_diseases(life_data, gen_data):

    life_scaled = life_scaler.transform(np.array(life_data).reshape(1, -1))
    gen_scaled  = gen_scaler.transform(np.array(gen_data).reshape(1, -1))

    combined = np.hstack((life_scaled, gen_scaled))

    diabetes_prob = safe_prob(model_diabetes, combined)[1]
    heart_prob    = safe_prob(model_heart, combined)[1]

    return {
        "Diabetes Risk": f"{diabetes_prob * 100:.2f}%",
        "Heart Disease Risk": f"{heart_prob * 100:.2f}%"
    }

if __name__ == "__main__":
    print("\nEnter Patient Data")

    print(f"Enter {X_life.shape[1]} lifestyle features (comma-separated):")
    life_data = [float(x.strip()) for x in input().split(",")]

    print(f"Enter {X_gen.shape[1]} genomic features (comma-separated):")
    gen_data = [float(x.strip()) for x in input().split(",")]

    result = predict_diseases(life_data, gen_data)

    print("\nPrediction Results\n")
    print("Chances of Diabetes:", result["Diabetes Risk"])
    print("Chances of Heart Disease:", result["Heart Disease Risk"])
