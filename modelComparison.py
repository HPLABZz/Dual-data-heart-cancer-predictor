import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

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
X_gen  = df_gen.select_dtypes(include=[np.number]).iloc[:min_len, :]

life_scaler = StandardScaler()
gen_scaler = StandardScaler()

X_life_scaled = life_scaler.fit_transform(X_life.values)
X_gen_scaled  = gen_scaler.fit_transform(X_gen.values)

X = np.hstack((X_life_scaled, X_gen_scaled))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    "Logistic Regression": {
        "diabetes": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "heart": LogisticRegression(max_iter=1000, class_weight="balanced")
    },
    "Support Vector Machine": {
        "diabetes": SVC(kernel="rbf", probability=True, class_weight="balanced"),
        "heart": SVC(kernel="rbf", probability=True, class_weight="balanced")
    },
    "Random Forest": {
        "diabetes": RandomForestClassifier(
            n_estimators=200, max_depth=10,
            min_samples_leaf=5, class_weight="balanced",
            random_state=42
        ),
        "heart": RandomForestClassifier(
            n_estimators=200, max_depth=10,
            min_samples_leaf=5, class_weight="balanced",
            random_state=42
        )
    }
}

print("MODEL COMPARISON RESULTS")
print(" ")

for model_name, model_set in models.items():
    print(" ")
    print(f"Model: {model_name}")

    model_set["diabetes"].fit(X_train, y_train[:, 0])
    diab_train_acc = accuracy_score(
        y_train[:, 0], model_set["diabetes"].predict(X_train)
    )
    diab_test_acc = accuracy_score(
        y_test[:, 0], model_set["diabetes"].predict(X_test)
    )

    model_set["heart"].fit(X_train, y_train[:, 1])
    heart_train_acc = accuracy_score(
        y_train[:, 1], model_set["heart"].predict(X_train)
    )
    heart_test_acc = accuracy_score(
        y_test[:, 1], model_set["heart"].predict(X_test)
    )

    print(f"Diabetes Train: {diab_train_acc*100:.2f}% | Test: {diab_test_acc*100:.2f}%")
    print(f"Heart Train: {heart_train_acc*100:.2f}% | Test: {heart_test_acc*100:.2f}%")
