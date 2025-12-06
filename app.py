from flask import Flask, render_template, request, flash, redirect, url_for
import joblib
import numpy as np

app = Flask(__name__)
app.secret_key = "supersecretkey"

model_diabetes = joblib.load("models/diabetes_model.pkl")
model_heart = joblib.load("models/heart_model.pkl")
life_scaler = joblib.load("models/life_scaler.pkl")
gen_scaler = joblib.load("models/gen_scaler.pkl")

def classify_risk(prob):
    if prob < 35:
        return "Low Risk", "green"
    elif prob < 65:
        return "Medium Risk", "yellow"
    else:
        return "High Risk", "red"

def generate_recommendations(diabetes_prob, heart_prob):
    rec = []

    if diabetes_prob >= 65:
        rec.append("Reduce sugar and refined carbs immediately.")
        rec.append("Start 30–45 minutes brisk walking daily.")
        rec.append("Increase fiber intake (vegetables, oats, millets).")
        rec.append("Maintain proper sleep schedule.")
    elif diabetes_prob >= 35:
        rec.append("Limit sugary drinks and junk food.")
        rec.append("Exercise at least 4–5 days a week.")
        rec.append("Include leafy greens and low-GI foods.")
    else:
        rec.append("Maintain balanced diet with regular physical activity.")
        rec.append("Stay hydrated and monitor body weight.")

    if heart_prob >= 65:
        rec.append("Avoid oily and deep-fried foods completely.")
        rec.append("Reduce salt intake significantly.")
        rec.append("Include yoga/meditation for stress control.")
        rec.append("Do regular cardio (walking, cycling).")
    elif heart_prob >= 35:
        rec.append("Reduce high-sodium processed foods.")
        rec.append("Manage stress and maintain sleep cycle.")
        rec.append("Exercise 4–5 days a week.")
    else:
        rec.append("Maintain active lifestyle and heart-healthy diet.")

    return rec


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    def safe_prob(model, X):
        if len(model.classes_) == 1:
            cls = model.classes_[0]
            return np.array([1.0, 0.0]) if cls == 0 else np.array([0.0, 1.0])
        return model.predict_proba(X)[0]

    life_data = request.form.get("life_data")
    gen_data = request.form.get("gen_data")

    def trigger_error(message):
        flash(message, "error")
        return redirect(url_for("home"))

    if not life_data or not gen_data:
        return trigger_error("Please enter both Lifestyle and Genomic data")

    try:
        life_values = [float(x.strip()) for x in life_data.split(",")]
        gen_values = [float(x.strip()) for x in gen_data.split(",")]
    except:
        return trigger_error("Only numbers allowed, separated by commas.")

    if len(life_values) != 132:
        return trigger_error(f"Lifestyle data must contain 132 values. You entered {len(life_values)}.")

    if len(gen_values) != 22:
        return trigger_error(f"Genomic data must contain 22 values. You entered {len(gen_values)}.")

    life_arr = np.array(life_values).reshape(1, -1)
    gen_arr = np.array(gen_values).reshape(1, -1)

    life_scaled = life_scaler.transform(life_arr)
    gen_scaled = gen_scaler.transform(gen_arr)

    combined = np.hstack((life_scaled, gen_scaled))

    diabetes_prob = safe_prob(model_diabetes, combined)[1] * 100
    heart_prob = safe_prob(model_heart, combined)[1] * 100

    diabetes_risk, diabetes_color = classify_risk(diabetes_prob)
    heart_risk, heart_color = classify_risk(heart_prob)

    recommendations = generate_recommendations(diabetes_prob, heart_prob)

    return render_template(
        "result.html",
        diabetes=f"{diabetes_prob:.2f}%",
        heart=f"{heart_prob:.2f}%",
        diabetes_risk=diabetes_risk,
        heart_risk=heart_risk,
        diabetes_color=diabetes_color,
        heart_color=heart_color,
        recommendations=recommendations
    )


if __name__ == "__main__":
    app.run(debug=True)
