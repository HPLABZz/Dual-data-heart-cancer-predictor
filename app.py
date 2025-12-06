from flask import Flask, render_template, request, flash, redirect, url_for
import joblib
import numpy as np

app = Flask(__name__)
app.secret_key = "supersecretkey"

model_diabetes = joblib.load("models/diabetes_model.pkl")
model_heart = joblib.load("models/heart_model.pkl")
life_scaler = joblib.load("models/life_scaler.pkl")
gen_scaler = joblib.load("models/gen_scaler.pkl")

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
        gen_values  = [float(x.strip()) for x in gen_data.split(",")]
    except:
        return trigger_error("Only numbers allowed, separated by commas.")

    if len(life_values) != 132:
        return trigger_error(f"Lifestyle data must contain 132 values. You entered {len(life_values)}.")

    if len(gen_values) != 22:
        return trigger_error(f"Genomic data must contain 22 values. You entered {len(gen_values)}.")

    life_arr = np.array(life_values).reshape(1, -1)
    gen_arr  = np.array(gen_values).reshape(1, -1)

    life_scaled = life_scaler.transform(life_arr)
    gen_scaled  = gen_scaler.transform(gen_arr)

    combined = np.hstack((life_scaled, gen_scaled))

    diabetes_prob = safe_prob(model_diabetes, combined)[1] * 100
    heart_prob    = safe_prob(model_heart, combined)[1] * 100

    return render_template("result.html",
                           diabetes=f"{diabetes_prob:.2f}%",
                           heart=f"{heart_prob:.2f}%")

if __name__ == "__main__":
    app.run(debug=True)
