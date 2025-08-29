# coding: utf-8
import os
import pickle
import pandas as pd
from flask import Flask, request, render_template

# ---------- Paths & sanity checks ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
CSV_PATH = os.path.join(BASE_DIR, "first_telc.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model.sav")

print("CWD:", os.getcwd())
print("App base:", BASE_DIR)
print("CSV exists?:", os.path.exists(CSV_PATH))
print("Model exists?:", os.path.exists(MODEL_PATH))
print("Template exists?:", os.path.exists(os.path.join(TEMPLATES_DIR, "home.html")))

# ---------- Flask app ----------
app = Flask(__name__, template_folder="templates")

# ---------- Load artifacts at startup ----------
try:
    df_1 = pd.read_csv(CSV_PATH)
except Exception as e:
    # Keep app up so at least "/" loads and shows template errors clearly
    print("ERROR reading first_telc.csv:", e)
    df_1 = None

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except Exception as e:
    print("ERROR loading model.sav:", e)
    model = None

# ---------- Routes ----------
@app.route("/")
def loadPage():
    # If anything is missing, you can still see the form and a helpful hint
    missing_bits = []
    if df_1 is None:
        missing_bits.append("first_telc.csv")
    if model is None:
        missing_bits.append("model.sav")
    hint = ""
    if missing_bits:
        hint = f"Warning: Missing files → {', '.join(missing_bits)}. Predictions will fail until these are fixed."
    return render_template("home.html", query="", hint=hint)

@app.route("/", methods=["POST"])
def predict():
    # Guard against missing artifacts
    if df_1 is None or model is None:
        return render_template(
            "home.html",
            output1="Server not ready",
            output2="Either first_telc.csv or model.sav could not be loaded. Check console logs.",
            **_echo_form_fields(request)
        )

    # Read and type-cast inputs
    # Numeric: SeniorCitizen (0/1), MonthlyCharges, TotalCharges, tenure
    # Categorical: others (strings like Yes/No, etc.)
    try:
        inputQuery1  = int(request.form["query1"])            # SeniorCitizen
        inputQuery2  = float(request.form["query2"])          # MonthlyCharges
        inputQuery3  = float(request.form["query3"])          # TotalCharges
        inputQuery4  = request.form["query4"]                 # gender
        inputQuery5  = request.form["query5"]                 # Partner
        inputQuery6  = request.form["query6"]                 # Dependents
        inputQuery7  = request.form["query7"]                 # PhoneService
        inputQuery8  = request.form["query8"]                 # MultipleLines
        inputQuery9  = request.form["query9"]                 # InternetService
        inputQuery10 = request.form["query10"]                # OnlineSecurity
        inputQuery11 = request.form["query11"]                # OnlineBackup
        inputQuery12 = request.form["query12"]                # DeviceProtection
        inputQuery13 = request.form["query13"]                # TechSupport
        inputQuery14 = request.form["query14"]                # StreamingTV
        inputQuery15 = request.form["query15"]                # StreamingMovies
        inputQuery16 = request.form["query16"]                # Contract
        inputQuery17 = request.form["query17"]                # PaperlessBilling
        inputQuery18 = request.form["query18"]                # PaymentMethod
        inputQuery19 = int(request.form["query19"])           # tenure (months)
    except Exception as e:
        return render_template(
            "home.html",
            output1="Invalid input",
            output2=f"Please check your numbers. Error: {e}",
            **_echo_form_fields(request)
        )

    # Build the prediction row
    data = [[
        inputQuery1, inputQuery2, inputQuery3, inputQuery4, inputQuery5, inputQuery6, inputQuery7,
        inputQuery8, inputQuery9, inputQuery10, inputQuery11, inputQuery12, inputQuery13, inputQuery14,
        inputQuery15, inputQuery16, inputQuery17, inputQuery18, inputQuery19
    ]]

    new_df = pd.DataFrame(
        data,
        columns=[
            "SeniorCitizen", "MonthlyCharges", "TotalCharges", "gender",
            "Partner", "Dependents", "PhoneService", "MultipleLines", "InternetService",
            "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
            "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
            "PaymentMethod", "tenure"
        ]
    )

    # Combine with base df to build consistent dummies
    try:
        base_df = df_1.copy()
        df_2 = pd.concat([base_df, new_df], ignore_index=True)

        # tenure bins (same logic as your original)
        labels = [f"{i} - {i + 11}" for i in range(1, 72, 12)]
        df_2["tenure_group"] = pd.cut(df_2.tenure.astype(int), range(1, 80, 12), right=False, labels=labels)

        # Drop raw tenure (your model expects the binned column instead)
        df_2.drop(columns=["tenure"], axis=1, inplace=True)

        # One-hot encode same set of columns
        features = [
            "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
            "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
            "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
            "Contract", "PaperlessBilling", "PaymentMethod", "tenure_group"
        ]
        new_df_dummies = pd.get_dummies(df_2[features])

        # Take only the last row (the user input)
        X = new_df_dummies.tail(1)

        # Predict
        single = model.predict(X)
        probability = model.predict_proba(X)[:, 1]

        if int(single[0]) == 1:
            o1 = "This customer is likely to be churned!!"
        else:
            o1 = "This customer is likely to continue!!"
        o2 = f"Confidence: {probability[0] * 100:.2f}%"

        return render_template("home.html", output1=o1, output2=o2, **_echo_form_fields(request))
    except Exception as e:
        # Catch any shape/column issues etc.
        return render_template(
            "home.html",
            output1="Prediction failed",
            output2=f"Error while preparing features or predicting: {e}",
            **_echo_form_fields(request)
        )

# ---------- Helpers ----------
def _echo_form_fields(req):
    """Return the 19 fields back to the template so form retains values."""
    ctx = {}
    for i in range(1, 20):
        key = f"query{i}"
        ctx[key] = req.form.get(key, "")
    return ctx

# ---------- Run ----------
if __name__ == "__main__":
    # Use 6060 so it doesn't clash with other processes; debug for auto-reload & error pages
    app.run(host="127.0.0.1", port=6060, debug=True)
