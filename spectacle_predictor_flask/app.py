from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the trained model
with open("onset_model.pkl", "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        try:
            # Get input values from the form
            sleep = float(request.form["sleep"])
            screen = float(request.form["screen"])
            activity = float(request.form["activity"])
            diet = float(request.form["diet"])
            family = request.form["family"]
            genetic = float(request.form["genetic"])
            study = float(request.form["study"])

            # Convert Yes/No to 1/0
            family = 1 if family.lower() == "yes" else 0

            # Prepare the input
            features = np.array([[sleep, screen, activity, diet, family, genetic, study]])

            # Make prediction
            prediction = model.predict(features)[0]
            prediction = round(prediction, 2)

            return render_template("index.html", result=f"Predicted spectacle onset age: {prediction} years")

        except:
            return render_template("index.html", result="Error in input. Please enter valid values.")
    return render_template("index.html", result=None)

if __name__ == "__main__":
    app.run(debug=True)
