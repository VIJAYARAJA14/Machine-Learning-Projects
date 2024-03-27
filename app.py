from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained Decision Tree Regressor model
model = joblib.load(r"d:\CADD Technologies\SNS Projects\Belarus Car Price Prediction\Main\DTR_Model.pkl")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        # Get the form data
        year = int(request.form["year"])
        condition = int(request.form["condition"])
        mileage = float(request.form["mileage"])
        fuel_type = int(request.form["fuel_type"])
        volume = float(request.form["volume"])
        color = int(request.form["color"])
        transmission = int(request.form["transmission"])
        drive_unit = int(request.form["drive_unit"])
        make_segment = int(request.form["make_segment"])

        # Make prediction
        input_data = np.array([[year, condition, mileage, fuel_type, volume, color, transmission, drive_unit, make_segment]])
        predicted_price = model.predict(input_data)

        return render_template("result.html", predicted_price=predicted_price)

if __name__ == "__main__":
    app.run(debug=True)
