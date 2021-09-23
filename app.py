from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
app = Flask(__name__)
model = pickle.load(open('Wine.pkl', 'rb'))


@app.route('/',methods=['GET'])
def Home():
    return render_template('result.html')


standard_to = StandardScaler()
@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':


        fixed_acidity= float(request.form["fixed_acidity"])
        volatile_acidity = float(request.form["volatile_acidity"])
        citric_acid = float(request.form["citric_acid"])
        residual_sugar = float(request.form["residual_sugar"])
        chlorides = float(request.form["chlorides"])
        free_sulphur_dioxide = float(request.form["free_sulphur_dioxide"])
        total_sulphur_dioxide = float(request.form["total_sulphur_dioxide"])
        density = float(request.form["density"])
        pH = float(request.form["ph"])
        sulphates = float(request.form["sulphates"])
        alcohol = float(request.form["alcohol"])


        prediction = model.predict([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                                     chlorides, free_sulphur_dioxide, total_sulphur_dioxide, density,
                                     pH, sulphates, alcohol]])

        output = prediction
        if output >= 5:
            return render_template('result.html', prediction_text="Wine quality is GOOD")
        else:
            return render_template('result.html', prediction_text="Wine quality is BAD")



if __name__=="__main__":
    app.run(debug=True)



