from flask import Flask, request, render_template
import numpy as np
import pickle

# Initialize the Flask app
app = Flask(__name__, static_folder='static')

# Load models and scalers
diabetesModel = pickle.load(open('DibetiesModel.pkl','rb'))
heartModel = pickle.load(open('HeartDModel.pkl','rb'))
kidneyModel = pickle.load(open('KidneyDModel.pkl','rb'))
scaler = pickle.load(open('scaler.pkl','rb'))
scaler2 = pickle.load(open('scaler2.pkl','rb'))

# Define routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/model')
def model():
    return render_template('model.html')

@app.route('/intution')
def intution():
    return render_template('intution.html')

@app.route('/model/diabetes')
def diabetes():
    return render_template('diabetes.html')

@app.route('/model/heart')
def heart():
    return render_template('heart.html')

@app.route('/model/kidney')
def kidney():
    return render_template('kidney.html')

# Prediction routes
@app.route('/predictDiabeties', methods=['POST'])
def predictDiabeties():
    int_features = [float(x) for x in request.form.values()]
    data = np.asarray(int_features).reshape(1, -1)
    std_data = scaler.transform(data)
    prediction = diabetesModel.predict(std_data)

    if prediction[0] == 1:
        prediction = "You are a Diabetic Person !!!"
        color = 'red'
    else:
        prediction = "You are not a Diabetic Person !!!"
        color = 'green'

    return render_template('result.html', title=prediction, clr=color)

@app.route('/HeartPredictionResult', methods=['POST'])
def HeartPredictionResult():
    int_features = [float(x) for x in request.form.values()]
    data = np.asarray(int_features).reshape(1, -1)
    prediction = heartModel.predict(data)

    if prediction[0] == 1:
        prediction = "You Have a Heart Disease!!!"
        color = 'red'
    else:
        prediction = "You Don't Have a Heart Disease!!!"
        color = 'green'

    return render_template('result.html', title=prediction, clr=color)

@app.route('/KidneyPredictionResult', methods=['POST'])
def KidneyPredictionResult():
    int_features = [float(x) for x in request.form.values()]
    data = np.asarray(int_features).reshape(1, -1)
    std_data = scaler2.transform(data)
    prediction = kidneyModel.predict(std_data)

    if prediction[0] == 0:
        prediction = "You Don't Have Chronic Kidney Disease !!!"
        color = 'green'
    else:
        prediction = "You Have Chronic Kidney Disease !!!"
        color = 'red'

    return render_template('result.html', title=prediction, clr=color)

if __name__ == "__main__":
    app.run(debug=True, port=5001)  # For development
