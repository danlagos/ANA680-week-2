from flask import Flask, render_template, request
import numpy as np
#import pickle
import joblib

app = Flask(__name__)

filename = 'KNN.pkl'

model = joblib.load(filename)

@app.route('/')

def index(): 
    return render_template('index.html')

@app.route('/predict', methods=['POST'])

def predict():
    radius_Mean = request.form['radius_mean']
    perimeter_Mean = request.form['perimeter_mean']
    area_Mean = request.form['area_mean']
    compactness_Mean = request.form['compactness_mean']
          
    pred = model.predict(np.array([[radius_Mean, perimeter_Mean, area_Mean, compactness_Mean]], dtype=float))
    print(pred)
    return render_template('index.html', predict=str(pred))

if __name__ == '__main__':
    app.run(debug=True)