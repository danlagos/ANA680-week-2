from flask import Flask, render_template, request
import numpy as np
import pickle
import joblib

app = Flask(__name__)

filename = 'KNN.pkl'

model = joblib.load(filename)

@app.route('/')

def index(): 
    return render_template('index.html')

@app.route('/predict', methods=['POST'])

def predict():
    Radius_Mean = request.form['radius_mean']
    Perimeter_Mean = request.form['perimeter_mean']
    Area_Mean = request.form['area_mean']
    Compactness_Mean = request.form['compactness_mean']
    Concavity_Mean = request.form['concavity_mean']
    Concave_Points_Mean = request.form['concave_points_mean']
    Radius_Se = request.form['radius_se']
    Perimeter_Se = request.form['perimeter_se']
    Area_Se = request.form['area_se']
    Radius_Worst = request.form['radius_worst']
    Perimeter_Worst = request.form['perimeter_worst']
    Area_Worst = request.form['area_worst']
    Compactness_Worst = request.form['compactness_worst']
    Concavity_Worst = request.form['concavity_worst']
    Concave_Points_Worst = request.form['concave_points_worst']
          
    pred = model.predict(np.array([[Radius_Mean, Perimeter_Mean, Area_Mean, Compactness_Mean, Concavity_Mean, Concave_Points_Mean, Radius_Se, Perimeter_Se, Area_Se, Radius_Worst, Perimeter_Worst, Area_Worst, Compactness_Worst, Concavity_Worst, Concave_Points_Worst]]))
    print(pred)
    return render_template('index.html', predict=str(pred))

if __name__ == '__main__':
    app.run(debug=False)