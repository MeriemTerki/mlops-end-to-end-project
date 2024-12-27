from flask import Flask, render_template, request
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from mlProject.pipeline.prediction import PredictionPipeline

app = Flask(__name__)  # Initializing a Flask app

# Route to display the home page
@app.route('/', methods=['GET'])
def homePage():
    return render_template("index.html")

# Route to train the pipeline
@app.route('/train', methods=['GET'])
def training():
    os.system("python main.py")  # Executes the training pipeline
    return "Training Successful!"

# Route to show the predictions in a web UI
@app.route('/predict', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        try:
            # Reading the inputs given by the user
            longitude = float(request.form['longitude'])
            latitude = float(request.form['latitude'])
            housing_median_age = float(request.form['housing_median_age'])
            total_rooms = float(request.form['total_rooms'])
            total_bedrooms = float(request.form['total_bedrooms'])
            population = float(request.form['population'])
            households = float(request.form['households'])
            median_income = float(request.form['median_income'])
            ocean_proximity = request.form['ocean_proximity']  # Categorical column
            
            # Construct input data in a dictionary format
            input_data = {
                "longitude": longitude,
                "latitude": latitude,
                "housing_median_age": housing_median_age,
                "total_rooms": total_rooms,
                "total_bedrooms": total_bedrooms,
                "population": population,
                "households": households,
                "median_income": median_income,
                "ocean_proximity": ocean_proximity
            }

            # Convert input data into a DataFrame
            input_df = pd.DataFrame([input_data])

            # Encode the categorical column ('ocean_proximity') using LabelEncoder
            label_encoder = LabelEncoder()
            input_df['ocean_proximity'] = label_encoder.fit_transform(input_df['ocean_proximity'])

            # Ensure other preprocessing is handled in your prediction pipeline
            obj = PredictionPipeline()
            predict = obj.predict(input_df)  # Pass the preprocessed DataFrame to the pipeline

            return render_template('results.html', prediction=str(predict))

        except Exception as e:
            print('The Exception message is: ', e)
            return 'Something went wrong while making predictions.'

    else:
        return render_template('index.html')


if __name__ == "__main__":
    # Run the Flask app
    app.run(host="0.0.0.0", port=8080)
