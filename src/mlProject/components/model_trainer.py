import pandas as pd
import os
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
import joblib
from mlProject.entity.config_entity import ModelTrainerConfig


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        # Load the training and test data
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)

        # Encode categorical 'ocean_proximity' column to numeric
        label_encoder = LabelEncoder()
        train_data['ocean_proximity'] = label_encoder.fit_transform(train_data['ocean_proximity'])
        test_data['ocean_proximity'] = label_encoder.transform(test_data['ocean_proximity'])

        # Save the LabelEncoder to a file
        joblib.dump(label_encoder, os.path.join(self.config.root_dir, 'label_encoder.pkl'))

        # Split the data into features and target
        train_x = train_data.drop([self.config.target_column], axis=1)
        test_x = test_data.drop([self.config.target_column], axis=1)
        train_y = train_data[[self.config.target_column]]
        test_y = test_data[[self.config.target_column]]

        # Handle missing values using SimpleImputer (replace NaNs with the column mean)
        imputer = SimpleImputer(strategy='mean')

        # Create a pipeline: Imputer -> ElasticNet model
        model = make_pipeline(imputer, ElasticNet(alpha=self.config.alpha, l1_ratio=self.config.l1_ratio, random_state=42))

        # Train the model
        model.fit(train_x, train_y)

        # Save the model to a file
        joblib.dump(model, os.path.join(self.config.root_dir, self.config.model_name))
