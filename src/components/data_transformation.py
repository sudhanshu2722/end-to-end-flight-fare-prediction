import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
import os

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifact','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            categorical_columns = [
                'airline', 
                'source_city', 
                'departure_time', 
                'stops', 
                'arrival_time',
                'destination_city', 
                'class'
                ]
            numerical_columns = [
                'duration',
                'days_left'
                ]
            
            '''
            numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder()

preprocessor= ColumnTransformer(
    [
        ('StandardScaler',numerical_transformer,num_features),
        ('OneHotEncoding',categorical_transformer,cat_features)
    ]
)
            '''

            num_pipeline = Pipeline(
                steps= [
                    ('scaler',StandardScaler())
                    ]
            )

            cat_pipeline = Pipeline(
                steps = [
                    ('one_hot_encoder',OneHotEncoder())
                ]
            )

            # Combining num and cat columns pipeline together
            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline',num_pipeline,numerical_columns),
                    ('cat_pipeline',cat_pipeline,categorical_columns)
                ]
            )

            logging.info("Numeric columns standarization is completed")
            logging.info("categoric columns encoding  is completed")

            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)
        