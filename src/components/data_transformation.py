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
from src.utils import save_object

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
            
            num_pipeline = Pipeline(
                steps= [
                    ('scaler',StandardScaler())
                    ]
            )

            cat_pipeline = Pipeline(
                steps = [
                    ('one_hot_encoder',OneHotEncoder(handle_unknown='ignore'))
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
    
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            #print(train_df.shape)  (240122, 12)
            #print(test_df.shape)  (60031, 12)

            logging.info("Read train and test data completed")
            logging.info("Obtaining the preprocessing object")

            preprocessing_obj = self.get_data_transformation_object()

            target_column_name = "price"
            numerical_columns = ['duration','days_left']

            input_feature_train_df = train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df = train_df[target_column_name].values

            input_feature_test_df = test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df = test_df[target_column_name].values
            
            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe.")

            print(input_feature_train_df.head())

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df).toarray()
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df).toarray()

            # Convert arrays to DataFrames before concatenation
            input_feature_train_df_transformed = pd.DataFrame(input_feature_train_arr)
            input_feature_test_df_transformed = pd.DataFrame(input_feature_test_arr)

            # Combine features and targets
            train_arr = np.column_stack([input_feature_train_df_transformed, target_feature_train_df])
            test_arr = np.column_stack([input_feature_test_df_transformed, target_feature_test_df])

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            logging.info(f"saved preprocessing object.")

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)
        