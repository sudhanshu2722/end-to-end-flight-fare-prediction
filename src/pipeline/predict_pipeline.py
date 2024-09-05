import os
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
from src.logger import logging

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path = os.path.join('artifact','model.pkl')
            preprocessor_path = os.path.join('artifact','preprocessor.pkl')

            logging.info("Before loading pickle files")

            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            logging.info("After loading pickle file")

          
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)

            return preds
        
        except Exception as e:
            raise CustomException(e,sys)

class CustomData:
    def __init__(
            self,
            airline:str,
            source_city:str,
            destination_city:str,
            departure_time:str,
            arrival_time:str,
            class_of_journey:str,
            stops:str,
            duration:float,
            days_left:int):
        
        self.airline=airline
        self.source_city=source_city
        self.destination_city=destination_city
        self.departure_time=departure_time
        self.arrival_time=arrival_time
        self.class_of_journey=class_of_journey
        self.stops=stops
        self.duration=duration
        self.days_left=days_left

    def get_data_as_data_frame(self):
        try:
            custom_data_input = {
                'airline' : [self.airline],
                'source_city' : [self.source_city],
                'destination_city' : [self.destination_city],
                'departure_time' : [self.departure_time],
                'arrival_time' : [self.arrival_time],
                'class' : [self.class_of_journey],
                'stops' : [self.stops],
                'duration' : [self.duration],
                'days_left' : [self.days_left]
            }

            return pd.DataFrame(custom_data_input)

        except Exception as e:
            raise CustomException(e,sys)
        


