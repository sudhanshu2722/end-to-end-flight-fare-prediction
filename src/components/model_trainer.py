import os
import sys
from dataclasses import dataclass

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge,Lasso,ElasticNet
from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
from xgboost import XGBRegressor 
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_models


@ dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifact','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Spliting training and test input data")
            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                "Linear Regression": LinearRegression(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "Elastic Net": ElasticNet(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(max_depth=100, min_samples_split=30, min_samples_leaf=10,random_state=42),
                "Bagging Regressor": BaggingRegressor(n_jobs=-1),
            }

            params = {
                "Decision Tree":{},
                "Linear Regression":{},
                "Lasso":{},
                "Ridge":{},
                "Elastic Net":{},
                "Bagging Regressor":{},
                "K-Neighbors Regressor" :{}
            }

            model_report:dict = evaluate_models(
                X_train = X_train,
                y_train = y_train,
                X_test = X_test,
                y_test = y_test,
                models = models,
                param = params
            )

            ## get best model score from dictionary(value)
            best_model_score = max(sorted(model_report.values()))

            ## get best model name from dictionary(key)
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            predicted = best_model.predict(X_test)

            r2_square = r2_score(y_test,predicted)
            
            return r2_square

        except Exception as e:
            raise CustomException(e,sys)
        


        