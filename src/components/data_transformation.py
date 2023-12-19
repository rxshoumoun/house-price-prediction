import os 
import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd 
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.exception import CustomException
from src.logger import logging

@dataclass
class DataTransformConfig:
    preprocessor_obj_file=os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformConfig()

    def get_data_transformer_obj(self):
        try:
            numerical_columns=['Area','No. of Bedrooms','Gymnasium',
                               'Lift Available','Car Parking','New/Resale','Swimming Pool']
            categorical_columns=['Location']
        except:
            pass