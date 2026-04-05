
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

class PropertyPreprocessor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    # Custome preprocessing thanks to insights found during the EDA :
    # cleaning of 'bathrooms' and 'bedrooms', merge some property types, merge some cities
    def transform(self, X):
        X_copy = X.copy()

        X_copy['bedrooms'] = X_copy['bedrooms'].replace(['7+', 'studio'], ['8', '0'])
        X_copy['bathrooms'] = X_copy['bathrooms'].replace(['7+', 'none'], ['8', '0'])

        prop_map = {'iVilla': 'Villa', 'Hotel Apartment': 'Apartment', 'Twin House': 'Townhouse'}
        X_copy['property_type'] = X_copy['property_type'].replace(prop_map)

        others = ['Qalyubia', 'South Sainai', 'Matrouh', 'Sharqia', 'Demyat', 'Al Daqahlya']
        X_copy['city'] = X_copy['city'].replace(others, 'Other Cities')

        return X_copy