#import sys,os
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

class cls_features:
    def __init__(self,dataframe):
        self.dataframe = dataframe

    def _fun_feature_na(self):
        #Here with find the percentage of nan values present in each feature
        #Step-1 : Make the list of features which has missing values
        feature_na = [features for features in self.dataframe.columns if self.dataframe[features].isnull().sum()>1 and self.dataframe[feature].dtypes=='O']

        #Step-2 : Print the Feature name and percentage of missing values
        features = []
        missing = []
        for feature in feature_na:
            #return feature
            _feature , _Missing = feature,np.round(self.dataframe[feature].isnull().mean(),4)
            features.append(_feature)
            missing.app(_Missing)
        return features,missing
    
    def _fun_replace_cat_feature(self,features_nan):
        data = self.dataframe
        data[features_nan]=data[features_nan].fillna('MISSING')
        return data

    def _fun_numerical_feature_na(self):
        #Here with find the percentage of nan values present in each numerical feature
        #Step-1 : Make the list of features which has missing values
        feature_na = [features for features in self.dataframe.columns if self.dataframe[features].isnull().sum()>1 and self.dataframe[features].dtypes!='O' and 'year' not in features]
        #Step-2 : Print the Feature name and percentage of missing values
        features = []
        missing = []
        for feature in feature_na:
            #return feature
            _feature , _Missing = feature,np.round(self.dataframe[feature].isnull().mean(),4)
            features.append(_feature)
            missing.app(_Missing)
        return features,missing
    
    def _fun_replace_num_feature(self,features_nan):
        data = self.dataframe
        median_value = data[features_nan].median()
        data[features_nan]=data[features_nan].fillna(median_value,inplace=True)
        return data
    
    def _fun_discreteFeature_encoding(self,df):
        data = df
        #Initialize OneHotEncoder
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        # Apply one-hot encoding to the categorical columns
        one_hot_encoded = encoder.fit_transform(data)
        one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(data.columns))
        return one_hot_df