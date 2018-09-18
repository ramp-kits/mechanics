import numpy as np

label_names = np.array(['A', 'B', 'C', 'D'])


class FeatureExtractor():
    def __init__(self):
        pass

    def fit(self, X_df, y):
        pass

    def transform(self, X_df):
        X_array = X_df.values
        return X_array
