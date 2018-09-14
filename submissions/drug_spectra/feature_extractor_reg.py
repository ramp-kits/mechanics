import numpy as np

label_names = np.array(['A', 'B', 'C', 'D'])


class FeatureExtractor():
    def __init__(self):
        pass

    def fit(self, X_df, y):
        pass

    def transform(self, X_df):
        print("X_df fe reg: ", X_df)
        X_array = X_df.values
#       X_array = np.roll(X_df.values, -4, axis=1)
#        X_array = np.concatenate([X_array, X_df[label_names].values], axis=1)
        print("X_array fe reg: ", X_array)
        return X_array
