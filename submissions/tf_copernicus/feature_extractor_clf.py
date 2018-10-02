class FeatureExtractor():
    def __init__(self):
        pass

    def fit(self, X_df, y):
        pass

    def transform(self, X_df):
        print("X_df fe clf: ", X_df)
        X_array = X_df.values
        print("X_array fe clf: ", X_array)
        return X_array
