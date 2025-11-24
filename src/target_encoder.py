import pandas as pd

class TargetEncoder:
    def __init__(self, columns, target_column):
        self.columns = columns
        self.target_column = target_column
        self.target_means = {}

    def fit(self, X, y):
        X = X.copy()
        X[self.target_column] = y
        for col in self.columns:
            self.target_means[col] = X.groupby(col)[self.target_column].mean()
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            X[col] = X[col].map(self.target_means[col])
            # Handle new categories not seen during fit by filling with the global mean of the target
            if X[col].isnull().any():
                global_mean = self.target_means[col].mean()
                X[col].fillna(global_mean, inplace=True)
        return X

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)
