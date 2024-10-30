from sklearn.ensemble import RandomForestClassifier 

class RandomForest:

    # Setup model
    def __init__(self, n_tree=100, max_depth=None, min_split=2, n_feature=None, class_weight = None):
        self.rf = RandomForestClassifier(
            n_estimators=n_tree,
            max_depth=max_depth,
            min_samples_split=min_split,
            max_features=n_feature,
            n_jobs=-1,
            verbose=1,
            class_weight=class_weight
        )
        self.all_score = None

    # Train the model
    def fit(self, X, Y):
        self.rf.fit(X, Y)
        self.all_score = self.rf.feature_importances_

    # Get class of data
    def predict(self, X):
        return self.rf.predict(X)

    # Get class probability
    def predict_proba(self, X):
        return self.rf.predict_proba(X)

    # Get importance score of features
    def get_feature_importance(self):
        return self.all_score