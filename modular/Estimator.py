from sklearn.ensemble import RandomForestClassifier, ExtraTreesRegressor, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier


class Estimator:
    def __init__(self, estimator, section_name):
        self.et = estimator
        self.section_name = section_name
        self.vscore = None
        self.vpredict = None

    def get(self):
        return self.et    

    def set_params(self, params):
        self.et.set_params(**params)

    def fit(self, X, y):
        self.et.fit(X, y)

    def score(self, X, y):
        self.vscore =  self.et.score(X, y)
    
    def predict(self, test):
        self.vpredict =  self.et.predict(test)
        return self.vpredict

    def getfi(self):
        return self.et.feature_importances_