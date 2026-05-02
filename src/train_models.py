from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

def train_models(X_train, y_train):

    models = {
        "logistic": LogisticRegression(max_iter=1000),
        "knn": KNeighborsClassifier(),
        "decision_tree": DecisionTreeClassifier(),
        "svm": SVC()
    }

    for name in models:
        models[name].fit(X_train, y_train)

    return models