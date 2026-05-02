import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np

def plot_model_comparison(results):
    plt.figure()
    plt.bar(results.keys(), results.values())
    plt.title("Model Comparison")
    plt.savefig("outputs/model_comparison.png")
    plt.show()

def plot_confusion_matrix(model, X_test, y_test):
    pred = model.predict(X_test)
    cm = confusion_matrix(y_test, pred)

    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title("Confusion Matrix")
    plt.savefig("outputs/confusion_matrix.png")
    plt.show()

def plot_feature_importance(model, feature_names):
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
        indices = np.argsort(importance)[::-1]

        plt.figure()
        plt.bar(range(len(importance)), importance[indices])
        plt.xticks(range(len(importance)),
                   [feature_names[i] for i in indices],
                   rotation=45)
        plt.title("Feature Importance")
        plt.savefig("outputs/feature_importance.png")
        plt.show()