import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

def plot_model_comparison(results):
    """
    Plot model accuracy comparison
    """
    plt.figure(figsize=(10, 6))
    models = list(results.keys())
    accuracies = list(results.values())
    
    plt.bar(models, accuracies, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.title('Model Comparison')
    plt.ylim([0, 1])
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('outputs/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(model, X_test, y_test):
    """
    Plot confusion matrix for best model
    """
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('outputs/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_feature_importance(model, feature_names):
    """
    Plot feature importance (works with tree-based models)
    """
    try:
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(importances)), importances[indices])
        plt.xlabel('Feature Index')
        plt.ylabel('Importance')
        plt.title('Feature Importance')
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
        plt.tight_layout()
        plt.savefig('outputs/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
    except AttributeError:
        print("Model does not support feature importance plotting")