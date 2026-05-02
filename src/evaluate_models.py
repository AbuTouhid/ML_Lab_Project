from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_models(models, X_test, y_test):
    """
    Evaluate all models and return accuracy scores
    """
    results = {}
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy
        
        print(f"{name}:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
        print(f"  Recall: {recall_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
        print(f"  F1-Score: {f1_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
        print()
    
    return results