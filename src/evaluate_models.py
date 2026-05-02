from sklearn.metrics import accuracy_score

def evaluate_models(models, X_test, y_test):

    results = {}

    for name, model in models.items():
        pred = model.predict(X_test)
        acc = accuracy_score(y_test, pred)
        results[name] = acc
        print(f"{name}: {acc:.2f}")

    return results