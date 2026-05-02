from src.data_preprocessing import load_data, preprocess_data
from src.train_models import train_models
from src.evaluate_models import evaluate_models
from src.visualize import plot_model_comparison, plot_confusion_matrix, plot_feature_importance

import pickle

# Load data
df = load_data("data/dataset.csv")

# Preprocess
X_train, X_test, y_train, y_test, scaler, le = preprocess_data(df)

# Train
models = train_models(X_train, y_train)

# Evaluate
results = evaluate_models(models, X_test, y_test)

# Best model
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]

print("Best Model:", best_model_name)

# Save everything
pickle.dump(best_model, open("models/best_model.pkl", "wb"))
pickle.dump(scaler, open("models/scaler.pkl", "wb"))
pickle.dump(le, open("models/label_encoder.pkl", "wb"))

# Visualizations
plot_model_comparison(results)
plot_confusion_matrix(best_model, X_test, y_test)

feature_names = ["age","study_hours","attendance","math","science","english","overall"]
plot_feature_importance(best_model, feature_names)