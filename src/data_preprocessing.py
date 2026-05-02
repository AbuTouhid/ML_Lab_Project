import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_data(path):
    return pd.read_csv(r"C:\Users\TOUHID\Programming\mlp\data\dataset.csv")

def preprocess_data(df):

    df = df[[
        "age", "study_hours", "attendance_percentage",
        "math_score", "science_score", "english_score",
        "overall_score", "final_grade"
    ]]

    X = df.drop("final_grade", axis=1)
    y = df["final_grade"]

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(y)

    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test, scaler, le