import pandas as pd

DATASET_PATH = "datasets/heart-disease-data/heart_disease_uci.csv"

def clean_dataset(path):
    # Handle possible missing values markers like '?' used in UCI datasets
    df = pd.read_csv(path, na_values="?")

    print("Initial dataset Info:")
    print(f"Initial dataset shape: {df.shape}")
    
    # --- ERROR FIX 1: Correct column name typo ---
    # The standard column name is 'thalach', not 'thalch'
    if 'thalch' in df.columns:
        df.rename(columns={'thalch': 'thalach'}, inplace=True)
    
    # --- ERROR FIX 2: Target variable cleaning ---
    # Ensure target is strictly 0 or 1 (some datasets have 1,2,3,4 for severity)
    # The original lambda was fine, but we ensure 'num' exists first
    if "num" in df.columns:
        df["num"] = df["num"].apply(lambda x: 0 if x == 0 else 1)

    # --- ERROR FIX 3: Robust Sex Mapping ---
    # If the dataset uses numbers (1=Male), your original code (x == "male") 
    # would turn everyone into 1 (Female). This handles both strings and numbers.
    # Note: Your logic sets Male to 0. Standard convention is often Male=1, 
    # but I have kept your logic (0 if Male).
    if "sex" in df.columns:
        df["sex"] = df["sex"].apply(lambda x: 0 if str(x).lower() in ['male', 'm', '1'] else 1)

    df = df.drop(columns=["id", "dataset", "ca"], errors="ignore")

    # Updated list with correct spelling 'thalach'
    numeric_cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    # Removed 'sex' from categorical_cols list because we manually encoded it above
    categorical_cols = ["cp", "restecg", "slope", "thal", "fbs", "exang"]

    # Impute Numeric
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Impute Categorical
    for col in categorical_cols:
        if col in df.columns:
            # Check if mode exists to avoid index error
            if not df[col].mode().empty:
                df[col] = df[col].fillna(df[col].mode()[0])

    # --- ERROR FIX 4: Encoding Categorical Data ---
    # Neural Networks cannot handle string categories (e.g. "typical angina").
    # We must convert them to numbers using One-Hot Encoding.
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    print("\nCleaned dataset info:")
    print(df.isnull().sum().sum(), "missing values") # summary of missing
    print("Final dataset shape:", df.shape)

    return df