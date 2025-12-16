import pandas as pd
import re
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans

# LOAD DATASET (FORCE STRING + CLEAN)
df = pd.read_csv("data/recommendation_dataset.csv", dtype=str)

df.columns = df.columns.str.strip()

# FORCE CLEAN CustomerID
df['CustomerID'] = (
    df['CustomerID']
    .str.strip()
    .str.upper()
)

# Convert numeric columns properly
for col in df.columns:
    if col != 'CustomerID':
        df[col] = pd.to_numeric(df[col], errors='ignore')

# FEATURE SELECTION
numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = df.select_dtypes(include=['object']).columns.tolist()

remove_cols = [
    'CustomerID',
    'Loan_type',
    'credit_cardtype',
    'investment_type',
    'savings_plan_type'
]

for col in remove_cols:
    if col in numeric_features:
        numeric_features.remove(col)
    if col in categorical_features:
        categorical_features.remove(col)

# PREPROCESSING
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# KMEANS CLUSTERING
X = preprocessor.fit_transform(df[numeric_features + categorical_features])

kmeans = KMeans(n_clusters=5, random_state=42)
df['cluster'] = kmeans.fit_predict(X)

# RECOMMENDATION CORE
def recommend_from_cluster(customer_id, product_col, top_n=3):

    customer_id = str(customer_id).strip().upper()

    customer = df[df['CustomerID'] == customer_id]

    if customer.empty:
        return None

    cluster_id = customer['cluster'].values[0]
    cluster_data = df[df['cluster'] == cluster_id]

    recommendations = (
        cluster_data[product_col]
        .value_counts()
        .dropna()
        .head(top_n)
        .index
        .tolist()
    )

    return recommendations

# CHAT ENTRY FUNCTION (USED BY app.py)
def get_recommendation(user_input):

    text = user_input.lower()

    # -------- Detect product --------
    if "loan" in text:
        product_col = "Loan_type"
        product_name = "loan"
    elif "credit" in text:
        product_col = "credit_cardtype"
        product_name = "credit card"
    elif "investment" in text:
        product_col = "investment_type"
        product_name = "investment"
    elif "saving" in text:
        product_col = "savings_plan_type"
        product_name = "savings"
    else:
        return "Please mention loan, credit card, investment, or savings."

    # Extract Customer ID (STRICT FORMAT)
    match = re.search(r'[cC]\d{7}', user_input)

    if not match:
        return "Please provide a valid Customer ID (Example: C1234567)."

    customer_id = match.group().upper().strip()

    # Get Recommendation
    result = recommend_from_cluster(customer_id, product_col)

    if not result:
        return f"Customer ID : {customer_id}\nCustomer ID not found."

    return (
        f"Customer ID : {customer_id}\n"
        f"For your {product_name}, we suggest : {', '.join(result)}"
    )