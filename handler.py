import os
import re
import pickle
import pandas as pd

from flask import Flask, request, Response
from rossmann.Rossmann import Rossmann


# ============================================================
# Paths
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# model_path
model_path = os.path.join(BASE_DIR, 'model', 'model_rossmann.pkl')

# dataset path
store_path = os.path.join(BASE_DIR, 'dataset', 'store.csv')

# ============================================================
# Load files
# ============================================================

# load model
model = pickle.load(open(model_path, 'rb'))

# load store dataset
df_store_raw = pd.read_csv(store_path)

# convert CamelCase to snake_case
df_store_raw = df_store_raw.rename(
    columns=lambda c: re.sub(r'(?<!^)(?=[A-Z])', '_', c).lower()
)

# ensure dtype
df_store_raw['store'] = df_store_raw['store'].astype(int)

# ============================================================
# initialize API
# ============================================================

app = Flask(__name__)

# ============================================================
# Prediction Endpoint
# ============================================================

@app.route('/predict', methods=['POST'])
def rossmann_predict():
    test_json = request.get_json()

    if test_json:
        if isinstance(test_json, dict):
            test_raw = pd.DataFrame(test_json, index=[0])
        else:
            test_raw = pd.DataFrame(test_json)
        
        # lowercase request columns
        test_raw.columns = map(str.lower, test_raw.columns)
        
        # ensure same dtype
        test_raw['store'] = test_raw['store'].astype(int)        

        # merge store dataset
        test_raw = pd.merge(
            test_raw,
            df_store_raw,
            how='left',
            on='store'
        )        

        # pipeline
        pipeline = Rossmann()

        df1 = pipeline.data_cleaning(test_raw)
        df2 = pipeline.feature_engineering(df1)
        df3 = pipeline.data_preparation(df2)

        # prediction
        df_response = pipeline.get_prediction(model, test_raw, df3)

        return df_response

    return Response({}, status=200, mimetype='application/json')

# =========================================================
# Main
# =========================================================

if __name__ == '__main__':
    port = os.environ.get('PORT', 5000)
    app.run('0.0.0.0', port = port)
