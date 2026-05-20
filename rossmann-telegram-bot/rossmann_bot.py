
import re
import requests
import os
import numpy as np

from flask import Flask, request, Response

# constants
TOKEN = os.environ.get('TELEGRAM_TOKEN')

def send_message( chat_id, text ):

    url = 'https://api.telegram.org/bot{}/'.format(TOKEN)
    url = url + 'sendMessage?chat_id={}'.format( chat_id ) 

    r = requests.post(url, json={'text': text})
    print('Status Code {}'.format(r.status_code))
    
    return None


def load_dataset( store_id ):

    # ============================================================
    # Paths
    # ============================================================

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # path test dataset
    df10 = os.path.join(BASE_DIR, 'dataset', 'test.csv')
    
    # loading test dataset
    df_test = pd.read_csv(df10)

    # convert CamelCase to snake_case
    df_test = df_test.rename(
        columns=lambda c: re.sub(r'(?<!^)(?=[A-Z])', '_', c).lower()
    )
    
    # choose store for prediction
    df_test = df_test[df_test['store'] == store_id]

    if not df_test.empty:
        # remove closed days
        df_test = df_test[df_test['open'] !=0]
        df_test = df_test[~df_test['open'].isnull()]
        df_test = df_test.drop('id', axis = 1)

        data = (
            df_test
            .replace({np.nan: None})
            .to_dict(orient='records')
        )
    
    else:
         data = 'error'

    return data


def predict ( data ):
    
    # api call
    url = 'https://rossmann-sales-predictor-05it.onrender.com/predict'

    try:

        r = requests.post(url, json=data, timeout=120)

        print(r.status_code)
        print(r.text)

        # verificacao se a API respondeu corretamente
        if r.status_code != 200:
            return 'error'        
              
        return r.json()
    
    except requests.exceptions.Timeout:
        print('Request Timeout')
        return 'error'

    except requests.exceptions.ConnectionError:
        print('Connection Error')
        return 'error'

    except Exception as e:
        print(f'Unexpected Error: {e}')
        return 'error'

def parse_message( message ):
    chat_id = message['message']['chat']['id']
    text = message['message']['text']

    store_id = text.replace('/', '')

    try:
        store_id = int( store_id)
    
    except ValueError:
        store_id = 'error'


    return chat_id, store_id

# API initialize
app = Flask(__name__)

@app.route('/predict', methods =['GET','POST'] )

def index():
    if request.method == 'POST':
        message = request.get_json()

        print(message)

        chat_id, store_id = parse_message( message )

        if store_id != 'error':

            # loading data
            data = load_dataset( store_id )

            if data != 'error':

                # prediction
                d1 = predict( data )

                if isinstance(d1, str):
                    send_message(chat_id, 'Prediction service unavailable')

                    return Response('OK', status=200)
                
                # send message
                msg = (
                    f"📈 Store {d1['store']}\n\n"
                    f"Predicted Sales (6 weeks): R${d1['prediction']:,.2f}\n\n"
                    f"📉 Worst Scenario: R${d1['worst_scenario']:,.2f}\n"
                    f"📈 Best Scenario: R${d1['best_scenario']:,.2f}"
                )

                send_message(chat_id, msg)

                return Response( 'Ok', status=200)
            
            else:
                send_message( chat_id, 'Store Not Available')

                return Response('OK', status=200)

        else:
            send_message(chat_id, 'Store ID is Wrong')

            return Response( 'Ok', status=200)
    
    
    else:
        return '<h1> Rossman Telegram BOT <h>'


# =========================================================
# Main
# =========================================================
if __name__ == '__main__':
    port = os.environ.get ('PORT', 5000)
    app.run(host='0.0.0.0', port=port)