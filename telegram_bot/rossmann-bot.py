
import re
import requests
import json
import os

import pandas as pd
import numpy as np

from flask import Flask, request, Response

# contants
TOKEN = '8627283937:AAHkSpBpZesmkd3_ZbVRD0a2ifERiphzy1g'

# # info about the Bot
# #https://api.telegram.org/bot8627283937:AAHkSpBpZesmkd3_ZbVRD0a2ifERiphzy1g/getMe


# # get updates
# https://api.telegram.org/bot8627283937:AAHkSpBpZesmkd3_ZbVRD0a2ifERiphzy1g/getUpdates

# # Wehook
# https://api.telegram.org/bot8627283937:AAHkSpBpZesmkd3_ZbVRD0a2ifERiphzy1g/setWebhook?url=https://a5f3322f6a16ff.lhr.life

# # send message
# https://api.telegram.org/bot8627283937:AAHkSpBpZesmkd3_ZbVRD0a2ifERiphzy1g/sendMessage?chat_id=2135610318&text=Hi Deivisson, I am doing good, thanks!


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

    r = requests.post(url, json=data)

    print(r.status_code)
    print(r.text)

    d1 = pd.DataFrame( r.json(), columns=r.json()[0].keys() )
    
    return d1

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

                # calculation
                d2 = d1[['store', 'prediction']].groupby( 'store' ).sum().reset_index()

                # send message
                msg = 'Store Number {} will sell R${:,.2f} in the next 6 weeks'.format(
                                d2.loc[0, 'store'],
                                d2.loc[0, 'prediction']) 
                
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