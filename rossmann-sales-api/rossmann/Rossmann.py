import pickle
import pandas as pd
import numpy as np
import re
import os
import math
import datetime

class Rossmann (object):
    def __init__( self ):

        BASE_DIR = os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__)
            )
        )

        parameter_path = os.path.join(BASE_DIR, 'parameter')   


        self.competition_distance_scaler   = pickle.load(open(os.path.join(parameter_path, 'competition_distance_scaler.pkl'), 'rb' ))
        self.competition_time_month_scaler = pickle.load(open(os.path.join(parameter_path, 'competition_time_month_scaler.pkl'), 'rb' ))
        self.promo_time_week_scaler        = pickle.load(open(os.path.join(parameter_path, 'promo_time_week_scaler.pkl'), 'rb' ))
        self.year_scaler                   = pickle.load(open(os.path.join(parameter_path, 'year_scaler.pkl'), 'rb' ))
        self.store_type_scaler             = pickle.load(open(os.path.join(parameter_path, 'store_type_scaler.pkl'), 'rb' ))

    def data_cleaning (self, df1):

        # renomear colunas 

        df1 = df1.rename(columns=lambda c: re.sub(r'(?<!^)(?=[A-Z])', '_', c).lower())
        
        # Data Types 
        df1['date'] = pd.to_datetime(df1['date'])

        # Fillout NA
        
        # =============================
        # competition_distance
        # =============================
        df1['competition_distance'] = df1['competition_distance'].fillna(200000.0)

        # =============================
        # competition_open_since_month
        # =============================
        df1['competition_open_since_month'] = df1['competition_open_since_month'].fillna(df1['date'].dt.month)

        # =============================
        # competition_open_since_year
        # =============================
        df1['competition_open_since_year'] = df1['competition_open_since_year'].fillna(df1['date'].dt.year)

        # =============================
        # promo2_since_week
        # =============================
        df1['promo2_since_week'] = df1['promo2_since_week'].fillna(df1['date'].dt.isocalendar().week)

        # =============================
        # promo2_since_year
        # =============================
        df1['promo2_since_year'] = df1['promo2_since_year'].fillna(df1['date'].dt.year)

        # =============================
        # promo_interval
        # =============================
        month_map = {
            1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr',
            5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug',
            9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
        }

        df1['promo_interval'] = df1['promo_interval'].fillna(0)
        df1['month_map'] = df1['date'].dt.month.map(month_map)

        # =============================
        # is_promo
        # =============================

        df1['is_promo'] = df1.apply(
            lambda x: 0 if x['promo_interval'] == 0 
            else 1 if x['month_map'] in x['promo_interval'].split(',') 
            else 0,
            axis=1
        )

        # Change Data Types

        # competiton
        df1['competition_open_since_month'] = df1['competition_open_since_month'].astype(int)
        df1['competition_open_since_year'] = df1['competition_open_since_year'].astype(int)

        # promo2
        df1['promo2_since_week'] = df1['promo2_since_week'].astype( int )
        df1['promo2_since_year'] = df1['promo2_since_year'].astype( int )

        return df1
    
    def feature_engineering (self, df2):
        
        # date
        df2['year'] = df2['date'].dt.year
        df2['month'] = df2['date'].dt.month
        df2['day'] = df2['date'].dt.day
    
        # week_of_year
        df2['week_of_year'] = df2['date'].dt.isocalendar().week.astype(int)

        # year_week
        df2['year_week'] = df2['date'].dt.strftime('%Y-%W')
    
        # competition_since
        df2['competition_since'] = pd.to_datetime(
            dict(year=df2['competition_open_since_year'],
                month=df2['competition_open_since_month'],
                day=1)
        )
    
        # competition_time_month
        df2['competition_time_month'] = (
            (df2['date'] - df2['competition_since']).dt.days / 30
        ).fillna(0)

        df2['competition_time_month'] = df2['competition_time_month'].clip(lower=0).astype(int)
        
        # promo since
        df2['promo_since'] = (
            df2['promo2_since_year'].astype(str) + '-' +
            df2['promo2_since_week'].astype(str) + '-1'
        )

        df2['promo_since'] = pd.to_datetime(
            df2['promo_since'],
            format='%Y-%W-%w',
            errors='coerce'
        )
    
        # promo_time_week
        df2['promo_time_week'] = (
            (df2['date'] - df2['promo_since']) / np.timedelta64(1, 'W')
        ).fillna(0).astype(int)

        # assortment
        df2['assortment_type'] = df2['assortment'].map({
            'a':'basic','b':'extra','c':'extended'
        })

        # state holiday
        df2['state_holiday_desc'] = df2['state_holiday'].map({
            'a':'public_holiday',
            'b':'easter_holiday',
            'c':'christmas',
            '0':'regular_day'
        })
        # promo month
        df2['month_map'] = df2['date'].dt.strftime('%b')

        df2['promo_month'] = df2.apply(
            lambda x: 0 if x['promo_interval'] == 0
            else 1 if x['month_map'] in x['promo_interval']
            else 0,
        axis=1
        )

        # is promo
        df2['is_promo'] = df2['promo'] * df2['promo_month']

        # competition distance log
        df2['competition_distance_log'] = np.log1p(df2['competition_distance'])

        # Passo 03 - Filtragem de variáveis

        df2 = df2[(df2['open'] !=0)]

        cols_drop = ['open', 'promo_interval', 'month_map']
        df2 = df2.drop( cols_drop, axis=1, errors='ignore' )

        return df2
    
    def data_preparation (self, df4):
        
        # Rescaling
        # competition distance
        df4['competition_distance'] = self.competition_distance_scaler.transform( df4[['competition_distance']].values )
        
        # competition time month
        df4['competition_time_month'] = self.competition_time_month_scaler.transform(df4[['competition_time_month']].values )        

        # promo time week
        df4['promo_time_week'] = self.promo_time_week_scaler.transform( df4[['promo_time_week']].values )        

        # year
        df4['year'] = self.year_scaler.transform( df4[['year']].values )

        # Encoding
        
        # store_type - Label Encoding        
        df4['store_type'] = self.store_type_scaler.transform( df4['store_type'] )        

        # assortment - Ordinal Encoding
        assortment_dict = {'basic': 1, 'extra': 2, 'extended': 3}
        df4['assortment_type'] = df4['assortment_type'].map( assortment_dict )

        # Nature Transformation
    
        # day of week (assumindo ciclo de 7 dias)
        df4['day_of_week_sin'] = np.sin(df4['day_of_week'] * (2. * np.pi / 7))
        df4['day_of_week_cos'] = np.cos(df4['day_of_week'] * (2. * np.pi / 7))

        # month (ciclo de 12 meses)
        df4['month_sin'] = np.sin(df4['month'] * (2. * np.pi / 12))
        df4['month_cos'] = np.cos(df4['month'] * (2. * np.pi / 12))

        # day (usando 30 como aproximação média para os dias do mês)
        df4['day_sin'] = np.sin(df4['day'] * (2. * np.pi / 30))
        df4['day_cos'] = np.cos(df4['day'] * (2. * np.pi / 30))

        # week of year (ciclo de 52 semanas no ano)
        df4['week_of_year_sin'] = np.sin(df4['week_of_year'] * (2. * np.pi / 52))
        df4['week_of_year_cos'] = np.cos(df4['week_of_year'] * (2. * np.pi / 52))

        selected_features = ['store', 'competition_distance', 'competition_open_since_year', 'competition_open_since_month', 'store_type', 'competition_distance_log', 'day_of_week_sin',
                             'promo2_since_week', 'assortment_type', 'promo2_since_year', 'promo_time_week', 'competition_time_month', 'promo2', 'day_of_week_cos', 'day_sin', 'day_cos', 'promo',
                             'week_of_year_cos']

        return df4[selected_features]
    
    def get_prediction(self, model, original_data, test_data):
        # prediction
        pred = model.predict (test_data)

        # join pred into the original data
        original_data['prediction'] = np.expm1(pred)

        return original_data