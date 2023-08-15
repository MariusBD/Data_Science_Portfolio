#1.LIBRERIAS
import numpy as np
import pandas as pd
import pickle

from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import MinMaxScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingRegressor

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline


#4.FUNCIONES DE SOPORTE
def calidad_datos(temp):
    temp['antigüedad_empleo'] = temp['antigüedad_empleo'].fillna('desconocido')
    temp.select_dtypes('number').fillna(0,inplace=True)
    return(temp)

def creacion_variables(df):
    temp = df.copy()
    temp.vivienda = temp.vivienda.replace(['ANY','NONE','OTHER'],'MORTGAGE')
    temp.finalidad = temp.finalidad.replace(['wedding','educational','renewable_energy'],'otros')
    return(temp)


#Ejecutamos el proceso solamente cuando el usuario haga click en el boton
def ejecutar_modelos(df):
    #5.CALIDAD Y CREACION DE VARIABLES
    x_pd = creacion_variables(calidad_datos(df))
    x_ead = creacion_variables(calidad_datos(df))
    x_lgd = creacion_variables(calidad_datos(df))


    #6.CARGA PIPES DE EJECUCION
    #al estar en la misma carpeta, ya encuentra ese objeto
    with open('pipe_ejecucion_pd.pickle', mode='rb') as file:
       pipe_ejecucion_pd = pickle.load(file)

    with open('pipe_ejecucion_ead.pickle', mode='rb') as file:
       pipe_ejecucion_ead = pickle.load(file)

    with open('pipe_ejecucion_lgd.pickle', mode='rb') as file:
       pipe_ejecucion_lgd = pickle.load(file)


    #7.EJECUCION
    scoring_pd = pipe_ejecucion_pd.predict_proba(x_pd)[:, 1]
    ead = pipe_ejecucion_ead.predict(x_ead)
    lgd = pipe_ejecucion_lgd.predict(x_lgd)


    #8.RESULTADO
    principal = x_pd.principal
    EL = pd.DataFrame({'principal':principal,
                       'pd':scoring_pd,
                       'ead':ead,
                       'lgd':lgd
                       })
    EL['perdida_esperada'] = round(EL.pd * EL.principal * EL.ead * EL.lgd,2)

    return(EL)
