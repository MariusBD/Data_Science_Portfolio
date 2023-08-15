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


#2.CARGA DATOS
ruta_proyecto = 'C:/Users/isaac/Google Drive/DS4B/CursoMachineLearningPython/03_MACHINE_LEARNING/08_CASOS/03_RIESGOS'
nombre_fichero_datos = 'validacion.csv'
ruta_completa = ruta_proyecto + '/02_Datos/02_Validacion/' + nombre_fichero_datos
df = pd.read_csv(ruta_completa,index_col='id_cliente').drop(columns='Unnamed: 0')


#3.VARIABLES Y REGISTROS FINALES
variables_finales = ['ingresos_verificados',
                     'vivienda',
                     'finalidad',
                     'num_cuotas',
                     'antigüedad_empleo',
                     'rating',
                     'ingresos',
                     'dti',
                     'num_lineas_credito',
                     'porc_uso_revolving',
                     'principal',
                     'tipo_interes',
                     'imp_cuota',
                     'num_derogatorios'
                  ]
a_eliminar = df.loc[df.ingresos > 300000].index.values
df = df[~df.index.isin(a_eliminar)]
df = df[variables_finales]


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


#5.CALIDAD Y CREACION DE VARIABLES
x_pd = creacion_variables(calidad_datos(df))
x_ead = creacion_variables(calidad_datos(df))
x_lgd = creacion_variables(calidad_datos(df))


#6.CARGA PIPES DE EJECUCION
ruta_pipe_ejecucion_pd = ruta_proyecto + '/04_Modelos/pipe_ejecucion_pd.pickle'
ruta_pipe_ejecucion_ead = ruta_proyecto + '/04_Modelos/pipe_ejecucion_ead.pickle'
ruta_pipe_ejecucion_lgd = ruta_proyecto + '/04_Modelos/pipe_ejecucion_lgd.pickle'

with open(ruta_pipe_ejecucion_pd, mode='rb') as file:
   pipe_ejecucion_pd = pickle.load(file)

with open(ruta_pipe_ejecucion_ead, mode='rb') as file:
   pipe_ejecucion_ead = pickle.load(file)

with open(ruta_pipe_ejecucion_lgd, mode='rb') as file:
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
