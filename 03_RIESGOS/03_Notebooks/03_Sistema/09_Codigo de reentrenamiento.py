#!/usr/bin/env python
# coding: utf-8

# ## CODIGO DE RE-ENTRENAMIENTO

# *NOTA: Para poder usar este código de entrenamiento hay que lanzarlo desde exactamente el mismo entorno en el que fue creado.*
#
# *Se puede instalar ese entorno en la nueva máquina usando el environment.yml que creamos en el set up del proyecto*
#
# *Copiar el riesgos.yml al directorio y en el terminal o anaconda prompt ejecutar:*
#
# conda env create --file riesgos.yml --name riesgos

# In[2]:


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
nombre_fichero_datos = 'prestamos.csv'
ruta_completa = ruta_proyecto + '/02_Datos/01_Originales/' + nombre_fichero_datos
df = pd.read_csv(ruta_completa,index_col=0)


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
                     'num_derogatorios',
                     'estado',
                     'imp_amortizado',
                     'imp_recuperado'
                  ]
a_eliminar = df.loc[df.ingresos > 300000].index.values
df = df[~df.index.isin(a_eliminar)]
df = df[variables_finales]


#4.FUNCIONES DE SOPORTE
def calidad_datos(temp):
    temp['antigüedad_empleo'] = temp['antigüedad_empleo'].fillna('desconocido')
    temp.select_dtypes('number').fillna(0,inplace=True)
    return(temp)

def creacion_variables_pd(df):
    temp = df.copy()
    temp['target_pd'] = np.where(temp.estado.isin(['Charged Off','Does not meet the credit policy. Status:Charged Off','Default']), 1, 0)
    temp.vivienda = temp.vivienda.replace(['ANY','NONE','OTHER'],'MORTGAGE')
    temp.finalidad = temp.finalidad.replace(['wedding','educational','renewable_energy'],'otros')
    #Eliminamos las variables que ya no usaremos
    temp.drop(columns = ['estado','imp_amortizado','imp_recuperado'],inplace = True)
    #Separamos entre predictoras y target
    temp_x = temp.iloc[:,:-1]
    temp_y = temp.iloc[:,-1]
    return(temp_x,temp_y)

def creacion_variables_ead(df):
    temp = df.copy()
    temp['pendiente'] = temp.principal - temp.imp_amortizado
    temp['target_ead'] = temp.pendiente / temp.principal
    temp.vivienda = temp.vivienda.replace(['ANY','NONE','OTHER'],'MORTGAGE')
    temp.finalidad = temp.finalidad.replace(['wedding','educational','renewable_energy'],'otros')
    #Eliminamos las variables que ya no usaremos
    temp.drop(columns = ['estado','imp_amortizado','imp_recuperado','pendiente'],inplace = True)
    #Separamos entre predictoras y target
    temp_x = temp.iloc[:,:-1]
    temp_y = temp.iloc[:,-1]
    return(temp_x,temp_y)

def creacion_variables_lgd(df):
    temp = df.copy()
    temp['pendiente'] = temp.principal - temp.imp_amortizado
    temp['target_lgd'] = 1 - (temp.imp_recuperado / temp.pendiente)
    temp['target_lgd'].fillna(0,inplace=True)
    temp.vivienda = temp.vivienda.replace(['ANY','NONE','OTHER'],'MORTGAGE')
    temp.finalidad = temp.finalidad.replace(['wedding','educational','renewable_energy'],'otros')
    #Eliminamos las variables que ya no usaremos
    temp.drop(columns = ['estado','imp_amortizado','imp_recuperado','pendiente'],inplace = True)
    #Separamos entre predictoras y target
    temp_x = temp.iloc[:,:-1]
    temp_y = temp.iloc[:,-1]
    return(temp_x,temp_y)


#5.CALIDAD Y CREACION DE VARAIBLES
x_pd, y_pd = creacion_variables_pd(calidad_datos(df))
x_ead, y_ead = creacion_variables_ead(calidad_datos(df))
x_lgd, y_lgd = creacion_variables_pd(calidad_datos(df))


#6.CARGA PIPES DE ENTRENAMIENTO
ruta_pipe_entrenamiento_pd = ruta_proyecto + '/04_Modelos/pipe_entrenamiento_pd.pickle'
ruta_pipe_entrenamiento_ead = ruta_proyecto + '/04_Modelos/pipe_entrenamiento_ead.pickle'
ruta_pipe_entrenamiento_lgd = ruta_proyecto + '/04_Modelos/pipe_entrenamiento_lgd.pickle'

with open(ruta_pipe_entrenamiento_pd, mode='rb') as file:
   pipe_entrenamiento_pd = pickle.load(file)

with open(ruta_pipe_entrenamiento_ead, mode='rb') as file:
   pipe_entrenamiento_ead = pickle.load(file)

with open(ruta_pipe_entrenamiento_lgd, mode='rb') as file:
   pipe_entrenamiento_lgd = pickle.load(file)


#7.ENTRENAMIENTO
pipe_ejecucion_pd = pipe_entrenamiento_pd.fit(x_pd,y_pd)
pipe_ejecucion_ead = pipe_entrenamiento_ead.fit(x_ead,y_ead)
pipe_ejecucion_lgd = pipe_entrenamiento_lgd.fit(x_lgd,y_lgd)


#8.GUARDA MODELOS ENTRENADOS EN PIPE DE EJECUCION
ruta_pipe_ejecucion_pd = ruta_proyecto + '/04_Modelos/pipe_ejecucion_pd.pickle'
ruta_pipe_ejecucion_ead = ruta_proyecto + '/04_Modelos/pipe_ejecucion_ead.pickle'
ruta_pipe_ejecucion_lgd = ruta_proyecto + '/04_Modelos/pipe_ejecucion_lgd.pickle'

with open(ruta_pipe_ejecucion_pd, mode='wb') as file:
   pickle.dump(pipe_ejecucion_pd, file)

with open(ruta_pipe_ejecucion_ead, mode='wb') as file:
   pickle.dump(pipe_ejecucion_ead, file)

with open(ruta_pipe_ejecucion_lgd, mode='wb') as file:
   pickle.dump(pipe_ejecucion_lgd, file)


# In[ ]:
