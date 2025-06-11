import datetime
import numpy as np
import pandas as pd
import traceback
from python_scripts.clases import *
from datetime import timedelta
from xgboost import XGBRegressor
from warnings import simplefilter
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, train_test_split
import json
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sqlalchemy import create_engine
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


def cargar_modelos():
    """
    Carga los modelos XGBoost pre-entrenados para las diferentes etapas del proceso.
    
    Returns:
        tuple: Una tupla que contiene cuatro modelos XGBoost:
            - mod_a: Modelo para la etapa A
            - mod_b: Modelo para la etapa B
            - mod_c: Modelo para la etapa C
            - mod_: Modelo para el tiempo final de picking
    """
    mod_a = XGBRegressor()
    mod_a.load_model('df/input/models/best_xgboost_model_Dif_a.json')

    mod_b = XGBRegressor()
    mod_b.load_model('df/input/models/best_xgboost_model_Dif_b.json')

    mod_c = XGBRegressor()
    mod_c.load_model('df/input/models/best_xgboost_model_Dif_c.json')

    mod_ = XGBRegressor()
    mod_.load_model('df/input/models/xgb_model_Fin_pck-mov_hora_control.json')
    return mod_a, mod_b, mod_c, mod_


def cargar_productos_voluminosos():
    """
    Carga el archivo de productos voluminosos desde un archivo Excel.
    
    Returns:
        pandas.DataFrame: DataFrame que contiene la información de productos voluminosos
    """
    return pd.read_excel('df/input/prod_voluminosos.xlsx')


def get_database_url(resultados=True):
    """
    Genera la URL de conexión a la base de datos según el entorno especificado.
    
    Args:
        resultados (bool, optional): Si es True, usa credenciales para la base de datos de resultados.
                                   Si es False, usa credenciales para la base de datos de producción.
                                   Por defecto es True.
    
    Returns:
        str: URL de conexión a la base de datos formateada según el motor de base de datos
    """
    if resultados:
        username = 'biwiser'
        password = 'bw2024Elfle'
        hostname = 'elfle-srv09.elfle.local'
        database = 'BIWISER'
        database_url = f"mssql+pyodbc://{username}:{password}@{hostname}/{database}?driver=ODBC+Driver+17+for+SQL+Server"
    else:
        username = 'usrview'
        password = 'EkwU96YRyDKJdr8'
        hostname = '192.168.1.8'
        database = 'panal_wms'
        database_url = f"mysql+mysqlconnector://{username}:{password}@{hostname}/{database}"
    return database_url


def fetch_data_from_view(view_name, engine, query=None):
    """
    Ejecuta una consulta SQL en una vista o tabla de la base de datos.
    
    Args:
        view_name (str): Nombre de la vista o tabla de la que se obtendrán los datos
        engine: Objeto de conexión a la base de datos SQLAlchemy
        query (str, optional): Consulta SQL personalizada. Si no se proporciona,
                             se usará 'SELECT * FROM view_name'. Por defecto es None.
    
    Returns:
        pandas.DataFrame: DataFrame con los resultados de la consulta, o None si ocurre un error
    """
    try:
        if query:
            df = pd.read_sql(query, engine)
        else:
            query = f"SELECT * FROM {view_name}"
            df = pd.read_sql(query, engine)
        return df
    except Exception as e:
        print(f"Error al traer los datos de la vista {view_name}: {e}")
        return None


def fetch_latest_results(n=100):
    try:
        # Obtener la URL de conexión
        database_url = get_database_url(resultados=True)

        # Crear el motor de conexión
        engine = create_engine(database_url)

        # Definir la consulta
        query = f"SELECT TOP {n} * FROM prediccion_pck ORDER BY version_control DESC"

        # Ejecutar la consulta y traer los datos en un DataFrame de pandas
        df = pd.read_sql(query, engine)

        return df

    except Exception as e:
        print(f"Error al traer los datos: {e}")
        return None


def keep_last_n_rows(df, n=40):
    """
    Mantiene solo las últimas n filas de un DataFrame y resetea el índice.

    Parámetros:
    df (pd.DataFrame): El DataFrame original.
    n (int): El número de filas a mantener. Por defecto es 40.

    Retorna:
    pd.DataFrame: El DataFrame actualizado con las últimas n filas y el índice reseteado.
    """
    # Mantener solo las últimas n filas
    df = df.tail(n)

    # Resetear el índice
    df.reset_index(drop=True, inplace=True)

    return df


def cargar_informacion(sim):
    """
    Params:
    sim : Boolean. Si es verdadero considerara simulacion en base a sabanas de datos, Falso indica conexion a datos BBDD
    Funcion que actualmente carga excel. Posteriormente podria involucrar la conexion a BBDD
    :return: 3 df con info
    """
    if sim:
        print('Cargando información')
        df_pasillo = pd.read_excel('df/input/informe_cantidades_pck.xlsx')
        tiempos_pck = pd.read_excel('df/input/tiempo_operaciones_v2.xlsx', skiprows=1).dropna(subset="mov_llamado")
        prod_vol = pd.read_excel('df/input/prod_voluminosos.xlsx')
        tiempos_pck = tiempos_pck.sort_values(by='mov_llamado')
        tiempos_pck["mov_llamado"] = pd.to_datetime(tiempos_pck["mov_llamado"])
        return df_pasillo, tiempos_pck, prod_vol
    else:
        print('Inicializando Conexion a BBDD')
        # prod_vol = pd.read_excel('df/input/prod_voluminosos.xlsx')
        # cargar info pasillo y tiempos_pck
        database_url = get_database_url(resultados=False)
        engine = create_engine(database_url)
        # PRIMERO CARGA TIEMPO
        df_dia = fetch_data_from_view('vw_pck_docs', engine,
                                      query='select * from vw_pck_docs where mov_entregado is null')
        #df_dia = df_dia[(df_dia.mov_entregado.isnull())]
        if df_dia is None:
            return None, None
        if len(df_dia) == 0 or df_dia.empty:
            print('WARNING query a vw_pck_docs no trae información')
            return None, None
        df_dia = df_dia.sort_values(by='mov_llamado')
        df_dia["mov_llamado"] = pd.to_datetime(df_dia["mov_llamado"])
        df_dia["mov_entregado"] = pd.to_datetime(df_dia["mov_entregado"])
        df_dia = df_dia[df_dia['mov_llamado'].dt.date == datetime.date.today()]
        df_dia['difA'] = df_dia['difA'].apply(time_to_seconds)
        df_dia['difB'] = df_dia['difB'].apply(time_to_seconds)
        df_dia['difC'] = df_dia['difC'].apply(time_to_seconds)
        df_dia['Fin_pck'] = pd.NaT
        df_dia['dif_ini_fin'] = np.nan
        #try:
        #    df_dia['Fin_pck'] = df_dia[['finpck_A', 'finpck_B', 'finpck_C']].max(axis=1)
        #    df_dia['dif_ini_fin'] = (df_dia['Fin_pck'] - df_dia['ini_pck']).dt.total_seconds()
        #except Exception as e:
        #    print(e)
        #print('Error al crear columna dif_ini_fin',df_dia['Fin_pck'],df_dia['ini_pck'])
        df_dia.rename({'difA': 'Dif_a', "difB": 'Dif_b', "difC": 'Dif_c', 'ini_pck': 'Ini_pck'}, axis=1, inplace=True)
        folios_relevantes = df_dia.mov_folio.unique()
        folios_relevantes = ','.join([f"'{folio}'" for folio in folios_relevantes])
        query = f"""
                SELECT * 
                FROM vw_pck_pas_docs
                WHERE Folio IN ({folios_relevantes})
                """
        # CARGA PASILLO , solo lso folios validos de df_dia
        df_pasillo = fetch_data_from_view('vw_pck_pas_docs', engine, query=query)
        if (df_pasillo is None) or (len(df_pasillo) == 0) or (df_pasillo.empty):
            print('WARNING query a vw_pck_pas_docs no trae información')
            return None, None
        #Revisar FIN_PCK -> cuantos pasillos espero  de esos si tienen fin entonces puedo sacar max else nada.
        print('calculando_finck')
        df_dia['Fin_pck'] = df_dia.apply(lambda x: fin_pck_alcanzado(x, df_pasillo[df_pasillo.Folio == x['mov_folio']]),
                                         axis=1)
        folio_dia = set(df_dia.mov_folio.unique())
        folio_pasillo = set(df_pasillo.Folio.unique())
        print(f'Folio no encontrado en vw_pck_pas_docs {folio_dia- folio_pasillo}')
        inter = folio_dia.intersection(folio_pasillo)

        df_dia = df_dia[df_dia.mov_folio.isin(list(inter))]
        df_pasillo = df_pasillo[df_pasillo.Folio.isin(list(inter))]
        print('Finalizando Conexión a BBDD')
        return df_dia, df_pasillo


def fin_pck_alcanzado(row_dia, pasillos):
    """
    Determina si se ha alcanzado el tiempo final de picking para un pedido en todos los pasillos utilizados.
    
    Args:
        row_dia (pandas.Series): Fila del DataFrame con los tiempos de los pasillos.
        pasillos (pandas.DataFrame): DataFrame con la información de los pasillos utilizados.
    
    Returns:
        datetime or None: La máxima fecha de finalización si todos los pasillos han terminado,
                        None en caso contrario o si no hay pasillos.
    """
    pasillos_utilizados = []
    finalizado = True
    if len(pasillos) > 0:
        for index, row in pasillos.iterrows():
            if row['Pasillo_A'] > 0:
                pasillos_utilizados.append('finpck_A')
            elif row['Pasillo_B'] > 0:
                pasillos_utilizados.append('finpck_B')
            else:
                pasillos_utilizados.append('finpck_C')
        for pas in set(pasillos_utilizados):
            if pd.isnull(row_dia[pas]):
                finalizado = False
        if finalizado:

            return row_dia[list(set(pasillos_utilizados))].max(skipna=True)
        else:
            return pd.NaT
    else:
        return pd.NaT


def cargar_productos_pasillos():
    """
       Carga la información de productos y pasillos desde un archivo JSON.

       Returns:
           dict: Un diccionario con la información de productos y pasillos.
       """
    print('Cargando Productos pasillos')
    f = open('df/input/models/productos_pasillos.json', 'r')
    prod_pass = json.load(f)
    return prod_pass


def cargar_columnas_df():
    """
        Carga la información de las columnas del DataFrame desde un archivo JSON.

        Returns:
            dict: Un diccionario con la información de las columnas del DataFrame.
        """

    print('Cargando columnas dataset')
    f = open('df/input/models/colsDF.json', 'r')
    dict_cols = json.load(f)
    return dict_cols


def contar_productos_mayores_que_cero(row):
    """
        Cuenta el número de productos mayores que cero en una fila.

        Args:
            row: Una fila del DataFrame.

        Returns:
            int: El número de productos mayores que cero en la fila.
        """
    return sum(row > 0)


def intervalo_hora(df, freq=1, target="Dif_a"):
    """
        Crea intervalos de hora en el DataFrame y asigna etiquetas a cada intervalo.

        Args:
            df (DataFrame): El DataFrame al que se aplicarán los intervalos de hora.
            freq (int): La frecuencia de los intervalos de hora.
            target (str): La columna objetivo en la que se basarán los intervalos.

        Returns:
            DataFrame: El DataFrame con los intervalos de hora añadidos.
        """
    # Definir los intervalos de 2 horas
    intervalos = range(7, 19 + freq, freq)

    # Utilizar pd.cut para asignar etiquetas a cada intervalo de 2 horas
    df['Intervalo_Hora'] = pd.cut(df['hora'], bins=intervalos, right=False, include_lowest=True,
                                  labels=[f'{i}-{i + freq}' for i in intervalos[:-1]])
    # print(df['Intervalo_Hora'])
    # print(type(df['Intervalo_Hora']))
    intervalo_encoded = pd.get_dummies(df['Intervalo_Hora'], prefix='Intervalo')
    #print(intervalo_encoded)
    df_encoded = pd.concat([df, intervalo_encoded], axis=1)
    # target_ = df_encoded.pop(target)
    # df_encoded.insert(len(df_encoded.columns), target, target_)
    df_encoded.pop('Intervalo_Hora')
    return df_encoded


def filt_day(df, column, day):
    """
        Filtra un DataFrame por día en una columna específica.

        Args:
            df (DataFrame): El DataFrame a filtrar.
            column (str): La columna en la que se basará el filtro.
            day: El día para el filtro.

        Returns:
            DataFrame: El DataFrame filtrado por día.
        """
    return df[(df[column] > day) & (df[column] <= day + datetime.timedelta(hours=23))].sort_values(by='mov_llamado')


def xgb_predict(df, model):
    """
       Realiza predicciones utilizando un modelo XGBoost.

       Args:
           df (DataFrame): El DataFrame con los datos de entrada para la predicción.
           model: El modelo XGBoost utilizado para la predicción.

       Returns:
           DataFrame: El DataFrame con las predicciones añadidas.
       """
    # y = df[y_label]
    X = df[df.columns.tolist()[2:-1]]  # .drop(['Dif_a'], axis=1)  # Elimina la columna de tiemp
    y_pred = model.predict(X)
    df['y_pred'] = y_pred
    return df


def prediction_time(mod, pasillo='A', p=None):
    """
        Realiza predicciones de tiempo para un pasillo específico.

        Args:
            pasillo (str): El pasillo para el que se realizará la predicción.
            p: El objeto Pedido utilizado para la predicción.

        Returns:
            int: El tiempo de predicción para el pasillo especificado.
        """
    if pasillo == 'A':
        # mod = XGBRegressor()
        # mod.load_model('df/input/models/xgb_model_Dif_a.json')
        df = p.dfA
        X = df[df.columns.tolist()[2:]]  # .drop(['Dif_a'], axis=1)  # Elimina la columna de tiemp
        y_pred = mod.predict(X)
    elif pasillo == 'B':
        # mod = XGBRegressor()
        # mod.load_model('df/input/models/xgb_model_Dif_b.json')
        df = p.dfB
        X = df[df.columns.tolist()[2:]]  # .drop(['Dif_a'], axis=1)  # Elimina la columna de tiemp
        y_pred = mod.predict(X)

    elif pasillo == 'C':
        # mod = XGBRegressor()
        # mod.load_model('df/input/models/xgb_model_Dif_c.json')
        df = p.dfC
        X = df[df.columns.tolist()[2:]]  # .drop(['Dif_a'], axis=1)  # Elimina la columna de tiemp
        y_pred = mod.predict(X)

    elif pasillo == '_':
        # mod = XGBRegressor()
        # mod.load_model('df/input/models/xgb_model_Fin_pck-mov_hora_control.json')
        df = p.df_
        X = df[df.columns.tolist()[2:]]  # .drop(['Dif_a'], axis=1)  # Elimina la columna de tiemp
        y_pred = mod.predict(X)
    if int(y_pred[0]) < 1:
        print(f'Modelo pred {X}')
        print(int(y_pred[0]))
    return int(y_pred[0])


def convertir_a_int(elemento):
    if isinstance(elemento, pd.Series):
        return elemento.iloc[0]
    else:
        #asumiendo que si no es series entonces si es int
        return elemento


def nuevo_pedido(i, df_dia, df_pasillo, prod_pass, dict_cols, lista_productos, mod_a, mod_b, mod_c, simulado=True,
                 resultado_bbdd=None, productos_pasillos_arreglado=None):
    """
        Crea un nuevo objeto Pedido.

        Args:
            i (int): El índice del pedido.
            df_dia (DataFrame): El DataFrame del día.
            df_pasillo (DataFrame): El DataFrame de pasillos.
            prod_pass (list): La lista de productos en el pasillo.
            dict_cols (dict): Un diccionario con información de columnas.
            lista_productos (list): La lista de productos.
            simulado (bool): Indica si el pedido es simulado o no.

        Returns:
            Pedido: El objeto Pedido creado.
            :param mod_a:
        """
    row = df_dia.iloc[i]
    df_pasillos = df_pasillo[df_pasillo.Folio == row['mov_folio']]
    dict_results = get_real_info(row, resultado_bbdd)
    dfA, dfB, dfC, df_ = creacion_pre_forecast(df_pasillos, row.to_frame().T, prod_pass, dict_cols, lista_productos)
    productos = df_pasillos.Producto.tolist()
    if not any(prod in productos_pasillos_arreglado['A'] for prod in productos):
        dfA = pd.DataFrame()
    if not any(prod in productos_pasillos_arreglado['B'] for prod in productos):
        dfB = pd.DataFrame()
    if not any(prod in productos_pasillos_arreglado['C'] for prod in productos):
        dfC = pd.DataFrame()

    p = Pedido(folio=row['mov_folio'], doc=row['Doc'], hora_meson=row['HoraMeson'], hora_llamado=row['mov_llamado'],
               dfA=dfA, dfB=dfB, dfC=dfC, df_=df_)
    if len(resultado_bbdd[resultado_bbdd.mov_folio == row['mov_folio']]) > 0:
        for key, value in dict_results.items():
            setattr(p, key, value)
    else:
        p.pred_ini_pck = int(np.random.triangular(5, 12, 40))
        p.hora_ini_pck = generar_time(p.hora_llamado, p.pred_ini_pck)
        p.pred_ = int(np.random.triangular(20, 250, 500))
        
        if len(p.dfA) > 0:
            p.predA = prediction_time(mod_a, 'A', p)
        if len(p.dfB) > 0:
            p.predB = prediction_time(mod_b, 'B', p)
        if len(p.dfC) > 0:
            p.predC = prediction_time(mod_c, 'C', p)

    if simulado:
        p.REAL_pred_ini_pck = (row['Ini_pck'] - row['mov_llamado']).total_seconds()
        if not isinstance(row['Dif_a'], float):
            p.REAL_predA = datetime.timedelta(hours=row['Dif_a'].hour, minutes=row['Dif_a'].minute,
                                              seconds=row['Dif_a'].second).total_seconds()
        if not isinstance(row['Dif_b'], float):
            p.REAL_predB = datetime.timedelta(hours=row['Dif_b'].hour, minutes=row['Dif_b'].minute,
                                              seconds=row['Dif_b'].second).total_seconds()
        if not isinstance(row['Dif_c'], float):
            p.REAL_predC = datetime.timedelta(hours=row['Dif_c'].hour, minutes=row['Dif_c'].minute,
                                              seconds=row['Dif_c'].second).total_seconds()
            # print(p.REAL_predA,p.REAL_predB,p.REAL_predC)

        p.REAL_pred_max = max([p.REAL_predA, p.REAL_predB, p.REAL_predC])
        p.REAL_pred_ = (row['mov_entregado'] - row['Fin_pck']).total_seconds()
        p.REAL_hora_ini_pck = row['Ini_pck']
        p.REAL_hora_ini_pckA = row['inipck_A']
        p.REAL_hora_ini_pckB = row['inipck_B']
        p.REAL_hora_ini_pckC = row['inipck_C']
        p.REAL_hora_fin_pck = row['Fin_pck']
        p.REAL_hora_fin_pckA = row['finpck_A']
        p.REAL_hora_fin_pckB = row['finpck_B']
        p.REAL_hora_fin_pckC = row['finpck_C']
        p.REAL_fecha_termino = row['mov_entregado']

    return p


def get_real_info(info, bbdd):
    """
    Si es que la información está en info utilizo la real, en caso contrario utilizo la predicha previamente en interaciones previas y
    guardads en bbdd. En caso de las peuabs actuales le entrego la hora para saber que información deberia saber.
    :param info:
    :param bbdd:
    :param hora:
    :return:
    """

    mapeo = {'hora_ini_pck': 'Ini_pck', 'predA': 'Dif_a', 'hora_ini_pckA': 'inipck_A',
             'hora_fin_pckA': 'finpck_A', 'predB': 'Dif_b', 'hora_ini_pckB': 'inipck_B',
             'hora_fin_pckB': 'finpck_B', 'predC': 'Dif_c', 'hora_ini_pckC': 'finpck_C', 'hora_fin_pckC': 'finpck_C',
             'hora_fin_pck': 'Fin_pck', 'pred_': None, 'mov_entregado': 'mov_entregado'}
    dict_return = {'hora_ini_pck': None, 'predA': None, 'hora_ini_pckA': None,
                   'hora_fin_pckA': None, 'predB': None, 'hora_ini_pckB': None,
                   'hora_fin_pckB': None, 'predC': None, 'hora_ini_pckC': None, 'hora_fin_pckC': None,
                   'hora_fin_pck': None, 'pred_': None, 'mov_entregado': None}
    bd = bbdd[bbdd.mov_folio == info['mov_folio']]
    for p_col, col in mapeo.items():

        try:
            if (col is None) or (pd.isnull(info[col])):
                # predict
                if len(bd) > 0:
                    dict_return[p_col] = bd[p_col].iloc[0]
                else:
                    # No está en bbdd
                    pass

            else:
                dict_return[p_col] = info[col]
                #if isinstance(info[col], datetime.datetime):
                #    if len(bd) > 0:
                #        dict_return[p_col] = bd[p_col].iloc[0]

                #else:
                #dict_return[p_col] = info[col]
                #  valor real
        except Exception as e:
            print(e)
            if len(bd) > 0:
                dict_return[p_col] = bd[p_col].iloc[0]
            print(traceback.format_exc())
    return dict_return


def time_to_seconds(x):
    if type(x) == datetime.time:
        return datetime.timedelta(hours=x.hour, minutes=x.minute, seconds=x.second).total_seconds()
    else:
        return None


def generar_time(hora, seg):
    """
        Genera una nueva marca de tiempo a partir de una hora inicial y una cantidad de segundos.

        Args:
            hora: La hora inicial.
            seg (int): La cantidad de segundos.

        Returns:
            datetime: La nueva marca de tiempo generada.
        """
    return hora + datetime.timedelta(seconds=seg)


def calcular_resultados(completados):
    """
        Calcula los resultados de los pedidos completados.

        Args:
            completados (list): La lista de pedidos completados.

        Returns:
            dict: Un diccionario con los resultados calculados.
        """
    atributos = ['predA', 'predB', 'predC', 'pred_']
    resultados = {}
    no_aceptados = 0
    aceptados = 0
    try:
        for at in atributos:
            pred = []
            real = []
            for p in completados:
                real_value = getattr(p, f'REAL_{at}')
                pred_value = getattr(p, at)
                if real_value > 0 and pred_value > 0:
                    if np.abs(pred_value - real_value) < 2000:
                        pred.append(pred_value)
                        real.append(real_value)
                        aceptados += 1
                    else:
                        no_aceptados += 1
            res = mean_absolute_error(real, pred)
            res1 = mean_squared_error(real, pred, squared=False)
            res2 = np.mean([pred[i] - real[i] for i in range(len(real))])
            resultados[at] = res
            resultados[f"{at}_sq"] = res1
            resultados[f"{at}_noabs"] = res2
        dif = []
        dif1 = []
        for p in completados:
            real_value = getattr(p, f'REAL_mov_entregado')
            pred_value = getattr(p, "mov_entregado")

            delta = np.abs((pred_value - real_value).total_seconds())
            if delta < 2500:
                dif.append(delta)
                dif1.append((pred_value - real_value).total_seconds())
                aceptados += 1
            else:
                no_aceptados += 1
        resultados['mov_entregado'] = np.mean(dif)
        resultados['mov_entregado_noabs'] = np.mean(dif1)
        dif = []
        dif1 = []
        for p in completados:
            real_value = getattr(p, f'REAL_hora_fin_pck')
            pred_value = getattr(p, "hora_fin_pck")
            # if not pred_value:
            #    print(pedido)
            # if np.abs((pred_value - real_value).total_seconds())>2400:
            #    print(pedido)
            delta = np.abs((pred_value - real_value).total_seconds())
            if delta < 2500:
                dif.append(delta)
                dif1.append((pred_value - real_value).total_seconds())
                aceptados += 1
            else:
                no_aceptados += 1
        resultados['hora_fin_pck'] = np.mean(dif)
        resultados['hora_fin_pck_noabs'] = np.mean(dif1)
        dif = []
        dif1 = []
        for p in completados:
            real_value = getattr(p, f'REAL_hora_ini_pck')
            pred_value = getattr(p, "hora_ini_pck")
            # if not pred_value:
            #    print(pedido)
            # if np.abs((pred_value - real_value).total_seconds())>2400:
            #    print(pedido)
            delta = np.abs((pred_value - real_value).total_seconds())
            if delta < 2000:
                dif.append(delta)
                dif1.append((pred_value - real_value).total_seconds())
                aceptados += 1
            else:
                no_aceptados += 1
        resultados['hora_ini_pck'] = np.mean(dif)
        resultados['hora_ini_pck_noabs'] = np.mean(dif1)
        print(f'NO aceptados {no_aceptados} de {no_aceptados + aceptados} ')
        for key, value in resultados.items():
            print(f'Para el atributo {key}: MAE {value}')
        return resultados
    except Exception as e:
        print(e)
        print(p)
        # var_dif.append(np.abs(pred_value - real_value)/real_value)

def creacion_pre_forecast(df_pasillos, row, prod_pass, dict_cols, lista_productos, ajustar_modelos=False):
    """
        Realiza la preparación previa al pronóstico de creación.

        Args:
            df_pasillos (DataFrame): El DataFrame de pasillos.
            row (DataFrame): Una fila del DataFrame.
            prod_pass (list): La lista de productos en el pasillo.
            dict_cols (dict): Un diccionario con información de columnas.
            lista_productos (list): La lista de productos.

        Returns:
            tuple: Un tuple que contiene DataFrames para el pronóstico de creación.
        """
    dfA = df_simulacion_creacion('A', df_pasillos, row, lista_productos, freq=1, corte=0, prod_pass=prod_pass,
                                 dict_cols=dict_cols, ajustar_modelos=ajustar_modelos)
    dfB = df_simulacion_creacion('B', df_pasillos, row, lista_productos, freq=1, corte=0, prod_pass=prod_pass,
                                 dict_cols=dict_cols, ajustar_modelos=ajustar_modelos)
    dfC = df_simulacion_creacion('C', df_pasillos, row, lista_productos, freq=1, corte=0, prod_pass=prod_pass,
                                 dict_cols=dict_cols, ajustar_modelos=ajustar_modelos)
    df_ = df_simulacion_creacion('', df_pasillos, row, lista_productos, freq=1, corte=0, prod_pass=prod_pass,
                                 dict_cols=dict_cols, ajustar_modelos=ajustar_modelos)
    return dfA, dfB, dfC, df_

def df_simulacion_creacion(label, df_pasillo, tiempos_pck, lista_productos, freq=1, corte=0, prod_pass=None,
                           dict_cols=None, ajustar_modelos=False):
    """
        Crea un DataFrame para simulación de creación.

        Args:
            label (str): La etiqueta del pasillo.
            df_pasillo (DataFrame): El DataFrame de pasillos.
            tiempos_pck (DataFrame): El DataFrame de tiempos de paquete.
            lista_productos (list): La lista de productos.
            freq (int): La frecuencia de los intervalos de hora.
            corte (int): El valor de corte para los datos.
            prod_pass (list): La lista de productos en el pasillo.
            dict_cols (dict): Un diccionario con información de columnas.

        Returns:
            DataFrame: El DataFrame creado para simulación de creación.
        """
    if ajustar_modelos:
        if label != "":
            df_melt = pd.melt(df_pasillo[df_pasillo[f'Pasillo_{label}'] > 0],
                              id_vars=['Folio', 'Producto'], value_vars=[f'Pasillo_{label}'], var_name="Pasillo").dropna(
                subset='value')
            if len(df_melt) == 0:
                return []
            df_pivot = df_melt.pivot_table(index=['Folio', 'Pasillo'], columns='Producto', values='value').reset_index()
            df_pivot = df_pivot.fillna(0)
            df_pivot['#prod'] = df_pivot.drop(['Folio', "Pasillo"], axis=1).apply(contar_productos_mayores_que_cero, axis=1)

            df_pivot = df_pivot.merge(tiempos_pck[['mov_llamado', 'mov_folio']], left_on='Folio',
                                      right_on='mov_folio', how='left')
            # .dropna(subset=[f'Dif_{label.lower()}']))
            df_pivot["mov_llamado"] = pd.to_datetime(df_pivot["mov_llamado"])
            df_pivot['hora'] = df_pivot['mov_llamado'].dt.hour
            col = df_pivot.pop('hora')
            df_pivot.insert(len(df_pivot.columns) - 3, 'hora', col)
            df_pivot.pop('mov_llamado')
            cols = [c for c in df_pivot.columns if c in lista_productos]
            df_pivot["#ProdVol"] = df_pivot[cols].sum(axis=1)
            col = df_pivot.pop('#ProdVol')
            df_pivot.insert(len(df_pivot.columns) - 3, '#ProdVol', col)
            df_pivot.pop('mov_folio')
            df_pivot = intervalo_hora(df_pivot, freq=freq, target=f'Dif_{label.lower()}')

        else:

            df_melt = pd.melt(df_pasillo, id_vars=['Folio', 'Producto'], value_vars=['Total_PCK'],
                              var_name="Pasillo").dropna(subset='value')
            df_pivot = df_melt.pivot_table(index=['Folio', 'Pasillo'], columns='Producto', values='value').reset_index()
            df_pivot = df_pivot.fillna(0)
            tiempos_pck["mov_llamado"] = pd.to_datetime(tiempos_pck["mov_llamado"])

            tiempos_pck['hora'] = tiempos_pck['mov_llamado'].dt.hour
            df_pivot = df_pivot.merge(tiempos_pck[['hora', 'mov_folio']], left_on='Folio',
                                      right_on='mov_folio', how='left')
            # .dropna(subset=['Fin_pck-mov_hora_control']))
            # for p in dict_cols[f'col_'][2:-14]:
            #     if not p in col:
            #         df_pivot[p] = 0
            # cols = []
            # for c in df_pivot.columns:
            #     if c in lista_productos:
            #         cols.append(c)
            # reemplazar con isnumeric
            cols = [c for c in df_pivot.columns if c in lista_productos]
            df_pivot["#ProdVol"] = df_pivot[cols].sum(axis=1)
            col = df_pivot.pop('#ProdVol')
            df_pivot.pop('mov_folio')
            df_pivot.insert(len(df_pivot.columns) - 1, '#ProdVol', col)

            df_pivot = intervalo_hora(df_pivot, freq=freq, target='Fin_pck-mov_hora_control')

            # dict_cols = dict_cols[f'col_']
            # if len(df_pivot.columns) != len(dict_cols):
            #     print(
            #         f'ERROR : columns modelo entrenado diferente al modelo generado.\nColumnas Obtenidas {len(df_pivot.columns)} esperadas {len(dict_cols)} ')
            #     print(set(dict_cols) - set(df_pivot.columns.tolist()))
            #     print(set(df_pivot.columns.tolist()) - set(dict_cols))
            # else:
            #     df_pivot = df_pivot[dict_cols]
    else:
        if label != "":
            if not prod_pass:
                prod_pass = list(df_pasillo.dropna(subset=f'Pasillo_{label}').Producto.unique())
            else:
                prod_pass = prod_pass[label]
            # print(f'Pasillo {label} tiene {len(prod_pass)} productos.')
            df_melt = pd.melt(df_pasillo[df_pasillo['Producto'].isin(dict_cols[f'col{label}'][2:-16])],
                              id_vars=['Folio', 'Producto'], value_vars=[f'Pasillo_{label}'], var_name="Pasillo").dropna(
                subset='value')
            if len(df_melt) == 0:
                return []
            df_pivot = df_melt.pivot_table(index=['Folio', 'Pasillo'], columns='Producto', values='value').reset_index()
            df_pivot = df_pivot.fillna(0)
            col = df_pivot.columns.tolist()
            #for p in dict_cols[f'col{label}'][2:-15]:
            for p in prod_pass:
                if not p in col:
                    df_pivot[p] = 0
            # for p in prod_pass:
            #    if not p in col:
            #        df_pivot[p] = 0
            df_pivot['#prod'] = df_pivot.drop(['Folio', "Pasillo"], axis=1).apply(contar_productos_mayores_que_cero, axis=1)

            df_pivot = df_pivot.merge(tiempos_pck[['mov_llamado', 'mov_folio']], left_on='Folio',
                                      right_on='mov_folio', how='left')
            # .dropna(subset=[f'Dif_{label.lower()}']))
            df_pivot["mov_llamado"] = pd.to_datetime(df_pivot["mov_llamado"])
            df_pivot['hora'] = df_pivot['mov_llamado'].dt.hour
            col = df_pivot.pop('hora')
            df_pivot.insert(len(df_pivot.columns) - 3, 'hora', col)
            df_pivot.pop('mov_llamado')
            # df_pivot[f'Dif_{label.lower()}'] = df_pivot[f'Dif_{label.lower()}'].apply(
            #     lambda x: timedelta(hours=x.hour, minutes=x.minute, seconds=x.second).total_seconds() if x else None)
            cols = []
            # if corte > 0:
            #     df_pivot = df_pivot[(df_pivot[f'Dif_{label.lower()}'] < corte) & (df_pivot[f'Dif_{label.lower()}'] > 15)]
            for c in df_pivot.columns:
                if c in lista_productos:
                    cols.append(c)
            df_pivot["#ProdVol"] = df_pivot[cols].sum(axis=1)
            col = df_pivot.pop('#ProdVol')
            df_pivot.insert(len(df_pivot.columns) - 3, '#ProdVol', col)
            df_pivot.pop('mov_folio')
            df_pivot = intervalo_hora(df_pivot, freq=freq, target=f'Dif_{label.lower()}')
            dict_cols = dict_cols[f'col{label}']
            columnas_pivot = df_pivot.columns.tolist()
            if len(columnas_pivot) != len(dict_cols):
                print(
                    f'ERROR : columns modelo entrenado diferente al modelo generado.\nColumnas Obtenidas {len(df_pivot.columns)} esperadas {len(dict_cols)} ')
                print(set(dict_cols) - set())
                print(set(columnas_pivot) - set(dict_cols))
                if len(set(columnas_pivot) - set(dict_cols)) > 0:
                    producto_invalidos = set(columnas_pivot) - set(dict_cols)
                    for producto_invalido in producto_invalidos:
                        cantidad_invalida = df_pivot[producto_invalido]
                        producto_destino = np.random.choice(prod_pass)
                        df_pivot[producto_destino] += cantidad_invalida
                    print(df_pivot.shape)
                    df_pivot = df_pivot[dict_cols]
                    print(df_pivot.shape)
            else:
                df_pivot = df_pivot[dict_cols]
        else:

            df_melt = pd.melt(df_pasillo, id_vars=['Folio', 'Producto'], value_vars=['Total_PCK'],
                              var_name="Pasillo").dropna(subset='value')
            df_pivot = df_melt.pivot_table(index=['Folio', 'Pasillo'], columns='Producto', values='value').reset_index()
            df_pivot = df_pivot.fillna(0)
            col = df_pivot.columns.tolist()
            # tiempos_pck["Fin_pck"] = pd.to_datetime(tiempos_pck["Fin_pck"])
            # tiempos_pck["mov_hora_control"] = pd.to_datetime(tiempos_pck["mov_hora_control"])
            # tiempos_pck['Fin_pck-mov_hora_control'] = abs(
            #     tiempos_pck['Fin_pck'] - tiempos_pck['mov_hora_control']).dt.total_seconds()
            # if corte > 0:
            #     tiempos_pck = tiempos_pck[
            #         (tiempos_pck['Fin_pck-mov_hora_control'] < corte) & (tiempos_pck['Fin_pck-mov_hora_control'] > 15)]
            tiempos_pck["mov_llamado"] = pd.to_datetime(tiempos_pck["mov_llamado"])

            tiempos_pck['hora'] = tiempos_pck['mov_llamado'].dt.hour
            df_pivot = df_pivot.merge(tiempos_pck[['hora', 'mov_folio']], left_on='Folio',
                                      right_on='mov_folio', how='left')
            # .dropna(subset=['Fin_pck-mov_hora_control']))
            productos_validos = dict_cols[f'col_'][2:-14]
            for p in productos_validos:
                if not p in col:
                    df_pivot[p] = 0
            cols = []
            for c in df_pivot.columns:
                if c in lista_productos:
                    cols.append(c)
            df_pivot["#ProdVol"] = df_pivot[cols].sum(axis=1)
            col = df_pivot.pop('#ProdVol')
            df_pivot.insert(len(df_pivot.columns) - 3, '#ProdVol', col)
            df_pivot.pop('mov_folio')
            df_pivot = intervalo_hora(df_pivot, freq=freq, target='Fin_pck-mov_hora_control')
            dict_cols = dict_cols[f'col_']
            columnas_pivot = df_pivot.columns.tolist()
            if len(columnas_pivot) != len(dict_cols):
                print(
                    f'ERROR : columns modelo entrenado diferente al modelo generado.\nColumnas Obtenidas {len(df_pivot.columns)} esperadas {len(dict_cols)} ')
                print(set(dict_cols) - set())
                print(set(columnas_pivot) - set(dict_cols))
                if len(set(columnas_pivot) - set(dict_cols)) > 0:
                    producto_invalidos = set(columnas_pivot) - set(dict_cols)
                    for producto_invalido in producto_invalidos:
                        cantidad_invalida = df_pivot[producto_invalido]
                        producto_destino = np.random.choice(productos_validos)
                        df_pivot[producto_destino] += cantidad_invalida
                    print(df_pivot.shape)
                    df_pivot = df_pivot[dict_cols]
                    print(df_pivot.shape)
            else:
                df_pivot = df_pivot[dict_cols]
    return df_pivot

def actualizar_modelos_pck():
    print('Iniciando Actualizar Modelo PCK')
    database_url = get_database_url(resultados=True)
    engine = create_engine(database_url)

    df_pasillo = pd.read_sql('select * from pasillo_historico', engine)
    tiempos_pck = pd.read_sql('select * from tiempos_pck_historico', engine)
    prod_vol = cargar_productos_voluminosos()


    # Esto por la info de panal.
    database_url = get_database_url(resultados=False)
    engine = create_engine(database_url)
    querys = """
    SELECT * 
    FROM vw_pck_docs 
    WHERE mov_entregado IS NOT NULL 
    AND DATE(mov_llamado) BETWEEN DATE_SUB(CURRENT_DATE, INTERVAL 4 DAY) AND CURRENT_DATE
    """
    querys = 'SELECT * FROM vw_pck_docs WHERE mov_entregado IS NOT NULL AND DATE(mov_llamado) = CURRENT_DATE'
    tiempos_pck2 = fetch_data_from_view('vw_pck_docs', engine,
                                       query=querys)
    if tiempos_pck2 is None:
        print("vw_pck_docs no trajo nada ")
        return None
    if len(tiempos_pck2)==0:
        print("vw_pck_docs no trajo nada ")
        return None

    tiempos_pck2.rename({'difA': 'Dif_a', "difB": 'Dif_b', "difC": 'Dif_c', 'ini_pck': 'Ini_pck'}, axis=1, inplace=True)
    tiempos_pck2['Dif_a'] = tiempos_pck2['Dif_a'].apply(time_to_seconds)
    tiempos_pck2['Dif_b'] = tiempos_pck2['Dif_b'].apply(time_to_seconds)
    tiempos_pck2['Dif_c'] = tiempos_pck2['Dif_c'].apply(time_to_seconds)
    folios_relevantes = tiempos_pck2.mov_folio.unique()
    print(f'Se han cargado {len(folios_relevantes)} folios nuevos')
    folios_relevantes = ','.join([f"'{folio}'" for folio in folios_relevantes])

    df_pasillo2 = fetch_data_from_view('vw_pck_pas_docs', engine,
                                       query=f'SELECT * FROM vw_pck_pas_docs WHERE Folio IN ({folios_relevantes})')
    tiempos_pck2["mov_llamado"] = pd.to_datetime(tiempos_pck2["mov_llamado"])
    tiempos_pck2 = tiempos_pck2.dropna(subset="mov_llamado")

    # <editor-fold desc="SOlo FOlios con toda la info">
    folios = set(tiempos_pck2.mov_folio.unique()).intersection(set(df_pasillo2.Folio.unique()))

    tiempos_pck2 = tiempos_pck2[tiempos_pck2.mov_folio.isin(folios)]
    df_pasillo2 = df_pasillo2[df_pasillo2.Folio.isin(folios)]

    # </editor-fold>

    tiempos_pck = pd.concat([tiempos_pck, tiempos_pck2]).drop_duplicates(subset=['mov_folio', 'mov_llamado'])
    df_pasillo = pd.concat([df_pasillo, df_pasillo2]).drop_duplicates()
    folios = set(tiempos_pck.mov_folio.unique()).intersection(set(df_pasillo.Folio.unique()))

    tiempos_pck = tiempos_pck[tiempos_pck.mov_folio.isin(folios)]
    df_pasillo = df_pasillo[df_pasillo.Folio.isin(folios)]

    lista_productos = prod_vol['Cod. Producto'].unique().tolist()


    dfA, dfB, dfC, df_ = creacion_pre_forecast(df_pasillo, tiempos_pck, prod_pass=None, dict_cols=None,
                                               lista_productos=lista_productos, ajustar_modelos=True)
    # <editor-fold desc="guardado_ dict to json">
    todos_productos = df_pasillo.Producto.unique().tolist()

    dict_cols_new = {'colA': dfA.columns.tolist(), 'colB': dfB.columns.tolist(), 'colC': dfC.columns.tolist(),
                     'col_': df_.columns.tolist()}
    prod_pass_new = {'A': [c for c in dfA.columns.tolist() if c in todos_productos],
                     'B': [c for c in dfB.columns.tolist() if c in todos_productos],
                     'C': [c for c in dfC.columns.tolist() if c in todos_productos]}
    with open("df/input/models/colsDF.json", "w") as outfile:
        json.dump(dict_cols_new, outfile)
    with open("df/input/models/productos_pasillos.json", "w") as outfile:
        json.dump(prod_pass_new, outfile)
    # </editor-fold>

    # <editor-fold desc="Creacion dfs">
    dfA = dfA.merge(tiempos_pck[['mov_folio', 'Dif_a']], left_on='Folio', right_on='mov_folio', how='left').drop(
        'mov_folio', axis=1)
    dfA = dfA[~dfA.Dif_a.isna()]
    dfB = dfB.merge(tiempos_pck[['mov_folio', 'Dif_b']], left_on='Folio', right_on='mov_folio', how='left').drop(
        'mov_folio', axis=1)
    dfB = dfB[~dfB.Dif_b.isna()]
    dfC = dfC.merge(tiempos_pck[['mov_folio', 'Dif_c']], left_on='Folio', right_on='mov_folio', how='left').drop(
        'mov_folio', axis=1)
    dfC = dfC[~dfC.Dif_c.isna()]
    # </editor-fold>

    labels = ['Dif_a', 'Dif_b', 'Dif_c']

    for label in labels:
        print(f'Iniciando entrenamiento para {label}')
        if 'Dif_a' == label:
            df_ = dfA[(dfA.Dif_a > 5)]
        elif 'Dif_b' == label:
            df_ = dfB[(dfB.Dif_b > 5)]
        else:
            df_ = dfC[(dfC.Dif_c > 10)]
        y = df_[label]
        X = df_[df_.columns.tolist()[2:-1]]  # .drop(['Dif_a'], axis=1)  # Elimina la columna de tiempo
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
        model = xgb.XGBRegressor(objective='reg:squarederror')
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 4, 5, 6],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
            'gamma': [0, 0.1, 0.2, 0.3],
            'min_child_weight': [1, 3, 5]
        }
        grid_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            n_iter=200,  # Number of parameter settings sampled
            scoring='neg_mean_squared_error',  # For regression
            cv=3,  # Number of folds in cross-validation
            verbose=1,
            n_jobs=-1,  # Use all available cores
            random_state=42
        )
        grid_search.fit(X_train, y_train)
        print("Best parameters found: ", grid_search.best_params_)
        print("Best RMSE: ", (-grid_search.best_score_) ** 0.5)

        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        mae = mean_absolute_error(y_test, y_pred)
        print("Test set RMSE: ", rmse)
        print("Test set mae: ", mae)
        best_model.save_model(f"df/input/models/best_xgboost_model_{label}.json")
        print("*"*20)

    database_url = get_database_url(resultados=True)
    engine = create_engine(database_url)
    print(f'Guardando {df_pasillo2.shape =}')
    df_pasillo2.to_sql('pasillo_historico', engine,index=False, if_exists='append')
    print(f'Guardando {tiempos_pck2.shape =}')
    tiempos_pck2.to_sql('tiempos_pck_historico',engine, index=False, if_exists='append')


def test_actualizar_modelos_pck():

    database_url = get_database_url(resultados=True)
    engine = create_engine(database_url)
    df_pasillo = pd.read_sql('select * from pasillo_historico', engine)
    tiempos_pck = pd.read_sql('select * from tiempos_pck_historico', engine)
    prod_vol = cargar_productos_voluminosos()


    # Esto por la info de panal.
    database_url = get_database_url(resultados=False)
    engine = create_engine(database_url)

    tiempos_pck2 = fetch_data_from_view('vw_pck_docs', engine,
                                       query='SELECT * FROM vw_pck_docs WHERE mov_entregado IS NOT NULL AND DATE(mov_llamado) = CURRENT_DATE')

    folios_relevantes = tiempos_pck2.mov_folio.unique()
    folios_relevantes = ','.join([f"'{folio}'" for folio in folios_relevantes])

    df_pasillo2 = fetch_data_from_view('vw_pck_pas_docs', engine,
                                       query=f'SELECT * FROM vw_pck_pas_docs WHERE Folio IN ({folios_relevantes})')
    tiempos_pck2["mov_llamado"] = pd.to_datetime(tiempos_pck2["mov_llamado"])
    tiempos_pck2 = tiempos_pck2.dropna(subset="mov_llamado")

    # <editor-fold desc="SOlo FOlios con toda la info">
    folios = set(tiempos_pck2.mov_folio.unique()).intersection(set(df_pasillo2.Folio.unique()))

    tiempos_pck2 = tiempos_pck2[tiempos_pck2.mov_folio.isin(folios)]
    df_pasillo2 = df_pasillo2[df_pasillo2.Folio.isin(folios)]
    # </editor-fold>

    tiempos_pck = pd.concat([tiempos_pck, tiempos_pck2]).drop_duplicates(subset=['mov_folio', 'mov_llamado'])
    df_pasillo = pd.concat([df_pasillo, df_pasillo2])

    lista_productos = prod_vol['Cod. Producto'].unique().tolist()


    dfA, dfB, dfC, df_ = creacion_pre_forecast(df_pasillo, tiempos_pck, prod_pass=None, dict_cols=None,
                                               lista_productos=lista_productos, ajustar_modelos=True)

    # <editor-fold desc="guardado_ dict to json">
    todos_productos = df_pasillo.Producto.unique().tolist()

    dict_cols_new = {'colA': dfA.columns.tolist(), 'colB': dfB.columns.tolist(), 'colC': dfC.columns.tolist(),
                     'col_': df_.columns.tolist()}
    prod_pass_new = {'A': [c for c in dfA.columns.tolist() if c in todos_productos],
                     'B': [c for c in dfB.columns.tolist() if c in todos_productos],
                     'C': [c for c in dfC.columns.tolist() if c in todos_productos]}
    with open("df/input/models/colsDF2.json", "w") as outfile:
        json.dump(dict_cols_new, outfile)
    with open("df/input/models/productos_pasillos2.json", "w") as outfile:
        json.dump(prod_pass_new, outfile)
    # </editor-fold>

    # <editor-fold desc="Creacion dfs">
    dfA = dfA.merge(tiempos_pck[['mov_folio', 'Dif_a']], left_on='Folio', right_on='mov_folio', how='left').drop(
        'mov_folio', axis=1)
    dfA = dfA[~dfA.Dif_a.isna()]
    dfB = dfB.merge(tiempos_pck[['mov_folio', 'Dif_b']], left_on='Folio', right_on='mov_folio', how='left').drop(
        'mov_folio', axis=1)
    dfB = dfB[~dfB.Dif_b.isna()]
    dfC = dfC.merge(tiempos_pck[['mov_folio', 'Dif_c']], left_on='Folio', right_on='mov_folio', how='left').drop(
        'mov_folio', axis=1)
    dfC = dfC[~dfC.Dif_c.isna()]

    dfA[f'Dif_a'] = dfA[f'Dif_a'].apply(
        lambda x: timedelta(hours=x.hour, minutes=x.minute, seconds=x.second).total_seconds() if x else None)
    dfB[f'Dif_b'] = dfB[f'Dif_b'].apply(
        lambda x: timedelta(hours=x.hour, minutes=x.minute, seconds=x.second).total_seconds() if x else None)
    dfC[f'Dif_c'] = dfC[f'Dif_c'].apply(
        lambda x: timedelta(hours=x.hour, minutes=x.minute, seconds=x.second).total_seconds() if x else None)
    # </editor-fold>



    #labels = ['Dif_a', 'Dif_b', 'Dif_c']
    labels = ['Dif_a']

    for label in labels:
        if 'Dif_a' == label:
            df_ = dfA[(dfA.Dif_a > 5)]
        elif 'Dif_b' == label:
            df_ = dfB[(dfB.Dif_b > 5)]
        else:
            df_ = dfC[(dfC.Dif_c > 10)]
        y = df_[label]
        X = df_[df_.columns.tolist()[2:-1]]  # .drop(['Dif_a'], axis=1)  # Elimina la columna de tiempo
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
        model = xgb.XGBRegressor(objective='reg:squarederror')
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 4, 5, 6],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
            'gamma': [0, 0.1, 0.2, 0.3],
            'min_child_weight': [1, 3, 5]
        }
        grid_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            n_iter=200,  # Number of parameter settings sampled
            scoring='neg_mean_squared_error',  # For regression
            cv=3,  # Number of folds in cross-validation
            verbose=1,
            n_jobs=-1,  # Use all available cores
            random_state=42
        )
        grid_search.fit(X_train, y_train)
        print("Best parameters found: ", grid_search.best_params_)
        print("Best RMSE: ", (-grid_search.best_score_) ** 0.5)

        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        mae = mean_absolute_error(y_test, y_pred)
        print("Test set RMSE: ", rmse)
        print("Test set mae: ", mae)
        best_model.save_model(f"df/input/models/best_xgboost_model_{label}.json")


def producto_pasillo_ultimo():
    database_url = get_database_url(resultados=True)
    engine = create_engine(database_url)
    
    df_pasillo = pd.read_sql('select TOP 15000 Folio, Producto, Pasillo_A, Pasillo_B, Pasillo_C from pasillo_historico', engine)
    tiempos_pck = pd.read_sql('select TOP 15000 mov_folio, mov_fecha from tiempos_pck_historico', engine)

    print(df_pasillo.shape)
    print(tiempos_pck.shape)
    df_pasillo = df_pasillo.merge(tiempos_pck[['mov_folio', 'mov_fecha']], left_on='Folio', right_on='mov_folio', how='left')
    print("After merge",df_pasillo.shape)
    # Obtener índices de la fila con la última fecha por producto
    df_pasillo['mov_fecha'] = pd.to_datetime(df_pasillo['mov_fecha'])
    df_pasillo = df_pasillo[df_pasillo["mov_fecha"].notna()]
    print("After filter na",df_pasillo.shape)
    idx = df_pasillo.groupby("Producto")["mov_fecha"].idxmax()
    print("After idx",idx)
    # Seleccionar las filas correspondientes
    df_ultima = df_pasillo.loc[idx].reset_index(drop=True)
    df_ultima["Pasillo"] = df_ultima[["Pasillo_A", "Pasillo_B", "Pasillo_C"]].idxmax(axis=1)
    df_ultima = df_ultima[['Producto','Pasillo']]
    prod_pass = cargar_productos_pasillos()
    A = set(prod_pass['A'])
    B = set(prod_pass['B'])
    C = set(prod_pass['C'])
    AB = A.intersection(B)
    AC = A.intersection(C)
    BC = B.intersection(C)
    prod_problemas_repeticion = AB|AC|BC
    for p in prod_problemas_repeticion:
        ultima_aparicion_prod_p = df_ultima[df_ultima.Producto == p]
        if len(ultima_aparicion_prod_p) == 1:
            if ultima_aparicion_prod_p.Pasillo.values[0] == 'A':
                print(f'Arreglando {p}, pertenece a A')
                if p in prod_pass['B']:
                    prod_pass['B'].remove(p)
                if p in prod_pass['C']:
                    prod_pass['C'].remove(p)
            elif ultima_aparicion_prod_p.Pasillo.values[0] == 'B':
                print(f'Arreglando {p}, pertenece a B')
                if p in prod_pass['A']:
                    prod_pass['A'].remove(p)
                if p in prod_pass['C']:
                    prod_pass['C'].remove(p)
            else:
                print(f'Arreglando {p}, pertenece a C')
                if p in prod_pass['A']:
                    prod_pass['A'].remove(p)
                if p in prod_pass['B']:
                    prod_pass['B'].remove(p)
    return prod_pass
            
    
    
    
    