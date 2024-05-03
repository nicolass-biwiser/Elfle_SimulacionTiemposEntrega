import numpy as np
import pandas as pd
from python_scripts.clases import *
from datetime import timedelta
from xgboost import XGBRegressor
import json
from warnings import simplefilter
from sklearn.metrics import mean_absolute_error, mean_squared_error

simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


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
        prod_vol = pd.read_excel('df/input/prod_voluminosos.xlsx')
        print('Finalizando Conexión a BBDD')
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

    df_encoded = pd.concat([df, intervalo_encoded], axis=1)
    target_ = df_encoded.pop(target)
    df_encoded.insert(len(df_encoded.columns), target, target_)
    df_encoded.pop('Intervalo_Hora')
    return df_encoded


def df_simulacion_creacion(label, df_pasillo, tiempos_pck, lista_productos, freq=1, corte=0, prod_pass=None,
                           dict_cols=None):
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
        for p in dict_cols[f'col{label}'][2:-16]:
            if not p in col:
                df_pivot[p] = 0
        # for p in prod_pass:
        #    if not p in col:
        #        df_pivot[p] = 0
        df_pivot['#prod'] = df_pivot.drop(['Folio', "Pasillo"], axis=1).apply(contar_productos_mayores_que_cero, axis=1)

        df_pivot = df_pivot.merge(tiempos_pck[['mov_llamado', 'mov_folio', f'Dif_{label.lower()}']], left_on='Folio',
                                  right_on='mov_folio', how='left').dropna(subset=[f'Dif_{label.lower()}'])
        df_pivot["mov_llamado"] = pd.to_datetime(df_pivot["mov_llamado"])
        df_pivot['hora'] = df_pivot['mov_llamado'].dt.hour
        col = df_pivot.pop('hora')
        df_pivot.insert(len(df_pivot.columns) - 3, 'hora', col)
        df_pivot.pop('mov_llamado')
        df_pivot[f'Dif_{label.lower()}'] = df_pivot[f'Dif_{label.lower()}'].apply(
            lambda x: timedelta(hours=x.hour, minutes=x.minute, seconds=x.second).total_seconds() if x else None)
        cols = []
        if corte > 0:
            df_pivot = df_pivot[(df_pivot[f'Dif_{label.lower()}'] < corte) & (df_pivot[f'Dif_{label.lower()}'] > 15)]
        for c in df_pivot.columns:
            if c in lista_productos:
                cols.append(c)
        df_pivot["#ProdVol"] = df_pivot[cols].sum(axis=1)
        col = df_pivot.pop('#ProdVol')
        df_pivot.insert(len(df_pivot.columns) - 3, '#ProdVol', col)
        df_pivot.pop('mov_folio')
        df_pivot = intervalo_hora(df_pivot, freq=freq, target=f'Dif_{label.lower()}')
        dict_cols = dict_cols[f'col{label}']
        if len(df_pivot.columns) != len(dict_cols):
            print(
                f'ERROR : columns modelo entrenado diferente al modelo generado.\nColumnas Obtenidas {len(df_pivot.columns)} esperadas {len(dict_cols)} ')
            print(set(dict_cols) - set(df_pivot.columns.tolist()))
        else:
            df_pivot = df_pivot[dict_cols]
    else:

        df_melt = pd.melt(df_pasillo, id_vars=['Folio', 'Producto'], value_vars=['Total_PCK'],
                          var_name="Pasillo").dropna(subset='value')
        df_pivot = df_melt.pivot_table(index=['Folio', 'Pasillo'], columns='Producto', values='value').reset_index()
        df_pivot = df_pivot.fillna(0)
        col = df_pivot.columns.tolist()
        tiempos_pck["Fin_pck"] = pd.to_datetime(tiempos_pck["Fin_pck"])
        tiempos_pck["mov_hora_control"] = pd.to_datetime(tiempos_pck["mov_hora_control"])
        tiempos_pck['Fin_pck-mov_hora_control'] = abs(
            tiempos_pck['Fin_pck'] - tiempos_pck['mov_hora_control']).dt.total_seconds()
        if corte > 0:
            tiempos_pck = tiempos_pck[
                (tiempos_pck['Fin_pck-mov_hora_control'] < corte) & (tiempos_pck['Fin_pck-mov_hora_control'] > 15)]
        tiempos_pck["mov_llamado"] = pd.to_datetime(tiempos_pck["mov_llamado"])

        tiempos_pck['hora'] = tiempos_pck['Fin_pck'].dt.hour
        df_pivot = df_pivot.merge(tiempos_pck[['Fin_pck-mov_hora_control', 'hora', 'mov_folio']], left_on='Folio',
                                  right_on='mov_folio', how='left').dropna(subset=['Fin_pck-mov_hora_control'])
        for p in dict_cols[f'col_'][2:-15]:
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
        if len(df_pivot.columns) != len(dict_cols):
            print(
                f'ERROR : columns modelo entrenado diferente al modelo generado.\nColumnas Obtenidas {len(df_pivot.columns)} esperadas {len(dict_cols)} ')
            print(set(dict_cols) - set(df_pivot.columns.tolist()))
        else:
            df_pivot = df_pivot[dict_cols]
    return df_pivot


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
    return df[(df[column] > day) & (df[column] <= day + timedelta(hours=23))].sort_values(by='mov_llamado')


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


def prediction_time(pasillo='A', p=None):
    """
        Realiza predicciones de tiempo para un pasillo específico.

        Args:
            pasillo (str): El pasillo para el que se realizará la predicción.
            p: El objeto Pedido utilizado para la predicción.

        Returns:
            int: El tiempo de predicción para el pasillo especificado.
        """
    if pasillo == 'A':
        mod = XGBRegressor()
        mod.load_model('df/input/models/xgb_model_Dif_a.json')
        df = p.dfA
        X = df[df.columns.tolist()[2:-1]]  # .drop(['Dif_a'], axis=1)  # Elimina la columna de tiemp
        y_pred = mod.predict(X)
        return int(y_pred[0])
    elif pasillo == 'B':
        mod = XGBRegressor()
        mod.load_model('df/input/models/xgb_model_Dif_b.json')
        df = p.dfB
        X = df[df.columns.tolist()[2:-1]]  # .drop(['Dif_a'], axis=1)  # Elimina la columna de tiemp
        y_pred = mod.predict(X)
        return int(y_pred[0])
    elif pasillo == 'C':
        mod = XGBRegressor()
        mod.load_model('df/input/models/xgb_model_Dif_c.json')
        df = p.dfC
        X = df[df.columns.tolist()[2:-1]]  # .drop(['Dif_a'], axis=1)  # Elimina la columna de tiemp
        y_pred = mod.predict(X)
        return int(y_pred[0])
    elif pasillo == '_':
        mod = XGBRegressor()
        mod.load_model('df/input/models/xgb_model_Fin_pck-mov_hora_control.json')
        df = p.df_
        X = df[df.columns.tolist()[2:-1]]  # .drop(['Dif_a'], axis=1)  # Elimina la columna de tiemp
        y_pred = mod.predict(X)
        return int(y_pred[0])


def creacion_pre_forecast(df_pasillos, row, prod_pass, dict_cols, lista_productos):
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
                                 dict_cols=dict_cols)
    dfB = df_simulacion_creacion('B', df_pasillos, row, lista_productos, freq=1, corte=0, prod_pass=prod_pass,
                                 dict_cols=dict_cols)
    dfC = df_simulacion_creacion('C', df_pasillos, row, lista_productos, freq=1, corte=0, prod_pass=prod_pass,
                                 dict_cols=dict_cols)
    df_ = df_simulacion_creacion('', df_pasillos, row, lista_productos, freq=1, corte=0, prod_pass=prod_pass,
                                 dict_cols=dict_cols)
    return dfA, dfB, dfC, df_


def nuevo_pedido(i, df_dia, df_pasillo, prod_pass, dict_cols, lista_productos, simulado=True):
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
        """
    row = df_dia.iloc[i]
    df_pasillos = df_pasillo[df_pasillo.Folio == row['mov_folio']]
    dfA, dfB, dfC, df_ = creacion_pre_forecast(df_pasillos, row.to_frame().T, prod_pass, dict_cols, lista_productos)
    p = Pedido(folio=row['mov_folio'], doc=row['Doc'], hora_meson=row['HoraMeson'], hora_llamado=row['mov_llamado'],
               pred_ini_pck=int(np.random.triangular(5, 12, 40)), dfA=dfA, dfB=dfB, dfC=dfC, df_=df_)
    p.hora_ini_pck = generar_time(p.hora_llamado, p.pred_ini_pck)
    if simulado:
        p.REAL_pred_ini_pck = (row['Ini_pck'] - row['mov_llamado']).total_seconds()
        if not isinstance(row['Dif_a'], float):
            p.REAL_predA = timedelta(hours=row['Dif_a'].hour, minutes=row['Dif_a'].minute,
                                     seconds=row['Dif_a'].second).total_seconds()
        if not isinstance(row['Dif_b'], float):
            p.REAL_predB = timedelta(hours=row['Dif_b'].hour, minutes=row['Dif_b'].minute,
                                     seconds=row['Dif_b'].second).total_seconds()
        if not isinstance(row['Dif_c'], float):
            p.REAL_predC = timedelta(hours=row['Dif_c'].hour, minutes=row['Dif_c'].minute,
                                     seconds=row['Dif_c'].second).total_seconds()
            # print(p.REAL_predA,p.REAL_predB,p.REAL_predC)
        p.pred_ = int(np.random.triangular(20, 250, 500))
        if len(p.dfA) > 0:
            p.predA = prediction_time('A', p)
        if len(p.dfB) > 0:
            p.predB = prediction_time('B', p)
        if len(p.dfC) > 0:
            p.predC = prediction_time('C', p)
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


def generar_time(hora, seg):
    """
        Genera una nueva marca de tiempo a partir de una hora inicial y una cantidad de segundos.

        Args:
            hora: La hora inicial.
            seg (int): La cantidad de segundos.

        Returns:
            datetime: La nueva marca de tiempo generada.
        """
    return hora + timedelta(seconds=seg)


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
            real_value = getattr(p, f'REAL_fecha_termino')
            pred_value = getattr(p, "fecha_termino")

            delta = np.abs((pred_value - real_value).total_seconds())
            if delta < 2500:
                dif.append(delta)
                dif1.append((pred_value - real_value).total_seconds())
                aceptados += 1
            else:
                no_aceptados += 1
        resultados['fecha_termino'] = np.mean(dif)
        resultados['fecha_termino_noabs'] = np.mean(dif1)
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

# 1) Data sea un dia completo : LISTO FILTRO_DAY
# 2) lista con el llamado de cada uno de los pedidos (EVentos llegada)

# 2.1) generar hora inicial . Ya está sorted del filt_day
# 2.2) crear el objeto Pedido
# 3) Ingreso de pedido y envio a los pasillos respectivos
# 4) Ejecucion de forecast para dicho requerimiento
# Para eso se debe crear el dfA,dfB, dfC
# 5) Cambiar las horas de pred
# 6) COrrer simulacion y cuando termina cada proceso cambiar las horas del objeto en cuestion.
