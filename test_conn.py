from python_scripts.functions import *
import pandas as pd
import datetime
from sqlalchemy import create_engine


test0 = False
test1 = False
test2 = False
test3 = False
test4 = False
#df_pasillo, tiempos_pck, prod_vol = cargar_informacion(True)
# TEST CONEXION BBDD ELFLE + GUARDADO
if test0:
    database_url = get_database_url(resultados=True)

    # Crear el motor de conexión
    engine = create_engine(database_url)
    df = pd.read_sql('select * from prediccion_pck', engine)


if test1:
    database_url = get_database_url(resultados=True)

    # Crear el motor de conexión
    engine = create_engine(database_url)

    df = pd.read_excel('resultados_bbdd.xlsx')
    print('Iniciando envio a BBDD')
    df.to_sql('prediccion_pck', engine, schema='dbo', if_exists='replace', index=False)
    print('Enviado a BBDD')
    resultados_bbdd = fetch_latest_results(30)
    print(f'resultados_bbdd -> {resultados_bbdd.shape}')
    print(resultados_bbdd.head())

# TEST CONEXION VISTA PANAL
if test2:
    database_url = get_database_url(resultados=False)

    # Crear el motor de conexión
    engine = create_engine(database_url)
    df_dia, df_pasillo = cargar_informacion(False)
    df_pasillo_teorico,df_dia_teorico, _ = cargar_informacion(True)
    print(f'df_dia -> {df_dia.shape}')
    print(df_dia.head())
    # columnas esperadas
    cols_recibidas = set(df_dia.columns)
    cols_esperadas = set(df_dia_teorico.columns)
    if len(cols_esperadas - cols_recibidas) > 0:
        print(f'Diferencia columnas {cols_esperadas - cols_recibidas}')
    if len(cols_recibidas - cols_esperadas) > 0:
        print(f'Recibo más columnas {cols_recibidas - cols_esperadas}')
    print("-"*20)
    print(f'df_pasillo -> {df_pasillo.shape}')
    print(df_pasillo.head())
    # columnas esperadas
    cols_recibidas = set(df_pasillo.columns)
    cols_esperadas = set(df_pasillo_teorico.columns)
    if len(cols_esperadas - cols_recibidas) > 0:
        print(f'Diferencia columnas {cols_esperadas - cols_recibidas}')
    if len(cols_recibidas - cols_esperadas) > 0:
        print(f'Recibo más columnas {cols_recibidas - cols_esperadas}')


if test3:
    print('Inicializando Conexion a BBDD')

    database_url = get_database_url(resultados=False)
    engine = create_engine(database_url)
    # PRIMERO CARGA TIEMPO
    df_dia = fetch_data_from_view('vw_pck_docs', engine)

    database_url = get_database_url(resultados=False)
    engine = create_engine(database_url)
    # PRIMERO CARGA TIEMPO
    df_pasillo = fetch_data_from_view('vw_pck_pas_docs', engine)


if test4:
    df_pasillo, tiempos_pck, prod_vol = cargar_informacion(True)
    df_pasillo2 = pd.read_excel("df/input/tiempo_operaciones (9).xlsx", sheet_name="cantidades_picking")
    tiempos_pck2 = pd.read_excel("df/input/tiempo_operaciones (9).xlsx", sheet_name="detalle_tiempos")
    tiempos_pck2["mov_llamado"] = pd.to_datetime(tiempos_pck2["mov_llamado"])
    tiempos_pck2 = tiempos_pck2.dropna(subset="mov_llamado")
    folios = set(tiempos_pck2.mov_folio.unique()).intersection(set(df_pasillo2.Folio.unique()))
    tiempos_pck2 = tiempos_pck2[tiempos_pck2.mov_folio.isin(folios)]
    df_pasillo2 = df_pasillo2[df_pasillo2.Folio.isin(folios)]
    tiempos_pck = pd.concat([tiempos_pck, tiempos_pck2]).drop_duplicates(subset=['mov_folio','mov_llamado'])
    df_pasillo = pd.concat([df_pasillo, df_pasillo2])
    database_url = get_database_url(resultados=True)
    engine = create_engine(database_url)
    df_pasillo = df_pasillo.drop(columns='prod_voluminoso')
    tiempos_pck = tiempos_pck[tiempos_pck2.columns.tolist()]
    df_pasillo.to_sql('pasillo_historico', engine, index=False, if_exists='replace')
    tiempos_pck.to_sql('tiempos_pck_historico', engine, index=False, if_exists='replace')