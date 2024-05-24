from python_scripts.functions import *
import pandas as pd
import datetime

test1 = False
test2 = False
df_pasillo, tiempos_pck, prod_vol = cargar_informacion(True)
# TEST CONEXION BBDD ELFLE + GUARDADO
if test1:
    database_url = get_database_url(resultados=True)

    # Crear el motor de conexión
    engine = create_engine(database_url)

    df = pd.read_excel('resultados_bbdd.xlsx')
    hora_control = datetime.datetime.now()
    df['version_control'] = hora_control
    df.to_sql('prediccion_pck', engine, if_exists='replace')

    resultados_bbdd = fetch_latest_results(30)
    print(f'resultados_bbdd -> {resultados_bbdd.shape}')
    print(resultados_bbdd.head())

# TEST CONEXION VISTA PANAL
if test2:
    database_url = get_database_url(resultados=False)

    # Crear el motor de conexión
    engine = create_engine(database_url)
    df_dia, df_pasillo = cargar_informacion(False)
    print(f'df_dia -> {df_dia.shape}')
    print(df_dia.head())
    # columnas esperadas
    cols_recibidas = set(df_dia.columns())
    cols_esperadas = set([])
    if len(cols_esperadas - cols_recibidas) > 0:
        print(f'Diferencia columnas {cols_esperadas - cols_recibidas}')
    if len(cols_recibidas - cols_esperadas) > 0:
        print(f'Recibo más columnas {cols_recibidas - cols_esperadas}')

    print(f'df_pasillo -> {df_pasillo.shape}')
    print(df_pasillo.head())
    # columnas esperadas
    cols_recibidas = set(df_pasillo.columns())
    cols_esperadas = set([])
    if len(cols_esperadas - cols_recibidas) > 0:
        print(f'Diferencia columnas {cols_esperadas - cols_recibidas}')
    if len(cols_recibidas - cols_esperadas) > 0:
        print(f'Recibo más columnas {cols_recibidas - cols_esperadas}')
