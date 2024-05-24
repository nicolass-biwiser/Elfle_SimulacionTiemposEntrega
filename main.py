import pandas as pd
from python_scripts.Simulador_clase import *
from python_scripts.functions import *
from datetime import timedelta, datetime, date
import time

# TODO: Si prod_voluminos no cambia, mejor no leerlo todo el tiempo y tenerlo afuera de la iteracion while.
# todo: Lo mismo con la carga de los modelos y columnas df/input/models
simulado = False
# PEDIDOS SIM

mod_a, mod_b, mod_c, mod_ = cargar_modelos()
if simulado:
    df_pasillo, tiempos_pck, prod_vol = cargar_informacion(simulado)
    cantidad_de_dias = 2
    lista_productos = prod_vol['Cod. Producto'].unique().tolist()
    dict_cols = cargar_columnas_df()
    prod_pass = cargar_productos_pasillos()

    days = [i for i in range(cantidad_de_dias)]
    pedidos_completados = []
    resultados_dia = {}
    # atributos = ['predA', 'predB', 'predC', 'pred_', 'fecha_termino']
    resultados_bbdd = pd.DataFrame(
        columns=['Doc', 'mov_folio', 'mov_llamado', 'hora_ini_pck', 'predA', 'hora_ini_pckA', 'hora_fin_pckA',
                 'predB', 'hora_ini_pckB', 'hora_fin_pckB',
                 'predC', 'hora_ini_pckC', 'hora_fin_pckC',
                 'hora_fin_pck',
                 'pred_',
                 'hora_estimada_entrega'])
    for day in days:
        dia = datetime(2023, 12, 4) + timedelta(days=+day)
        df_dia = filt_day(tiempos_pck, 'mov_llamado', dia)
        df_dia = df_dia.drop_duplicates(subset="mov_folio", keep='last')
        df_dia = df_dia.dropna(subset='mov_entregado')
        if len(df_dia) > 0:
            sim = Simulador(df_dia, df_pasillo, prod_pass, dict_cols, lista_productos, mod_a, mod_b, mod_c, mod_,
                            simulado)
            t00 = time.time()
            sim.run()
            pedidos_completados.extend(sim.completados)
            # r = calcular_resultados(sim.completados)
            # resultados_dia[day] = r
            t11 = time.time()
            print(f'Tiempo transcurrido {round(t11 - t00, 2)}')
    print("INICIANDO CALCULO COMPLETO")

    for p in sim.completados:
        resultados_bbdd.loc[len(resultados_bbdd)] = [p.doc, p.folio, p.hora_llamado, p.hora_ini_pck,
                                                     p.predA, p.hora_ini_pckA, p.hora_fin_pckA,
                                                     p.predB, p.hora_ini_pckB, p.hora_fin_pckB,
                                                     p.predC, p.hora_ini_pckC, p.hora_fin_pckC,
                                                     p.hora_fin_pck,
                                                     p.pred_,
                                                     p.fecha_termino]
    resultados_bbdd.to_excel('resultados_bbdd.xlsx', index=False)
    r = calcular_resultados(pedidos_completados)
    # resultados_dia['total'] = r
    # df = pd.DataFrame.from_dict(resultados_dia, orient='index')
    # df.to_excel("resultados_simulacion 60dias.xlsx")

# CASO REAL
# SE ASUME que las 3 dataset vienen ya filtrado pro dia, si no, entonces se debe aplicar lo coemntado

# dia = datetime(2023,12,4) + timedelta(days=+day)
# df_dia = filt_day(tiempos_pck, 'mov_llamado', dia)
# df_dia = df_dia.drop_duplicates(subset="mov_folio")
else:
    # resultados_bbdd_iteracion = pd.DataFrame(
    #     columns=['Doc', 'mov_folio', 'mov_llamado', 'pred_ini_pck', 'hora_ini_pck', 'predA', 'hora_ini_pckA',
    #              'hora_fin_pckA',
    #              'predB', 'hora_ini_pckB', 'hora_fin_pckB',
    #              'predC', 'hora_ini_pckC', 'hora_fin_pckC',
    #              'hora_fin_pck',
    #              'pred_',
    #              'mov_entregado', 'version_control'])
    # i = 0
    iterar = True


    prod_vol = cargar_productos_voluminosos()
    # Esto solo para probar

    # ESTA LINEA DEBE CAMBIAR POR LA QUERY

    # añadir query resultados ultimos 30 para utilizar info
    resultados_bbdd = fetch_latest_results(30)
    dict_cols = cargar_columnas_df()
    prod_pass = cargar_productos_pasillos()
    lista_productos = prod_vol['Cod. Producto'].unique().tolist()

    while True:
        df_dia, df_pasillo = cargar_informacion(simulado)

        resultados_bbdd_iteracion = pd.DataFrame(
            columns=['Doc', 'mov_folio', 'mov_llamado', 'pred_ini_pck', 'hora_ini_pck', 'predA', 'hora_ini_pckA',
                     'hora_fin_pckA',
                     'predB', 'hora_ini_pckB', 'hora_fin_pckB',
                     'predC', 'hora_ini_pckC', 'hora_fin_pckC',
                     'hora_fin_pck',
                     'pred_',
                     'mov_entregado', 'version_control'])
        folios_listos = resultados_bbdd.mov_folio.unique.tolist()
        hora_dia = datetime.now()
        print(f'Hora dia {hora_dia}')
        # CARGAR INFO EN ESTE CASO DEBERIA CONECTARSE A BBDD
        # df_pasillo, tiempos_pck = cargar_informacion(simulado)
        df_dia = df_dia.drop_duplicates(subset="mov_folio", keep='last')
        # df_dia = df_dia[df_dia['mov_entregado'].isnull()]
        # ESTO NO DEBERIA IR DESPUES
        # df_dia_ = df_dia[(df_dia.mov_llamado < hora_dia) & (df_dia.mov_entregado > hora_dia)]
        # print(f"len df_dia: {len(df_dia_)}")

        folios = set(df_dia.mov_folio.unique())
        df_pasillo_ = df_pasillo[df_pasillo.Folio.isin(folios)]
        folio_bbdd = set(resultados_bbdd.mov_folio.unique())
        # SI ESTÁ LA INFO ENTONCES USO LA INFO, SI NO DATA ANTIGUA YA PREDICHA
        if len(folios - folio_bbdd) > 0:
            print(f'set folios {folios - folio_bbdd} = {folios}\n{folio_bbdd}')
            sim = Simulador(df_dia, df_pasillo_, prod_pass, dict_cols, lista_productos, mod_a, mod_b, mod_c, mod_,
                            simulado,
                            resultados_bbdd, hora_dia)
            t00 = time.time()
            sim.run()
            for p in sim.completados:
                if p.folio not in folios_listos:
                    resultados_bbdd_iteracion.loc[len(resultados_bbdd_iteracion)] = [p.doc, p.folio, p.hora_llamado,
                                                                                     p.pred_ini_pck,
                                                                                     p.hora_ini_pck,
                                                                                     p.predA, p.hora_ini_pckA,
                                                                                     p.hora_fin_pckA,
                                                                                     p.predB, p.hora_ini_pckB,
                                                                                     p.hora_fin_pckB,
                                                                                     p.predC, p.hora_ini_pckC,
                                                                                     p.hora_fin_pckC,
                                                                                     p.hora_fin_pck,
                                                                                     p.pred_,
                                                                                     p.mov_entregado, hora_dia]

            if len(resultados_bbdd_iteracion) > 0:
                database_url = get_database_url(resultados=True)
                engine = create_engine(database_url)
                resultados_bbdd_iteracion.to_sql('prediccion_pck', engine, if_exists='append',index=False)
                resultados_bbdd = pd.concat([resultados_bbdd, resultados_bbdd_iteracion])
                resultados_bbdd = keep_last_n_rows(resultados_bbdd)
            t11 = time.time()

            print(f'Tiempo transcurrido {round(t11 - t00, 2)}')
        # FALTA REVISAR TABLAS PARA ENVIAR RESULTADOS
        else:
            # No hay información nueva.
            time.sleep(4)
