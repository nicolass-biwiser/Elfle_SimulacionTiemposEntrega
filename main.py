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

else:

    prod_vol = cargar_productos_voluminosos()

    resultados_bbdd = fetch_latest_results(30)
    dict_cols = cargar_columnas_df()
    prod_pass = cargar_productos_pasillos()
    lista_productos = prod_vol['Cod. Producto'].unique().tolist()
    i = 0
    #hora_dia = datetime.now()

    while datetime.now().hour < 18:

        df_dia, df_pasillo = cargar_informacion(simulado)
        hora_dia = datetime.now()
        print(f"**************** ITER {i} : {hora_dia} | {hora_dia.hour}****************")
        if (df_dia is None) or (df_pasillo is None):
            print('NO SE CAPTURÓ INFORMACIÓN - WAIT SEC')
            # No hay información nueva.
            time.sleep(20)
            i += 1
        else:
            resultados_bbdd_iteracion = pd.DataFrame(
                columns=['Doc', 'mov_folio', 'mov_llamado', 'pred_ini_pck', 'hora_ini_pck','inicio_colab', 'predA', 'hora_ini_pckA',
                         'hora_fin_pckA',
                         'predB', 'hora_ini_pckB', 'hora_fin_pckB',
                         'predC', 'hora_ini_pckC', 'hora_fin_pckC',
                         'hora_fin_pck',
                         'pred_',
                         'mov_entregado', 'pred_total', 'version_control'])
            folios_listos = resultados_bbdd.mov_folio.unique().tolist()


            df_dia = df_dia.drop_duplicates(subset="mov_folio", keep='last')


            folios = set(df_dia.mov_folio.unique())
            df_pasillo_ = df_pasillo[df_pasillo.Folio.isin(folios)]
            folio_bbdd = set(folios_listos)
            # SI ESTÁ LA INFO ENTONCES USO LA INFO, SI NO DATA ANTIGUA YA PREDICHA
            if len(folios - folio_bbdd) > 0:
                print('INICIANDO SIMULADOR')
                print(folios - folio_bbdd)
                #print(f'set folios {folios - folio_bbdd} = {folios}\n{folio_bbdd}')
                sim = Simulador(df_dia, df_pasillo_, prod_pass, dict_cols, lista_productos, mod_a, mod_b, mod_c, mod_,
                                simulado,
                                resultados_bbdd, hora_dia)
                t00 = time.time()
                sim.run()
                for p in sim.completados:
                    if p.folio not in folios_listos:
                        datetime_series = pd.Series([p.hora_ini_pckA,  p.hora_ini_pckB, p.hora_ini_pckC])
                        resultados_bbdd_iteracion.loc[len(resultados_bbdd_iteracion)] = [p.doc, p.folio, p.hora_llamado,
                                                                                         p.pred_ini_pck,
                                                                                         p.hora_ini_pck,
                                                                                         datetime_series.min(skipna=True),
                                                                                         p.predA, p.hora_ini_pckA,
                                                                                         p.hora_fin_pckA,
                                                                                         p.predB, p.hora_ini_pckB,
                                                                                         p.hora_fin_pckB,
                                                                                         p.predC, p.hora_ini_pckC,
                                                                                         p.hora_fin_pckC,
                                                                                         p.hora_fin_pck,
                                                                                         p.pred_,
                                                                                         p.mov_entregado,
                                                                                         int((p.mov_entregado - p.hora_llamado).total_seconds()),
                                                                                         hora_dia]
                    else:
                        print(f'{p.folio} ya está en folioslistos {folios_listos}')
                if len(resultados_bbdd_iteracion) > 0:
                    database_url = get_database_url(resultados=True)
                    engine = create_engine(database_url)
                    resultados_bbdd_iteracion.to_sql('prediccion_pck', engine, if_exists='append',index=False)
                    resultados_bbdd = pd.concat([resultados_bbdd, resultados_bbdd_iteracion])
                    print(f'Se envian a BBDD {resultados_bbdd_iteracion.shape}')
                    # resultados_bbdd = keep_last_n_rows(resultados_bbdd)
                t11 = time.time()

                print(f'Tiempo transcurrido {round(t11 - t00, 2)}')
                i += 1
            else:
                print('NO HAY INFORMACIÓN NUEVA')
                # No hay información nueva.
                time.sleep(10)
                i+= 1
            if datetime.now().hour >= 18:
                break
    print("Dia finalizado")
    tin = time.time()
    actualizar_modelos_pck()
    tfin = time.time()
    print(f'Tiempo transcurrido {round(tfin - tin, 2)}')

