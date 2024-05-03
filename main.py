from python_scripts.Simulador_clase import *
from python_scripts.functions import *
from datetime import timedelta, datetime, date
import time

# TODO: Si prod_voluminos no cambia, mejor no leerlo todo el tiempo y tenerlo afuera de la iteracion while.
# Lo mismo con la carga de los modelos y columnas df/input/models
sim = False
# PEDIDOS SIM

if sim:
    df_pasillo, tiempos_pck, prod_vol = cargar_informacion(sim)

    lista_productos = prod_vol['Cod. Producto'].unique().tolist()
    dict_cols = cargar_columnas_df()
    prod_pass = cargar_productos_pasillos()

    days = [i for i in range(10)]
    pedidos_completados = []
    resultados_dia = {}
    for day in days:
        dia = datetime(2023,12,4) + timedelta(days=+day)
        df_dia = filt_day(tiempos_pck, 'mov_llamado', dia)
        df_dia = df_dia.drop_duplicates(subset="mov_folio" ,keep='last')
        df_dia = df_dia.dropna(subset='mov_entregado')
        if len(df_dia)>0:
            sim = Simulador(df_dia, df_pasillo, prod_pass, dict_cols, lista_productos)
            t00 = time.time()
            sim.run()
            pedidos_completados.extend(sim.completados)
            r = calcular_resultados(sim.completados)
            resultados_dia[day] = r
            t11 = time.time()
            print(f'Tiempo transcurrido {round(t11-t00,2)}')
    print("INICIANDO CALCULO COMPLETO")
    r = calcular_resultados(pedidos_completados)
    resultados_dia['total'] = r
    df = pd.DataFrame.from_dict(resultados_dia, orient='index')
    df.to_excel("resultados_simulacion 60dias.xlsx")

# CASO REAL
# SE ASUME que las 3 dataset vienen ya filtrado pro dia, si no, entonces se debe aplicar lo coemntado

# dia = datetime(2023,12,4) + timedelta(days=+day)
# df_dia = filt_day(tiempos_pck, 'mov_llamado', dia)
# df_dia = df_dia.drop_duplicates(subset="mov_folio")
else:
    i = 0
    iterar = True
    while iterar:
        # CARGAR INFO EN ESTE CASO DEBERIA CONECTARSE A BBDD
        df_pasillo, tiempos_pck, prod_vol = cargar_informacion(sim)

        lista_productos = prod_vol['Cod. Producto'].unique().tolist()
        # Dict_cols y prod_pass se mantiene
        dict_cols = cargar_columnas_df()
        prod_pass = cargar_productos_pasillos()

        fecha_hoy = date.today()
        # dia = datetime(2023,12,4) + timedelta(days=+day)
        # df_dia = filt_day(tiempos_pck, 'mov_llamado', dia)
        df_dia = tiempos_pck[tiempos_pck['Fecha'].dt.date == fecha_hoy]
        df_dia = df_dia.drop_duplicates(subset="mov_folio", keep='last')
        # Quedarme con los que no están entregados
        df_dia = df_dia[df_dia['mov_entregado'].isnull()]
        sim = Simulador(df_dia, df_pasillo, prod_pass, dict_cols, lista_productos)
        t00 = time.time()
        sim.run()
        # FALTA REVISAR TABLAS PARA ENVIAR RESULTADOS
        #pedidos_completados.extend(sim.completados)
        #r = calcular_resultados(sim.completados)
        #resultados_dia[day] = r
        t11 = time.time()
        print(f'Tiempo transcurrido {round(t11-t00,2)}')
        i += 1
        print(f'Finalizado iteración {i}')
        if i == 10:
            iterar = False
        
        time.sleep(1)

