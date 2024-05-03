import numpy as np
import random
from collections import deque
from python_scripts.functions import *
import time
random.seed(42)

class Pasillo:
    def __init__(self, nombre):
        """
        Inicializa un objeto Pasillo.

        Parámetros:
        - nombre (str): El nombre del Pasillo.

        Atributos:
        - nombre (str): El nombre del Pasillo.
        - busqueda (bool): Indica si el Pasillo está actualmente buscando un pedido.
        - pedido_actual: El pedido actual que se está procesando en el Pasillo.
        - cola (deque): Cola de pedidos esperando ser procesados en el Pasillo.
        """
        self.nombre = nombre
        self.busqueda = False
        self.pedido_actual = None
        self.cola = deque()

    def fin_busqueda(self, hora_inicio, tiempo):
        """
                Finaliza la búsqueda del pedido actual en el Pasillo.

                Parámetros:
                - hora_inicio: Hora de inicio de la búsqueda.
                - tiempo: Tiempo tomado para la búsqueda.

                Retorna:
                - pedido_finalizado: El pedido completado.
                """
        pedido_finalizado = self.pedido_actual
        if self.nombre == "A":
            pedido_finalizado.hora_fin_pckA = generar_time(hora_inicio, tiempo)
            pedido_finalizado.fin_pasA = True
        elif self.nombre == "B":
            pedido_finalizado.hora_fin_pckB = generar_time(hora_inicio, tiempo)
            pedido_finalizado.fin_pasB = True
        elif self.nombre == "C":
            pedido_finalizado.hora_fin_pckC = generar_time(hora_inicio, tiempo)
            pedido_finalizado.fin_pasC = True
        else:
            print(f'Nombre Pasillo no mapeado {self.nombre}')
        self.cargar_nuevo_pedido(hora_inicio, tiempo)
        return pedido_finalizado
        # REVISO SI PEDIDO ESTÁ LISTO PARA SER ENVIADO A MESON

    def cargar_nuevo_pedido(self, hora_inicio, tiempo):
        """
                Carga un nuevo pedido para ser procesado en el Pasillo.

                Parámetros:
                - hora_inicio: Hora de inicio del procesamiento del nuevo pedido.
                - tiempo: Tiempo tomado para el procesamiento.

                Retorna:
                Ninguno
                """
        if len(self.cola) > 0:
            p = self.cola.popleft()
            if self.nombre == "A":
                p.hora_ini_pckA = generar_time(hora_inicio, tiempo)
            elif self.nombre == "B":
                p.hora_ini_pckB = generar_time(hora_inicio, tiempo)
            elif self.nombre == "C":
                p.hora_ini_pckC = generar_time(hora_inicio, tiempo)
            else:
                print(f'Nombre Pasillo no mapeado {self.nombre}')
            self.pedido_actual = p
            self.busqueda = True
        else:
            # NO HAY PRODUCTOS EN COLA
            self.pedido_actual = None
            self.busqueda = False


class Meson:
    def __init__(self):
        """
                Inicializa un objeto Meson.

                Atributos:
                - revision (bool): Indica si el Meson está actualmente bajo revisión.
                - pedido_actual: El pedido actual que se está procesando en el Meson.
                - cola (deque): Cola de pedidos esperando ser procesados en el Meson.
                """
        self.revision = False
        self.pedido_actual = None
        self.cola = deque()

    def nuevo_pedido(self, pedido_finalizado, hora_inicio, tiempo):
        """
                Recibe un nuevo pedido completado para procesarlo en el Meson.

                Parámetros:
                - pedido_finalizado: El pedido completado que se va a procesar.
                - hora_inicio: Hora de inicio del procesamiento del pedido.
                - tiempo: Tiempo necesario para el procesamiento.

                Retorna:
                - bool: True si el pedido se procesa correctamente, False en caso contrario.
                """
        if self.revision:
            self.cola.append(pedido_finalizado)
            return False
        else:
            self.revision = True
            pedido_finalizado.hora_ini_revision = generar_time(hora_inicio, tiempo)
            self.pedido_actual = pedido_finalizado

            return True

    def pedido_finalizado(self, hora_inicio, tiempo):
        """
                Marca el pedido actual en el Meson como completado.

                Parámetros:
                - hora_inicio: Hora de inicio de la finalización del pedido.
                - tiempo: Tiempo necesario para la finalización.

                Retorna:
                - pedido_completado: El pedido completado.
                """
        pedido_completado = self.pedido_actual
        pedido_completado.fecha_termino = generar_time(hora_inicio, tiempo)
        pedido_completado.hora_fin_revision = generar_time(hora_inicio, tiempo)
        if len(self.cola) > 0:
            p = self.cola.popleft()
            p.hora_ini_revision = generar_time(hora_inicio, tiempo)
            self.revision = True
            self.pedido_actual = p
        else:
            self.revision = False
            self.pedido_actual = None
        return pedido_completado


class Simulador:
    def __init__(self, df_dia, df_pasillo, prod_pass, dict_cols, lista_productos):
        """
                Inicializa un objeto Simulador.

                Parámetros:
                - df_dia: DataFrame que contiene los datos diarios.
                - df_pasillo: DataFrame que contiene los datos del Pasillo.
                - prod_pass: Contraseña del producto.
                - dict_cols: Diccionario de columnas.
                - lista_productos: Lista de productos.

                Atributos:
                - eventos (list): Lista de eventos de la simulación.
                - pasillos (dict): Diccionario de objetos Pasillo.
                - tiempo: Tiempo actual de la simulación.
                - df_dia: DataFrame que contiene los datos diarios.
                - total_pedidos: Número total de pedidos en la simulación.
                - hora_inicio: Hora de inicio de la simulación.
                - meson: Objeto Meson para el procesamiento de pedidos.
                - df_pasillo: DataFrame que contiene los datos del Pasillo.
                - prod_pass: Contraseña del producto.
                - lista_productos: Lista de productos.
                - dict_cols: Diccionario de columnas.
                - verificar_meson: Diccionario para rastrear el estado del Meson para cada pedido.
                - completados: Lista de pedidos completados.
                - resultados: Diccionario para almacenar los resultados de la simulación.
                """
        self.eventos = []  # (nombre, tiempo, params)
        self.pasillos = {nombre: Pasillo(nombre) for nombre in ['A', 'B', 'C']}
        self.tiempo = 0
        self.df_dia = df_dia.sort_values(by="mov_llamado")
        # eventos_llegada_sim = self.df_dia.mov_llamado.tolist()
        self.total_pedidos = len(self.df_dia)
        self.hora_inicio = self.df_dia.mov_llamado.tolist()[0]
        self.meson = Meson()
        self.df_pasillo = df_pasillo
        self.prod_pass = prod_pass
        self.lista_productos = lista_productos
        self.dict_cols = dict_cols
        self.verificar_meson = {}
        self.completados = []
        self.resultados = {}

    def proximo_evento(self):
        """
                Encuentra el próximo evento en la simulación.

                Retorna:
                - posicion_min_tiempo: Posición del evento con el tiempo mínimo.
                - tupla_menor_tiempo: Tupla que contiene la información del próximo evento.
                """

        posicion_min_tiempo = min(range(len(self.eventos)), key=lambda i: self.eventos[i][1])
        tupla_menor_tiempo = self.eventos.pop(posicion_min_tiempo)
        return posicion_min_tiempo, tupla_menor_tiempo

    def accion(self, tupla):
        """
                Realiza una acción basada en el evento dado.

                Parámetros:
                - tupla: Tupla que contiene la información del evento.

                Retorna:
                None
                """

        nombre, tiempo, p = tupla

        if nombre == 'llega_pedido':
            self.verificar_meson[p.folio] = [False, False, False]
            if len(p.dfA) > 0:
                if self.pasillos['A'].busqueda:
                    self.pasillos['A'].cola.append(p)
                else:
                    # No hay cola, pasa directamente a ser procesado
                    self.pasillos['A'].busqueda = True
                    p.hora_ini_pckA = generar_time(self.hora_inicio, self.tiempo)
                    self.pasillos['A'].pedido_actual = p
                    self.eventos.append(('pasilloA_finalizado', self.tiempo + p.predA, p))

            else:
                p.fin_pasA = True
                self.verificar_meson[p.folio][0] = True
            if len(p.dfB) > 0:

                if self.pasillos['B'].busqueda:
                    self.pasillos['B'].cola.append(p)
                else:
                    # No hay cola, pasa directamente a ser procesado
                    self.pasillos['B'].busqueda = True
                    p.hora_ini_pckB = generar_time(self.hora_inicio, self.tiempo)
                    self.eventos.append(('pasilloB_finalizado', self.tiempo + p.predB, p))
                    self.pasillos['B'].pedido_actual = p
            else:
                p.fin_pasB = True
                self.verificar_meson[p.folio][1] = True

            if len(p.dfC) > 0:
                if self.pasillos['C'].busqueda:
                    self.pasillos['C'].cola.append(p)
                else:
                    # No hay cola, pasa directamente a ser procesado
                    self.pasillos['C'].busqueda = True
                    p.hora_ini_pckC = generar_time(self.hora_inicio, self.tiempo)
                    self.eventos.append(('pasilloC_finalizado', self.tiempo + p.predC, p))
                    self.pasillos['C'].pedido_actual = p
            else:
                p.fin_pasC = True
                self.verificar_meson[p.folio][2] = True
            # reviso los pasillos involucrados en el pasillo

        elif nombre == 'pedido_entregado':
            pedido_completado = self.meson.pedido_finalizado(self.hora_inicio, self.tiempo)
            self.completados.append(pedido_completado)
            if self.meson.revision:
                p = self.meson.pedido_actual
                self.eventos.append(('pedido_entregado', self.tiempo + p.pred_, p))
            print(f'Pedidos completados {len(self.completados)}/{len(self.df_dia)}')

        elif nombre == 'pasilloA_finalizado':
            pedido_finalizado = self.pasillos['A'].fin_busqueda(self.hora_inicio, self.tiempo)
            self.verificar_meson[pedido_finalizado.folio][0] = True
            if self.pasillos['A'].busqueda:
                p = self.pasillos['A'].pedido_actual
                self.eventos.append(('pasilloA_finalizado', self.tiempo + p.predA, p))

            # REVISO SI PEDIDO ESTÁ LISTO PARA SER ENVIADO A MESON
            # a_meson  = self.busqueda_completa(pedido_finalizado)
            if all(self.verificar_meson[pedido_finalizado.folio]):
                # self.verificar_meson.pop(pedido_finalizado.folio)
                pedido_finalizado.hora_fin_pck = generar_time(self.hora_inicio, self.tiempo)
                nuevo_pedido = self.meson.nuevo_pedido(pedido_finalizado, self.hora_inicio, self.tiempo)
                if nuevo_pedido:
                    self.eventos.append(('pedido_entregado', self.tiempo + pedido_finalizado.pred_, pedido_finalizado))

        elif nombre == 'pasilloB_finalizado':
            pedido_finalizado = self.pasillos['B'].fin_busqueda(self.hora_inicio, self.tiempo)
            self.verificar_meson[pedido_finalizado.folio][1] = True
            if self.pasillos['B'].busqueda:
                p = self.pasillos['B'].pedido_actual
                self.eventos.append(('pasilloB_finalizado', self.tiempo + p.predB, p))

            # REVISO SI PEDIDO ESTÁ LISTO PARA SER ENVIADO A MESON

            if all(self.verificar_meson[pedido_finalizado.folio]):
                # self.verificar_meson.pop(pedido_finalizado.folio)
                pedido_finalizado.hora_fin_pck = generar_time(self.hora_inicio, self.tiempo)
                nuevo_pedido = self.meson.nuevo_pedido(pedido_finalizado, self.hora_inicio, self.tiempo)
                if nuevo_pedido:
                    self.eventos.append(('pedido_entregado', self.tiempo + pedido_finalizado.pred_, pedido_finalizado))

        elif nombre == 'pasilloC_finalizado':
            pedido_finalizado = self.pasillos['C'].fin_busqueda(self.hora_inicio, self.tiempo)
            self.verificar_meson[pedido_finalizado.folio][2] = True
            if self.pasillos['C'].busqueda:
                p = self.pasillos['C'].pedido_actual
                self.eventos.append(('pasilloC_finalizado', self.tiempo + p.predB, p))

            # REVISO SI PEDIDO ESTÁ LISTO PARA SER ENVIADO A MESON
            if all(self.verificar_meson[pedido_finalizado.folio]):
                # self.verificar_meson.pop(pedido_finalizado.folio)
                pedido_finalizado.hora_fin_pck = generar_time(self.hora_inicio, self.tiempo)
                nuevo_pedido = self.meson.nuevo_pedido(pedido_finalizado, self.hora_inicio, self.tiempo)
                if nuevo_pedido:
                    self.eventos.append(('pedido_entregado', self.tiempo + pedido_finalizado.pred_, pedido_finalizado))

        else:
            print("Nombre accion no mapeada")

    def busqueda_completa(self, p):
        """
                Verifica si la búsqueda de un pedido está completa.

                Parámetros:
                - p: El pedido a verificar.

                Retorna:
                - bool: True si la búsqueda está completa, False en caso contrario.
                """

        if (p.fin_pasA) and (p.fin_pasB) and (p.fin_pasC):
            return True
        return False

    def add_evento_llegada(self):
        """
                Agrega eventos de llegada para todos los pedidos en la simulación.

                Retorna:
                None
                """
        print('Añadiendo todos los eventos llegada')
        for i in range(len(self.df_dia)):
            p = nuevo_pedido(i, self.df_dia, self.df_pasillo, self.prod_pass ,self.dict_cols ,self.lista_productos)
            delta_time = p.hora_llamado - self.hora_inicio
            one_second = np.timedelta64(1000000000, 'ns')
            seconds = delta_time / one_second
            self.eventos.append(('llega_pedido', seconds, p))
        print('Eventos llegada finalizado')

    def run(self):
        """
                Ejecuta la simulación.

                Retorna:
                None
                """
        print('INICIO SIMULADOR')

        continuar = True
        i = 0
        t0 = time.time()
        self.add_evento_llegada()  # añado todos los eventos de llegada.
        t1 = time.time()
        print(f'Tiempo transcurrido en add_evento_llegada {round(t1 - t0, 2)}')
        print(f'Pedidos completados {len(self.completados)}/{len(self.df_dia)}')
        while continuar:
            # eventos
            # EN ESTAS PRUEBAS CONVIENE PONER TODOS LOS PEDIDOS POR LLEGAR INICIALMENTE A EVENTO
            # escojo nuevo evento
            if len(self.eventos) > 0:
                pos, tupla = self.proximo_evento()
                print(f'TIEMPO ACTUAL {self.tiempo} --> PROXIMO EVENTO {tupla[1]} - {tupla[0]}')
                # acciono evento y cambios relacionados
                self.tiempo = tupla[1]
                self.accion(tupla)
                # actualizo tiempo

                # finalizo while?
            else:
                continuar = False

    def crosscheck(self):
        """
                Realiza una verificación cruzada para validar los resultados de la simulación.

                Retorna:
                None
                """
        for p in self.completados:
            if not (p.hora_fin_revision > p.hora_fin_pck):
                print(
                    f'[ERROR] Pedido {p.folio}: hora_fin_revision {p.hora_fin_revision} v/s hora_fin_pck {p.hora_fin_pck} ')
            if not (p.hora_fin_pck > p.hora_ini_pck):
                print(f'[ERROR] Pedido {p.folio}: hora_fin_pck {p.hora_fin_pck} v/s hora_ini_pck {p.hora_ini_pck} ')
            if p.hora_ini_pckA and not (p.hora_fin_pckA > p.hora_ini_pckA):
                print(f'[ERROR] Pedido {p.folio}: hora_fin_pckA {p.hora_fin_pckA} v/s hora_ini_pckA {p.hora_ini_pckA} ')
            if p.hora_ini_pckB and not (p.hora_fin_pckB > p.hora_ini_pckB):
                print(f'[ERROR] Pedido {p.folio}: hora_fin_pckB {p.hora_fin_pckB} v/s hora_ini_pckB {p.hora_ini_pckB} ')
            if p.hora_ini_pckC and not (p.hora_fin_pckC > p.hora_ini_pckC):
                print(f'[ERROR] Pedido {p.folio}: hora_fin_pckC {p.hora_fin_pckC} v/s hora_ini_pckC {p.hora_ini_pckC} ')


