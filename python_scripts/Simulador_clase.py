from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import random
import time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple, Union, Deque, TypeVar, Generic, Type
from datetime import datetime
import pandas as pd
import random

# Import local functions
from python_scripts.functions import (
    convertir_a_int, generar_time, nuevo_pedido
)

# Set random seed for reproducibility
random.seed(42)

# Type aliases
T = TypeVar('T')
DataFrame = pd.DataFrame

@dataclass
class Pedido:
    """Data class representing an order in the warehouse system."""
    folio: str
    dfA: DataFrame = field(default_factory=pd.DataFrame)
    dfB: DataFrame = field(default_factory=pd.DataFrame)
    dfC: DataFrame = field(default_factory=pd.DataFrame)
    predA: float = 0.0
    predB: float = 0.0
    predC: float = 0.0
    pred_: float = 0.0
    hora_ini_pck: Optional[datetime] = None
    hora_fin_pck: Optional[datetime] = None
    hora_ini_pckA: Optional[datetime] = None
    hora_fin_pckA: Optional[datetime] = None
    hora_ini_pckB: Optional[datetime] = None
    hora_fin_pckB: Optional[datetime] = None
    hora_ini_pckC: Optional[datetime] = None
    hora_fin_pckC: Optional[datetime] = None
    hora_ini_revision: Optional[datetime] = None
    hora_fin_revision: Optional[datetime] = None
    mov_entregado: Optional[datetime] = None
    fin_pasA: bool = False
    fin_pasB: bool = False
    fin_pasC: bool = False
    
    def __post_init__(self):
        # Ensure DataFrame attributes are always DataFrames
        self.dfA = pd.DataFrame() if self.dfA is None or self.dfA.empty else self.dfA
        self.dfB = pd.DataFrame() if self.dfB is None or self.dfB.empty else self.dfB
        self.dfC = pd.DataFrame() if self.dfC is None or self.dfC.empty else self.dfC

class Pasillo:
    """Represents an aisle in the warehouse where order picking occurs.
    
    Attributes:
        nombre: Identifier for the aisle (A, B, or C)
        busqueda: Whether the aisle is currently processing an order
        pedido_actual: The order currently being processed in this aisle
        cola: Queue of orders waiting to be processed in this aisle
    """
    
    def __init__(self, nombre: str) -> None:
        """Initialize a new Pasillo instance.
        
        Args:
            nombre: Identifier for the aisle (A, B, or C)
        """
        self.nombre: str = nombre
        self.busqueda: bool = False
        self.pedido_actual: Optional[Pedido] = None
        self.cola: Deque[Pedido] = deque()

    def fin_busqueda(self, hora_inicio: datetime, tiempo: float) -> Optional[Pedido]:
        """Complete the current order picking in the aisle.
        
        Args:
            hora_inicio: Start time of the picking operation
            tiempo: Duration of the picking operation in seconds
            
        Returns:
            The completed Pedido if any, None otherwise
            
        Raises:
            ValueError: If the aisle name is not recognized
        """
        if self.pedido_actual is None:
            return None
            
        pedido_finalizado = self.pedido_actual
        
        # Update the appropriate completion time based on the aisle
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
            raise ValueError(f'Nombre de pasillo no reconocido: {self.nombre}')
            
        # Load the next order if available
        self.cargar_nuevo_pedido(hora_inicio, tiempo)
        return pedido_finalizado
        # REVISO SI PEDIDO ESTÁ LISTO PARA SER ENVIADO A MESON

    def cargar_nuevo_pedido(self, hora_inicio: datetime, tiempo: float) -> None:
        """Load the next order from the queue for processing in the aisle.
        
        Args:
            hora_inicio: Start time of the picking operation
            tiempo: Current simulation time in seconds
            
        Raises:
            ValueError: If the aisle name is not recognized
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
            # No more orders in the queue
            self.pedido_actual = None
            self.busqueda = False


class Meson:
    """Represents a checking station where orders are verified after picking.
    
    Attributes:
        revision: Whether the station is currently verifying an order
        pedido_actual: The order currently being verified
        cola: Queue of orders waiting to be verified
    """
    
    def __init__(self) -> None:
        """Initialize a new Meson instance."""
        self.revision: bool = False
        self.pedido_actual: Optional[Pedido] = None
        self.cola: Deque[Pedido] = deque()

    def nuevo_pedido(self, pedido_finalizado: Pedido, hora_inicio: datetime, 
                    tiempo: float) -> bool:
        """Add a new completed order to the checking station.
        
        Args:
            pedido_finalizado: The completed order to be verified
            hora_inicio: Start time of the verification
            tiempo: Current simulation time in seconds
            
        Returns:
            bool: True if the order starts verification immediately, 
                 False if it's added to the queue
        """
        if self.revision:
            self.cola.append(pedido_finalizado)
            return False
            
        self.revision = True
        pedido_finalizado.hora_ini_revision = generar_time(hora_inicio, tiempo)
        self.pedido_actual = pedido_finalizado
        return True

    def pedido_finalizado(self, hora_inicio: datetime, tiempo: float) -> Optional[Pedido]:
        """Mark the current order as completed and start the next one if available.
        
        Args:
            hora_inicio: Start time of the completion
            tiempo: Current simulation time in seconds
            S
        Returns:
            The completed Pedido if any, None otherwise
        """
        if self.pedido_actual is None:
            return None
            
        pedido_completado = self.pedido_actual
        pedido_completado.mov_entregado = generar_time(hora_inicio, tiempo)
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
    """Main simulation class that coordinates the warehouse order picking process.
    
    This class manages the simulation of order processing through different warehouse
    aisles and a final checking station. It handles event scheduling, order processing,
    and tracks simulation metrics.
    
    Attributes:
        eventos: List of simulation events as (event_name, time, params) tuples
        pasillos: Dictionary of aisle objects (A, B, C)
        tiempo: Current simulation time in seconds
        df_dia: DataFrame containing daily order data
        total_pedidos: Total number of orders to process
        hora_inicio: Simulation start time
        meson: Checking station instance
        df_pasillo: DataFrame with aisle-specific data
        prod_pass: Product password/identifier
        lista_productos: List of product identifiers
        dict_cols: Dictionary of column mappings
        verificar_meson: Tracks order status for the checking station
        completados: List of completed orders
        resultados: Dictionary for storing simulation results
        mod_a: Model for aisle A predictions
        mod_b: Model for aisle B predictions
        mod_c: Model for aisle C predictions
        mod_: General model for final processing
        sim: Whether to run in simulation mode
        hora_dia: Current day time
        resultado_bbdd: Database results storage
    """
    
    def __init__(self, 
                 df_dia: pd.DataFrame,
                 df_pasillo: pd.DataFrame,
                 prod_pass: Any,
                 dict_cols: Dict[str, Any],
                 lista_productos: List[str],
                 mod_a: Any,
                 mod_b: Any,
                 mod_c: Any,
                 mod_: Any,
                 sim: bool,
                 resultado_bbdd: Any,
                 hora_dia: datetime,
                 productos_pasillos_arreglado: Dict[str, List[str]]) -> None:
        """Initialize the Simulador with the given parameters.
        
        Args:
            df_dia: DataFrame containing daily order data
            df_pasillo: DataFrame with aisle-specific product data
            prod_pass: Product password/identifier
            dict_cols: Dictionary mapping column names
            lista_productos: List of product identifiers
            mod_a: Prediction model for aisle A
            mod_b: Prediction model for aisle B
            mod_c: Prediction model for aisle C
            mod_: General prediction model
            sim: Whether to run in simulation mode
            resultado_bbdd: Database results storage
            hora_dia: Current day time
            productos_pasillos_arreglado: Dictionary of aisle-specific products
        """
        # Initialize simulation state
        self.eventos: List[Tuple[str, float, Any]] = []  # (event_name, time, params)
        self.pasillos: Dict[str, Pasillo] = {nombre: Pasillo(nombre) for nombre in ['A', 'B', 'C']}
        self.tiempo: float = 0.0
        
        # Store input data and parameters
        self.df_dia: pd.DataFrame = df_dia.sort_values(by="mov_llamado")
        self.df_pasillo: pd.DataFrame = df_pasillo
        self.prod_pass: Any = prod_pass
        self.lista_productos: List[str] = lista_productos
        self.dict_cols: Dict[str, Any] = dict_cols
        
        # Initialize models
        self.mod_a: Any = mod_a
        self.mod_b: Any = mod_b
        self.mod_c: Any = mod_c
        self.mod_: Any = mod_
        
        # Simulation state tracking
        self.total_pedidos: int = len(self.df_dia)
        self.hora_inicio: datetime = self.df_dia.mov_llamado.tolist()[0]
        self.meson: Meson = Meson()
        self.verificar_meson: Dict[str, List[bool]] = {}
        self.completados: List[Pedido] = []
        self.resultados: Dict[str, Any] = {}
        
        # Additional parameters
        self.sim: bool = sim
        self.hora_dia: datetime = hora_dia
        self.resultado_bbdd: Any = resultado_bbdd
        self.productos_pasillos_arreglado: Dict[str, List[str]] = productos_pasillos_arreglado
    def proximo_evento(self) -> Tuple[int, Tuple[str, float, Any]]:
        """Find and remove the next event from the simulation queue.
        
        The events are processed in chronological order based on their scheduled time.
        
        Returns:
            A tuple containing:
            - The index of the next event in the original events list
            - A tuple with (event_name, event_time, event_params)
            
        Raises:
            IndexError: If there are no events in the queue
        """
        if not self.eventos:
            raise IndexError("No events in the simulation queue")
            
        posicion_min_tiempo = min(
            range(len(self.eventos)), 
            key=lambda i: self.eventos[i][1]  # Sort by event time (index 1)
        )
        tupla_menor_tiempo = self.eventos.pop(posicion_min_tiempo)
        return posicion_min_tiempo, tupla_menor_tiempo

    def accion(self, tupla: Tuple[str, float, Any]) -> None:
        """Process an event from the simulation queue.
        
        This method handles different types of events in the simulation, including:
        - Order arrivals ('llega_pedido')
        - Order completions ('pedido_entregado')
        - Aisle completions ('pasilloA_finalizado', 'pasilloB_finalizado', 'pasilloC_finalizado')
        
        Args:
            tupla: A tuple containing (event_name, event_time, event_params)
            
        Raises:
            ValueError: If the event name is not recognized
        """
        nombre, tiempo, p = tupla

        if nombre == 'llega_pedido':
            self.verificar_meson[p.folio] = [False, False, False]
            if len(p.dfA) > 0:
                if self.pasillos['A'].busqueda:
                    # Si hay cola, pasa a la cola
                    self.pasillos['A'].cola.append(p)
                else:
                    # No hay cola, pasa directamente a ser procesado
                    self.pasillos['A'].busqueda = True
                    p.hora_ini_pckA = generar_time(self.hora_inicio, self.tiempo)
                    self.pasillos['A'].pedido_actual = p
                    self.eventos.append(('pasilloA_finalizado', self.tiempo + p.predA, p))

            else:
                # Mark as completed for aisle A if not needed
                p.fin_pasA = True
                self.verificar_meson[p.folio][0] = True
            if len(p.dfB) > 0:
                # Si está en Pasillo B
                if self.pasillos['B'].busqueda:
                    # Si hay cola, pasa a la cola
                    self.pasillos['B'].cola.append(p)
                else:
                    # No hay cola, pasa directamente a ser procesado
                    self.pasillos['B'].busqueda = True
                    p.hora_ini_pckB = generar_time(self.hora_inicio, self.tiempo)
                    self.eventos.append(('pasilloB_finalizado', self.tiempo + p.predB, p))
                    self.pasillos['B'].pedido_actual = p
            else:
                # Mark as completed for aisle B if not needed
                p.fin_pasB = True
                self.verificar_meson[p.folio][1] = True

            if len(p.dfC) > 0:
                # Si está en Pasillo C
                if self.pasillos['C'].busqueda:
                    # Si hay cola, pasa a la cola
                    self.pasillos['C'].cola.append(p)
                else:
                    # No hay cola, pasa directamente a ser procesado
                    self.pasillos['C'].busqueda = True
                    p.hora_ini_pckC = generar_time(self.hora_inicio, self.tiempo)
                    self.eventos.append(('pasilloC_finalizado', self.tiempo + p.predC, p))
                    self.pasillos['C'].pedido_actual = p
            else:
                # Mark as completed for aisle C if not needed
                p.fin_pasC = True
                self.verificar_meson[p.folio][2] = True
                
            # Check if order is ready to be sent to the checking station

        elif nombre == 'pedido_entregado':
            # Process order completion at the checking station
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
        # NECESITO HORA ACTUAL O DE INICIO PARA FILTRAR INFO QUE TENGO REAL VS PRED. PRED DEBO ASUMIR QUE LO TENGO GUARDADO
        # teoricamente esa tabla va a ser lo que guarde en bbdd.
        # INICIO Y FIN DE CADA ETAPA.
        #
        for i in range(len(self.df_dia)):
            p = nuevo_pedido(i, self.df_dia, self.df_pasillo, self.prod_pass ,self.dict_cols ,
                             self.lista_productos, self.mod_a, self.mod_b, self.mod_c, simulado=self.sim,
                             resultado_bbdd=self.resultado_bbdd, productos_pasillos_arreglado=self.productos_pasillos_arreglado)
            delta_time = p.hora_ini_pck - self.hora_inicio
            one_second = np.timedelta64(1000000000, 'ns')
            seconds = convertir_a_int(delta_time / one_second)
            # NO TODOS SON LLEGA_PEDIDO
            self.eventos.append(('llega_pedido', seconds, p))
        print(f'Se añadieron {len(self.eventos)} en un inicio')
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
        # print(f'Pedidos completados {len(self.completados)}/{len(self.df_dia)}')
        cant_pedidos = len(self.df_dia)
        print(f'Pedidos completados {len(self.completados)}/{len(self.df_dia)}')
        while continuar:

            # eventos
            # EN ESTAS PRUEBAS CONVIENE PONER TODOS LOS PEDIDOS POR LLEGAR INICIALMENTE A EVENTO
            # escojo nuevo evento
            if len(self.eventos) > 0:
                for e in self.eventos:
                    if not (isinstance(e[1],int) or isinstance(e[1],float)):
                        print(e[0], e[1])
                pos, tupla = self.proximo_evento()
                print(f'TIEMPO ACTUAL {self.tiempo} --> PROXIMO EVENTO {tupla[2].folio} {tupla[1]} - {tupla[0]}')
                # acciono evento y cambios relacionados
                self.tiempo = tupla[1]
                self.accion(tupla)
                # actualizo tiempo

                # finalizo while?
            else:
                print('here continuar false ')
                print(f'Pedidos completados {len(self.completados)}/{len(self.df_dia)}')
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