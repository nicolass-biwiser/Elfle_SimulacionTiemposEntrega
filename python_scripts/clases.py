from dataclasses import dataclass, field
from typing import Dict
from datetime import datetime
from pandas import DataFrame

@dataclass
class Producto:
    nombre: str = ""
    # Nombre y unidad al parecer no es necesario
    codigo: str = ''
    cantidad: bool = False
    unidad: str = ""
    pasillo: str = ''


@dataclass
class Operario:
    nombre: str = ""
    tiempo_empresa: float = 0.0
    acceso_pasillo: list = field(default_factory=list)
@dataclass
class Pasillo:
    fecha_in: datetime
    fecha_out: datetime
    nombre: str = ""
    productos: Dict[str, Producto] = field(default_factory=dict)
    operario = None

@dataclass
class Pedido:
    hora_meson: datetime
    hora_llamado: datetime
    folio: str = ""
    # nombre: str = ""
    doc: str = ''
    pasillos: list = field(default_factory=list)
    dfA : DataFrame = None
    dfB : DataFrame = None
    dfC : DataFrame = None
    df_ : DataFrame = None
    fin_pasA: bool = True
    fin_pasB: bool = True
    fin_pasC: bool = True
    pred_ini_pck: float = 0.0
    predA: float = 0.0
    predB: float = 0.0
    predC: float = 0.0
    pred_max: float  = 0.0
    pred_: float = 0.0
    hora_ini_pck: datetime = None
    hora_ini_pckA: datetime = None
    hora_ini_pckB: datetime = None
    hora_ini_pckC: datetime = None
    hora_fin_pck: datetime = None
    hora_fin_pckA: datetime = None
    hora_fin_pckB: datetime = None
    hora_fin_pckC: datetime = None
    hora_ini_revision: datetime = None
    hora_fin_revision: datetime = None

    hora_fin_pckSIMA: datetime = None
    hora_fin_pckSIMB: datetime = None
    hora_fin_pckSIMC: datetime = None
    fecha_termino: datetime = None
    # finpck_termino: datetime = None
    REAL_pred_ini_pck: float = 0.0
    REAL_predA: float = 0.0
    REAL_predB: float = 0.0
    REAL_predC: float = 0.0
    REAL_pred_max: float = 0.0
    REAL_pred_: float = 0.0
    REAL_hora_ini_pck: datetime = None
    REAL_hora_ini_pckA: datetime = None
    REAL_hora_ini_pckB: datetime = None
    REAL_hora_ini_pckC: datetime = None
    REAL_hora_fin_pck: datetime = None
    REAL_hora_fin_pckA: datetime = None
    REAL_hora_fin_pckB: datetime = None
    REAL_hora_fin_pckC: datetime = None
    REAL_fecha_termino: datetime = None

    time_A: datetime = None
    time_B: datetime = None
    time_C: datetime = None

    # total_items: int = 0
    # total_cantidad: int = 0
    # productos : Dict[str, Producto] = field(default_factory=dict)
    # pasillos: Dict[str, Pasillo] = field(default_factory=dict)

    #







