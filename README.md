# Sistema de Simulación de Pedidos

## Descripción
Este proyecto es un sistema de simulación de pedidos que modela el flujo de trabajo en un almacén o centro de distribución. El sistema utiliza modelos predictivos y simulación para optimizar el manejo de pedidos en diferentes pasillos y mesones.

## Estructura del Proyecto

```
ELFLE/
├── main.py               # Script principal de ejecución
├── python_scripts/       # Scripts de Python
│   ├── Simulador_clase.py  # Clases principales de simulación
│   └── functions.py      # Funciones auxiliares
├── df/                  # Directorio para datos
├── settings/            # Configuraciones
├── presentaciones/      # Presentaciones y documentación
├── requirements.txt     # Dependencias del proyecto
└── Pipfile             # Dependencias para Poetry
```

## Dependencias

El proyecto utiliza las siguientes dependencias principales:

- pandas==2.2.2
- numpy==1.26.4
- scikit-learn==1.4.2
- xgboost==2.0.3
- SQLAlchemy==2.0.29
- pyodbc==5.1.0
- openpyxl==3.1.2

## Componentes Principales

### 1. Clases Principales

- **Simulador**: Clase principal que maneja la simulación completa
- **Pasillo**: Representa los diferentes pasillos de trabajo (A, B, C)
- **Meson**: Maneja la revisión final de los pedidos

### 2. Funcionalidades

- Simulación de flujo de pedidos
- Predicción de tiempos de procesamiento
- Manejo de colas y prioridades
- Integración con base de datos
- Generación de reportes y análisis

## Modo de Uso

1. Instalar dependencias:
```bash
pip install -r requirements.txt
```

2. Ejecutar el simulador:
```bash
python main.py
```

## Funcionalidades Específicas

- Simulación de pedidos en tiempo real
- Manejo de diferentes tipos de productos
- Predicción de tiempos de procesamiento
- Optimización de rutas de trabajo
- Generación de estadísticas y métricas

## Desarrollo

El proyecto está diseñado para ser extensible y modular, permitiendo fácilmente:
- Añadir nuevos tipos de productos
- Modificar los modelos predictivos
- Aumentar la capacidad de simulación
- Integrar nuevos sistemas de reporte

## Contribución

Para contribuir al proyecto:
1. Clonar el repositorio
2. Crear una rama para nuevas características
3. Realizar los cambios
4. Hacer pull request
