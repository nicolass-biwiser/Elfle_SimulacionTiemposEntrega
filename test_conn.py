import pyodbc


def verificar_conexion_sql(hostname, usuario, password, base_datos):
    try:
        # Establecer la conexión
        conn = pyodbc.connect(
            'DRIVER={SQL Server};SERVER=' + hostname + ';DATABASE=' + base_datos + ';UID=' + usuario + ';PWD=' + password)

        # Si la conexión se establece con éxito, imprimir mensaje de éxito
        print("Conexión exitosa a la base de datos:", base_datos)

        # Verificar permisos ejecutando una consulta simple
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        row = cursor.fetchone()

        # Si la consulta se ejecuta correctamente, imprimir mensaje de permisos
        print("Permisos de SQL verificados: Puede acceder a la base de datos")

        # Cerrar la conexión
        conn.close()

    except pyodbc.Error as e:
        # Si hay algún error en la conexión o permisos, imprimir el mensaje de error
        print("Error al conectar a la base de datos:", e)


# Datos de conexión
hostname = 'elfle-srv09.elfle.local'
usuario = 'biwiser'
password = 'bw2024Elfle'
base_datos = 'BIWISER'

# Llamada a la función para verificar la conexión y permisos
verificar_conexion_sql(hostname, usuario, password, base_datos)
