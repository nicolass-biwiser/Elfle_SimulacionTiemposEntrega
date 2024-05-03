from pathlib import Path
from os import path


class FILES:
    input_dir = "input/"
    output_dir = "output/"
    input_modelo = 'input/Input_modelo.xlsx'
    output_modelo = "output/Salidas_Modelo.xlsx"
    output_sinopt = "output/df_SinOptimizar.png"
    output_hs = "output/df_Optimizando_Horas_Salida.png"
    output_as= "output/df_Optimizando_Asignación.png"
    INPUT_DIR = '/data/inputs'
    OUTPUT_DIR = '/data/outputs'

def make_dir_exist(file_path: str):
    destination_path = Path(file_path).parent.absolute()
    if destination_path.exists() is False:
        destination_path.mkdir(777, True, True)

    return destination_path.exists()


def getPath(_path):
    if _path != None and _path != '':
        if _path in list(vars(FILES).values()):
            currentPath = Path(__file__).parent.absolute()
            resultPath = path.join(currentPath, _path)
            if make_dir_exist(resultPath) is True:
                return resultPath
            raise Exception('No se pudo crear Directorio')
        raise Exception('query no registrada en mantenedor QUERY')
    raise Exception('parametro Query invalido')