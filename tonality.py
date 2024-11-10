import json

#Intervalos para las escalas mayores y menores
INTERVALOS_MAYOR = [2, 2, 1, 2, 2, 2, 1]
INTERVALOS_MENOR = [2, 1, 2, 2, 1, 2, 2]

NOTAS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

class PrediccionTonalidad:
    def __init__(self, notas_grabadas):
        #Carga el archivo JSON de las notas grabadas
        self.notas_grabadas = self.cargar_json(notas_grabadas)
        self.escalas = self.generar_escalas()

    def cargar_json(self, archivo):
        """Función para cargar archivos JSON."""
        with open(archivo, 'r') as file:
            return json.load(file)

    def generar_escala(self, nota_base, intervalos):
        """Genera una escala a partir de una nota base y una lista de intervalos."""
        indice = NOTAS.index(nota_base)
        escala = [nota_base]
        for intervalo in intervalos:
            indice = (indice + intervalo) % len(NOTAS)
            escala.append(NOTAS[indice])
        return escala

    def generar_escalas(self):
        """Genera todas las escalas mayores y menores posibles."""
        escalas = {}
        for nota in NOTAS:
            escalas[f"{nota} Major"] = self.generar_escala(nota, INTERVALOS_MAYOR)
            escalas[f"{nota} Minor"] = self.generar_escala(nota, INTERVALOS_MENOR)
        return escalas

    def calcular_coincidencias(self, notas, escala):
        """Calcula cuántas notas de las más comunes coinciden con la escala."""
        coincidencias = 0
        for nota in notas:
            if nota in escala:
                coincidencias += self.notas_grabadas[nota]
        return coincidencias

    def predecir_tonalidad(self):
        """Predice la tonalidad más probable."""
        mejor_puntaje = -1
        escalas_detectadas = []

        notas_grabadas_keys = self.notas_grabadas.keys()

        for tonalidad, escala in self.escalas.items():
            #Calcular el puntaje para cada tonalidad
            puntaje = self.calcular_coincidencias(notas_grabadas_keys, escala)

            if puntaje > mejor_puntaje:
                mejor_puntaje = puntaje
                escalas_detectadas = [tonalidad]  #Limpiar y agregar la mejor tonalidad
            elif puntaje == mejor_puntaje:
                escalas_detectadas.append(tonalidad)  #Agregar si hay empate

        self.guardar_tonalidades(escalas_detectadas)

        return escalas_detectadas

    def guardar_tonalidades(self, escalas_detectadas):
        """Guarda las escalas y sus notas en un archivo JSON."""
        tonalidades = {tonalidad: self.escalas[tonalidad] for tonalidad in escalas_detectadas}

        with open('record_data/tonalities.json', 'w') as json_file:
            json.dump(tonalidades, json_file, indent=4)