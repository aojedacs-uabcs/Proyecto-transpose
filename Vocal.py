import json
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
class RangoVocal:
    def __init__(self, data):
        if isinstance(data, dict):
            self.data = data
        else:
            raise ValueError("Los datos proporcionados deben ser un diccionario.")

        #Define los rangos vocales con sus frecuencias en Hz (min, max)
        self.rangos = {
            "bajo": (82.41, 349.23),  #E2 a F4
            "baritono": (110.00, 440.00),  #A2 a A4
            "tenor": (130.81, 493.88),  #C3 a B4
            "contratenor": (164.81, 659.26),  #E3 a E5
            "mezzosoprano": (220.00, 880.00),  #A3 a A5
            "soprano": (261.63, 1046.50),  #C4 a C6
            "contralto": (174.61, 698.46)  #F3 a F5
        }

        self.frecuencia_minima = 50  ##Umbral minimo de frecuencia para evitar ruido
        self.frecuencia_maxima = 1100  #Umbral máximo de frecuencia para evitar ruido
        self.distancia_minima = 10  #Distancia mínima en Hz entre notas para considerarlas ruido

        #Elimina archivos PNG existentes en la carpeta "graphs"
        for filename in os.listdir(os.path.join(os.getcwd(), 'graphs')):
            if filename.endswith(".png"):
                os.remove(os.path.join(os.getcwd(), 'graphs', filename))

    def gaussian_activation(self, x, mu, sigma):
        """Función de activación gaussiana"""
        return tf.exp(-0.5 * tf.square((x - mu) / sigma))

    def generar_campanas(self):

        x_vals = np.linspace(self.frecuencia_minima, self.frecuencia_maxima, 1000)  #Frecuencias desde 50 Hz hasta 1100 Hz
        plt.figure(figsize=(15, 6))

        conteo_notas = {rango: 0 for rango in self.rangos}

        #Grafica la función de activación para cada rango vocal
        for rango, (min_f, max_f) in self.rangos.items():
            centro = (min_f + max_f) / 2
            sigma = (max_f - min_f) / 6
            y_vals = self.gaussian_activation(x_vals, centro, sigma)

            #Convierte tensores a arrays para graficar
            y_vals = y_vals.numpy()
            plt.plot(x_vals, y_vals, label=f'{rango} [{min_f:.2f} Hz - {max_f:.2f} Hz]')

            #Cuenta las notas dentro del rango vocal y dentro de los límites de frecuencia permitidos
            for nota, info in self.data.items():
                freq = info['frequency']
                if min_f <= freq <= max_f and self.frecuencia_minima <= freq <= self.frecuencia_maxima:
                    conteo_notas[rango] += 1 #Sirve para saber el rango vocal predominante

        #Filtra las frecuencias dentro de los rangos vocales
        filtered_frequencies = []
        for nota, info in self.data.items():
            freq = info['frequency']

            #Verifica que la frecuencia esté dentro de los límites y no demasiado cerca de otras frecuencias
            if self.frecuencia_minima <= freq <= self.frecuencia_maxima and any(
                    min_f <= freq <= max_f for min_f, max_f in self.rangos.values()):
                if all(abs(freq - f) >= self.distancia_minima for f in filtered_frequencies):
                    filtered_frequencies.append(freq)
                    plt.axvline(x=freq, color='black', linestyle=':', linewidth=2)

        #Detalles de la gráfica
        plt.title('Rangos Vocales')
        plt.xlabel('Frecuencia (Hz)')
        plt.ylabel('Activación')
        plt.legend()
        plt.grid(True)

        #Guarda el gráfico en la carpeta 'graphs'
        plt.savefig('graphs/vocal_ranges.png')
        plt.close()

        #Encuentra el rango con más notas
        rango = max(conteo_notas, key=conteo_notas.get)

        #Grafica el rango vocal con más notas
        self.graficar_rango(rango, filtered_frequencies)

    def graficar_rango(self, rango, filtered_frequencies):
        """Graficar el rango vocal que contiene la mayor cantidad de notas"""
        min_f, max_f = self.rangos[rango]

        #Definie los límites de la gráfica extendidos en ±100 Hz
        x_min = max(self.frecuencia_minima, min_f - 100)  # No menos de 50 Hz
        x_max = min(self.frecuencia_maxima, max_f + 100)  # No más de 1100 Hz
        x_vals = np.linspace(x_min, x_max, 1000)

        centro = (min_f + max_f) / 2
        sigma = (max_f - min_f) / 6

        plt.figure(figsize=(10, 6))
        y_vals = self.gaussian_activation(x_vals, centro, sigma).numpy()
        plt.plot(x_vals, y_vals, label=f'{rango} [{min_f:.2f} Hz - {max_f:.2f} Hz]', color='orange')

        #Marca las notas en el rango vocal, en rojo si están cerca del centro
        notas_en_rango = [] #Para guardar las notas del rango vocal
        for nota, info in self.data.items():
            freq = info['frequency']
            if min_f <= freq <= max_f:
                #Define el color para las notas centrales
                color = 'red' if abs(freq - centro) <= sigma else 'black'
                label = f"{nota} ({freq:.2f} Hz)" if color == 'red' else None  #Etiqueta las frecuencias centrales con nombre y frecuencia
                plt.axvline(x=freq, color=color, linestyle=':', linewidth=2, label=label)

                #Agrega la nota y frecuencia a la lista
                notas_en_rango.append({"note": nota, "frequency": freq})

        #Ordena las notas por frecuencia ascendente
        notas_en_rango.sort(key=lambda x: x["frequency"])

        # Guarda las notas en un archivo JSON
        with open(f'record_data/vocal_range.json', 'w') as json_file:
            json.dump({rango: notas_en_rango}, json_file, indent=4)

        #Detalles de la gráfica
        plt.title(f'Rango Vocal Predominante: {rango}')
        plt.xlabel('Frecuencia (Hz)')
        plt.ylabel('Activación')
        plt.legend()
        plt.grid(True)

        #Escala de grafica
        plt.xticks(np.arange(x_min, x_max + 1, 50))

        #Guarda el gráfico en la carpeta 'graphs'
        plt.savefig(f'graphs/{rango}.png')
        plt.close()