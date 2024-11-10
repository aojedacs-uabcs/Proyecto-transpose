import os
import sys
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write

class GrabarAudio:
    def __init__(self, output_folder='records', sample_rate=44100):
        self.output_folder = output_folder
        self.sample_rate = sample_rate
        self.grabando = False
        self.grabacion = np.array([])  # Usamos un array para almacenar el audio

        # Crea la carpeta si no existe
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

    def grabar_voz(self):
        self.grabando = True

        with sd.InputStream(samplerate=self.sample_rate, channels=1, dtype='float32') as stream:
            while self.grabando:
                #Graba en bloques de 1 segundo
                audio_bloque = stream.read(self.sample_rate)[0]  # Esto devuelve un array 2D
                self.grabacion = np.concatenate((self.grabacion, audio_bloque.flatten()))  # Aplanamos el bloque

    def detener_grabacion(self, nombre_archivo="record"):
        input("\033[91m¡Grabando!\033[0m"
              "\nPresiona Enter para detener la grabación...\n")
        self.grabando = False

        #Verifica si hay grabaciones antes de intentar guardar
        if self.grabacion.size == 0:
            print("\033[91mError: Grabación demasiado corta.\033[0m")
            sys.exit()

        #Guarda el archivo WAV
        wav_path = os.path.join(self.output_folder, nombre_archivo + ".wav")
        write(wav_path, self.sample_rate, self.grabacion)
        print(f"Grabación guardada en: {wav_path}")
        return wav_path

