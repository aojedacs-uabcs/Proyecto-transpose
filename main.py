import json
import threading
from collections import defaultdict
from scipy.fftpack import fft
from scipy.io import wavfile
import os
import plotly.graph_objects as go
import numpy as np
import tqdm
import warnings
import subprocess

from Record import GrabarAudio
from Vocal import RangoVocal

warnings.filterwarnings("ignore", category=wavfile.WavFileWarning)  #Ignora advertencias sobre archivos WAV

#Configuración
FPS = 30  #Fotogramas
FFT_WINDOW_SECONDS = 0.25  #Duración en segundos de cada ventana FFT
FREQ_MIN = 50  #Frecuencia mínima
FREQ_MAX = 1100  #Frecuencia máxima
TOP_NOTES = 5  #Número máximo de notas a detectar
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]  #Notas
RESOLUTION = (1280, 720)  #Resolución del gráfico
SCALE = 0.5  #Factor de escala de resolución (0.5=QHD, 1=HD, 2=4K)

frame_folder = 'frames'
if not os.path.exists(frame_folder):
    os.makedirs(frame_folder)

graphs_folder = 'graphs'
if not os.path.exists(graphs_folder):
    os.makedirs(graphs_folder)

record_data_folder = 'record_data'
if not os.path.exists(record_data_folder):
    os.makedirs(record_data_folder)

#Graba audio desde el micrófono
microfono = GrabarAudio()

#Inicia grabación en un hilo separado
grabar_thread = threading.Thread(target=microfono.grabar_voz)
grabar_thread.start()

#Detener grabación
archivo_wav = microfono.detener_grabacion()
grabar_thread.join()

#Ruta de audio
AUDIO_FILE = 'records/c_scale.wav'
AUDIO_FILE = os.path.abspath(AUDIO_FILE)

#Verifica si el archivo de audio existe
if not os.path.exists(AUDIO_FILE):
    print(f"Error: El archivo de audio no se encuentra en la ruta especificada: {AUDIO_FILE}")
    exit()

#Graficar espectro de frecuencias
def plot_fft(p, xf, fs, notes, dimensions=(960, 540)):
    layout = go.Layout(
        title="Espectro de Frecuencias",
        autosize=False,
        width=dimensions[0],  #Ancho de gráfico
        height=dimensions[1],  #Altura de gráfico
        xaxis_title="Frecuencia (nota)",  #Eje X
        yaxis_title="Magnitud",  #Eje Y
        font={'size': 24}  #Tamaño de fuente
    )

    #Crea figura de Plotly
    fig = go.Figure(layout=layout,
                    layout_xaxis_range=[FREQ_MIN, FREQ_MAX],
                    layout_yaxis_range=[0, 1]
                    )

    #Datos de frecuencia
    fig.add_trace(go.Scatter(
        x=xf,
        y=p))

    #Anotaciones para notas detectadas
    for note in notes:
        fig.add_annotation(x=note[0] + 10, y=note[2],
                           text=note[1],
                           font={'size': 48},
                           showarrow=False)
    return fig

#Extrae muestras de audio
def extract_sample(audio, frame_number):
  end = frame_number * FRAME_OFFSET
  begin = int(end - FFT_WINDOW_SIZE)

  if end == 0:
    #Si no hay audio, devuelve un array de ceros
    return np.zeros((np.abs(begin)),dtype=float)
  elif begin<0:
    #Si hay menos audio del necesario, completa con ceros
    return np.concatenate([np.zeros((np.abs(begin)),dtype=float),audio[0:end]])
  else:
    #Devuelve el siguiente fragmento de audio
    return audio[begin:end]

#Diccionarios para almacenar los datos de las notas
notes = defaultdict(lambda: {'count': 0, 'magnitude': 0, 'frequency': 0})
def find_top_notes(fft, num):
    if np.max(fft.real) < 0.001:
        return []

    #Enumera frecuencias y magnitudes
    lst = [x for x in enumerate(fft.real)]
    lst = sorted(lst, key=lambda x: x[1], reverse=True)  #Ordena por magnitud

    idx = 0
    found = []
    found_note = set()
    max_magnitude_note = None

    while (idx < len(lst)) and (len(found) < num):
        f = xf[lst[idx][0]]  # Frecuencia
        y = lst[idx][1]  # Magnitud
        n = freq_to_number(f)

        if n == float('inf'):
            idx += 1
            continue

        n0 = int(round(n))
        name = note_name(n0)

        #Si la magnitud es inferior a 0.05, ignorar la nota
        if y < 0.05:
            idx += 1
            continue

        #Añade la nota si tiene una magnitud superior a 0.6
        if y > 0.6:
            found_note.add(name)
            s = [f, note_name(n0), y]
            found.append(s)

            #Actualiza la nota de mayor magnitud
            if max_magnitude_note is None or y > max_magnitude_note[2]:
                max_magnitude_note = s

        elif name not in found_note:
            found_note.add(name)
            s = [f, note_name(n0), y]
            found.append(s)

            if max_magnitude_note is None or y > max_magnitude_note[2]:
                max_magnitude_note = s

        idx += 1

    #Añade la nota de mayor magnitud
    if max_magnitude_note:
        note_name_max = max_magnitude_note[1]
        freq_max = max_magnitude_note[0]
        mag_max = max_magnitude_note[2]

        #Si la nota ya está en el diccionario, actualiza sus valores
        notes[note_name_max]['count'] += 1
        notes[note_name_max]['magnitude'] = max(notes[note_name_max]['magnitude'], mag_max)  #Guarda la mayor magnitud
        notes[note_name_max]['frequency'] = freq_max  #Actualiza la frecuencia

    return found

#Convierte la frecuencia a número de nota
def freq_to_number(f):
    if f <= 0:
        return float('inf')
    return 69 + 12 * np.log2(f / 440.0)

#Convierte el número de nota a frecuencia
def number_to_freq(n): return 440 * 2.0**((n-69)/12.0)

#Obtiene el nombre de la nota
def note_name(n): return NOTE_NAMES[n % 12] + str(int(n/12 - 1))

#Guarda datos importante
def save_data():
    #Ordena las notas por el número de veces contadas ('count'), de mayor a menor
    sorted_notes = sorted(notes.items(), key=lambda x: x[1]['count'], reverse=True)

    notas_dict = {
        note: {
            "count": data['count'],
            "magnitude": data['magnitude'],
            "frequency": data['frequency']
        }
        for note, data in sorted_notes
    }

    #Guarda los datos en un archivo JSON
    archivo_json = os.path.join('record_data', 'notes.json')
    with open(archivo_json, 'w') as f:
        json.dump(notas_dict, f, indent=4)

#Procesamiento principal del archivo de audio
fs, data = wavfile.read(AUDIO_FILE)  #Lee archivo WAV
audio = data.T[0] if data.ndim > 1 else data  #Manejo de mono y estéreo
FRAME_STEP = int(round(fs / FPS))  #Calcular paso de los frames
FFT_WINDOW_SIZE = int(fs * FFT_WINDOW_SECONDS)  #Tamaño de ventana FFT
AUDIO_LENGTH = len(audio) / fs  #Duración de audio en segundos

#Crea ventana de Hann para FFT
window = 0.5 * (1 - np.cos(np.linspace(0, 2 * np.pi, FFT_WINDOW_SIZE, False)))
xf = np.fft.rfftfreq(FFT_WINDOW_SIZE, 1 / fs)  #Calcula frecuencias FFT
FRAME_COUNT = int(AUDIO_LENGTH * FPS)  #Cuenta frames totales
FRAME_OFFSET = int(len(audio) / FRAME_COUNT)  #Offset frame

print(f"Duración del audio: {AUDIO_LENGTH} segundos")
print(f"Número de frames a procesar: {FRAME_COUNT}")

#Encuentra la amplitud máxima en los datos FFT
mx = 0
for frame_number in range(FRAME_COUNT):
  sample = extract_sample(audio, frame_number)

  fft = np.fft.rfft(sample * window)
  fft = np.abs(fft).real
  mx = max(np.max(fft),mx)

print(f"Amplitud máxima: {mx}")

#Elimina archivos PNG existentes en la carpeta "frames"
for filename in os.listdir(os.path.join(os.getcwd(), frame_folder)):
    if filename.endswith(".png"):
        os.remove(os.path.join(os.getcwd(), frame_folder, filename))

#Crea gráficos para cada frame de audio
for frame_number in tqdm.tqdm(range(FRAME_COUNT)):

    sample = extract_sample(audio, frame_number)

    fft = np.fft.rfft(sample * window)
    fft = np.abs(fft) / mx  #Normaliza

    #Encuentra las principales notas del frame actual
    notas_frame = find_top_notes(fft, TOP_NOTES)

    #Genera el gráfico FFT para el frame actual, mostrando el acorde recomendado
    fig = plot_fft(fft.real, xf, fs, notas_frame, RESOLUTION)

    fig.update_layout(
        title=f"Espectro de Frecuencia / Frame {frame_number}"
    )

    #Guardar el gráfico con el acorde en la carpeta de frames
    fig.write_image(os.path.join(frame_folder, f"frame{frame_number}.png"), scale=2)

#Ruta de salida
ruta_salida = os.path.abspath(os.path.join(os.getcwd(), graphs_folder, 'frequency.mp4'))

save_data()

#Generar el video utilizando ffmpeg
generarVideo = [
    "ffmpeg",
    "-y",  #Sobrescribe archivos existentes
    "-r", str(FPS),  #Frames por segundo
    "-f", "image2",  #Indica que las entradas son imágenes
    "-s", f"{RESOLUTION[0]}x{RESOLUTION[1]}",  #Resolución del video
    "-i", os.path.join(frame_folder, "frame%d.png"),  #Imágenes de entrada
    "-i", AUDIO_FILE,  #Archivo de audio
    "-c:v", "libx264",  #Códec de video
    "-pix_fmt", "yuv420p",  #Formato de píxel
    ruta_salida
]

#Ejecuta ffmpeg
result = subprocess.run(generarVideo, cwd=os.getcwd(), capture_output=True)
print(result.stdout.decode())  #salida estándar
print(result.stderr.decode())  #Salida de error

#Genera los graficos de Rango Vocal
rango_vocal_activacion = RangoVocal(notes)
rango_vocal_activacion.generar_campanas()