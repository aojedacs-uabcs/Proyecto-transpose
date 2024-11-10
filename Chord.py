import librosa

class Chord:
    def __init__(self, audio_file_path, hop_length, n_fft):
        self.audio_file_path = audio_file_path
        self.y, self.sr = librosa.load(audio_file_path, sr=None)  # Carga el audio sin cambiar la frecuencia
        self.chromagram = librosa.feature.chroma_stft(y=self.y, sr=self.sr, hop_length=hop_length, n_fft=n_fft)
        self.chroma_to_key = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

        chord_list = {
            # Acordes mayores
            "C": ["C", "E", "G"],
            "C#": ["C#", "F", "G#"],
            "D": ["D", "F#", "A"],
            "D#": ["D#", "G", "A#"],
            "E": ["E", "G#", "B"],
            "F": ["F", "A", "C"],
            "F#": ["F#", "A#", "C#"],
            "G": ["G", "B", "D"],
            "G#": ["G#", "C", "D#"],
            "A": ["A", "C#", "E"],
            "A#": ["A#", "D", "F"],
            "B": ["B", "D#", "F#"],

            # Acordes menores
            "Cm": ["C", "D#", "G"],
            "C#m": ["C#", "E", "G#"],
            "Dm": ["D", "F", "A"],
            "D#m": ["D#", "F#", "A#"],
            "Em": ["E", "G", "B"],
            "Fm": ["F", "G#", "C"],
            "F#m": ["F#", "A", "C#"],
            "Gm": ["G", "A#", "D"],
            "G#m": ["G#", "B", "D#"],
            "Am": ["A", "C", "E"],
            "A#m": ["A#", "C#", "F"],
            "Bm": ["B", "D", "F#"]
        }

        print("Chromagram shape:", self.chromagram.shape)  # Verifica la forma del chromagram

    def notes_from_frame(self, frame):
        notes_detected = []
        for i, chroma_value in enumerate(frame):
            if chroma_value > 0.5:  # Ajustable umbral
                notes_detected.append((self.chroma_to_key[i], chroma_value))
        return notes_detected

    def predict_chords(self):
        chords_by_frame = []

        for frame_number, frame in enumerate(self.chromagram.T):
            notes_detected = self.notes_from_frame(frame)

            if notes_detected:
                sorted_notes = sorted(notes_detected, key=lambda x: x[1], reverse=True)
                top_notes = [note[0] for note in sorted_notes[:3]]
                chord_prediction = "-".join(top_notes)
            else:
                chord_prediction = "N/A"

            chords_by_frame.append((frame_number, chord_prediction))

        return chords_by_frame