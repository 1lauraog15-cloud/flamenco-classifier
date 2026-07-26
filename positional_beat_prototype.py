"""
Prototipo -- fase 3: beat-tracking posicionado en el ciclo de 12 pulsos,
sobre un track real de soleá (cante100, track id 22: Tio Gregorio Borrico,
"Estan tocando a rebato").

Objetivo de este prototipo (no es aun el pipeline final):
  1. Localizar el audio dentro de la carpeta local.
  2. Detectar el pulso (beat-tracking) y ver a que resolucion detecta
     librosa los pulsos -- si coincide, a grandes rasgos, con los "12
     tiempos" flamencos o si detecta un pulso mas grueso (eso hay que
     comprobarlo empiricamente, no asumirlo).
  3. Extraer croma por pulso detectado y ver la nota/grado dominante.
  4. Imprimir un diagnostico visual, no una clasificacion automatica.
     Este paso es para VALIDAR el metodo antes de escalarlo a 100 tracks.

Requiere: librosa, numpy  (pip install librosa --break-system-packages)
"""

import os
import glob
import numpy as np
import librosa

from palo_templates import PALO_TEMPLATES  # el diccionario que ya construimos


# ---------------------------------------------------------------------
# 1. Localizar el archivo de audio por palabras clave, sin asumir
#    convencion de nombres.
# ---------------------------------------------------------------------
def find_audio_file(folder, keywords, extensions=(".wav", ".mp3", ".flac", ".m4a")):
    """Busca en `folder` (recursivo) un archivo cuyo nombre contenga
    TODAS las palabras clave (sin distinguir mayus/minus/acentos simples).
    Devuelve la lista de candidatos -- si hay mas de uno, hay que revisar
    a mano cual es."""
    keywords_norm = [k.lower() for k in keywords]
    candidates = []
    for ext in extensions:
        for path in glob.glob(os.path.join(folder, "**", f"*{ext}"), recursive=True):
            name = os.path.basename(path).lower()
            if all(k in name for k in keywords_norm):
                candidates.append(path)
    return candidates


# ---------------------------------------------------------------------
# 2 y 3. Beat-tracking + croma por pulso
# ---------------------------------------------------------------------
def estimate_tonic_pitch_class(chroma):
    """Placeholder simple: pitch class con mas energia acumulada.
    SI YA TIENES una funcion de deteccion de tonica en flamenco.py
    (la que usas para las relative chroma profiles), sustituye esta
    por esa -- no tiene sentido mantener dos versiones distintas."""
    return int(np.argmax(chroma.sum(axis=1)))


def analyze_track(audio_path, sr=22050):
    print(f"Cargando: {audio_path}")
    y, sr = librosa.load(audio_path, sr=sr, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)
    print(f"Duracion: {duration:.1f}s, sr={sr}")

    # --- Beat tracking ---
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    # En versiones recientes de librosa, tempo puede venir como np.ndarray
    # (p.ej. array([90.]) ) en vez de escalar -- lo normalizamos aqui.
    tempo = float(np.asarray(tempo).squeeze())
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    n_beats = len(beat_times)
    print(f"Tempo estimado: {tempo:.1f} BPM")
    print(f"Pulsos detectados: {n_beats}")

    # Chequeo de sanidad: si el compas de soleá tiene 12 pulsos y dura
    # ~X segundos por ciclo, cuantos ciclos de 12 caben en n_beats?
    ciclos_aprox = n_beats / 12
    print(f"Eso equivaldria a ~{ciclos_aprox:.1f} ciclos de 12 pulsos "
          f"-- revisa si tiene sentido para la duracion del track "
          f"({duration:.1f}s). Si el numero de pulsos por segundo es "
          f"muy alto o muy bajo comparado con lo que esperarias al "
          f"escuchar el track, es señal de que librosa esta detectando "
          f"un pulso mas fino o mas grueso que el pulso flamenco real.")

    # --- Croma por pulso detectado ---
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_frames = librosa.time_to_frames(beat_times, sr=sr)
    chroma_frames = chroma_frames[chroma_frames < chroma.shape[1]]

    tonic_pc = estimate_tonic_pitch_class(chroma)
    pitch_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    print(f"\nTonica estimada (placeholder): {pitch_names[tonic_pc]}")

    print("\nGrado dominante por pulso detectado (relativo a la tonica estimada):")
    print("pulso_idx | tiempo(s) | grado_cromatico")
    degrees = []
    for i, cf in enumerate(chroma_frames):
        frame_chroma = chroma[:, cf]
        dominant_pc = int(np.argmax(frame_chroma))
        relative_degree = (dominant_pc - tonic_pc) % 12
        degrees.append(relative_degree)
        # posicion dentro de un ciclo hipotetico de 12 (asumiendo que el
        # pulso 0 detectado == pulso 1 flamenco -- ESTO hay que validarlo,
        # no darlo por hecho)
        pos_en_ciclo = (i % 12) + 1
        print(f"{i:9d} | {beat_times[i]:9.2f} | grado {relative_degree:2d} "
              f"(pos. hipotetica en ciclo: {pos_en_ciclo})")

    return {
        "tempo": tempo,
        "beat_times": beat_times,
        "degrees": degrees,
    }


# ---------------------------------------------------------------------
# 4. Comparacion (muy preliminar) contra la plantilla de soleá
# ---------------------------------------------------------------------
def compare_to_template(result, palo="soleá"):
    template = PALO_TEMPLATES[palo]
    print(f"\n--- Comparacion contra plantilla de {palo} ---")
    tension_beat = template.get('tension_beat', None)
    tension_degree = template.get('tension_degree', None)
    resolution_beat = template.get('resolution_beat', None)
    resolution_degree = template.get('resolution_degree', None)
    if tension_beat is not None:
        print(f"Tension esperada en pulso {tension_beat} (grado {tension_degree})")
    if resolution_beat is not None:
        print(f"Reposo esperado en pulso {resolution_beat} (grado {resolution_degree})")
    print("\nEsto todavia NO es una comparacion automatica -- es para que "
          "revises a ojo si el grado detectado en las posiciones 3 y 10 "
          "(dentro de cada ciclo hipotetico de 12) se parece a lo esperado. "
          "Si no cuadra, antes de tocar el codigo hay que revisar el "
          "supuesto mas fragil de todos: que el pulso 0 detectado por "
          "librosa coincide con el pulso 1 flamenco. Puede que haga falta "
          "anotar a mano el primer pulso de al menos un ciclo para anclar "
          "el conteo.")

def extract_positional_features(result, palo_template, offset_beats=-2):
    """
    Extrae features posicionales del ciclo de 12 pulsos para un track.
    Devuelve un diccionario con 3 features nuevos.
    """
    import collections

    degrees = result['degrees']
    cycle_length = palo_template['cycle_length']
    accents = palo_template['accents']  # posiciones clave [3, 6, 8, 10, 12]

    # Para cada posición del ciclo, recoge los grados observados
    pos_degrees = collections.defaultdict(list)
    for i, deg in enumerate(degrees):
        pos_en_ciclo = ((i - offset_beats) % cycle_length) + 1
        pos_degrees[pos_en_ciclo].append(deg)

    # Feature 1: frecuencia de grado 0 (tónica) en pos 10
    pos10 = pos_degrees.get(10, [])
    freq_tonica_pos10 = pos10.count(0) / len(pos10) if pos10 else 0.0

    # Feature 2: frecuencia de grado 1 (bII) en pos 3
    pos3 = pos_degrees.get(3, [])
    freq_bII_pos3 = pos3.count(1) / len(pos3) if pos3 else 0.0

    # Feature 3: entropía media del perfil armónico en posiciones de acento
    from scipy.stats import entropy
    entropies = []
    for pos in accents:
        degs = pos_degrees.get(pos, [])
        if len(degs) > 1:
            counts = np.bincount(degs, minlength=12)
            entropies.append(float(entropy(counts + 1e-9)))
    harmonic_entropy_mean = np.mean(entropies) if entropies else 0.0

    return {
        'freq_tonica_pos10': round(freq_tonica_pos10, 3),
        'freq_bII_pos3': round(freq_bII_pos3, 3),
        'harmonic_entropy_mean': round(harmonic_entropy_mean, 3),
    }

if __name__ == "__main__":
    DESKTOP_FOLDER = os.path.expanduser("~/Desktop/cante100audio")

    candidates = find_audio_file(DESKTOP_FOLDER, keywords=["037"])
    if not candidates:
     candidates = find_audio_file(DESKTOP_FOLDER, keywords=["rebato"])

    if not candidates:
        print(f"No he encontrado el track en {DESKTOP_FOLDER}. "
              f"Revisa el nombre real del archivo y ajusta la palabra "
              f"clave en find_audio_file(), o pasa la ruta directamente.")
    elif len(candidates) > 1:
        print("Varios candidatos encontrados, elige uno:")
        for c in candidates:
            print(" -", c)
    else:
            result = analyze_track(candidates[0])
            compare_to_template(result, palo="seguiriya")
            feats = extract_positional_features(
                result, PALO_TEMPLATES['seguiriya'], offset_beats=-2)
            print("\n--- Features posicionales extraídos ---")
            for k, v in feats.items():
                print(f"  {k}: {v}")

# Offset anotado manualmente: el "3" flamenco cae en t=0s (inicio del audio)
# Duración media del ciclo medida a oído: ~6s
# Con 12 pulsos por ciclo y pulso cada ~0.48s, el primer beat de librosa
# (t=0.39s) es aproximadamente el pulso 1 flamenco real

            OFFSET_BEATS = -2  # audio empieza en flamenco "3", faltan pulsos 1 y 2
            CYCLE_DURATION_S = 6.0  # segundos por ciclo de 12 pulsos

            print("\n--- Reancla con offset manual ---")
            print("Posiciones 3 y 10 en cada ciclo (grado armónico):")
            for ciclo in range(25):
                base = ciclo * 12 + OFFSET_BEATS
                for pos_flamenca in [3, 10]:
                    idx = base + (pos_flamenca - 1)
                    if idx < len(result['degrees']):
                        t = result['beat_times'][idx]
                        g = result['degrees'][idx]
                        print(f"  Ciclo {ciclo+1:2d} | pos {pos_flamenca:2d} | t={t:.2f}s | grado {g:2d}")