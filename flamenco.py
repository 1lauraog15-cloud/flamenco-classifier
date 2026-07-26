#!/usr/bin/env python3
"""
Extrae features de audio musicológicamente informadas del dataset cante100.
Una fila por track → cante100_features.csv
"""

import os
import warnings
warnings.filterwarnings("ignore")

import collections
import numpy as np
import pandas as pd
import librosa
from scipy.stats import entropy
from tqdm import tqdm

from palo_templates import PALO_TEMPLATES

# ── Config ──────────────────────────────────────────────────────────────────
AUDIO_DIR  = os.path.expanduser("~/Desktop/cante100audio")
OUTPUT_CSV = os.path.expanduser("~/Desktop/cante100_features_v3.csv")
SR         = 22050   # resample estándar librosa
HOP        = 512
N_FFT      = 2048
N_MFCC     = 13

# Normalización de etiquetas (colapsa variantes ortográficas)
LABEL_MAP = {
    "soleares":             "soleares",
    "seguiriyas":           "seguiriyas",
    "bulerias":             "bulerias",
    "buleria":              "bulerias",   # typo en 1 archivo
    "buerias":              "bulerias",   # typo en 1 archivo
    "alegrias":             "alegrias",
    "tientostangos":        "tientostangos",
    "fandangos":            "fandangos",
    "cantesmineros":        "cantesmineros",
    "cantesamericanos":     "cantesamericanos",
    "tonas":                "tonas",
    "malaguenasgraninas":   "malaguenasgraninas",
    "malagenasgranainas":   "malaguenasgraninas",
    "malaguenasgranainas":  "malaguenasgraninas",
    "antoniomairenatonas":  "tonas",
}

TARGET_4 = {"soleares", "seguiriyas", "bulerias", "alegrias"}
PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Mapeo label normalizado → clave en PALO_TEMPLATES; fallback a soleá
LABEL_TO_TEMPLATE = {
    "soleares":  "soleá",
    "seguiriyas": "seguiriya",
    "bulerias":  "bulería",
    "alegrias":  "alegrías",
}
FALLBACK_TEMPLATE = "soleá"


def parse_label(filename):
    """Extrae palo del nombre: 027_ManuelSotoSordero_Soleares.mp3 → soleares."""
    stem = os.path.splitext(os.path.basename(filename))[0]
    raw  = stem.split("_")[-1].lower()
    return LABEL_MAP.get(raw, raw)


def extract_positional_features(degrees, palo_template, offset_beats=-2):
    """Features posicionales del ciclo de 12 pulsos (freq tónica en pos 10,
    freq bII en pos 3, entropía armónica media en posiciones de acento)."""
    cycle_length = palo_template['cycle_length']
    accents      = palo_template['accents']

    pos_degrees = collections.defaultdict(list)
    for i, deg in enumerate(degrees):
        pos_en_ciclo = ((i - offset_beats) % cycle_length) + 1
        pos_degrees[pos_en_ciclo].append(deg)

    pos10 = pos_degrees.get(10, [])
    freq_tonica_pos10 = pos10.count(0) / len(pos10) if pos10 else 0.0

    pos3 = pos_degrees.get(3, [])
    freq_bII_pos3 = pos3.count(1) / len(pos3) if pos3 else 0.0

    entropies = []
    for pos in accents:
        degs = pos_degrees.get(pos, [])
        if len(degs) > 1:
            counts = np.bincount(degs, minlength=12)
            entropies.append(float(entropy(counts + 1e-9)))
    harmonic_entropy_mean = np.mean(entropies) if entropies else 0.0

    return {
        'freq_tonica_pos10':    round(freq_tonica_pos10, 3),
        'freq_bII_pos3':        round(freq_bII_pos3, 3),
        'harmonic_entropy_mean': round(harmonic_entropy_mean, 3),
    }


def extract_features(path: str, palo_template: dict | None = None) -> dict:
    y, sr = librosa.load(path, sr=SR, mono=True)

    # ── Chroma CQT → tónica y perfil relativo ───────────────────────────────
    chroma      = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=HOP)
    chroma_mean = chroma.mean(axis=1)                     # (12,)

    tonica      = int(np.argmax(chroma_mean))             # pitch class 0–11
    chroma_rel  = np.roll(chroma_mean, -tonica)           # rota: tónica → idx 0
    chroma_rel  = chroma_rel / (chroma_rel.sum() + 1e-8)  # normaliza a suma 1

    # índice 1  = bII  (Phrygian 2nd: característica de soleá/seguiriya/bulería)
    # índice 10 = bVII (grado mixolidio: presente en alegrías)
    bII_energy  = float(chroma_rel[1])
    bVII_energy = float(chroma_rel[10])

    # ── pyin F0 (solo frames voiced) ────────────────────────────────────────
    f0, voiced_flag, _ = librosa.pyin(
        y, sr=sr,
        fmin=librosa.note_to_hz("C2"),   # ~65 Hz
        fmax=librosa.note_to_hz("C6"),   # ~1047 Hz
        hop_length=HOP,
    )
    voiced_f0  = f0[voiced_flag & ~np.isnan(f0)]
    pct_voiced = float(np.mean(voiced_flag))

    if len(voiced_f0) > 0:
        f0_median = float(np.median(voiced_f0))
        f0_range  = float(voiced_f0.max() - voiced_f0.min())
        f0_std    = float(voiced_f0.std())
    else:
        f0_median = f0_range = f0_std = float("nan")

    # ── MFCCs 1–13 ──────────────────────────────────────────────────────────
    mfcc     = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC,
                                    n_fft=N_FFT, hop_length=HOP)
    mfcc_m   = mfcc.mean(axis=1)
    mfcc_s   = mfcc.std(axis=1)

    # ── Spectral features ───────────────────────────────────────────────────
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr,
                                                  n_fft=N_FFT, hop_length=HOP)[0]
    rolloff  = librosa.feature.spectral_rolloff(y=y, sr=sr,
                                                 n_fft=N_FFT, hop_length=HOP)[0]
    zcr      = librosa.feature.zero_crossing_rate(y, hop_length=HOP)[0]
    rms      = librosa.feature.rms(y=y, hop_length=HOP)[0]

    # ── Tempo + beat frames (reutilizados para features posicionales) ─────────
    tempo_arr, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=HOP)
    tempo = float(np.atleast_1d(tempo_arr)[0])

    # ── Features posicionales (requieren beat_frames y chroma ya computados) ─
    if palo_template is not None:
        valid_frames = beat_frames[beat_frames < chroma.shape[1]]
        degrees = [
            int((np.argmax(chroma[:, cf]) - tonica) % 12)
            for cf in valid_frames
        ]
        pos_feats = extract_positional_features(degrees, palo_template, offset_beats=-2)
    else:
        pos_feats = {
            'freq_tonica_pos10':    float('nan'),
            'freq_bII_pos3':        float('nan'),
            'harmonic_entropy_mean': float('nan'),
        }

    # ── Ensambla fila ────────────────────────────────────────────────────────
    row = {
        "tonica":              tonica,
        "tonica_name":         PITCH_NAMES[tonica],
        "bII_energy":          bII_energy,
        "bVII_energy":         bVII_energy,
        "f0_median_hz":        f0_median,
        "f0_range_hz":         f0_range,
        "f0_std_hz":           f0_std,
        "pct_voiced":          pct_voiced,
        "spec_centroid_mean":  float(centroid.mean()),
        "spec_centroid_std":   float(centroid.std()),
        "spec_rolloff_mean":   float(rolloff.mean()),
        "zcr_mean":            float(zcr.mean()),
        "rms_mean":            float(rms.mean()),
        "rms_std":             float(rms.std()),
        "tempo_bpm":           tempo,
        **pos_feats,
    }
    # Perfil chroma relativo (12 features: idx 0 = tónica)
    for i, v in enumerate(chroma_rel):
        row[f"chroma_rel_{i}"] = float(v)
    # III mayor / bIII menor relativo a la tónica (discrimina modos mayor/menor/frigio)
    bIII = float(chroma_rel[3])
    row["ratio_mayor_menor"] = float(chroma_rel[4] / bIII) if bIII != 0.0 else float("nan")
    # MFCCs
    for i in range(N_MFCC):
        row[f"mfcc_{i+1}_mean"] = float(mfcc_m[i])
        row[f"mfcc_{i+1}_std"]  = float(mfcc_s[i])

    return row


def main():
    mp3s = sorted([
        os.path.join(AUDIO_DIR, f)
        for f in os.listdir(AUDIO_DIR)
        if f.lower().endswith(".mp3")
    ])
    print(f"Archivos encontrados: {len(mp3s)}")

    rows, errors = [], []

    for path in tqdm(mp3s, desc="Extrayendo features", unit="track"):
        fname = os.path.basename(path)
        label = parse_label(fname)
        try:
            template_key = LABEL_TO_TEMPLATE.get(label, FALLBACK_TEMPLATE)
            palo_template = PALO_TEMPLATES[template_key]
            feats            = extract_features(path, palo_template=palo_template)
            feats["filename"]     = fname
            feats["label"]        = label
            feats["label_4class"] = label if label in TARGET_4 else "other"
            rows.append(feats)
        except Exception as exc:
            errors.append((fname, str(exc)))
            tqdm.write(f"  ERROR {fname}: {exc}")

    df   = pd.DataFrame(rows)
    cols = ["filename", "label", "label_4class"] + [
        c for c in df.columns if c not in ("filename", "label", "label_4class")
    ]
    df   = df[cols]
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\nGuardado: {OUTPUT_CSV}  ({df.shape[0]} filas × {df.shape[1]} columnas)")
    if errors:
        print(f"Errores ({len(errors)}): {[e[0] for e in errors]}")

    print("\nDistribución de palos:")
    print(df["label"].value_counts().to_string())
    print(f"\nPrimeras 5 filas (columnas clave):")
    key_cols = ["filename", "label", "tonica", "bII_energy", "bVII_energy",
                "f0_median_hz", "pct_voiced", "tempo_bpm"]
    print(df[key_cols].head().to_string(index=False))
    return df


if __name__ == "__main__":
    main()
