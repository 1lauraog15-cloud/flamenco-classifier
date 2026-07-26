# flamenco-classifier

Extracción de features de audio para clasificación automática de palos flamencos, sobre el dataset [cante100](https://mtg.upf.edu/research/datasets/cante100).

Desarrollado como parte de una investigación en musicología computacional del flamenco.

---

## ¿Qué hace?

El pipeline extrae features musicológicamente informadas de 100 grabaciones de cante flamenco (10 palos × 10 intérpretes), con el objetivo de entrenar clasificadores que distingan entre soleares, seguiriyas, bulerías, alegrías y otros palos.

Las features integran tanto características acústicas globales (MFCCs, F0, chroma) como features posicionales específicas del compás flamenco de 12 pulsos.

---

## Archivos

| Archivo | Descripción |
|---|---|
| `flamenco.py` | Pipeline principal: carga audio, extrae features, guarda CSV |
| `palo_templates.py` | Plantillas musicológicas por palo (compás, acentos, armonía) |
| `positional_beat_prototype.py` | Prototipo de beat-tracking posicionado en el ciclo de 12 pulsos |
| `flamenco_classifier.ipynb` | Notebook de exploración y visualización |
| `cante100_features_v3.csv` | Dataset de features extraídas (100 tracks × 60 columnas) |

---

## Features extraídas

### Globales
- **Tónica estimada** (pitch class 0–11, nombre)
- **bII\_energy** — energía del grado frigio (caracteriza soleá, seguiriya, bulería)
- **bVII\_energy** — grado mixolidio (caracteriza alegrías en modo mayor)
- **ratio\_mayor\_menor** — discrimina modo frigio / menor / mayor
- **Perfil chroma relativo** (12 dimensiones, normalizado a tónica)
- **F0 mediana, rango y desviación** (solo frames voiced, via pyin)
- **pct\_voiced** — porcentaje de frames con voz detectada
- **Spectral centroid, rolloff, ZCR, RMS** (media y std)
- **MFCCs 1–13** (media y std)
- **Tempo** estimado en BPM

### Posicionales (nuevas en v3)
Calculadas sobre el ciclo de 12 pulsos con `offset_beats=-2`, usando la plantilla del palo correspondiente como referencia:

- **freq\_tonica\_pos10** — frecuencia con que la tónica cae en el pulso 10 (posición de reposo en la familia soleá)
- **freq\_bII\_pos3** — frecuencia del grado frigio en el pulso 3 (tensión característica de soleá y bulería)
- **harmonic\_entropy\_mean** — entropía armónica media en las posiciones de acento del compás

---

## Results

| Model | CV F1-macro | Std |
|---|---|---|
| Random Forest (v3) | 0.645 | ±0.115 |
| Gradient Boosting | 0.610 | ±0.218 |
| SVM (RBF) | 0.580 | ±0.164 |

Best model: Random Forest. Seguiriyas and bulerías are best classified; soleares is the weakest class (shared Phrygian mode with seguiriyas, shared 12-beat compás with bulerías and alegrías).

---

## Uso

```bash
pip install librosa scipy pandas tqdm
```

Coloca los audios del cante100 en `~/Desktop/cante100audio/` y ejecuta:

```bash
python flamenco.py
```

Genera `cante100_features_v3.csv` con 100 filas × 60 columnas.

---

## Dataset

[cante100](https://mtg.upf.edu/research/datasets/cante100) — Music Technology Group, Universitat Pompeu Fabra.  
100 grabaciones de cante flamenco, 10 palos × 10 intérpretes.

---

## Palos cubiertos

`soleares` · `seguiriyas` · `bulerias` · `alegrias` · `fandangos` · `tientostangos` · `tonas` · `cantesmineros` · `cantesamericanos` · `malaguenasgraninas`

---

## Author

Laura Ortega — professional pianist (classical and flamenco), Data Scientist and AI.
