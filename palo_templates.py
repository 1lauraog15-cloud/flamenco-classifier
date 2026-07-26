"""
Plantillas de compás y armonía por familia de palo.
Fuente: Asensio (apuntes ESMUC), Castro Buendía (2014), Cañizares
(Patrones Estructurales de la Seguiriya / Soleá / Alegrías).

Uso previsto: comparar la secuencia armónica extraída del audio
(chroma + beat-tracking, posicionada en el ciclo de 12 pulsos)
contra estos valores esperados, como feature de similitud por palo.
"""

PALO_TEMPLATES = {

    "soleá": {
        "cycle_length": 12,
        "count_style": "acéfalo: (12) 1 2 / 3 4 5 / 6 7 / 8 9 / 10 11",
        "accents": [3, 6, 8, 10, 12],
        "perceptual_downbeat": 3,  # Castro: "comienzo melódico con intención
                                   # de acento en el pulso 3"
        "mode": "phrygian",             # Mi frigio flamenco
        "tension_beat": 3,
        "tension_degree": "bII",
        "resolution_beat": 10,
        "resolution_degree": "I",
        "cierre_beats": [10, 11, 12],
        "cierre_degree": "I",
        "cierre_variant_beats": [4, 5, 6],  # variante documentada
        "typical_degrees": ["I", "bII", "bVI"],   # caídas más usuales (mi, fa, do)
        "secondary_variant_degrees": ["IV", "V", "III"],
        "chords": {"I": "Mi", "bII": "Fa", "bVI": "Do", "IVm_relative": "Lam"},
        "secondary_dominants": ["Do7", "Sol7", "Mi7"],
        "notes": "Llamada sobre grados I-II, transición en pulso 3 con 7a de "
                 "dominante (FM7/F). Relativo mayor frecuente hacia mitad del cante.",
    },

    "bulería": {
        "cycle_length": 12,
        "count_style": "2(=12) 1 2 3 4 5 6 7 8 9 10 11 -- acento en 6 a veces desplazado a 7",
        "accents": [12, 3, 6, 8, 10],
        "perceptual_downbeat": 3,  # misma convención de la familia de la soleá
        "mode": "phrygian_or_minor_or_major",  # variable según el estilo, pero
        "mode_predominant": "phrygian",        # el frigio flamenco es el más
                                                # frecuente en la práctica (dato
                                                # de campo confirmado por Laura)
        "tension_beat": 3,
        "tension_degree": "bII",           # V si el cante está en modo menor
        "tension_degree_minor_mode": "V",
        "resolution_beat": 10,
        "resolution_degree": "I",
        "cierre_beats": [10],
        "cierre_variant": "antes o después del 10; en el 4 si hay medio compás",
        "typical_degrees": ["I", "bII", "V"],
        "notes": "Compás más rico/variable que la soleá: puede alternar con "
                 "compás ternario heredado del jaleo. Tempo notablemente más "
                 "rápido -- el discriminador más fuerte hallado hasta ahora "
                 "(tempo_bpm) probablemente refleja esto.",
    },

    "alegrías": {
        "cycle_length": 12,
        "count_style": "anacrúsico, igual numeración que la soleá",
        "accents": [3, 6, 8, 10, 12],
        "perceptual_downbeat": 3,  # misma convención de la familia de la soleá
        "mode": "major",                # tonal, no modal -- clave frente a soleá
        "tension_beat": 6,               # dominante secundaria
        "tension_degree": "V/x",
        "tension_beat_2": 8,
        "resolution_beat": 10,
        "resolution_degree": "I",
        "cierre_beats": [10, 11, 12],
        "cierre_degree": "I",
        "cierre_variant_degrees": ["IV", "V"],  # variantes documentadas
        "llamada_A_degrees": ["I", "V"],
        "llamada_A_pulse_split": {"I": 5, "V": 7},  # reparto asimétrico de pulsos
        "llamada_cante_beats": [7, 8, 9, 10],        # deja 11-12 para respirar
        "typical_degrees": ["I", "IV", "V"],
        "notes": "Mismo compás de 12 que la soleá pero sistema TONAL (mayor), "
                 "no modal. Esta es la diferencia clave a nivel de features: "
                 "misma posición rítmica, distinta función armónica.",
    },

    "seguiriya": {
        "cycle_length": 12,
        "count_style": "inversión rítmica de la soleá empezando en el pulso 6 "
                        "(3/4 - 6/8 en vez de 6/8 - 3/4); base de 5 pulsos "
                        "(3 negras + 2 negras con puntillo)",
        "accents": [8, 10, 12, 3, 6],  # mismo conjunto de acentos que soleá/
                                       # bulería/alegrías pero en otro orden de
                                       # entrada al ciclo (confirmado por Laura)
        "perceptual_downbeat": 8,      # el cante/guitarra entra percibiendo el 8
                                       # como primer tiempo fuerte del compás y
                                       # de la frase -- clave para alinear el
                                       # beat-tracking, no es solo un reordenado
                                       # arbitrario de la lista de acentos
        "mode": "phrygian",   # cabal = variante en modo mayor
        "characteristic_bass_progression": ["D", "C", "Bb", "A"],
        "cierre_start": "segunda mitad del compás de 6/8",
        "cierre_end": "primer tiempo del compás de 3/4",
        "cierre_chord": "I (arpegio de tónica)",
        "suspensive_pattern_chord": "Gm (sol-sib-re-sib-re)",
        "suspensive_pattern_position": "primera mitad del 6/8, tras las falsetas",
        "family_related": ["liviana", "serrana", "cabal (modo mayor)"],
        "notes": "Estructuralmente la más distinta de las 4 -- no comparte "
                 "el patrón de acentos 3-6-8-10-12. Es la que más debería "
                 "diferenciarse ya solo por posición del cierre y por la "
                 "progresión de bajo D-C-Bb-A.",
    },
}


def to_dataframe(templates: dict):
    """Aplana el diccionario a formato 'largo' (palo, atributo, valor) para CSV,
    porque los campos son heterogéneos (listas, dicts, texto) y un CSV ancho
    perdería esa estructura o la forzaría de forma artificial."""
    import pandas as pd

    rows = []
    for palo, attrs in templates.items():
        for key, value in attrs.items():
            rows.append({"palo": palo, "atributo": key, "valor": str(value)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import pathlib
    df = to_dataframe(PALO_TEMPLATES)
    out = pathlib.Path(__file__).parent / "palo_templates_long.csv"
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
