#!/usr/bin/env bash
# Sweep fin de conformity_bias SANS archivage (--no_archive), pour les 3 distributions
# initiales disponibles (power_law / uniform / random) -- suite des sections 10-13 du
# notebook (code/archive_rate_conformity_exploration.ipynb) et de test_conformity_no_archive.py,
# via le CLI de code/simulation.py (--sweep_param conformity_bias, corrigé pour être pris en
# compte -- cf. section 13/14).
#
# Grille : 0.85 à 1.05 par pas de 0.01 (21 valeurs). Sans archivage, la population finale
# (t_max=1000) est déjà largement effondrée à conformity_bias=1.0 (neutre, dérive pure) et
# totalement fixée dès ~1.05 (cf. section 13) -- la zone intéressante est donc plus basse et
# plus large que pour la version archivée (section 4-5, 0.84-1.00 puis zoom 0.98-1.00).
# Ajuster CB_VALUES si le résultat suggère de resserrer/élargir la fenêtre.
#
# Usage :
#   bash code/run_conformity_no_archive_fine_sweep.sh

set -e

cd "$(dirname "$0")/.."   # racine du repo (CultureLab_article_repo/), quel que soit le cwd d'appel

PY="/Users/rolly/Documents/10-19_Université_et_scolarité/PhD/phd-env/bin/python"

N_CLASSES=176
N_INIT=4400
N_FINAL=8800
GENERATIONS=1000
N_REPEATS=8

CB_VALUES="0.85 0.86 0.87 0.88 0.89 0.90 0.91 0.92 0.93 0.94 0.95 0.96 0.97 0.98 0.99 1.00 1.01 1.02 1.03 1.04 1.05"

mkdir -p outputs

for DIST in power_law uniform random; do
    echo "=================================================================="
    echo "Sweep conformity_bias (fin, sans archivage) -- distribution=${DIST}"
    echo "=================================================================="
    $PY code/simulation.py \
        -N $N_CLASSES -ni $N_INIT -nf $N_FINAL -T $GENERATIONS \
        --distribution "$DIST" --no_archive --compare_empirical \
        --sweep_param conformity_bias --sweep_values $CB_VALUES --sweep_repeats $N_REPEATS \
        -o outputs/ --plot "outputs/conformity_no_archive_fine_${DIST}.png"
    # --sweep_param écrit outputs/sweep_conformity_bias.csv (+ _raw.csv) à chaque itération de
    # la boucle -- renommé tout de suite pour ne pas écraser la distribution suivante.
    mv outputs/sweep_conformity_bias.csv "outputs/conformity_no_archive_fine_${DIST}.csv"
    mv outputs/sweep_conformity_bias_raw.csv "outputs/conformity_no_archive_fine_${DIST}_raw.csv"
    echo
done

echo "Terminé. Fichiers produits dans outputs/ :"
echo "  conformity_no_archive_fine_{power_law,uniform,random}.png"
echo "  conformity_no_archive_fine_{power_law,uniform,random}.csv (+ _raw.csv)"
