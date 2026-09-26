#!/usr/bin/env bash
# Construit params_5M.csv pour le batch ~5M (conformity_bias x archive_rate x generations, fins,
# sans répétition de seed -- archive=true partout, pas d'axe archive on/off).
#
# Axes :
#   conformity_bias : 0.80 -> 1.20, pas 0.002 (201 valeurs)
#   archive_rate    : 0.01 -> 1.00, pas 0.01  (100 valeurs)
#   generations     : 500  -> 1000, pas 2     (251 valeurs)
#   n_seeds=1 (une seule seed par combinaison, pas de répétitions)
# Total : 201 x 100 x 251 x 1 = 5 045 100 lignes.
#
# Usage (depuis code/hpc/, avec le venv du serveur) :
#   PY=/home/alex/CultureLab_article_repo/wf_simulation/bin/python3 bash build_grid_5M.sh

set -e

PY="${PY:-/Users/rolly/Documents/10-19_Université_et_scolarité/PhD/phd-env/bin/python}"
cd "$(dirname "$0")"   # code/hpc/, quel que soit le cwd d'appel

# Arrays bash (pas de découpage de mots ambigu selon le shell de connexion -- lancer ce script
# via `bash build_grid_5M.sh`, pas en le sourçant depuis zsh).
CB=()
for i in $(seq 0 200); do
    CB+=("$($PY -c "print(f'{0.80 + 0.002*$i:.3f}')")")
done

AR=()
for i in $(seq 0 99); do
    AR+=("$($PY -c "print(f'{0.01 + 0.01*$i:.2f}')")")
done

GEN=()
for i in $(seq 0 250); do
    GEN+=("$((500 + 2*i))")
done

# Pas d'indexation négative (${arr[-1]}) : bash 3.2 (macOS par défaut) ne la supporte pas,
# contrairement à bash 5.x (les-lillas) -- ce script doit rester exécutable des deux côtés.
echo "conformity_bias : ${#CB[@]} valeurs (${CB[0]} ... ${CB[$((${#CB[@]} - 1))]})"
echo "archive_rate    : ${#AR[@]} valeurs (${AR[0]} ... ${AR[$((${#AR[@]} - 1))]})"
echo "generations     : ${#GEN[@]} valeurs (${GEN[0]} ... ${GEN[$((${#GEN[@]} - 1))]})"

$PY build_param_grid.py -N 176 -ni 4400 -nf 8800 \
    --conformity_biases "${CB[@]}" \
    --archive_rates "${AR[@]}" \
    --generations_values "${GEN[@]}" \
    --distribution uniform \
    --n_seeds 1 \
    -o params_5M.csv
