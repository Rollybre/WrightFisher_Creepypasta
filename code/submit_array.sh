#!/bin/bash -l
# ============================================================================
# Job array SGE (Myriad, UCL) pour simulation.py.
# Chaque tâche de l'array lit SA ligne (indexée par $SGE_TASK_ID) dans
# params.csv (généré par build_param_grid.py) et lance un run de
# simulation.py avec ces paramètres -> écrit son propre fichier JSON (-o).
# Aucun fichier partagé entre tâches : pas de risque de collision d'écriture.
#
# Préparation avant soumission :
#   python build_param_grid.py --archive_rates 0.01 0.05 0.1 0.2 0.4 --n_seeds 5
#   qsub -t 1-25 submit_array.sh      # 25 = nb de lignes de params.csv (sans le header)
# ============================================================================

#$ -S /bin/bash
#$ -N sim_sweep

# ── À COMPLÉTER avec la doc UCL Myriad / le script de ton groupe ──────────
#$ -wd /home/<UCL_ID>/Scratch/sim_output   # dossier de travail (logs stdout/stderr), doit exister
#$ -l h_rt=0:30:0                          # temps max PAR TÂCHE (hh:mm:ss) — ajuster à la durée réelle d'un run
#$ -l mem=2G                               # mémoire PAR TÂCHE
# #$ -P <project_id>                       # décommenter + compléter si ton groupe a un code de facturation
# ────────────────────────────────────────────────────────────────────────────

# L'array est déclaré à la soumission (qsub -t 1-N), pas ici en dur, pour ne
# pas avoir à éditer ce fichier à chaque fois que la taille de la grille change.

# ── À VÉRIFIER : nom/version exacts du module Python sur Myriad ───────────
module load python3/3.11
# ou, si tu utilises un venv plutôt que les modules du cluster :
# source /chemin/vers/ton/venv/bin/activate
# ────────────────────────────────────────────────────────────────────────────

PARAMS_FILE="params.csv"
OUTPUT_DIR="results/"
CODE_DIR="$(dirname "$0")"

# Ligne 1 = header -> la tâche $SGE_TASK_ID lit la ligne $SGE_TASK_ID+1
# Colonnes de params.csv (build_param_grid.py) : n_classes,initial_pop,final_pop,generations,
# archive_rate,conformity_bias,seed -- si tu régénères params.csv avec un ordre différent,
# adapte le `read` ci-dessous en conséquence.
LINE=$(sed -n "$((SGE_TASK_ID + 1))p" "$PARAMS_FILE")
IFS=',' read -r N_CLASSES INITIAL_POP FINAL_POP GENERATIONS ARCHIVE_RATE CONFORMITY_BIAS SEED <<< "$LINE"

echo "Task $SGE_TASK_ID : n_classes=$N_CLASSES archive_rate=$ARCHIVE_RATE conformity_bias=$CONFORMITY_BIAS seed=$SEED"

python "$CODE_DIR/simulation.py" \
    -N "$N_CLASSES" -ni "$INITIAL_POP" -nf "$FINAL_POP" \
    -T "$GENERATIONS" -alpha "$ARCHIVE_RATE" -q "$CONFORMITY_BIAS" -s "$SEED" \
    -o "$OUTPUT_DIR"
