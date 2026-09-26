#!/bin/bash -l
# ============================================================================
# Job array SGE (Myriad, UCL) pour simulation.py.
# Chaque tâche de l'array lit SA ligne (indexée par $SGE_TASK_ID) dans
# params.csv (généré par build_param_grid.py) et lance un run de
# simulation.py avec ces paramètres -> écrit son propre fichier JSON (-o).
# Aucun fichier partagé entre tâches : pas de risque de collision d'écriture.
#
# Préparation avant soumission (depuis code/hpc/) :
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
# simulation.py vit dans code/, un niveau au-dessus de code/hpc/ (où est ce script)
SIM_SCRIPT="$(dirname "$0")/../simulation.py"

# Ligne 1 = header -> la tâche $SGE_TASK_ID lit la ligne $SGE_TASK_ID+1
# Colonnes de params.csv (build_param_grid.py) : n_classes,initial_pop,final_pop,generations,
# archive_rate,conformity_bias,archive,distribution,distribution_path,seed -- si tu régénères
# params.csv avec un ordre différent, adapte le `read` ci-dessous en conséquence.
LINE=$(sed -n "$((SGE_TASK_ID + 1))p" "$PARAMS_FILE")
IFS=',' read -r N_CLASSES INITIAL_POP FINAL_POP GENERATIONS ARCHIVE_RATE CONFORMITY_BIAS ARCHIVE DISTRIBUTION DISTRIBUTION_PATH SEED <<< "$LINE"

echo "Task $SGE_TASK_ID : n_classes=$N_CLASSES archive_rate=$ARCHIVE_RATE conformity_bias=$CONFORMITY_BIAS archive=$ARCHIVE distribution=$DISTRIBUTION seed=$SEED"

# archive=false -> --no_archive ; archive=true (ou colonne absente/vide) -> comportement
# historique inchangé (archivage cumulatif, cf. simulation.run_simulation).
NO_ARCHIVE_FLAG=""
if [ "$ARCHIVE" = "false" ] || [ "$ARCHIVE" = "False" ]; then
    NO_ARCHIVE_FLAG="--no_archive"
fi

DISTRIBUTION_PATH_FLAG=""
if [ -n "$DISTRIBUTION_PATH" ]; then
    DISTRIBUTION_PATH_FLAG="--distribution_path $DISTRIBUTION_PATH"
fi

python "$SIM_SCRIPT" \
    -N "$N_CLASSES" -ni "$INITIAL_POP" -nf "$FINAL_POP" \
    -T "$GENERATIONS" -alpha "$ARCHIVE_RATE" -q "$CONFORMITY_BIAS" -s "$SEED" \
    --distribution "${DISTRIBUTION:-power_law}" $DISTRIBUTION_PATH_FLAG $NO_ARCHIVE_FLAG \
    -o "$OUTPUT_DIR"
