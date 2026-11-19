#!/usr/bin/env bash
#SBATCH --job-name=apertus-zero-shot
#SBATCH --partition=students
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --signal=B:TERM@120

# Set --time at submission: 03:00:00 for UMWP, 04:00:00 for SUM.
set -Eeuo pipefail
if [[ $# -ne 2 || ( $1 != umwp && $1 != sum ) ]]; then
    echo "Usage: sbatch --time=03:00:00 slurm/evaluate_apertus.sh umwp RUN_ID" >&2
    echo "       sbatch --time=04:00:00 slurm/evaluate_apertus.sh sum RUN_ID" >&2
    exit 2
fi

cd "${SLURM_SUBMIT_DIR:?Submit with sbatch from the project root}"
if [[ ! -f src/26_evaluate_apertus.py ]]; then
    echo "Submit from the CDS1092_NLP_Project root." >&2
    exit 2
fi

conda_executable=${CONDA_EXE:-}
if [[ ! -x "$conda_executable" ]]; then
    conda_executable=$(command -v conda || true)
fi
if [[ ! -x "$conda_executable" ]]; then
    for candidate in "$HOME/miniconda3/bin/conda" "$HOME/anaconda3/bin/conda" "$HOME/miniforge3/bin/conda"; do
        if [[ -x "$candidate" ]]; then
            conda_executable=$candidate
            break
        fi
    done
fi
if [[ ! -x "$conda_executable" ]]; then
    echo "Conda not found. Export CONDA_EXE before submission." >&2
    exit 1
fi
eval "$("$conda_executable" shell.bash hook)"
conda activate cds1092
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# exec lets SLURM's B:TERM reach the evaluator directly.
exec python -u src/26_evaluate_apertus.py --dataset "$1" --run-id "$2"
