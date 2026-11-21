#!/usr/bin/env bash
#SBATCH --job-name=cuda-transfer
#SBATCH --partition=students
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks=1
#SBATCH --time=00:05:00

set -Eeuo pipefail

cd "${SLURM_SUBMIT_DIR:?Submit with sbatch from the project root}"
if [[ ! -f src/diagnose_cuda_transfer.py ]]; then
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
exec python -u src/diagnose_cuda_transfer.py
