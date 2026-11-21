#!/usr/bin/env bash
#SBATCH --job-name=llama-zero-shot
#SBATCH --partition=students
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks=1
#SBATCH --time=08:00:00
#SBATCH --signal=B:TERM@120

# Override the default wall time with sbatch --time=HH:MM:SS.
set -Eeuo pipefail
case "$*" in
    smoke) args=(--smoke-test) ;;
    diagnose-forward) args=(--diagnose-forward) ;;
    "umwp run_01"|"sum run_01") args=(--dataset "$1" --run-id "$2") ;;
    *)
        echo "Usage: sbatch slurm/evaluate_llama.sh smoke | diagnose-forward | {umwp|sum} run_01" >&2
        exit 2
        ;;
esac

cd "${SLURM_SUBMIT_DIR:?Submit with sbatch from the project root}"
if [[ ! -f src/29_evaluate_llama.py ]]; then
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
exec python -u src/29_evaluate_llama.py "${args[@]}"
