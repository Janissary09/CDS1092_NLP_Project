#!/usr/bin/env bash
#SBATCH --job-name=phi4-standard-zero-shot
#SBATCH --partition=students
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --time=15:00:00
#SBATCH --signal=B:TERM@120

# For smoke, override the default with sbatch --time=00:15:00.
set -Eeuo pipefail
case "$*" in
    smoke) args=(--smoke-test) ;;
    "umwp run_01"|"sum run_01") args=(--dataset "$1" --run-id "$2") ;;
    "umwp run_01 --allow-failed-smoke"|"sum run_01 --allow-failed-smoke")
        args=(--dataset "$1" --run-id "$2" --allow-failed-smoke) ;;
    *)
        echo "Usage: sbatch --time=00:15:00 slurm/evaluate_phi4_standard.sh smoke" >&2
        echo "       sbatch slurm/evaluate_phi4_standard.sh {umwp|sum} run_01 [--allow-failed-smoke]" >&2
        exit 2
        ;;
esac

cd "${SLURM_SUBMIT_DIR:?Submit with sbatch from the project root}"
if [[ ! -f src/31_evaluate_phi4_standard.py ]]; then
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
exec python -u src/31_evaluate_phi4_standard.py "${args[@]}"
