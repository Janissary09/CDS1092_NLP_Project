#!/usr/bin/env bash
#SBATCH --partition=students
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
#SBATCH --signal=B:TERM@60

set -Eeuo pipefail
if [[ $# -ne 5 ]]; then
    echo "Usage: train_encoder.sh ROOT FAMILY RUN STATUS_DIRECTORY CODE_SNAPSHOT" >&2
    exit 2
fi
project_root=$1
family=$2
run=$3
status_dir=$4
code_snapshot=$5
started_epoch=$(date +%s)
started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
finish() {
    code=$?
    trap - EXIT
    ended_epoch=$(date +%s)
    ended_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    elapsed=$((ended_epoch - started_epoch))
    printf 'END family=%s run=%s job=%s start=%s end=%s exit=%s runtime_seconds=%s\n' \
        "$family" "$run" "${SLURM_JOB_ID:-unknown}" "$started_utc" "$ended_utc" "$code" "$elapsed"
    # One exclusive record per SLURM job. No model/result directory is touched.
    if ! (set -o noclobber
        printf 'family\trun\tjob_id\tstart_utc\tend_utc\texit_status\truntime_seconds\n%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$family" "$run" "${SLURM_JOB_ID:-unknown}" "$started_utc" "$ended_utc" "$code" "$elapsed" \
            > "$status_dir/${family}_${run}_${SLURM_JOB_ID:-unknown}.tsv"); then
        echo "Could not write exclusive job status record." >&2
        if [[ $code -eq 0 ]]; then code=1; fi
    fi
    exit "$code"
}
trap finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
printf 'START family=%s run=%s job=%s time=%s host=%s\n' \
    "$family" "$run" "${SLURM_JOB_ID:-unknown}" "$started_utc" "$(hostname)"
cd "$project_root"

# Batch shells do not necessarily inherit interactive Conda initialization.
if [[ -n ${CONDA_EXE:-} && -x ${CONDA_EXE} ]]; then
    conda_executable=$CONDA_EXE
elif command -v conda >/dev/null 2>&1; then
    conda_executable=$(command -v conda)
else
    conda_executable=
    for candidate in "$HOME/miniconda3/bin/conda" "$HOME/anaconda3/bin/conda" "$HOME/miniforge3/bin/conda"; do
        if [[ -x $candidate ]]; then conda_executable=$candidate; break; fi
    done
    if [[ -z $conda_executable ]]; then
        echo "Conda not found. Export the real CONDA_EXE before submission." >&2
        exit 1
    fi
fi
eval "$("$conda_executable" shell.bash hook)"
conda activate cds1092
export PYTHONUNBUFFERED=1
export ACCELERATE_MIXED_PRECISION=no
export CDS1092_PROJECT_ROOT=$project_root
# Keep SLURM's CUDA_VISIBLE_DEVICES unchanged. srun forwards the job exit status.
srun --ntasks=1 python -u "$code_snapshot/src/22_train_encoder.py" --model "$family" --run "$run"
