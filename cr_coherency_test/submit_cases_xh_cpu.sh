#!/bin/bash
#SBATCH --job-name=cr_coherency
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --partition=xhacnormalb
#SBATCH --cpus-per-task=128

set -eo pipefail

source /public/software/apps/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate opendis-gpu
module load nvidia/cuda/12.1
module unload compiler/devtoolset/7.3.1 2>/dev/null || true
module load compiler/gcc/12.2.0

set -u

project_root=/work/home/cjy306/openDIS
case_dir=$project_root/cr_coherency_test

export PYTHONPATH="$project_root/core/exadis/python:$project_root/core/pydis/python:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$project_root/build/core/exadis/kokkos/core/src:$project_root/build/core/exadis/src:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OMP_PROC_BIND=spread
export OMP_PLACES=threads

if [[ "${TASK_KIND:-}" == "prepare" ]]; then
    echo "=== 编译/验证/共享初态准备开始：$(date)；作业 $SLURM_JOB_ID ==="

    echo ">>> [1/4] 增量编译 pyexadis"
    cmake --build "$project_root/build" --target pyexadis -j "$SLURM_CPUS_PER_TASK"

    echo ">>> [2/4] 验证周期相干应力场"
    cd "$project_root"
    python -m unittest cr_coherency_test.test_coherency_field_unit -v

    echo ">>> [3/4] 验证 C++ 相干应力力模型"
    cd "$project_root/core/exadis/tests/unit_tests"
    python test_force.py coherency_zero
    python test_force.py coherency_uniform
    python test_force.py coherency_periodic
    python test_force.py coherency_sign

    echo ">>> [4/4] 生成两组共用的位错环与相干应力场"
    cd "$case_dir"
    python generate_coherency_case.py
    test -s init_coherency/init_config.data
    test -s init_coherency/coherency_stress.npz
    test -s init_coherency/metadata.json

    echo "=== 编译/验证/共享初态准备成功结束：$(date) ==="
    exit 0
fi

if [[ "${TASK_KIND:-}" != "case" ]]; then
    echo "错误：TASK_KIND 必须为 prepare 或 case。" >&2
    exit 2
fi
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "错误：case 模式必须通过 --array=0-1 提交。" >&2
    exit 2
fi

case_names=(no_coherency with_coherency)
test_scripts=(test_no_coherency.py test_with_coherency.py)
output_dirs=(output_no_coherency output_with_coherency)

case_name=${case_names[$SLURM_ARRAY_TASK_ID]}
test_script=${test_scripts[$SLURM_ARRAY_TASK_ID]}
output_dir=${output_dirs[$SLURM_ARRAY_TASK_ID]}

cd "$case_dir"
test -s init_coherency/init_config.data
test -s init_coherency/coherency_stress.npz

echo "=== ${case_name} 开始：$(date)；作业 ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} ==="
python "$test_script"
test -s "$output_dir/stress_strain_dens.dat"
echo "=== ${case_name} 成功结束：$(date) ==="
