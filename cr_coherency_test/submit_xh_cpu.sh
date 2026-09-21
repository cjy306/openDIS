#!/bin/bash
# 雄衡登录节点总控脚本。用法：bash submit_xh_cpu.sh

set -euo pipefail

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "错误：请在雄衡登录节点运行 bash submit_xh_cpu.sh，不要使用 sbatch submit_xh_cpu.sh。" >&2
    exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$script_dir"

if ! command -v sbatch >/dev/null 2>&1; then
    echo "错误：当前节点找不到 sbatch，请在雄衡登录节点运行。" >&2
    exit 1
fi

for required in submit_cases_xh_cpu.sh submit_post_xh_cpu.sh; do
    if [[ ! -f "$required" ]]; then
        echo "缺少文件：$script_dir/$required" >&2
        exit 1
    fi
done

bash -n submit_cases_xh_cpu.sh submit_post_xh_cpu.sh

prepare_submission=$(sbatch --parsable \
    --export=ALL,TASK_KIND=prepare \
    --output=slurm-coherency-prepare-%j.out \
    --error=slurm-coherency-prepare-%j.err \
    submit_cases_xh_cpu.sh)
prepare_job_id=${prepare_submission%%;*}
echo "编译/验证/生成共享初态作业已提交：$prepare_job_id"

cases_submission=$(sbatch --parsable \
    --dependency="afterok:${prepare_job_id}" \
    --array=0-1 \
    --export=ALL,TASK_KIND=case \
    --output=slurm-coherency-%A_%a.out \
    --error=slurm-coherency-%A_%a.err \
    submit_cases_xh_cpu.sh)
cases_job_id=${cases_submission%%;*}
echo "无/有相干应力数组作业已提交：$cases_job_id"

post_submission=$(sbatch --parsable \
    --dependency="afterok:${cases_job_id}" \
    submit_post_xh_cpu.sh)
post_job_id=${post_submission%%;*}
echo "plot/VTK 后处理作业已提交：$post_job_id"

echo "依赖链：${prepare_job_id} -> ${cases_job_id} -> ${post_job_id}"
echo "查看队列：squeue -j ${prepare_job_id},${cases_job_id},${post_job_id}"
