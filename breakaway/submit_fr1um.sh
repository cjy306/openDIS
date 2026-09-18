#!/bin/bash
# 登录节点总控脚本。用法：bash submit_fr1um.sh

set -euo pipefail

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "错误：请在登录节点运行 bash submit_fr1um.sh，不要使用 sbatch submit_fr1um.sh。" >&2
    exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$script_dir"

if ! command -v sbatch >/dev/null 2>&1; then
    echo "错误：当前节点找不到 sbatch，请在昆山登录节点运行。" >&2
    exit 1
fi

for required in submit_fr1um_cases_kun.sh submit_fr1um_post_kun.sh; do
    if [[ ! -f "$required" ]]; then
        echo "缺少文件：$script_dir/$required" >&2
        exit 1
    fi
done

bash -n submit_fr1um_cases_kun.sh submit_fr1um_post_kun.sh

cases_submission=$(sbatch --parsable submit_fr1um_cases_kun.sh)
cases_job_id=${cases_submission%%;*}
echo "无障碍/点障碍数组作业已提交：$cases_job_id"

post_submission=$(sbatch --parsable \
    --dependency="afterok:${cases_job_id}" \
    submit_fr1um_post_kun.sh)
post_job_id=${post_submission%%;*}
echo "曲线/VTK后处理作业已提交：$post_job_id"
echo "依赖关系：afterok:${cases_job_id}"
echo "查看队列：squeue -j ${cases_job_id},${post_job_id}"
