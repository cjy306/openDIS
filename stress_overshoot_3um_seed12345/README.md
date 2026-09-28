# 3 μm FR 版本，seed=12345

由 stress_overshoot_5runs/run_01 派生，独立目录，不覆盖原始五组。

- FCC Cu，5×5×5 μm 三向周期盒，43 条两端钉扎直线 FR，每条长 3 μm。
- 总线长 129 μm，实际密度 1.032e12 m⁻²，比原目标 1.024e12 高 0.78125%。
- 高 Schmid 8 系各 4 条，共 32 条；零 Schmid 系 ID 2、5、8、11 分别 3、3、3、2 条。高 Schmid 比例 74.4186%，为已确认的 A75 整数近似。
- seed=12345；位置均匀随机，线方向角 theta 均匀分布于 [0°,180°)，不是固定刃/螺比例。
- 源数量改变会改变滑移系随机分配；相同 seed 不意味着与原 128 源网络逐源对应。此对照同时改变臂长、源数量，并有小幅密度和滑移系比例差异。
- 加载及数值参数与原 run_01 相同：[001]、1000 s⁻¹、终止总应变 0.005（0.5%）、cross_slip=None、write_freq=1000。
- 保持 maxseg=200b，因此每条源有 60 个节点、59 段；43 条共 2580 节点、2537 段。

## 昆山运行

将整个目录上传至 `/public/home/cjy306/openDIS/stress_overshoot_3um_seed12345`，进入目录执行：

```bash
sbatch submit_kun.sh
```

沿用昆山 ksagnormal01、1 GPU、8 CPU、120 小时上限，依次执行 generate.py 和 test.py。
也可在已配置 pyexadis 的环境手动运行：

```bash
python generate.py
python test.py
```

初态输出至 `init_A75_3um_seed12345`，模拟输出至 `output_A75_3um_seed12345`。
初态 CSV 记录各源中心、滑移系和线方向角。

已验证 Python 语法、布局复现、各系数量、真实 FR 插入函数生成的面内线段、密度、钉扎端点、段长及与原 run_01 的加载参数一致性；提交脚本通过 bash -n。尚未运行 ExaDiS 或提交作业。
