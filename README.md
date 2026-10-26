# TrafficFlowBench Multi-Task Solution

**2026 IEEE Big Data - Traffic Flow Bench**: One Pipeline for All 4 Tasks.

## 1. 竞赛背景

本竞赛包含四个权重不同的任务，评测指标涵盖 RMSE、时空 IoU 以及物理一致性：

| Task | Weight | Predict | Scored by |
|---|---:|---|---|
| 1. State reconstruction | 35% | 隐藏路段的速度和流量 | RMSE |
| 2. Queue forecasting | 30% | T+5 至 T+30 的排队链路 | 时空 IoU |
| 3. Physics consistency | 15% | 无提交，使用 Task 1 结果 | 基本图 + 车辆守恒 |
| 4. OD estimation | 20% | 路径流量 | 匹配计数与 OD 结构 |

## 2. 方案概览

- **Task 1**: 3 x LightGBM + 2 x U-Net，按 70/30 融合输出速度与流量。物理投影（L-BFGS-B）基于车辆守恒约束微调流量，显著提升 Task 3 得分。
- **Task 2**: 
  - Ongoing 窗口：7 个 LightGBM 成员，概率阈值 0.40。
  - Onset 窗口：3 个 CNN 成员预测 T+30，概率阈值 0.50。
- **Task 3**: 将 U-Net 的损失函数中加入密度项（flow/speed），直接优化物理一致性。
- **Task 4**: 在每个 split 自身的 counts 和 prior 上求解 NNLS，使用 FISTA + L-BFGS-B 收敛。

## 3. 技术亮点

- **物理约束投影**：利用 `L-BFGS-B` 优化器，在有限时间内将预测流量投影到满足车辆守恒定律的流形上。
- **基于拆分的 OD 估计**：放弃使用训练集先验，转而基于当前 split 的数据求解 NNLS，单步带来 +0.031 提升。
- **模型家族特化**：CNN 在 onset 窗口上击败树模型，而在 ongoing 窗口上相反，按条件拆分开启了双赢。
- **严谨的消融实验**：见 `docs/experiment_log.md`，详细记录了 20 余次单步实验的正负收益。

## 4. 复现指南

```bash
pip install -r requirements.txt
# 运行物理投影脚本 (需要先准备好初始 submission.csv)
python -m src.main --data-root /path/to/data --submission submission.csv --output submission_physics.csv


5. 文件结构
text
src/
  task1_physics.py   # Task 1 物理约束投影与状态重建
  ensemble.py        # 多模型 rank 融合与目标级路由
  main.py            # CLI 入口
configs/
  default.yaml       # 模型与融合参数
notebooks/
  trafficflowbench-trees-u-net-cnn-lb-0-87.ipynb
docs/
  experiment_log.md
6. 我的贡献
独立实现 Task 1 的物理约束投影模块，通过 L-BFGS-B 优化流量残差，提高物理一致性。

设计了 Task 2 的条件分离架构，将 ongoing (LightGBM) 与 onset (CNN) 分别处理。

参与 Task 4 的 NNLS 求解调优，利用每个 split 的独立先验取得显著提升。
