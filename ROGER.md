如果完全抛开前面的设想，只看 ROGER 对当前三专家架构最小风险、最大收益的改进，我会选这三个，按优先级排序。

**1. 给最终 scorer 加 ListNet/KL 蒸馏损失**
这是最像 ROGER、也最小侵入的改法。

不改模型结构，只在原 loss 上加一项：

```text
L = L_original + γ * L_rank_distill
```

其中：

```text
student = TeamLoRA final candidate scores
teacher = XGBoost alpha=0.03 final scores
```

用每个 group 内 Top100 候选做 softmax 分布：

```text
teacher_dist = softmax(teacher_score / T)
student_log_dist = log_softmax(model_score / T)
L_rank_distill = CE(teacher_dist, student_log_dist)
```

推荐初始参数：

```text
γ = 0.1
T = 2.0
```

收益点：让 TeamLoRA 保留 XGBoost 已经有效的候选排序结构，减少训练时只盯正样本导致的排序波动。  
风险低：不改数据流、不改推理、不改三专家结构。

**2. 用 teacher score 做 hard negative weighting**
当前 hard negatives 主要按 rank 取。可以把 XGBoost teacher 分数高但不是 target 的候选作为更强 hard negative。

训练负样本采样改为：

```text
优先采样 teacher_score 高的负样本
```

或者 loss 加权：

```text
negative_weight_i = 1 + λ * normalized_teacher_score_i
```

收益点：模型更关注“看起来很像正样本”的负例，而不是浪费在容易负例上。  
风险低：只改采样/权重，不改架构。

**3. 保留 graph prior，限制 residual 幅度**
ROGER 的经验是不能完全丢掉原始目标，`γ=1` 会变差。对应我们这里就是：不要让 TeamLoRA 完全推翻 GraphRAG/XGBoost 排序。

当前你已经在用：

```text
use_graph_prior=true
residual_bound_mode=tanh
residual_bound_value=0.3
residual_l2=0.01
```

这方向是对的。后续可以固定，不建议放开 residual。

最稳配置：

```text
graph_prior = rank_log
residual_bound_mode = tanh
residual_bound_value = 0.3
residual_l2 = 0.01
```

**我不会优先做的**
暂时不建议先做这些：

```text
user × poi lookup 大矩阵
候选池外全局解码
复杂纠错因子
多阶段再解码
专家结构大改
```

这些潜在收益有，但风险和解释成本高，而且会干扰当前主线判断。

**最推荐的下一组实验**
等当前 baseline 出来后，直接做：

```text
XGBTop100 baseline:
原 TeamLoRA loss

XGBTop100 + ROGER-style:
原 TeamLoRA loss + ListNet/KL distill loss
γ=0.1, T=2.0
```

只改一个因素，最干净。  
如果它提升 Top1/Top5/MRR，再考虑专家级 distill 或 hard negative weighting。