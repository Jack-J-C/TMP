# experiment3 落地计划

本文档给出基于 experiment2 的可执行改进计划，供实现前审查。

## 1. 核心目标

experiment3 的目标不是替换候选池，也不是新增 POI 编码塔，而是：

```text
固定候选池不变
        ↓
匿名三专家 Llama 只编码用户轨迹
        ↓
得到 user mobility embedding
        ↓
轻量对齐模块把该表示对齐到候选池结构化锚点
        ↓
在固定 Top100 上评估排序效果
```

关键约束：

- 保留匿名三专家 routed TeamLoRA。
- LLM 输入不包含 candidate 级长文本。
- 不再把 `POI_HYPOTHESIS` 放入主输入。
- 对齐模块中也不直接使用长文本 `POI_HYPOTHESIS`。
- `GRAPH_CONTEXT` 不进入 LLM 主输入；如需使用，只保留为对齐模块侧的结构化锚点。
- 不引入独立 POI graph encoder。
- 不改 GraphRAG / learned Top100 候选池构建逻辑。
- RAAT、hard negatives 暂时不作为第一阶段主改动，后续通过超参数或消融单独确定。

## 2. 与 experiment2 的继承关系

沿用 experiment2 已经稳定的部分：

```text
retrieval_assets_clsprec/NYC/joined_poi_classification_exp2_top500_rescored/
  train_joined_top100.parquet
  val_joined_top100.parquet
  test_joined_top100.parquet
```

候选池仍采用：

```text
GraphRAG Top500 -> coverage-only linear scorer -> learned Top100
```

因此 candidate Hit@100 不作为 experiment3 的变量。experiment3 主要观察：

- 同一固定 Top100 内，轨迹编码 + 对齐模块能否产生有效排序；
- 匿名三专家是否学到稳定的轨迹表示；
- 轻量对齐模块是否必要。

## 3. 输入设计

### 3.1 LLM 轨迹编码输入

LLM 输入只描述用户侧信息：

```text
[TASK=MOBILITY_ENCODING]

[RECENT_CONTEXT]
compact raw trajectory / transition / nearby / temporal context

[USER_SEMANTIC_PROFILE]
Long-term category affinity
Revisit affinity
Geo routine
Temporal routine

[SIMILAR_USER_SEMANTIC_PROFILE]
仅当用户自身画像不足且相似画像有信号时注入

[MOBILITY_EMBED]
```

不进入 LLM 主输入：

```text
POI_HYPOTHESIS
candidate_id
candidate semantic_id
candidate category
candidate geo_cell
candidate graph rank text
```

这样可以避免模型滑回 candidate-wise reranker。

### 3.2 候选池结构化锚点

对齐模块使用固定 Top100 候选池中的结构化字段构建 candidate anchor，不使用长文本 `POI_HYPOTHESIS`。

候选 anchor 只保留离散/数值特征，不作为自然语言文本输入：

```text
categorical ids:
  category_id
  geo_cell_id
  semantic_cluster_id / local_index bucket

numeric / binary features:
  rank_invlog
  graph_score_log
  rank_norm
  source_count
  source flags
  train target popularity bucket
```

这些 anchor 可以由现有 parquet / semantic map / candidate details 直接构建，不需要新增 POI encoder，也不需要保留候选文本描述。

## 4. 模型设计

### 4.1 匿名三专家轨迹编码器

沿用当前 routed TeamLoRA 注入方式：

```text
Llama-3.2-1B-Instruct frozen
q/k/v/o/gate/up/down -> RoutedTeamLoRALinear
lora_num=3
expert_mode=anonymous
```

编码方式：

```text
LLM hidden at [MOBILITY_EMBED] or final token
        ↓
projection head
        ↓
user mobility embedding u
```

推荐 projection head：

```text
Linear(hidden_size -> 512)
GELU
Dropout
Linear(512 -> 128)
LayerNorm
```

第一阶段可以先用 final token hidden，避免新增特殊 token 带来的 tokenizer resize 和 embedding 初始化问题。若 final-token 效果不稳定，再考虑加入 `[MOBILITY_EMBED]` 特殊 token。

### 4.2 轻量对齐模块

候选 anchor 是结构化向量 `a_i`，不是 POI 文本编码。

建议从简单到复杂：

1. `linear dot`

```text
score_i = dot(W_u u, W_a a_i)
```

2. `bilinear`

```text
score_i = u^T W a_i
```

3. `low-rank bilinear`

```text
score_i = dot(Uu, Va_i)
```

第一版建议使用 `low-rank bilinear`，原因是它比单纯 linear 更有表达力，但仍然足够轻，不会变成 POI encoder。

## 5. 训练目标

第一阶段只做两个版本：

### 5.1 Baseline: 匿名三专家轨迹编码 + 固定线性打分

目的：验证轨迹 embedding 本身是否有监督信号。

训练：

```text
Top100 listwise cross entropy
positive = target POI in fixed Top100
negative = 同一 Top100 中其他候选
```

只训练：

```text
LoRA parameters
projection head
minimal scoring head
```

### 5.2 + Lightweight Alignment: 轨迹编码 + 轻量对齐模块

目的：验证轻量对齐是否能把 user mobility embedding 拉向候选池结构化语义。

训练：

```text
Top100 listwise cross entropy
+ optional embedding norm regularization
+ optional anchor alignment temperature
```

暂不加入：

```text
RAAT
hard negative extra loss
source dropout
rank noise
```

这些留到后续超参数/消融实验，避免第一阶段变量过多。

## 6. 评估设计

主评估仍然使用全量 test：

```text
top5
top10
ndcg5
ndcg10
mrr
candidate_hit@100
evaluated
missed
```

必须对比：

| 方法 | 说明 |
|---|---|
| GraphRAG Top100 order | 原始候选排序 |
| Learned Pool order | experiment2 learned Top100 原始排序 |
| experiment2 TeamLoRA reranker 800 | 当前最强 reranker 参考 |
| experiment3 anonymous trajectory encoder baseline | 新基线 |
| experiment3 + lightweight alignment | 主实验 |

如果 experiment3 的最终排序弱于 experiment2 reranker，但 mobility embedding 在分桶分析中更稳定，也可以作为后续方法模块，而不是直接替代 reranker。

## 7. 需要新增或修改的文件

建议新增，不覆盖 experiment2 脚本：

```text
src/poi_reranker/train_anonymous_mobility_encoder.py
src/poi_reranker/evaluate_anonymous_mobility_encoder.py
config/train_nyc_exp3_mobility_encoder_baseline.yaml
config/train_nyc_exp3_mobility_encoder_align.yaml
```

可选新增：

```text
src/poi_reranker/build_mobility_encoder_groups.py
reports/experiment3_anchor_feature_audit.json
reports/experiment3_expert_route_analysis.json
```

不建议修改：

```text
src/poi_reranker/train_teamlora_reranker_raat.py
src/data_build/build_joined_poi_classification_data.py
```

除非发现当前 parquet 缺少必要结构化字段。

## 8. 实现里程碑

### Milestone 0: 数据检查

检查当前 joined parquet 是否已经包含：

- target POI；
- Top100 candidates；
- candidate semantic id；
- candidate category；
- candidate geo cell；
- graph_features；
- source flags；
- user semantic profile；
- similar user semantic profile。

产出：

```text
reports/experiment3_anchor_feature_audit.json
```

### Milestone 1: 输入模板

新增 mobility encoder 输入模板：

```text
semantic_profile_simuser_mobility_v1
```

要求：

- 不包含 `POI_HYPOTHESIS`；
- 不包含 candidate 逐条描述；
- 只保留用户侧轨迹、画像、相似用户画像、轨迹侧图上下文。

### Milestone 2: 轨迹编码 baseline

实现：

```text
train_anonymous_mobility_encoder.py
```

先跑 NYC：

```text
max_steps=800
max_length=1024 or 1280
lora_num=3
lora_r=8
candidate_pool=learned Top100
```

输出：

```text
models/poi-mobility-encoder-nyc-exp3-anon3-baseline-...
```

### Milestone 3: 轻量对齐模块

加入 low-rank bilinear alignment head。

只比较：

- baseline scoring head；
- low-rank alignment head。

不要同时加入 RAAT / hard negatives，先保持变量干净。

### Milestone 4: 全量 test 对比

跑 test full Top100，生成：

```text
eval_best_test_full_top100.json
```

和 experiment2 的 800 steps reranker 同表比较。

### Milestone 5: 专家路由分析

统计：

- expert 使用频率；
- route entropy；
- tail / mid / head 分桶；
- profile insufficient 分桶；
- target rank bucket 分桶。

这个分析用于判断匿名三专家是否真的有功能分工。

## 9. 当前需要审查的决策点

实现前建议确认下面四点：

1. 第一版 trajectory embedding 使用 final-token hidden，还是新增 `[MOBILITY_EMBED]` 特殊 token？
2. candidate anchor 是否只用结构化字段，完全不使用 `POI_HYPOTHESIS`？
3. 第一阶段 max_length 设为 1024 还是 1280？
4. 轻量对齐模块第一版使用 `linear dot` 还是 `low-rank bilinear`？

我的建议：

```text
final-token hidden
不使用 POI_HYPOTHESIS
max_length=1280
low-rank bilinear
```
