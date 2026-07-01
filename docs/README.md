# TMP 当前流程

TMP 项目当前主线已从旧 LLMMRA 的 `LoRA-A + 8B LoRA-B + Selector` 切换为 **1B candidate-wise POI reranker**。

旧 8B 路线、selector 路线和 full-POI classifier 路线只保留为历史对照，不再作为当前主线。

## 当前主线

```text
原始 trajectory / history / geo / time
        ↓
LoRA-A-v2 离线生成 refined_prompt
        ↓
Semantic-ID GraphRAG 构建 Top100 candidates
        ↓
拆成 query-candidate pair / listwise group
        ↓
Llama-3.2-1B candidate-wise reranker
        ↓
Top1 / Top5 / Top10 / Top20 / MRR
```

目标不是让 1B 模型生成 POI 文本，而是让 1B 模型对 GraphRAG Top100 候选进行重排。

## 当前判断

`full-POI classifier + GraphRAG prior` 主架构效果不理想。它一次 forward 输出全 POI vocab logits，再把 GraphRAG rank/score 加成 prior。这种形式存在两个问题：

- 1B LoRA 很难稳定学习全 POI 类别空间。
- GraphRAG 候选没有被逐候选语义比较，只是被作为 logits prior 使用。

因此后续不继续围绕该主架构堆 RAAT 或调参。

`2-view RAAT` 的定位是增强模块：

```text
original + target-mask
loss = max(loss_original, loss_target_mask)
```

它可以作为 baseline 或后续 candidate-wise reranker 的鲁棒训练模块，但不能替代主架构切换。

## 关键数据

GraphRAG Top100 joined 数据：

```text
retrieval_assets/NewYork/joined_poi_classification/train_joined_top100.parquet
retrieval_assets/NewYork/joined_poi_classification/val_joined_top100.parquet
```

GraphRAG 原始候选：

```text
retrieval_assets/NewYork/double_llm/graphrag_semantic_edges_v2_top100_train_candidates.jsonl
retrieval_assets/NewYork/double_llm/graphrag_semantic_edges_v2_top100_val_candidates.jsonl
```

Semantic-ID 映射：

```text
retrieval_assets/NewYork/double_llm/semantic_poi_ids.jsonl
```

LoRA-A-v2 refined prompt：

```text
retrieval_assets/NewYork/refined_prompts_decision/
models/prompt-refiner-lora-llama32-1b-decision-v2-final/
```

## 下一步

已开始实现：

```text
src/poi_reranker/build_teamlora_reranker_groups.py
src/poi_reranker/train_teamlora_reranker_raat.py
```

第一版直接做匿名三专家 TeamLoRA + 2-view RAAT，不再把单专家 baseline 作为主线，也不显式规定每个专家必须学习哪类证据：

```text
same query-candidate input
        ↓
Shared Llama-3.2-1B + 3 anonymous LoRA experts
        ↓
Gate fusion + candidate graph features
        ↓
scalar relevance score
```

默认模式：

- 三个 LoRA expert 都接收同一个 query-candidate 输入。
- gate 自动学习专家组合，不手工指定专家语义。
- `--expert-mode named` 仅作为可选消融，会把输入拆成 pref/graph/refine 三个 view。
- 训练脚本可直接读取 `joined_top100.parquet`，不强制预先生成新 JSONL 数据集。

训练时默认只使用 `target_in_candidates=true` 的 train groups，因为候选集未命中时没有正候选，无法做 listwise CE。验证仍保留全量样本，并报告 `candidate_hit` 作为上限。

评估指标默认使用全量 val 口径：

- `top1/top5/top10/top20/mrr`: 全量指标，候选池未命中样本按错误计入。
- `conditional_top1/conditional_top5/conditional_top10/conditional_top20/conditional_mrr`: 只在 `target_in_candidates=true` 子集上计算。
- `candidate_hit`: GraphRAG TopK 召回上限。

训练增强支持：

- `--raat-mode target_mask_2view`: clean + target graph mask。
- `--raat-mode graph_3view`: clean + target-demotion + hard-negative-promotion，loss 取三者最大值。
