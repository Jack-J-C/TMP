# TMP Current Mainline Plan

本文档只记录当前有效主线。旧结论如果被推翻，直接覆盖，不保留流水账。

## 结论更新

当前 `full-POI classifier + GraphRAG prior` 主架构效果不理想，后续不再把它作为主要优化方向。

`2-view RAAT`、candidate dropout、target mask、rank noise 等机制只能增强鲁棒性，不能修复主架构表达不对齐的问题。若主架构本身不能有效利用 `query-candidate` 语义，再继续叠加 RAAT 只会增加训练复杂度，收益有限。

因此当前主线切换为：

```text
Semantic-ID GraphRAG Top100
        ↓
candidate-wise query-candidate reranker
        ↓
Top1 / Top5 / Top10 / Top20 / MRR
```

旧 `train_teamlora_poi_classifier.py` 保留为 baseline 和消融，不作为下一版主线。

## 目标主架构

```text
raw trajectory / user history / geo / time
        ↓
LoRA-A-v2 offline refined_prompt
        ↓
Semantic-ID GraphRAG Top100 candidates
        ↓
拆成 query-candidate pair
        ↓
Shared Llama-3.2-1B Backbone
        ↓
3 anonymous LoRA experts
        ↓
Expert Fusion / Gate
        ↓
Candidate Scoring Head
        ↓
rank Top100 candidates
```

核心变化：

- 从 `一次 forward 输出全 POI logits` 改成 `每个候选单独打 relevance score`。
- 从 `GraphRAG prior 加到 full logits` 改成 `GraphRAG rank/score/source 作为 candidate feature`。
- 从单一路径改成三 LoRA expert + gate fusion，但默认不手工规定每个 expert 学什么。
- 评估只在当前 Top100 candidate set 内排序，指标为 `top1/top5/top10/top20/MRR`。

## 为什么要切换

当前 full-classifier 的主要问题：

- 类别空间是全 POI vocab，1B 模型和小样本 LoRA 很难稳定学习所有 POI 的全局分类边界。
- GraphRAG 的 Top100 是强先验，但当前只是作为 logits prior 加进去，模型容易在训练后期用 MLP logits 扰乱 graph ranking。
- 输入里虽然有 GraphRAG 候选文本，但模型不是逐候选比较，无法精细建模“这个 candidate 是否匹配当前 trajectory”。
- RAAT 只能模拟候选噪声，不能让模型天然学会候选间排序。

candidate-wise reranker 更符合任务：

- 每次只判断一个候选和 query 是否匹配，训练目标更局部、更稳定。
- semantic_id、graph_rank、score、sources 可以直接进入 candidate 表示。
- 可以自然做 hard negative、target-mask、source dropout、rank noise 等增强。
- 与传统 POI reranking 论文更容易对齐。

## 2-view RAAT 的定位

`2-view RAAT` 已在旧 classifier 脚本中实现：

```text
original + target-mask
loss = max(loss_original, loss_target_mask)
```

它的定位是增强模块，不是主架构。

后续如果 candidate-wise reranker 已经有效，可以把 RAAT 迁移到 pair/listwise 训练中：

```text
clean candidate features
target-mask / hard-negative candidate features
loss = max(clean_listwise_loss, adversarial_listwise_loss)
```

在新主线达到合理 baseline 前，不优先继续强化旧 classifier 的 RAAT。

## 下一步实现顺序

### Step 1: 构建 pairwise/listwise 数据

输入：

```text
retrieval_assets/NewYork/joined_poi_classification/train_joined_top100.parquet
retrieval_assets/NewYork/joined_poi_classification/val_joined_top100.parquet
```

输出建议：

```text
retrieval_assets/NewYork/teamlora_reranker/train_pairs_top100.jsonl
retrieval_assets/NewYork/teamlora_reranker/val_pairs_top100.jsonl
```

每条 pair 至少包含：

```json
{
  "sample_id": "...",
  "candidate_poi_id": "...",
  "candidate_rank": 1,
  "candidate_score": 0.0,
  "candidate_sources": ["transition", "geo", "history"],
  "candidate_semantic_id": "...",
  "label": 0,
  "target_poi_id": "...",
  "query_text": "...",
  "candidate_text": "..."
}
```

训练时按 `sample_id` group，组内 Top100 计算 listwise softmax CE。

### Step 2: 直接训练匿名三专家 TeamLoRA + RAAT

不再把单专家 baseline 作为主线目标。单专家即便成功，论文创新点不足；当前直接实现匿名三专家结构，复用同一个 query-candidate 输入，由 gate 自动学习专家组合：

```text
Expert-1 / Expert-2 / Expert-3
    shared input = raw trajectory + refined evidence + candidate semantic graph evidence
```

融合方式：

```text
h = gate([h_pref, h_graph, h_refine])
score = scoring_head(h, candidate_features)
```

`--expert-mode named` 仅作为可选消融，可显式拆成 pref/graph/refine 三个 view；默认不使用。

训练目标使用 group/listwise CE。由于 Top100 未命中的样本没有正候选，默认只用 `target_in_candidates=true` 的 train groups 训练；验证仍保留全量，用 `candidate_hit` 标明候选上限。

### Step 3: RAAT / hard negative

第一版直接启用 2-view RAAT：

```text
clean listwise loss
hard view listwise loss
loss = max(clean_loss, hard_loss)
```

hard view 首先使用 target candidate 的 graph evidence mask，而不是删除 target candidate；删除 target 会让 listwise CE 失去正样本。

后续再加：

```text
source dropout
rank noise
same-category / same-geo hard negatives
```

## 当前不做

- 不继续把旧 `full-POI classifier + GraphRAG prior` 当主线刷参数。
- 不把单专家 reranker 当最终方案。
- 不优先跑 full 4-view RAAT。
- 不重新训练 generative C1。
- 不把 `target_in_candidates=false` 等泄露性字段输入模型。
- 不把 selector 作为当前阶段重点；selector 应在 reranker 主体有效后再讨论。

## 当前可保留的旧资产

- `src/poi_classification/train_teamlora_poi_classifier.py`: full-classifier baseline，可用于消融。
- `--raat-mode target_mask_2view`: 旧架构上的 RAAT baseline。
- `retrieval_assets/NewYork/joined_poi_classification/*top100*`: 新 reranker 可直接读取的数据来源，不强制重新生成 JSONL。
- `retrieval_assets/NewYork/double_llm/graphrag_semantic_edges_v2_top100_*_candidates.jsonl`: GraphRAG Top100 原始候选来源。
- `models/prompt-refiner-lora-llama32-1b-decision-v2-final`: LoRA-A-v2 refined prompt 生成权重。
