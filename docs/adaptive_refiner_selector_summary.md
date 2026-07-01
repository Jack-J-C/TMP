# TMP 当前方案摘要

本文档只记录当前有效方案。旧 LLMMRA 的 `LoRA-A + 8B LoRA-B + Selector` 路线在 TMP 中不再作为主线。

## 当前主线

当前主线是 1B-only 匿名三专家 candidate-wise reranker：

```text
LoRA-A-v2 refined_prompt offline generation
        ↓
Semantic-ID GraphRAG Top100 retrieval
        ↓
query-candidate pair / listwise reranking
        ↓
Llama-3.2-1B + 3 anonymous LoRA experts
        ↓
Gate fusion + candidate scoring head
        ↓
Top1 / Top5 / Top10 / Top20 / MRR
```

当前不再优先训练 8B LoRA-B，也不优先训练 selector。

## LoRA-A-v2 定位

LoRA-A-v2 继续保留，但角色固定为离线 evidence refiner：

- 输入：trajectory、transition evidence、geo evidence、history evidence。
- 输出：label-free `refined_prompt` 和 `confidence`。
- 用途：作为 query-candidate 输入的一部分。

LoRA-A-v2 不负责生成候选集，也不负责最终 POI 预测。

## GraphRAG 定位

Semantic-ID GraphRAG 负责构建 Top100 candidate set。

候选字段包括：

- `candidate_poi_id`
- `semantic_id`
- `graph_rank`
- `score`
- `sources`

GraphRAG 的作用是召回和提供结构化图证据。最终排序由 candidate-wise reranker 完成。

## 旧 full-classifier 结论

旧脚本：

```text
src/poi_classification/train_teamlora_poi_classifier.py
```

该路线是：

```text
input_text
  -> Llama-3.2-1B + TeamLoRA
  -> pooled hidden
  -> full POI vocab classifier
  -> logits + GraphRAG prior
```

当前判断：该主架构不作为后续主线。原因：

- full POI vocab 分类对 1B LoRA 难度过高。
- GraphRAG 只是 logits prior，没有成为逐候选比较证据。
- 训练后期 MLP logits 容易扰乱 GraphRAG 排序。
- RAAT 只能增强鲁棒性，不能修复主架构不对齐。

## RAAT 定位

`2-view RAAT` 已在旧 classifier 中实现：

```text
original + target-mask
loss = max(loss_original, loss_target_mask)
```

它是增强模块，不是主线架构。

当前直接在 candidate-wise reranker 中使用 listwise 2-view RAAT：

```text
clean Top100 listwise loss
target graph masked listwise loss
loss = max(clean_loss, hard_loss)
```

注意：hard view 不删除 target candidate，只 mask target candidate 的 graph evidence。否则 listwise CE 没有正样本。

候选增强可以包括：

- target mask
- candidate dropout
- source dropout
- rank noise
- same-category hard negative
- same-geo hard negative

## 下一步优先级

1. 构建 grouped Top100 reranker 数据。
2. 直接训练匿名三专家 TeamLoRA candidate-wise reranker。
3. 训练期启用 2-view RAAT。
4. 对比 GraphRAG 原始排序，确认是否提升 `top1/MRR`。
5. 后续再加 full hard-negative/source-dropout/rank-noise 消融。

## 暂不做

- 不继续围绕旧 full-POI classifier 大量调参。
- 不把单专家 reranker 当主线结果。
- 不手工规定每个 LoRA expert 必须学习 pref/graph/refine；默认让 gate 自动分工。
- 不优先训练 8B。
- 不优先训练 selector。
- 不训练 generative C1 输出 Top50。
- 不把 `target_in_candidates` 这类泄露字段输入模型。
