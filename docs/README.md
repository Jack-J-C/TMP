# TMP 当前主版本

本项目当前主线是 **Llama-3.2-1B + TeamLoRA candidate-wise POI reranker**。旧 LLMMRA 中的 `LoRA-A + 8B LoRA-B + selector` 路线、full-POI classifier 路线、C1 生成式候选压缩路线均只保留为历史对照，不作为当前主训练路径。

当前目标不是让模型生成 POI 文本，而是在 GraphRAG Top100 候选池内对候选 POI 打分重排，最终评价 `Top1 / Top5 / Top10 / Top20 / MRR`。

## 主架构

```text
NYC trajectory / history / geo / time
        ↓
LoRA-A-v2 离线生成 refined_prompt
        ↓
Semantic-ID GraphRAG 构建 Top100 candidates
        ↓
joined parquet: raw_text + refined_text + graph candidates
        ↓
compact raw_text 主训练样本
        ↓
query-candidate pair / listwise group
        ↓
Shared Llama-3.2-1B backbone
        ↓
3 anonymous TeamLoRA experts: pref / graph / refine
        ↓
Gate fusion + graph rank/score/source features
        ↓
scalar relevance score
        ↓
Rank Top100 candidates
```

当前默认使用 `expert_mode=anonymous`：三个 LoRA expert 接收同一个 query-candidate 输入，gate 自行学习融合，不手工指定每个 expert 学什么。`expert_mode=named` 仅作为消融模式。

## 关键结论

候选池命中率是当前上限瓶颈。GraphRAG Top100 的 `candidate_hit` 约为 0.625，所有全量 val 指标都不能超过这个上限。

当前训练输入采用 candidate 后置模板：

```text
[TASK=CANDIDATE_RERANK]
Score whether the candidate POI is the next location.

[QUERY]
raw_text

[REFINED_EVIDENCE]
refined_text

[CANDIDATE]
candidate_text
```

full raw 过长时，`[CANDIDATE]` 会被 `max_length=512` 截断。因此当前主训练数据已经改为 compact raw context。

## 关键文件

基础模型：

```text
models/Llama-3.2-1B-Instruct/
```

LoRA-A-v2 refine prompt 生成权重：

```text
models/prompt-refiner-lora-llama32-1b-decision-v2-final/
```

Semantic-ID 映射：

```text
retrieval_assets/NewYork/double_llm/semantic_poi_ids.jsonl
```

GraphRAG Top100 原始候选：

```text
retrieval_assets/NewYork/double_llm/graphrag_semantic_edges_v2_top100_train_candidates.jsonl
retrieval_assets/NewYork/double_llm/graphrag_semantic_edges_v2_top100_val_candidates.jsonl
```

joined 主训练数据：

```text
retrieval_assets/NewYork/joined_poi_classification/train_joined_top100.parquet
retrieval_assets/NewYork/joined_poi_classification/val_joined_top100.parquet
```

joined fullraw 备份数据：

```text
retrieval_assets/NewYork/joined_poi_classification/train_joined_top100_fullraw.parquet
retrieval_assets/NewYork/joined_poi_classification/val_joined_top100_fullraw.parquet
```

## 数据构建流程

当前数据构建固定为两阶段：

```text
1. 生成 fullraw joined parquet，仅作备份/审计，不作为主训练入口。
2. 从 fullraw 生成 compact raw-context parquet，作为主训练文件。
```

一键重建：

```bash
cd /mnt/data/yyl/TMP

/mnt/data/yyl/miniconda3/envs/poi_data/bin/python src/data_build/build_joined_poi_classification_pipeline.py \
  --base-dir retrieval_assets/NewYork \
  --output-dir retrieval_assets/NewYork/joined_poi_classification \
  --graph-top-k 100 \
  --splits train val \
  --tokenizer models/Llama-3.2-1B-Instruct \
  --overwrite
```

如果只想快速重建并跳过 token 统计：

```bash
cd /mnt/data/yyl/TMP

/mnt/data/yyl/miniconda3/envs/poi_data/bin/python src/data_build/build_joined_poi_classification_pipeline.py \
  --base-dir retrieval_assets/NewYork \
  --output-dir retrieval_assets/NewYork/joined_poi_classification \
  --graph-top-k 100 \
  --splits train val \
  --tokenizer models/Llama-3.2-1B-Instruct \
  --overwrite \
  --skip-token-stats
```

如只基于已有 `*_fullraw.parquet` 重新压缩：

```bash
cd /mnt/data/yyl/TMP

/mnt/data/yyl/miniconda3/envs/poi_data/bin/python src/data_build/compress_joined_raw_context.py \
  --input-dir retrieval_assets/NewYork/joined_poi_classification \
  --top-k 100 \
  --splits train val \
  --tokenizer models/Llama-3.2-1B-Instruct
```

默认 compact 策略：

```text
Trajectory: 最多 8 行
Transitions: top4
Nearby: top4
top_categories: top3
revisited_pois: top3
```

当前压缩后统计：

```text
train raw_text p50 ≈ 425 tokens, p95 ≈ 532
val   raw_text p50 ≈ 425 tokens, p95 ≈ 528

train [CANDIDATE] start p50 ≈ 615, p95 ≈ 733
val   [CANDIDATE] start p50 ≈ 616, p95 ≈ 731
```

因此当前建议 `--max-length 768` 起步；显存允许时使用 `1024` 更稳。`512` 仍可能截断部分候选文本。

## 训练脚本

主训练脚本：

```text
src/poi_reranker/train_teamlora_reranker_raat.py
```

默认行为：

```text
train 使用 target_in_candidates=true 样本
val 默认保留全量样本
train 每组采样 1 positive + N negatives
val 默认排序完整 Top100
保存 latest 和 best 两类 checkpoint
```

轻量验证配置，适合先看趋势：

```bash
cd /mnt/data/yyl/TMP

CUDA_VISIBLE_DEVICES=0 \
NCCL_P2P_DISABLE=1 \
NCCL_IB_DISABLE=1 \
nohup /mnt/data/yyl/miniconda3/envs/poi_data/bin/python src/poi_reranker/train_teamlora_reranker_raat.py \
  --train-joined retrieval_assets/NewYork/joined_poi_classification/train_joined_top100.parquet \
  --val-joined retrieval_assets/NewYork/joined_poi_classification/val_joined_top100.parquet \
  --semantic-map retrieval_assets/NewYork/double_llm/semantic_poi_ids.jsonl \
  --base-model /mnt/data/yyl/TMP/models/Llama-3.2-1B-Instruct \
  --output-dir models/poi-teamlora-reranker-anon3-compact-quick-v1 \
  --top-k 100 \
  --max-length 768 \
  --expert-mode anonymous \
  --train-negatives 15 \
  --hard-negatives 12 \
  --batch-groups 1 \
  --grad-accum 8 \
  --max-steps 200 \
  --lr 2e-4 \
  --lora-r 8 \
  --lora-alpha 16 \
  --lora-dropout 0.05 \
  --scorer-dropout 0.1 \
  --raat-mode none \
  --val-hit-only \
  --max-val-groups 1000 \
  --eval-candidate-limit 50 \
  --eval-steps 200 \
  --save-steps 200 \
  --bf16 \
  --gradient-checkpointing \
  --attn-implementation sdpa \
  > logs/train_poi_teamlora_reranker_anonymous_compact_quick_v1.log 2>&1 &
```

主实验配置，适合正式对比：

```bash
cd /mnt/data/yyl/TMP

CUDA_VISIBLE_DEVICES=0 \
NCCL_P2P_DISABLE=1 \
NCCL_IB_DISABLE=1 \
nohup /mnt/data/yyl/miniconda3/envs/poi_data/bin/python src/poi_reranker/train_teamlora_reranker_raat.py \
  --train-joined retrieval_assets/NewYork/joined_poi_classification/train_joined_top100.parquet \
  --val-joined retrieval_assets/NewYork/joined_poi_classification/val_joined_top100.parquet \
  --semantic-map retrieval_assets/NewYork/double_llm/semantic_poi_ids.jsonl \
  --base-model /mnt/data/yyl/TMP/models/Llama-3.2-1B-Instruct \
  --output-dir models/poi-teamlora-reranker-anon3-compact-raat-v1 \
  --top-k 100 \
  --max-length 768 \
  --expert-mode anonymous \
  --train-negatives 15 \
  --hard-negatives 12 \
  --batch-groups 1 \
  --grad-accum 8 \
  --max-steps 600 \
  --lr 2e-4 \
  --lora-r 8 \
  --lora-alpha 16 \
  --lora-dropout 0.05 \
  --scorer-dropout 0.1 \
  --raat-mode target_mask_2view \
  --eval-steps 200 \
  --save-steps 200 \
  --bf16 \
  --gradient-checkpointing \
  --attn-implementation sdpa \
  > logs/train_poi_teamlora_reranker_anonymous_compact_raat_v1.log 2>&1 &
```

如果显存不足，优先调整：

```text
--max-length 768 → 640
--train-negatives 15 → 7
--hard-negatives 12 → 6
--raat-mode target_mask_2view → none
--grad-accum 8 → 16
```

`grad_accum` 增大主要影响更新频率和训练时间，不直接降低单步显存；降低 `max-length`、负样本数、关闭 RAAT 才会明显降显存。

## 评估脚本

备用独立评估脚本：

```text
src/poi_reranker/evaluate_teamlora_reranker.py
```

评估 best checkpoint，全量 val，完整 Top100：

```bash
cd /mnt/data/yyl/TMP

CUDA_VISIBLE_DEVICES=0 \
NCCL_P2P_DISABLE=1 \
NCCL_IB_DISABLE=1 \
/mnt/data/yyl/miniconda3/envs/poi_data/bin/python src/poi_reranker/evaluate_teamlora_reranker.py \
  --model-dir models/poi-teamlora-reranker-anon3-compact-raat-v1 \
  --checkpoint best \
  --val-joined retrieval_assets/NewYork/joined_poi_classification/val_joined_top100.parquet \
  --semantic-map retrieval_assets/NewYork/double_llm/semantic_poi_ids.jsonl \
  --top-k 100 \
  --max-length 768 \
  --bf16 \
  --attn-implementation sdpa \
  --output-json models/poi-teamlora-reranker-anon3-compact-raat-v1/eval_best_fullval.json
```

快速评估命中样本子集：

```bash
cd /mnt/data/yyl/TMP

CUDA_VISIBLE_DEVICES=0 \
NCCL_P2P_DISABLE=1 \
NCCL_IB_DISABLE=1 \
/mnt/data/yyl/miniconda3/envs/poi_data/bin/python src/poi_reranker/evaluate_teamlora_reranker.py \
  --model-dir models/poi-teamlora-reranker-anon3-compact-raat-v1 \
  --checkpoint latest \
  --val-joined retrieval_assets/NewYork/joined_poi_classification/val_joined_top100.parquet \
  --semantic-map retrieval_assets/NewYork/double_llm/semantic_poi_ids.jsonl \
  --top-k 100 \
  --max-length 768 \
  --val-hit-only \
  --max-val-groups 1000 \
  --eval-candidate-limit 50 \
  --bf16 \
  --attn-implementation sdpa
```

## 指标口径

训练脚本和独立评估脚本当前使用同一套指标：

```text
top1 / top5 / top10 / top20 / mrr
```

这些是全量口径；候选池未命中的样本按错误计入。

```text
conditional_top1 / conditional_top5 / conditional_top10 / conditional_top20 / conditional_mrr
```

这些只在 `target_in_candidates=true` 的样本上计算，用来观察模型在候选池命中后的重排能力。

```text
candidate_hit
```

这是候选池召回上限，不是模型能力。若 `candidate_hit=0.625`，全量 `top20` 理论上不可能超过 0.625。

## RAAT 与噪声模块

当前支持三种训练模式：

```text
--raat-mode none
```

只训练 clean view，适合轻量 baseline。

```text
--raat-mode target_mask_2view
```

训练 clean + target graph mask，两路 loss 取最大值：

```text
loss = max(loss_clean, loss_target_mask)
```

这是当前推荐的 RAAT 版本。

```text
--raat-mode graph_3view
```

训练 clean + target-demotion + hard-negative-promotion，三路 loss 取最大值。该模式更重，优先用于后续增强实验，不建议作为第一轮验证。

## 当前代码入口

数据构建：

```text
src/data_build/build_joined_poi_classification_pipeline.py
src/data_build/build_joined_poi_classification_data.py
src/data_build/compress_joined_raw_context.py
```

GraphRAG：

```text
src/graphrag/build_semantic_poi_ids.py
src/graphrag/build_graphrag_topk_candidates.py
```

LoRA-A refine prompt：

```text
src/refine_prompt/train_prompt_refiner_lora.py
src/refine_prompt/generate_refined_prompts.py
src/refine_prompt/validate_decision_refined_prompts.py
```

TeamLoRA reranker：

```text
src/poi_reranker/build_teamlora_reranker_groups.py
src/poi_reranker/train_teamlora_reranker_raat.py
src/poi_reranker/evaluate_teamlora_reranker.py
```

历史 full-POI classifier：

```text
src/poi_classification/train_teamlora_poi_classifier.py
```

该路线不再作为当前主线。

## 当前实验原则

优先比较同一数据、同一模板、同一 `max-length` 下的模块差异。不要把 candidate 前置/后置、full raw/compact raw、512/768/1024、RAAT/no-RAAT 混在一起比较。

快速验证可以使用 `--val-hit-only`、`--max-val-groups`、`--eval-candidate-limit`。正式报告必须回到全量 val + 完整 Top100，并同时报告 `candidate_hit` 与 conditional 指标。
