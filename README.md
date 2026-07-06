# TMP POI Reranking Project

本项目当前主线是 **Semantic-ID GraphRAG Top100 + Llama-3.2-1B + 3-expert anonymous TeamLoRA candidate-wise reranker**。旧 README 和部分历史文档来自 LLMMRA 迁移期，里面有路径、实验结论和“当前推荐配置”过时的问题；本文件以 `src/` 代码、现有数据统计和最新模型 metadata 为准。

## 当前主线

```text
NYC trajectory / history / geo / time
        ↓
LoRA-A-v2 离线生成 refined_prompt
        ↓
Semantic-ID GraphRAG 构建 Top100 candidate POI
        ↓
joined parquet: compact raw_text + refined_text + graph candidates
        ↓
query-candidate group/listwise samples
        ↓
Shared Llama-3.2-1B backbone
        ↓
3 anonymous TeamLoRA experts: pref / graph / refine
        ↓
Gate fusion + graph rank/score/source features
        ↓
GraphRAG rank prior + bounded TeamLoRA residual correction
        ↓
Top100 candidate reranking
```

当前目标不是生成 POI 文本，也不是在全 POI vocab 上做分类，而是在 GraphRAG 候选池内给每个 candidate 打分并重排。正式指标应报告 `top1 / top5 / top10 / top20 / mrr`，并同时报告 `candidate_hit` 和 `conditional_*` 指标。

## 最新模型

最新训练模型位于：

```text
models/poi-teamlora-reranker-anon3-compact-l1024-priorres-alpha01-l201-tanh03-step600-v2/
```

关键配置来自该目录下的 `metadata.json` 和 `best/metadata.json`：

```text
base_model: models/Llama-3.2-1B-Instruct
train_joined: retrieval_assets/NewYork/joined_poi_classification/train_joined_top100.parquet
val_joined: retrieval_assets/NewYork/joined_poi_classification/val_joined_top100.parquet
semantic_map: retrieval_assets/NewYork/double_llm/semantic_poi_ids.jsonl
top_k: 100
max_length: 1024
expert_mode: anonymous
train_negatives: 7
hard_negatives: 6
lora_r / lora_alpha / lora_dropout: 8 / 16 / 0.05
scorer_dropout: 0.1
graph_feature_dim: 32
use_graph_prior: true
graph_prior_type: rank_log
residual_alpha_init: 0.1
residual_l2: 0.01
residual_bound_mode / value: tanh / 0.3
raat_mode: none
max_steps: 600
bf16: true
gradient_checkpointing: true
attn_implementation: sdpa
```

该模型保存的 best checkpoint 是 `best_step=600`。保存时的验证口径是快速验证设置：

```text
max_val_groups: 1000
val_hit_only: true
eval_candidate_limit: 50
eval_candidate_batch_size: 2
evaluated: 615
missed: 59
candidate_hit: 0.904065
top1 / top5 / top10 / top20 / mrr:
0.310569 / 0.598374 / 0.695935 / 0.775610 / 0.440998
conditional_top1 / conditional_top5 / conditional_top10 / conditional_top20 / conditional_mrr:
0.343525 / 0.661871 / 0.769784 / 0.857914 / 0.487795
residual_alpha: 0.106255
```

注意：以上不是全量 val + 完整 Top100 的正式口径。正式报告应重新跑独立评估，不加 `--val-hit-only`，不加 `--max-val-groups`，也不加 `--eval-candidate-limit`。

## 目录结构

```text
src/
  data_build/          数据构建、蒸馏输入、SFT/raw semantic 数据、joined parquet 和 raw_text 压缩
  refine_prompt/       LoRA-A-v2 decision-style refined_prompt 训练、生成、清洗
  graphrag/            Semantic-ID 构建、GraphRAG TopK 候选生成，旧 C1 生成式压缩脚本
  poi_reranker/        当前主线：TeamLoRA candidate-wise reranker 构组、训练、评估
  poi_classification/  历史 full-POI classifier baseline，不再作为主线

retrieval_assets/NewYork/
  evidence/                  sequence / geo / preference evidence 与 teacher distill 数据
  distill_decision/          LoRA-A-v2 decision refiner 蒸馏训练数据
  refined_prompts_decision/  LoRA-A-v2 全量 refined_prompt 输出
  semantic_poi_sft/          raw prompt 改写为 semantic POI 表示后的 SFT 数据
  double_llm/                semantic_poi_ids 与 GraphRAG Top100 candidate JSONL
  joined_poi_classification/ 当前 reranker 直接读取的 joined parquet

models/
  Llama-3.2-1B-Instruct/
  prompt-refiner-lora-llama32-1b-decision-v2-final/
  poi-teamlora-reranker-anon3-compact-l1024-priorres-alpha01-l201-tanh03-step600-v2/
```

## `src` 架构梳理

### `src/data_build`

数据构建分两层：

1. 早期 evidence / teacher / SFT 数据：
   - `build_all_evidence.py` 从轨迹、图和 POI 元信息构造 sequence、geo、preference evidence。
   - `build_teacher_distill_inputs.py` 将多路 evidence 合成 teacher/refiner 输入。
   - `call_teacher_prompt_distill_api.py` 调 teacher 生成蒸馏标签。
   - `sample_teacher_distill_subset.py`、`sample_teacher_hard_boundary_subset.py`、`sample_refiner_inputs_by_quality.py` 负责采样。
   - `build_poi_sft_data.py`、`build_semantic_poi_sft_data.py` 生成 POI SFT/raw semantic prompt。

2. 当前 reranker 直接使用的 joined parquet：
   - `build_joined_poi_classification_data.py` 将 semantic raw prompt、refined prompt、GraphRAG TopK 合并。
   - `compress_joined_raw_context.py` 将 full raw context 压缩成 compact raw context，避免 candidate 后置模板在短 `max_length` 下被截断。
   - `build_joined_poi_classification_pipeline.py` 是推荐重建入口：先写 `*_fullraw.parquet` 备份，再生成主训练用 compact `*_joined_top100.parquet`。

当前 compact 策略：

```text
Trajectory: 8 lines
Transitions: top4
Nearby: top4
top_categories: top3
revisited_pois: top3
```

现有 top100 compact 数据统计：

```text
train rows: 30918
val rows: 3959

train raw_text_after p50/p95/max: 425 / 532.15 / 696 tokens
val   raw_text_after p50/p95/max: 425 / 528.00 / 680 tokens

train candidate_start_after p50/p95/max: 615 / 733 / 930 tokens
val   candidate_start_after p50/p95/max: 616 / 731 / 894 tokens
```

因此当前训练推荐 `--max-length 1024`；`768` 可用于轻量实验，`512` 仍可能截断候选文本。

### `src/refine_prompt`

这里是 LoRA-A-v2 decision-style refine prompt 线路：

- `train_prompt_refiner_lora.py` 训练 refiner LoRA。
- `generate_refined_prompts.py` 使用 base model + LoRA 权重生成 `distilled_prompt`。
- `validate_decision_refined_prompts.py` 清洗过短、过长、泄露 target 等异常输出。
- `refiner_prompt_styles.py` 定义 decision 风格 prompt。

当前使用的 refiner 权重：

```text
models/prompt-refiner-lora-llama32-1b-decision-v2-final/
```

当前 joined 数据读取的是：

```text
retrieval_assets/NewYork/refined_prompts_decision/lora_a_decision_v2_train_full_outputs.jsonl
retrieval_assets/NewYork/refined_prompts_decision/lora_a_decision_v2_val_full_outputs.jsonl
```

### `src/graphrag`

GraphRAG 模块负责候选池上限：

- `build_semantic_poi_ids.py` 生成稳定 semantic id，形如 `NYC::<CATEGORY_TOKEN>::<GEO_CELL>::P####`。
- `build_graphrag_topk_candidates.py` 基于 transition、geo、history、semantic edge、covisit 等信号构建 TopK 候选。
- `build_c1_reranker_sft_data.py`、`train_c1_candidate_reranker_lora.py`、`generate_eval_c1_candidate_reranker.py` 是旧 C1 生成式候选压缩路线，仅保留作对照。

现有 GraphRAG Top100 召回：

```text
train hit@100: 0.615111
val   hit@100: 0.624653

val hit@1 / hit@5 / hit@10 / hit@20 / hit@50:
0.188179 / 0.367770 / 0.432432 / 0.487749 / 0.572114
```

`candidate_hit` 是候选池上限。全量 val 指标不可能超过 full-val `candidate_hit≈0.624653`，除非先扩大或改进候选池。

### `src/poi_reranker`

这是当前主线。

- `build_teamlora_reranker_groups.py` 可将 joined parquet 预构造成 grouped JSONL；训练脚本也可以直接从 joined parquet 动态构 group。
- `train_teamlora_reranker_raat.py` 训练 3-expert TeamLoRA candidate-wise reranker。
- `evaluate_teamlora_reranker.py` 加载保存的 `trainable_model.bin` 做独立评估。

训练样本格式由 `build_group()` 构造：

```text
sample_id
target_poi_id
target_in_candidates
pref_text      compact raw_text
refine_text    LoRA-A-v2 refined_text
graph_text     GraphRAG TopK textual evidence
candidates[]   poi_id / rank / score / sources / semantic_id / category / geo_cell / label
```

默认 `expert_mode=anonymous` 时，三个 expert 都看同一个 candidate-postfix 模板，由 gate 学习融合：

```text
[TASK=CANDIDATE_RERANK]
Score whether the candidate POI is the next location.

[QUERY]
{compact raw_text}

[REFINED_EVIDENCE]
{refined_text}

[CANDIDATE]
Candidate POI: {poi_id}|semantic_id=...|category=...|geo=...|coord=...
Graph evidence: rank=...; score=...; sources=...
```

模型结构：

```text
AutoModel(Llama-3.2-1B-Instruct, frozen)
  └─ q/k/v/o/gate/up/down proj 注入 MultiExpertLoRA
       experts = pref, graph, refine

每个 candidate:
  1. 用 pref expert 编码文本，取 last token hidden
  2. 用 graph expert 编码文本
  3. 用 refine expert 编码文本
  4. graph_features = [1/log1p(rank), log1p(score)/10, rank_norm, masked, source bits...]
  5. gate([h_pref, h_graph, h_refine, graph_features]) -> 3 expert weights
  6. scorer(fused_hidden, graph_features) -> residual score
  7. 若启用 --use-graph-prior:
       final_score = graph_rank_prior + bound(alpha * residual)
```

当前最新模型启用了 GraphRAG prior：

```text
graph_prior_type=rank_log: prior = -log(rank)
residual_bound_mode=tanh, residual_bound_value=0.3
residual_alpha≈0.106
```

训练目标是 group/listwise softmax CE。训练默认只使用 `target_in_candidates=true` 的样本；验证可保留全量样本，未命中候选池的样本按 miss 计入。

RAAT 相关参数仍在脚本中保留：

```text
--raat-mode none
--raat-mode target_mask_2view
--raat-mode graph_3view
```

但最新模型使用 `--raat-mode none`，不是旧 README 中推荐的 RAAT 主线。

### `src/poi_classification`

`train_teamlora_poi_classifier.py` 是历史 full-POI classifier baseline：一次 forward 输出全 POI logits，并可叠加 GraphRAG prior / RAAT。该路线已不作为当前主线，主要用于消融或历史对照。

## 重建 joined parquet

从项目根目录执行：

```bash
cd /mnt/data/users/yyl/TMP
PY=/mnt/data/users/yyl/miniconda3/envs/poi_data/bin/python

$PY src/data_build/build_joined_poi_classification_pipeline.py \
  --base-dir retrieval_assets/NewYork \
  --output-dir retrieval_assets/NewYork/joined_poi_classification \
  --graph-top-k 100 \
  --splits train val \
  --tokenizer models/Llama-3.2-1B-Instruct \
  --overwrite
```

如只想基于已有 `*_fullraw.parquet` 重新压缩：

```bash
cd /mnt/data/users/yyl/TMP
PY=/mnt/data/users/yyl/miniconda3/envs/poi_data/bin/python

$PY src/data_build/compress_joined_raw_context.py \
  --input-dir retrieval_assets/NewYork/joined_poi_classification \
  --top-k 100 \
  --splits train val \
  --tokenizer models/Llama-3.2-1B-Instruct
```

## 复现实验训练

以下命令复现最新模型的主要训练配置：

```bash
cd /mnt/data/users/yyl/TMP
PY=/mnt/data/users/yyl/miniconda3/envs/poi_data/bin/python

CUDA_VISIBLE_DEVICES=0 \
NCCL_P2P_DISABLE=1 \
NCCL_IB_DISABLE=1 \
nohup $PY src/poi_reranker/train_teamlora_reranker_raat.py \
  --train-joined retrieval_assets/NewYork/joined_poi_classification/train_joined_top100.parquet \
  --val-joined retrieval_assets/NewYork/joined_poi_classification/val_joined_top100.parquet \
  --semantic-map retrieval_assets/NewYork/double_llm/semantic_poi_ids.jsonl \
  --base-model /mnt/data/users/yyl/TMP/models/Llama-3.2-1B-Instruct \
  --output-dir models/poi-teamlora-reranker-anon3-compact-l1024-priorres-alpha01-l201-tanh03-step600-v2 \
  --top-k 100 \
  --max-length 1024 \
  --expert-mode anonymous \
  --train-negatives 7 \
  --hard-negatives 6 \
  --batch-groups 1 \
  --grad-accum 8 \
  --max-steps 600 \
  --lr 2e-4 \
  --lora-r 8 \
  --lora-alpha 16 \
  --lora-dropout 0.05 \
  --scorer-dropout 0.1 \
  --graph-feature-dim 32 \
  --use-graph-prior \
  --graph-prior-type rank_log \
  --residual-alpha-init 0.1 \
  --residual-l2 0.01 \
  --residual-bound-mode tanh \
  --residual-bound-value 0.3 \
  --raat-mode none \
  --val-hit-only \
  --max-val-groups 1000 \
  --eval-candidate-limit 50 \
  --eval-candidate-batch-size 2 \
  --eval-final-only \
  --eval-steps 600 \
  --save-steps 600 \
  --bf16 \
  --gradient-checkpointing \
  --attn-implementation sdpa \
  > logs/train_poi_teamlora_reranker_anon3_compact_l1024_priorres_alpha01_l201_tanh03_step600_v2.log 2>&1 &
```

如显存不足，优先降低这些项：

```text
--max-length 1024 -> 768
--train-negatives 7 -> 3
--hard-negatives 6 -> 3
--eval-candidate-batch-size 2 -> 1
```

`grad_accum` 主要影响更新频率和训练时间，不会显著降低单个 forward/backward 的显存峰值。

## 评估

复现模型目录中保存的快速验证口径：

```bash
cd /mnt/data/users/yyl/TMP
PY=/mnt/data/users/yyl/miniconda3/envs/poi_data/bin/python

CUDA_VISIBLE_DEVICES=0 \
NCCL_P2P_DISABLE=1 \
NCCL_IB_DISABLE=1 \
$PY src/poi_reranker/evaluate_teamlora_reranker.py \
  --model-dir models/poi-teamlora-reranker-anon3-compact-l1024-priorres-alpha01-l201-tanh03-step600-v2 \
  --checkpoint best \
  --val-joined retrieval_assets/NewYork/joined_poi_classification/val_joined_top100.parquet \
  --semantic-map retrieval_assets/NewYork/double_llm/semantic_poi_ids.jsonl \
  --top-k 100 \
  --max-length 1024 \
  --val-hit-only \
  --max-val-groups 1000 \
  --eval-candidate-limit 50 \
  --eval-candidate-batch-size 2 \
  --bf16 \
  --attn-implementation sdpa
```

正式 full-val 评估建议：

```bash
cd /mnt/data/users/yyl/TMP
PY=/mnt/data/users/yyl/miniconda3/envs/poi_data/bin/python

CUDA_VISIBLE_DEVICES=0 \
NCCL_P2P_DISABLE=1 \
NCCL_IB_DISABLE=1 \
$PY src/poi_reranker/evaluate_teamlora_reranker.py \
  --model-dir models/poi-teamlora-reranker-anon3-compact-l1024-priorres-alpha01-l201-tanh03-step600-v2 \
  --checkpoint best \
  --val-joined retrieval_assets/NewYork/joined_poi_classification/val_joined_top100.parquet \
  --semantic-map retrieval_assets/NewYork/double_llm/semantic_poi_ids.jsonl \
  --top-k 100 \
  --max-length 1024 \
  --eval-candidate-batch-size 2 \
  --bf16 \
  --attn-implementation sdpa \
  --output-json models/poi-teamlora-reranker-anon3-compact-l1024-priorres-alpha01-l201-tanh03-step600-v2/eval_best_fullval_top100.json
```

正式评估不要加：

```text
--val-hit-only
--max-val-groups
--eval-candidate-limit
```

## 指标解释

```text
candidate_hit
```

候选池召回上限。target 不在 TopK 里时，reranker 无法选中正确 POI。

```text
conditional_top1 / conditional_top5 / conditional_top10 / conditional_top20 / conditional_mrr
```

只在 target 已进入候选池的样本上计算，观察模型的重排能力。

```text
top1 / top5 / top10 / top20 / mrr
```

全量口径，等于 conditional 指标乘以 `candidate_hit`。如果 full-val `candidate_hit≈0.624653`，全量 `top20` 理论上不能超过这个值。

## 历史与实验目录

- `experiments/graphrag_bge_rerank/`：BGE reranker 对 GraphRAG 候选的重排实验。
- `experiments/lightgbm_candidate_ranker/`：sklearn / XGBoost candidate ranker 与 residual-prior 实验。
- `experiments/multisource_candidate_pool/`：多源候选池和 top500 -> top100 压缩实验。
- `legacy_scripts_snapshot/`：原 LLMMRA 脚本快照，仅作兼容和追溯。

这些实验可以作为候选池增强方向参考，但当前可复现主线仍是 `src/poi_reranker/train_teamlora_reranker_raat.py`。

## 重要注意

- 旧文档中的 `/mnt/data/yyl/TMP` 路径应改为当前项目路径 `/mnt/data/users/yyl/TMP`。
- `MANIFEST.md` 描述的是迁移清单，不代表当前目录一定包含所有旧数据集目录。
- `src/graphrag/README_GRAPHRAG.md` 中“大 GraphRAG candidate JSONL 未复制”的说法已过时；当前 `retrieval_assets/NewYork/double_llm/` 下存在 train/val Top100 candidate JSONL。
- 当前模型的 saved best 指标来自快速验证口径；论文或正式汇报必须重新跑 full-val 完整 Top100 评估。
