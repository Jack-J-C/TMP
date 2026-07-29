# TMP POI Reranking Project

TMP 当前主线在 `experiment2` 分支：先用 Semantic-ID GraphRAG 生成宽候选池，再用轻量 coverage-only 线性模型把宽池压回固定 Top100，最后用 Llama-3.2-1B routed TeamLoRA reranker 在 Top100 内重排。

当前目标不是全量 POI 生成，也不是直接预测所有 POI，而是提升固定 Top100 候选池质量，并在该 Top100 内学习重排。

## 当前架构

```text
TMP-format trajectory / POI metadata
        |
        v
Semantic-ID GraphRAG wide retrieval
  - transition / geo / history / semantic signals
  - NYC experiment2 使用 Top500 宽候选池
        |
        v
coverage-only linear pool scorer
  - input: GraphRAG rank / score / source flags
  - objective: target coverage in final Top100
  - output: fixed learned Top100 candidate pool
        |
        v
joined parquet
  - compact raw context
  - USER_SEMANTIC_PROFILE
  - conditional SIMILAR_USER_SEMANTIC_PROFILE
  - POI_HYPOTHESIS for each candidate
  - structured graph_features + graph prior
        |
        v
Frozen Llama-3.2-1B-Instruct
  q/k/v/o/gate/up/down Linear -> RoutedTeamLoRALinear
  lora_num=3 implicit routed LoRA experts
        |
        v
GraphRAG/learned-pool rank prior + bounded residual correction
        |
        v
Top100 candidate reranking
```

当前三专家不是显式 `pref/graph/refine` 三次 forward。现在采用 TeamLoRA 风格的层内隐式专家路由：每个 candidate 只跑一次输入文本，LoRA 专家在 adapted Linear 层内部动态加权。

## 当前主线数据

三城原始 v5 训练数据仍保留：

```text
retrieval_assets_clsprec/NYC/joined_poi_classification/
retrieval_assets_clsprec/SIN/joined_poi_classification/
retrieval_assets_getnext_clsprec/TKY/joined_poi_classification/
```

`experiment2` 当前重点是 NYC 的候选池增强数据：

```text
retrieval_assets_clsprec/NYC/joined_poi_classification_exp2_top500_rescored/
  train_joined_top100.parquet
  val_joined_top100.parquet
  test_joined_top100.parquet
  build_summary_top100.json
  compression_summary_top100.json
```

对应训练配置：

```text
config/train_nyc_exp2_top500_rescored_semprofile_simuser_v2.yaml
```

需要本地手动准备的模型权重：

```text
models/Llama-3.2-1B-Instruct
models/bge-m3        # 仅重建 similar user profile 时需要；直接训练 parquet 不需要重新跑
```

`models/` 默认不推送到 git。

CLSPRec 三城与 GETNext-TKY 数据从原始 check-in 到 `dataset_clsprec` / `dataset_getnext_clsprec`、`retrieval_assets_*` 和训练 parquet 的预处理/split 规则，见：

```text
docs/clsprec_dataset_pipeline.md
```

## 候选池重建策略

当前 NYC experiment2 实际采用的是：

```text
GraphRAG Top500 wide pool -> coverage-only linear scorer -> learned fixed Top100
```

脚本：

```text
scripts/experiment2_candidate_pool_quality.py
```

核心输入：

```text
retrieval_assets_clsprec/NYC/double_llm/graphrag_semantic_edges_v2_top500_train_Candidates.jsonl
retrieval_assets_clsprec/NYC/double_llm/graphrag_semantic_edges_v2_top500_val_Candidates.jsonl
retrieval_assets_clsprec/NYC/double_llm/graphrag_semantic_edges_v2_top500_test_Candidates.jsonl
```

最佳候选池报告：

```text
reports/experiment2_nyc_top500_to_top100_rescore_quality.json
```

已验证的覆盖率：

| Candidate pool | Val Hit@100 | Test Hit@100 | Test rescued/lost |
|---|---:|---:|---:|
| 原始 GraphRAG Top100 | 0.6839 | 0.6482 | - |
| Top500 oracle | 0.7807 | 0.7613 | - |
| Top500 -> learned Top100 | 0.7478 | 0.7206 | 74 / 10 |

对比过的宽池：

| Wide N | Test oracle | Test learned Top100 | 结论 |
|---:|---:|---:|---|
| 300 | 0.7432 | 0.6765 | 上限偏低 |
| 400 | 0.7545 | 0.6867 | 仍低于 Top500 |
| 500 | 0.7613 | 0.7206 | 当前最佳 |
| 600 | 0.7658 | 0.6787 | 压缩损失变大 |
| 1000 | 0.8020 | 0.6550 | oracle 高，但噪声太大 |

所以目前不采用 Top1000。Top1000 虽然包含更多答案，但当前线性 scorer 无法稳定把深层答案压进最终 Top100。

用于构建 joined parquet 的固定候选池 summary：

```text
retrieval_assets_clsprec/NYC/double_llm/experiment2_top500_best_rescored/apply_best_weights_summary.json
```

注意：该目录只推送了 summary，不推送大体积候选 JSONL。训练直接使用已推送的 compact parquet。

## Joined Parquet 构建

`build_joined_poi_classification_data.py` 已支持自定义候选目录：

```bash
cd /mnt/data/users/yyl/TMP
PY=/mnt/data/users/yyl/miniconda3/envs/poi_data/bin/python

CUDA_VISIBLE_DEVICES=0 $PY src/data_build/build_joined_poi_classification_data.py \
  --base-dir retrieval_assets_clsprec/NYC \
  --output-dir retrieval_assets_clsprec/NYC/joined_poi_classification_exp2_top500_rescored \
  --semantic-map retrieval_assets_clsprec/NYC/double_llm/semantic_poi_ids.jsonl \
  --graph-candidates-dir retrieval_assets_clsprec/NYC/double_llm/experiment2_top500_best_rescored \
  --graph-top-k 100 \
  --splits train val test \
  --format parquet \
  --overwrite \
  --allow-missing-refined \
  --similar-profile \
  --similar-profile-model models/bge-m3 \
  --similar-profile-local-files-only \
  --similar-profile-query-source recent_profile \
  --similar-profile-k 20 \
  --similar-profile-min-sim 0.35 \
  --similar-profile-batch-size 64 \
  --similar-profile-search-batch-size 256 \
  --similar-profile-max-length 256 \
  --similar-profile-device cuda
```

当前已构建结果：

| split | rows | learned Top100 target hit |
|---|---:|---:|
| train | 4502 | 0.750333 |
| val | 579 | 0.747841 |
| test | 884 | 0.720588 |

`allow_missing_refined=true` 是当前 CLSPRec NYC 版本的正常状态，`refined_text` 为空，不再依赖旧的 offline refiner 输出。

## Raw Context Compact

当前 compact 命令：

```bash
cd /mnt/data/users/yyl/TMP
PY=/mnt/data/users/yyl/miniconda3/envs/poi_data/bin/python

$PY src/data_build/compress_joined_raw_context.py \
  --input-dir retrieval_assets_clsprec/NYC/joined_poi_classification_exp2_top500_rescored \
  --top-k 100 \
  --splits train val test \
  --trajectory-lines 12 \
  --transitions 6 \
  --nearby 6 \
  --top-categories 4 \
  --revisited-pois 4 \
  --tokenizer models/Llama-3.2-1B-Instruct
```

压缩后 candidate 起始 token 统计：

| split | mean | p95 | max |
|---|---:|---:|---:|
| train | 770.47 | 968.00 | 1027 |
| val | 748.22 | 939.10 | 998 |
| test | 729.87 | 918.85 | 983 |

因此当前训练使用 `max_length: 1440` 是合理的，通常不会在 candidate hypothesis 前发生严重截断。

## 输入模板

当前训练配置：

```yaml
input_template: semantic_profile_simuser_v1
```

单 candidate 输入包含：

```text
[TASK=CANDIDATE_RERANK]
[RECENT_CONTEXT]
[USER_SEMANTIC_PROFILE]
[SIMILAR_USER_SEMANTIC_PROFILE]  # 仅触发时出现
[GRAPH_VIEW]
[POI_HYPOTHESIS]
```

`USER_SEMANTIC_PROFILE` 只保留四类长期画像：

```text
Long-term category affinity
Revisit affinity
Geo routine
Temporal routine
```

`SIMILAR_USER_SEMANTIC_PROFILE` 的触发逻辑：

- 数据构建阶段为所有样本增强相似用户画像。
- 模型输入阶段仅当用户自身画像不足且相似画像有信号时进入文本。
- 否则该字段为空。

GraphRAG / learned-pool 的 `rank / score / sources` 主要走结构化 `graph_features` 和 rank prior，不作为长文本冗余写入 prompt。

## Routed TeamLoRA 与 Graph Prior

训练脚本：

```text
src/poi_reranker/train_teamlora_reranker_raat.py
```

LoRA 注入模块：

```text
q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
```

当前核心参数：

```yaml
lora_r: 8
lora_alpha: 16
lora_num: 3
use_graph_prior: true
graph_prior_type: rank_log
residual_alpha_init: 0.3
residual_l2: 0.001
residual_bound_mode: tanh
residual_bound_value: 1.0
```

打分形式：

```text
final_score = graph_rank_prior + bound(alpha * residual_score)
```

这里的 `graph_rank_prior` 对 experiment2 数据来说已经是 learned Top100 候选池内的 rank prior，而不是原始 GraphRAG Top100 的 rank prior。

## RAAT

当前默认：

```yaml
raat_mode: target_mask_2view
raat_memory_mode: recompute_hard
```

含义：

```text
clean view:
  正常输入 + 正常 graph_features

target_mask view:
  target candidate 的 graph view 文本与 graph_features 被 mask

recompute_hard:
  先 no_grad 选择更难 view，再只对该 view 重算并反传
```

如果云端显存紧张，优先尝试：

```yaml
raat_mode: none
train_negatives: 11
hard_negatives: 8
max_length: 1280
```

## 训练命令

NYC experiment2 当前推荐：

```bash
cd /mnt/data/users/yyl/TMP
mkdir -p logs
PY=/mnt/data/users/yyl/miniconda3/envs/poi_data/bin/python

CUDA_VISIBLE_DEVICES=0 \
NCCL_P2P_DISABLE=1 \
NCCL_IB_DISABLE=1 \
nohup $PY src/poi_reranker/train_teamlora_reranker_raat.py \
  --config config/train_nyc_exp2_top500_rescored_semprofile_simuser_v2.yaml \
  > logs/train_nyc_exp2_top500_rescored_step800_eval600800_v1.log 2>&1 &
```

配置文件中的主要训练参数：

```yaml
top_k: 100
max_length: 1440
train_negatives: 15
hard_negatives: 12
batch_groups: 1
grad_accum: 8
max_steps: 800
lr: 0.0002
bf16: true
gradient_checkpointing: true
attn_implementation: sdpa
val_hit_only: true
eval_candidate_limit: null
eval_candidate_batch_size: 2
eval_final_only: false
eval_at_steps: [600, 800]
```

训练阶段 val 口径是 `val_hit_only=true`，用于观察 Top100 已召回样本内的重排能力。当前配置只在 600 和 800 step 做两次命中集评估，`best` checkpoint 仍按验证 MRR 自动保存最优一次。

## 独立评估

训练结束后评估 best checkpoint 的 val hit-only：

```bash
cd /mnt/data/users/yyl/TMP
PY=/mnt/data/users/yyl/miniconda3/envs/poi_data/bin/python

CUDA_VISIBLE_DEVICES=0 \
NCCL_P2P_DISABLE=1 \
NCCL_IB_DISABLE=1 \
$PY src/poi_reranker/evaluate_teamlora_reranker.py \
  --model-dir models/poi-teamlora-reranker-nyc-exp2-top500rescored-routed-lora3-semprofile-simuser-l1440-priorres-raatmem-step800-eval600800-v1 \
  --checkpoint best \
  --val-hit-only \
  --eval-candidate-batch-size 2 \
  --bf16 \
  --attn-implementation sdpa
```

最终 full-test 口径不要加 `--val-hit-only`：

```bash
cd /mnt/data/users/yyl/TMP
PY=/mnt/data/users/yyl/miniconda3/envs/poi_data/bin/python
MODEL=models/poi-teamlora-reranker-nyc-exp2-top500rescored-routed-lora3-semprofile-simuser-l1440-priorres-raatmem-step800-eval600800-v1

CUDA_VISIBLE_DEVICES=0 \
NCCL_P2P_DISABLE=1 \
NCCL_IB_DISABLE=1 \
$PY src/poi_reranker/evaluate_teamlora_reranker.py \
  --model-dir $MODEL \
  --checkpoint best \
  --val-joined retrieval_assets_clsprec/NYC/joined_poi_classification_exp2_top500_rescored/test_joined_top100.parquet \
  --semantic-map retrieval_assets_clsprec/NYC/double_llm/semantic_poi_ids.jsonl \
  --eval-candidate-batch-size 2 \
  --bf16 \
  --attn-implementation sdpa \
  --output-json $MODEL/eval_best_test_full_top100.json
```

全量 test 指标会同时反映候选池覆盖上限 `candidate_hit`。当前 experiment2 test candidate_hit 是 `0.720588`，这是 reranker 在固定 Top100 内能达到的上限。

## 指标口径

当前评估输出：

```text
top1 / top5 / top10 / top20
ndcg1 / ndcg5 / ndcg10
mrr
candidate_hit
conditional_top1 / conditional_top5 / conditional_top10 / conditional_top20
conditional_ndcg1 / conditional_ndcg5 / conditional_ndcg10
conditional_mrr
```

解释：

```text
candidate_hit
```

target 是否进入固定 Top100 候选池。target 不在候选池时，reranker 不可能命中。

```text
conditional_*
```

只在 target 已进入 Top100 的样本上计算，反映纯重排能力。

```text
top* / ndcg* / mrr
```

全量口径，受候选池覆盖率和重排能力共同影响。

## Baseline 对比状态

已完成的 NYC v5 是旧 GraphRAG Top100 数据上的模型：

```text
models/poi-teamlora-reranker-nyc-routed-lora3-semprofile-simuser-l1440-priorres-raatmem-step600-v5
```

旧 v5 full-test 结果：

| Model | evaluated | Top5 | Top10 | NDCG5 | NDCG10 | MRR |
|---|---:|---:|---:|---:|---:|---:|
| TMP v5 routed TeamLoRA | 884 | 0.3710 | 0.4525 | 0.2808 | 0.3068 | 0.2705 |
| GETNext TMP-adapted | 855 | 0.3368 | 0.4421 | 0.2498 | 0.2840 | 0.2457 |
| CLSPRec fair-best | 855 | 0.2795 | 0.3427 | 0.1990 | 0.2190 | 0.1852 |

experiment2 的 Top500-rescored TeamLoRA 还需要重新训练，不能直接用旧 v5 checkpoint 宣称最终提升。

## 跨机器拉取

已有仓库：

```bash
cd /path/to/TMP
git fetch origin experiment2
git checkout experiment2
git pull --ff-only origin experiment2
```

新机器：

```bash
git clone -b experiment2 https://github.com/Jack-J-C/TMP.git
cd TMP
```

本分支已包含直接训练所需的 NYC experiment2 compact parquet 和配置；不包含模型权重，也不包含大体积 wide candidate JSONL。
