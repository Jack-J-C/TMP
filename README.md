# TMP POI Reranking Project

TMP 当前主线是 **Semantic-ID GraphRAG Top100 候选池 + Llama-3.2-1B routed TeamLoRA reranker**。项目目标不是生成 POI，也不是全量 POI 分类，而是在每条样本的 GraphRAG Top100 候选内进行 candidate-wise/listwise 重排。

本 README 以当前 `src/` 代码、`config/` 配置和 `experiment` 分支训练流程为准。旧 NewYork 单城、显式 `pref/graph/refine` 三专家、`REFINED_EVIDENCE` 主线等描述已经过时。

## 当前架构

```text
raw check-in trajectory / POI metadata
        |
        v
city-level TMP-format data: NYC / SIN / TKY
        |
        v
Semantic-ID GraphRAG retrieval
  - transition / geo / history / semantic signals
  - produce Top100 candidate POIs
        |
        v
joined parquet
  - compact raw context
  - USER_SEMANTIC_PROFILE
  - optional SIMILAR_USER_SEMANTIC_PROFILE
  - POI_HYPOTHESIS for each candidate
  - structured graph rank / score / source features
        |
        v
single-input candidate text template
        |
        v
Frozen Llama-3.2-1B-Instruct
  q/k/v/o/gate/up/down Linear -> RoutedTeamLoRALinear
  lora_num=3 implicit LoRA experts
  router(hidden_state) -> expert weights
        |
        v
last-token hidden + graph_features
        |
        v
GraphRAG rank prior + bounded residual correction
        |
        v
Top100 candidate reranking
```

核心变化：当前版本不再对每个 candidate 执行 `pref`、`graph`、`refine` 三次 encoder forward。三专家概念已经改为 TeamLoRA 原版风格的 **层内隐式 LoRA 专家路由**，每个 candidate 只构造一份输入文本。

## 当前实验范围

当前主线三城：

```text
NYC: retrieval_assets_clsprec/NYC
SIN: retrieval_assets_clsprec/SIN
TKY: retrieval_assets_getnext_clsprec/TKY
```

训练配置：

```text
config/train_nyc_semprofile_simuser_v2.yaml
config/train_sin_semprofile_simuser_v2.yaml
config/train_tky_semprofile_simuser_v2.yaml
```

训练输出目录：

```text
models/poi-teamlora-reranker-nyc-routed-lora3-semprofile-simuser-l1440-priorres-raatmem-step600-v5
models/poi-teamlora-reranker-sin-routed-lora3-semprofile-simuser-l1440-priorres-raatmem-step600-v5
models/poi-teamlora-reranker-tky-routed-lora3-semprofile-simuser-l1440-priorres-raatmem-step600-v5
```

注意：`config/*.yaml` 表示当前准备继续跑的配置；已完成的 NYC v5 best checkpoint 以 checkpoint 内的 `metadata.json` 为准。NYC v5 best 实际训练时使用 `train_negatives=10`、`hard_negatives=8`，而当前 YAML 已调整为 `15/12`。如果要保留旧 v5 结果，继续训练前请改 `output_dir`。

`models/` 默认不推送到 git。另一台机器需要手动准备：

```text
models/Llama-3.2-1B-Instruct
```

## 关键目录

```text
config/
  train_nyc_semprofile_simuser_v2.yaml
  train_sin_semprofile_simuser_v2.yaml
  train_tky_semprofile_simuser_v2.yaml

src/
  data_build/
    数据转换、joined parquet 构建、raw context 压缩、similar user profile 构建
  graphrag/
    semantic POI id 构建、GraphRAG TopK 候选生成、历史 C1 reranker 实验
  poi_reranker/
    当前主线 reranker 的 group 构造、训练、独立评估
  refine_prompt/
    历史 prompt refiner 训练/生成/清洗流程
  poi_classification/
    历史 full-POI classifier baseline，不是当前主线

retrieval_assets_clsprec/
  NYC/
    joined_poi_classification/
    double_llm/semantic_poi_ids.jsonl
  SIN/
    joined_poi_classification/
    double_llm/semantic_poi_ids.jsonl

retrieval_assets_getnext_clsprec/
  TKY/
    joined_poi_classification/
    double_llm/semantic_poi_ids.jsonl

docs/
  experiment_branch_training.md
```

## 当前训练数据

训练脚本直接读取 joined parquet：

```text
retrieval_assets_clsprec/NYC/joined_poi_classification/train_joined_top100.parquet
retrieval_assets_clsprec/NYC/joined_poi_classification/val_joined_top100.parquet
retrieval_assets_clsprec/NYC/joined_poi_classification/test_joined_top100.parquet
retrieval_assets_clsprec/SIN/joined_poi_classification/train_joined_top100.parquet
retrieval_assets_clsprec/SIN/joined_poi_classification/val_joined_top100.parquet
retrieval_assets_clsprec/SIN/joined_poi_classification/test_joined_top100.parquet
retrieval_assets_getnext_clsprec/TKY/joined_poi_classification/train_joined_top100.parquet
retrieval_assets_getnext_clsprec/TKY/joined_poi_classification/val_joined_top100.parquet
retrieval_assets_getnext_clsprec/TKY/joined_poi_classification/test_joined_top100.parquet
```

每条样本会由 [build_teamlora_reranker_groups.py](src/poi_reranker/build_teamlora_reranker_groups.py) 转为 group：

```text
sample_id
city
target_poi_id
target_in_candidates
pref_text                         compact raw context
user_semantic_profile              用户自身长期画像
similar_user_semantic_profile      相似用户长期画像，按触发规则进入模型
candidates[]
  poi_id
  rank / score / sources
  semantic_id / category / geo_cell
  candidate_hypothesis_text
  label
```

`candidate_hypothesis_text` 是当前 candidate 的语义描述，形如：

```text
id=v123; semantic_id=...; category=...; geo_cell=...; coord=...
```

GraphRAG 的 `rank / score / sources` 主要作为结构化 `graph_features` 和 rank prior 使用，不再把详细 `rank/score/sources` 当成大段文本塞进 prompt。

## 输入模板

当前配置使用：

```yaml
input_template: semantic_profile_simuser_v1
```

单个 candidate 的输入大致为：

```text
[TASK=CANDIDATE_RERANK]
Score whether this POI is the user's next check-in...

[RECENT_CONTEXT]
compact raw trajectory context

[USER_SEMANTIC_PROFILE]
Long-term category affinity
Revisit affinity
Geo routine
Temporal routine

[SIMILAR_USER_SEMANTIC_PROFILE]
only appears when user profile is insufficient and similar profile has signal

[GRAPH_VIEW]
Structured graph rank, score, and source features are provided separately...

[POI_HYPOTHESIS]
candidate semantic id / category / geo / coord
```

`SIMILAR_USER_SEMANTIC_PROFILE` 的触发逻辑在训练脚本中完成：

- 用户自身画像不足时才考虑启用。
- 相似用户画像需要至少有足够语义信号。
- 否则该字段为空，不进入模型输入。

## Routed TeamLoRA

核心训练脚本：

```text
src/poi_reranker/train_teamlora_reranker_raat.py
```

当前 LoRA 注入目标：

```text
q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
```

当前 TeamLoRA 形式：

```text
RoutedTeamLoRALinear
  base Linear frozen
  shared lora_A: in_features -> r * lora_num
  lora_B[0..lora_num-1]: r -> out_features
  ShapleyRouter: hidden_state -> softmax(lora_num)
```

当前三城配置：

```yaml
lora_r: 8
lora_alpha: 16
lora_num: 3
lora_dropout: 0.05
```

这比旧版显式三路专家更省显存：旧版每个 candidate 要跑 `pref/graph/refine` 三次 encoder；当前 routed 版每个 candidate 只跑一次 encoder，专家协作发生在每个 Linear 层内部。

## Graph Prior 与 Residual

当前 reranker 不是完全替代 GraphRAG，而是在 GraphRAG 排名基础上学习 residual correction：

```yaml
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

`graph_features` 由候选的 rank、score、masked flag 和 source bits 构成：

```text
1/log1p(rank)
log1p(score)/10
rank_norm
masked
source bits: transition / geo / history / semantic
```

## RAAT

当前 YAML 默认：

```yaml
raat_mode: target_mask_2view
raat_memory_mode: recompute_hard
```

RAAT 保留为训练鲁棒性策略，但已经适配单路 routed TeamLoRA：

```text
clean view:
  正常输入 + 正常 graph_features

target_mask view:
  target candidate 的 graph view 文本与 graph_features 被 mask

loss:
  select the harder view with no_grad, then recompute only that view for backward
```

可选值：

```text
none
target_mask_2view
graph_3view
```

显存说明：旧的 joint RAAT 会同时保留 clean/masked 两个 view 的训练图。当前配置使用 `raat_memory_mode: recompute_hard`，先用 `no_grad` 选择更难 view，再只对该 view 重算并反传，避免两个反传图叠加。如果还需要进一步压显存，优先尝试：

```yaml
raat_mode: none
train_negatives: 11
hard_negatives: 8
max_length: 1280
```

## 当前训练参数

三城 YAML 的主要参数一致：

```yaml
top_k: 100
max_length: 1440
train_negatives: 15
hard_negatives: 12
batch_groups: 1
grad_accum: 8
max_steps: 600
lr: 0.0002
bf16: true
gradient_checkpointing: true
attn_implementation: sdpa
eval_final_only: true
eval_steps: 600
save_steps: 600
```

训练时每个 group 大约是：

```text
1 positive + 15 negatives = 16 candidates
```

已完成的 NYC v5 best checkpoint 是上一版 negatives 设置：

```text
train_negatives: 10
hard_negatives: 8
```

在 `max_length=1440` 且 `raat_mode=target_mask_2view` 下，单步显存主要由：

```text
candidates × max_length × RAAT views
```

决定，而不是 LoRA 参数量决定。

## 评估口径

当前 YAML 默认：

```yaml
val_hit_only: true
eval_candidate_limit: null
eval_candidate_batch_size: 2
```

含义：

- 只在 val 中 target 已经进入 GraphRAG Top100 的样本上评估。
- 每条样本评估完整 100 个候选。
- `eval_candidate_batch_size: 2` 只是降低评估显存峰值，不改变候选数量。

训练阶段的 val 评估口径是 `val_hit_only=true`，即只衡量 GraphRAG Top100 已召回样本上的重排能力。最终 test 对比使用全量样本口径，不加 `--val-hit-only`，因此会同时反映候选池召回上限 `candidate_hit`。

当前指标：

```text
top1 / top5 / top10 / top20
ndcg1 / ndcg5 / ndcg10
mrr
candidate_hit
conditional_top1 / conditional_top5 / conditional_top10 / conditional_top20
conditional_ndcg1 / conditional_ndcg5 / conditional_ndcg10
conditional_mrr
```

指标解释：

```text
candidate_hit
```

候选池召回上限。target 不在 Top100 中时，reranker 无法命中。

```text
conditional_*
```

只在 target 已经进入候选池的样本上计算，反映 reranker 的重排能力。

```text
top* / ndcg* / mrr
```

全量口径，等于 conditional 指标乘以 `candidate_hit`。如果启用 `val_hit_only: true`，则评估集合已经过滤为 Top100 命中样本。

## 启动训练

NYC：

```bash
cd /mnt/data/users/yyl/TMP
mkdir -p logs
PY=/mnt/data/users/yyl/miniconda3/envs/poi_data/bin/python

CUDA_VISIBLE_DEVICES=0 \
NCCL_P2P_DISABLE=1 \
NCCL_IB_DISABLE=1 \
nohup $PY src/poi_reranker/train_teamlora_reranker_raat.py \
  --config config/train_nyc_semprofile_simuser_v2.yaml \
  > logs/train_nyc_routed_lora3_semprofile_simuser_v2_yaml.log 2>&1 &
```

SIN：

```bash
cd /mnt/data/users/yyl/TMP
mkdir -p logs
PY=/mnt/data/users/yyl/miniconda3/envs/poi_data/bin/python

CUDA_VISIBLE_DEVICES=0 \
NCCL_P2P_DISABLE=1 \
NCCL_IB_DISABLE=1 \
nohup $PY src/poi_reranker/train_teamlora_reranker_raat.py \
  --config config/train_sin_semprofile_simuser_v2.yaml \
  > logs/train_sin_routed_lora3_semprofile_simuser_v2_yaml.log 2>&1 &
```

TKY：

```bash
cd /mnt/data/users/yyl/TMP
mkdir -p logs
PY=/mnt/data/users/yyl/miniconda3/envs/poi_data/bin/python

CUDA_VISIBLE_DEVICES=0 \
NCCL_P2P_DISABLE=1 \
NCCL_IB_DISABLE=1 \
nohup $PY src/poi_reranker/train_teamlora_reranker_raat.py \
  --config config/train_tky_semprofile_simuser_v2.yaml \
  > logs/train_tky_routed_lora3_semprofile_simuser_v2_yaml.log 2>&1 &
```

查看日志：

```bash
tail -f logs/train_nyc_routed_lora3_semprofile_simuser_v2_yaml.log
```

## 独立评估

训练结束后评估 best checkpoint：

```bash
cd /mnt/data/users/yyl/TMP
PY=/mnt/data/users/yyl/miniconda3/envs/poi_data/bin/python

CUDA_VISIBLE_DEVICES=0 \
NCCL_P2P_DISABLE=1 \
NCCL_IB_DISABLE=1 \
$PY src/poi_reranker/evaluate_teamlora_reranker.py \
  --model-dir models/poi-teamlora-reranker-nyc-routed-lora3-semprofile-simuser-l1440-priorres-raatmem-step600-v5 \
  --checkpoint best \
  --val-hit-only \
  --eval-candidate-batch-size 2 \
  --bf16 \
  --attn-implementation sdpa
```

NYC 全量 test 口径：

```bash
cd /mnt/data/users/yyl/TMP
PY=/mnt/data/users/yyl/miniconda3/envs/poi_data/bin/python

CUDA_VISIBLE_DEVICES=0 \
NCCL_P2P_DISABLE=1 \
NCCL_IB_DISABLE=1 \
$PY src/poi_reranker/evaluate_teamlora_reranker.py \
  --model-dir models/poi-teamlora-reranker-nyc-routed-lora3-semprofile-simuser-l1440-priorres-raatmem-step600-v5 \
  --checkpoint best \
  --val-joined retrieval_assets_clsprec/NYC/joined_poi_classification/test_joined_top100.parquet \
  --semantic-map retrieval_assets_clsprec/NYC/double_llm/semantic_poi_ids.jsonl \
  --eval-candidate-batch-size 2 \
  --bf16 \
  --attn-implementation sdpa \
  --output-json models/poi-teamlora-reranker-nyc-routed-lora3-semprofile-simuser-l1440-priorres-raatmem-step600-v5/eval_best_test_full_top100.json
```

如果要完整 val/test 口径，不要加：

```text
--val-hit-only
--eval-candidate-limit
--max-val-groups
```

## NYC Baseline 对比

当前 baseline 适配只改变数据层，不改 CLSPRec / GETNext 模型架构。适配脚本：

```text
scripts/adapt_nyc_baselines.py
scripts/run_clsprec_tmp_nyc.py
scripts/evaluate_getnext_tmp.py
scripts/watch_gpu0_run_clsprec_nyc.sh
scripts/watch_gpu1_run_getnext_nyc.sh
```

GETNext 使用 TMP NYC train/val/test split，并修正 `trajectory_id`，使其以 `user_id` 开头，因为 GETNext 源码会用 `trajectory_id.split("_")[0]` 取用户。

CLSPRec fair-best 版本使用 TMP NYC split，并放宽原始 CLSPRec 的强过滤，使训练/验证/测试样本数更接近 TMP/GETNext：

```text
current trajectory length >= 3
at least 1 previous trajectory
no 7-day history window
max recent history count = 7
no future trajectory leakage
```

NYC full-test 当前结果：

| Model | evaluated | Top5 | Top10 | NDCG5 | NDCG10 | MRR |
|---|---:|---:|---:|---:|---:|---:|
| TMP v5 routed TeamLoRA | 884 | 0.3710 | 0.4525 | 0.2808 | 0.3068 | 0.2705 |
| GETNext TMP-adapted | 855 | 0.3368 | 0.4421 | 0.2498 | 0.2840 | 0.2457 |
| CLSPRec fair-best | 855 | 0.2795 | 0.3427 | 0.1990 | 0.2190 | 0.1852 |

结果文件：

```text
models/poi-teamlora-reranker-nyc-routed-lora3-semprofile-simuser-l1440-priorres-raatmem-step600-v5/eval_best_test_full_top100.json
/mnt/data/users/yyl/GETNext/runs/train/tmp_nyc_getnext/test_metrics.json
/mnt/data/users/yyl/CLSPRec/results/TMP_NYC_CLSPRec_fair_best_mrr_e50_eval5_test_metrics.json
```

表中 TMP v5 是 GraphRAG Top100 reranker，GETNext / CLSPRec 是按 TMP NYC split 做的数据层适配 baseline。更稳妥的表述是：在当前 TMP NYC preprocessing / full-test evaluation protocol 下，TMP v5 超过这两个适配 baseline。

## Experiment 分支跨机器运行

详细说明见：

```text
docs/experiment_branch_training.md
```

当前 `experiment` 分支只要求能直接运行训练，不要求包含 raw data、预处理产物或模型权重。

最小训练资产：

```text
src/poi_reranker/train_teamlora_reranker_raat.py
src/poi_reranker/build_teamlora_reranker_groups.py
src/poi_reranker/evaluate_teamlora_reranker.py
config/*.yaml
retrieval_assets_clsprec/*/joined_poi_classification/{train,val,test}_joined_top100.parquet
retrieval_assets_clsprec/*/double_llm/semantic_poi_ids.jsonl
retrieval_assets_getnext_clsprec/TKY/joined_poi_classification/{train,val,test}_joined_top100.parquet
retrieval_assets_getnext_clsprec/TKY/double_llm/semantic_poi_ids.jsonl
```

另一台机器需要额外准备：

```text
models/Llama-3.2-1B-Instruct
```

## Conda 环境

当前环境可参考：

```text
python==3.10
torch==2.7.1
transformers==5.4.0
tokenizers==0.22.2
safetensors==0.7.0
accelerate==1.13.0
numpy==2.2.6
pandas==2.3.3
pyarrow==24.0.0
PyYAML==6.0.3
tqdm==4.67.3
scikit-learn==1.7.2
sentence-transformers==5.3.0
```

安装示例：

```bash
conda create -n poi_data python=3.10 -y
conda activate poi_data

pip install \
  torch==2.7.1 \
  transformers==5.4.0 \
  tokenizers==0.22.2 \
  safetensors==0.7.0 \
  accelerate==1.13.0 \
  numpy==2.2.6 \
  pandas==2.3.3 \
  pyarrow==24.0.0 \
  PyYAML==6.0.3 \
  tqdm==4.67.3 \
  scikit-learn==1.7.2 \
  sentence-transformers==5.3.0
```

如果目标机器 CUDA/PyTorch 版本不同，优先按目标机器安装匹配的 PyTorch，再安装其余 Python 包。

## 数据构建与压缩

当前 `experiment` 分支训练不要求重新预处理数据。若需要重建 joined parquet，核心脚本是：

```text
src/data_build/build_joined_poi_classification_data.py
src/data_build/compress_joined_raw_context.py
```

当前 compact 默认规则：

```text
trajectory_lines: 8
transitions: 4
nearby: 4
top_categories: 3
revisited_pois: 3
```

历史上也使用过更宽松压缩，例如：

```text
trajectory_lines: 12
transitions: 6
nearby: 6
top_categories: 4
revisited_pois: 4
```

当前三城已构建好的 parquet 可直接用于训练。

## Git 注意事项

不要推送：

```text
models/
logs/
raw_data/
dataset_clsprec/
dataset_getnext_clsprec/
retrieval_assets_*/evidence/
retrieval_assets_*/semantic_poi_sft/
retrieval_assets_*/poi_sft/
*_fullraw.parquet
```

可以推送当前训练所需 parquet 和 semantic map。由于 `.jsonl` 通常被忽略，semantic map 需要强制 add：

```bash
git add -f retrieval_assets_clsprec/NYC/double_llm/semantic_poi_ids.jsonl
git add -f retrieval_assets_clsprec/SIN/double_llm/semantic_poi_ids.jsonl
git add -f retrieval_assets_getnext_clsprec/TKY/double_llm/semantic_poi_ids.jsonl
```

提交当前 README 和 routed TeamLoRA 相关改动：

```bash
cd /mnt/data/users/yyl/TMP

git add README.md docs/experiment_branch_training.md
git add config/train_nyc_semprofile_simuser_v2.yaml config/train_sin_semprofile_simuser_v2.yaml config/train_tky_semprofile_simuser_v2.yaml
git add src/poi_reranker/train_teamlora_reranker_raat.py src/poi_reranker/evaluate_teamlora_reranker.py
git add scripts/adapt_nyc_baselines.py scripts/run_clsprec_tmp_nyc.py scripts/evaluate_getnext_tmp.py
git add scripts/watch_gpu0_run_clsprec_nyc.sh scripts/watch_gpu1_run_getnext_nyc.sh

git add retrieval_assets_clsprec/NYC/joined_poi_classification/train_joined_top100.parquet
git add retrieval_assets_clsprec/NYC/joined_poi_classification/val_joined_top100.parquet
git add retrieval_assets_clsprec/NYC/joined_poi_classification/test_joined_top100.parquet
git add retrieval_assets_clsprec/SIN/joined_poi_classification/train_joined_top100.parquet
git add retrieval_assets_clsprec/SIN/joined_poi_classification/val_joined_top100.parquet
git add retrieval_assets_clsprec/SIN/joined_poi_classification/test_joined_top100.parquet
git add retrieval_assets_getnext_clsprec/TKY/joined_poi_classification/train_joined_top100.parquet
git add retrieval_assets_getnext_clsprec/TKY/joined_poi_classification/val_joined_top100.parquet
git add retrieval_assets_getnext_clsprec/TKY/joined_poi_classification/test_joined_top100.parquet

git add -f retrieval_assets_clsprec/NYC/double_llm/semantic_poi_ids.jsonl
git add -f retrieval_assets_clsprec/SIN/double_llm/semantic_poi_ids.jsonl
git add -f retrieval_assets_getnext_clsprec/TKY/double_llm/semantic_poi_ids.jsonl

git status
git commit -m "Document routed TeamLoRA experiment workflow"
git push origin experiment
```
