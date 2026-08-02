# TMP POI 推荐实验项目

TMP 当前主线位于 `experiment3`。最新研究重点是：

> 在 `exp2 Top500` 候选池基础上做在线 Top500->Top100 压缩，并用 BERT routed TeamLoRA 编码用户轨迹；随后通过候选感知 token attention、DIN-lite 历史行为注意力和 DCNv2 对齐头融合候选结构特征，输出最终 Top100 排序。

需要特别注意：**当前最新 BERT v5 DIN-lite 仍未超过离线 learned_pool 原始顺序**。v5 的意义在于验证“在线候选池 + 轻量轨迹编码 + 行为对齐”方向开始有小幅正向变化，而不是已经替代 learned_pool。

## 当前结论

NYC test 全量样本数为 `884`。当前最强固定基线仍是离线 learned_pool：

| 方法 | step | top5 | top10 | ndcg5 | ndcg10 | mrr | candidate_hit |
|---|---:|---:|---:|---:|---:|---:|---:|
| 离线 learned_pool 原始顺序 | - | 0.395928 | 0.481900 | 0.292651 | 0.320440 | 0.280377 | 0.720588 |
| Llama online pool | 800 final | 0.347285 | 0.451357 | 0.266927 | 0.300680 | 0.261940 | 0.587104 |
| BERT v3 unfreeze2 | 400 best | 0.351810 | 0.453620 | 0.263244 | 0.296324 | 0.256885 | 0.640271 |
| BERT v4 balanced | 400 best | 0.346154 | 0.438914 | 0.256468 | 0.286555 | 0.248452 | 0.615385 |
| BERT v5 DIN-lite | 600 best | 0.360860 | 0.451357 | 0.267823 | 0.296921 | 0.258366 | 0.640271 |

对比结论：

- `BERT v5 DIN-lite` 相对 `BERT v3 unfreeze2` 有小幅提升：`top5 +0.009050`，`ndcg5 +0.004578`，`mrr +0.001481`。
- `BERT v5 DIN-lite` 的 `candidate_hit` 与 v3 持平，为 `0.640271`，说明新增 DIN-lite 没有进一步扩大 Top100 覆盖。
- `Llama online pool` 的 `candidate_hit` 更低，但命中后的排序能力更强，因此最终 `mrr` 仍高于 BERT v5。
- 离线 learned_pool 的 `candidate_hit=0.720588`，仍明显高于所有 online pool 版本，是当前最强可用基线。

## 最新模型：BERT v5 DIN-lite

最新配置：

- [config/train_nyc_exp3_online_pool_bert_dcnv2_dinlite_v5.yaml](/mnt/data/users/yyl/TMP/config/train_nyc_exp3_online_pool_bert_dcnv2_dinlite_v5.yaml)

最新输出：

- [models/poi-bert-mobility-encoder-nyc-exp3-online-pool-dcnv2-dinlite-l512-step800-eval200-v5/eval_history.jsonl](/mnt/data/users/yyl/TMP/models/poi-bert-mobility-encoder-nyc-exp3-online-pool-dcnv2-dinlite-l512-step800-eval200-v5/eval_history.jsonl)
- [models/poi-bert-mobility-encoder-nyc-exp3-online-pool-dcnv2-dinlite-l512-step800-eval200-v5/best/metadata.json](/mnt/data/users/yyl/TMP/models/poi-bert-mobility-encoder-nyc-exp3-online-pool-dcnv2-dinlite-l512-step800-eval200-v5/best/metadata.json)

核心参数：

| 参数 | 当前值 |
|---|---:|
| `base_model` | `models/bert-base-uncased` |
| `pool_k` | 500 |
| `top_k` | 100 |
| `max_length` | 512 |
| `lora_num` | 3 |
| `lora_r` | 16 |
| `lora_alpha` | 32 |
| `unfreeze_last_n_layers` | 2 |
| `relation_dim` | 32 |
| `event_numeric_dim` | 8 |
| `alignment_mode` | `dcnv2` |
| `candidate_token_attention` | true |
| `din_history_attention` | true |
| `keep_loss_weight` | 5.0 |
| `final_loss_weight` | 1.0 |

## 架构

当前 v5 主线：

```text
Top500 candidate pool
        |
        |  pool features
        v
online linear pool compressor
        |
        |  hard Top100 selection
        v
selected Top100 candidates
        |
        +---------------------------+
        |                           |
        v                           v
candidate anchor encoder      BERT mobility encoder
category / geo / semantic     structured trajectory tokens
rank / score / source         anonymous routed TeamLoRA
relation features             unfreeze last 2 BERT layers
        |                           |
        +-------------+-------------+
                      |
                      v
candidate-token attention
DIN-lite history attention
DCNv2 fusion scorer
                      |
                      v
final Top100 ranking
```

### BERT 轨迹编码

v5 使用 `bert-base-uncased` 作为用户轨迹编码器。BERT 主体冻结，但最后 2 层解冻；同时在 BERT 的 attention 和 FFN 线性层上注入匿名三专家 routed LoRA。

BERT LoRA 目标模块：

- `attention.self.query`
- `attention.self.key`
- `attention.self.value`
- `attention.output.dense`
- `intermediate.dense`
- `output.dense`

当前 BERT 版替换 `72` 个 LoRA 模块。三专家是匿名专家，专家没有显式语义标签，由输入动态路由。

### 行为输入

v5 不再把 `POI_HYPOTHESIS` 或候选长文本输入 LLM/BERT。BERT 只看用户行为相关信息。

结构化轨迹事件示例：

```text
EVT[-1] POI=v12605 CAT=OTHER_GREAT_OUTDOORS GEO=LAT40_71_LON_73_95 WEEKDAY=Fri SLOT=s40 GAP=gap_long PREV_CAT=BUS_STATION CAT_TRANS=BUS_STATION->OTHER_GREAT_OUTDOORS GEO_MOVE=move POI_FREQ=1 CAT_FREQ=1 GEO_FREQ=1
EVT[-2] POI=v15188 CAT=BUS_STATION GEO=LAT40_69_LON_73_92 WEEKDAY=Fri SLOT=s25 GAP=gap_tiny PREV_CAT=DRUGSTORE_PHARMACY CAT_TRANS=DRUGSTORE_PHARMACY->BUS_STATION GEO_MOVE=stay POI_FREQ=1 CAT_FREQ=1 GEO_FREQ=3
```

同时保留：

- `USER_SEMANTIC_PROFILE`
- `SIMILAR_USER_SEMANTIC_PROFILE`

相似用户画像不是无条件注入，只有用户自身画像不足且相似用户画像有信号时才注入。

### 候选结构特征

候选侧不输入长文本，而是使用结构化 anchor。

候选本体特征包括：

- rank prior
- rank norm
- graph score log
- source count
- source flags
- source family statistics
- popularity bucket
- category bucket
- geo bucket
- semantic id bucket

候选-用户关系特征包括：

- 是否等于 last / last2 POI。
- 是否与 last / last2 类别一致。
- 是否命中最近类别、最近 geo。
- 是否命中用户画像 category / geo / revisit / temporal。
- 是否命中相似用户画像信号。
- 候选 POI/category/geo 在历史轨迹中的出现次数。
- 候选与历史轨迹的 recency。
- 是否是同类迁移、同地异类、探索候选。
- last category 到候选 category 的转移信号。

### DIN-lite 历史注意力

v5 新增 DIN-lite 分支：

```text
candidate anchor = query
history event anchors = keys / values
attention(candidate, history)
        |
candidate-aware history vector
        |
gate(history vector, BERT token-attended user vector, candidate anchor)
        |
DCNv2 scorer
```

这个模块的目的不是扩大候选池覆盖，而是让候选在排序阶段主动关注相关历史行为。

## 数据口径

当前 NYC 使用 exp2 top500-rescored 数据资产。

训练/验证/测试 joined parquet：

- `retrieval_assets_clsprec/NYC/joined_poi_classification_exp2_top500_rescored/train_joined_top100.parquet`
- `retrieval_assets_clsprec/NYC/joined_poi_classification_exp2_top500_rescored/val_joined_top100.parquet`
- `retrieval_assets_clsprec/NYC/joined_poi_classification_exp2_top500_rescored/test_joined_top100.parquet`

Top500 在线候选池：

- `retrieval_assets_clsprec/NYC/double_llm/graphrag_semantic_edges_v2_top500_train_Candidates.jsonl`
- `retrieval_assets_clsprec/NYC/double_llm/graphrag_semantic_edges_v2_top500_val_Candidates.jsonl`
- `retrieval_assets_clsprec/NYC/double_llm/graphrag_semantic_edges_v2_top500_test_Candidates.jsonl`

semantic map：

- `retrieval_assets_clsprec/NYC/double_llm/semantic_poi_ids.jsonl`

全量样本统计：

| split | groups | Top500 target hit |
|---|---:|---:|
| train | 4502 | 0.771435 |
| val | 579 | 0.780656 |
| test | 884 | 0.761312 |

注意：

- `Top500 target hit` 是 wide pool oracle 命中率，不等于在线压缩后的 Top100 `candidate_hit`。
- v5 训练、验证、测试均为全量样本，不是只保留命中样本。

## 训练命令

```bash
cd /mnt/data/users/yyl/TMP
PY=/mnt/data/users/yyl/miniconda3/envs/poi_data/bin/python

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
nohup $PY src/poi_reranker/train_anonymous_mobility_encoder_online.py \
  --config config/train_nyc_exp3_online_pool_bert_dcnv2_dinlite_v5.yaml \
  > logs/train_nyc_exp3_online_pool_bert_dcnv2_dinlite_v5.log 2>&1 &
```

监控：

```bash
tail -f /mnt/data/users/yyl/TMP/logs/train_nyc_exp3_online_pool_bert_dcnv2_dinlite_v5.log
```

极小批量 smoke test：

```bash
cd /mnt/data/users/yyl/TMP
PY=/mnt/data/users/yyl/miniconda3/envs/poi_data/bin/python

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
$PY src/poi_reranker/train_anonymous_mobility_encoder_online.py \
  --config config/train_nyc_exp3_online_pool_bert_dcnv2_dinlite_v5.yaml \
  --output-dir /tmp/tmp_exp3_v5_dinlite_smoke \
  --max-train-groups 2 \
  --max-val-groups 2 \
  --max-test-groups 2 \
  --max-steps 1 \
  --eval-at-steps 1 \
  --save-steps 1 \
  --logging-steps 1 \
  --eval-candidate-batch-size 100
```

## v5 详细结果

v5 best checkpoint 按 `val mrr` 选择，最佳为 `step=600`。

| step | split | candidate_hit | top5 | top10 | ndcg5 | ndcg10 | mrr | conditional_mrr |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 200 | val | 0.656304 | 0.364421 | 0.462867 | 0.283061 | 0.314910 | 0.278386 | 0.424173 |
| 200 | test | 0.605204 | 0.348416 | 0.420814 | 0.264330 | 0.288210 | 0.256280 | 0.423461 |
| 400 | val | 0.683938 | 0.373057 | 0.480138 | 0.289104 | 0.323578 | 0.285450 | 0.417363 |
| 400 | test | 0.639140 | 0.357466 | 0.443439 | 0.269230 | 0.297071 | 0.261244 | 0.408742 |
| 600 | val | 0.694301 | 0.379965 | 0.476684 | 0.291151 | 0.322800 | 0.286050 | 0.411998 |
| 600 | test | 0.640271 | 0.360860 | 0.451357 | 0.267823 | 0.296921 | 0.258366 | 0.403526 |
| 800 | val | 0.694301 | 0.383420 | 0.476684 | 0.290596 | 0.320717 | 0.283394 | 0.408172 |
| 800 | test | 0.638009 | 0.359729 | 0.451357 | 0.265820 | 0.295184 | 0.255949 | 0.401168 |

解读：

- v5 的 `candidate_hit` 在 600 step 达到 `0.640271`，与 BERT v3 best 持平。
- v5 的 `top5/ndcg5/mrr` 相对 BERT v3 有小幅正向变化，说明 DIN-lite 对命中后的局部排序有帮助。
- v5 的 `conditional_mrr` 从 200 step 到 800 step 持续下降，说明继续训练仍会损伤命中后的排序质量。
- 800 step 不应作为主结果，主结果应使用 `best/metadata.json` 中的 `step=600`。

## 历史实验定位

### experiment2

`experiment2` 主要用于构建和比较候选池。当前最重要产物是：

- GraphRAG Top500
- 线性 learned_pool Top500->Top100
- exp2 top500-rescored joined parquet

离线 learned_pool 的 test 指标：

| top5 | top10 | ndcg5 | ndcg10 | mrr | candidate_hit |
|---:|---:|---:|---:|---:|---:|
| 0.395928 | 0.481900 | 0.292651 | 0.320440 | 0.280377 | 0.720588 |

### Llama online pool

Llama online pool 使用 Llama-3.2-1B-Instruct 编码轨迹，online Top500->Top100 后再融合 DCNv2 排序。

它的特点是：

- `candidate_hit` 低于 BERT。
- 命中后的排序能力强于 BERT。
- 最终 `mrr=0.261940`，仍高于当前 BERT v5。

### BERT v3 unfreeze2

BERT v3 是当前 BERT 分支的基础主干：

- BERT 最后 2 层解冻。
- LoRA rank 16。
- `keep_loss_weight=5.0`。
- `candidate_token_attention=true`。

v5 在 v3 基础上新增 enhanced behavior features 和 DIN-lite history attention。

## 当前问题

1. **online Top100 覆盖仍不足。**  
   离线 learned_pool 的 `candidate_hit=0.720588`，BERT v5 只有 `0.640271`。很多样本在在线压缩阶段已经被丢掉，后续排序器无法补救。

2. **BERT 命中后排序仍弱于 Llama。**  
   Llama online 的 `conditional_mrr=0.446156`，BERT v5 best 只有 `0.403526`。

3. **训练后期存在排序退化。**  
   v5 从 step 200 到 800，`candidate_hit` 上升，但 `conditional_mrr` 下降，说明 pool 保留目标与最终精排仍在拉扯。

4. **DIN-lite 带来的提升偏小。**  
   v5 证明候选感知历史注意力方向有用，但当前设计还没有形成显著增益。

## 下一步策略

优先方向：

1. **蒸馏离线 learned_pool。**  
   让 online pool scorer 先模仿离线 learned_pool 的 Top100 选择或分数分布，再引入最终排序损失。目标是把 online `candidate_hit` 拉近 `0.720588`。

2. **分阶段训练。**  
   第一阶段训练 pool/keep，第二阶段冻结或降低 pool scorer 学习率，重点训练 DIN-lite + DCNv2 final reranker，避免后期排序退化。

3. **强化 BERT 行为输入。**  
   当前结构化 token 已经加入 gap、transition、frequency，后续可以继续加入更明确的时间周期、工作日/周末、轨迹段落边界和重复访问模式。

4. **分析失败样本。**  
   分别统计 online pool 丢失样本、候选命中但 rank 靠后的样本，避免继续盲目堆结构。

5. **保留 learned_pool 作为强基线。**  
   所有新模型都必须和离线 learned_pool 原始顺序对比，不能只和 GraphRAG 原始顺序或较弱 online 版本对比。

## 常用文件

- [src/poi_reranker/train_anonymous_mobility_encoder_online.py](/mnt/data/users/yyl/TMP/src/poi_reranker/train_anonymous_mobility_encoder_online.py)
- [config/train_nyc_exp3_online_pool_bert_dcnv2_dinlite_v5.yaml](/mnt/data/users/yyl/TMP/config/train_nyc_exp3_online_pool_bert_dcnv2_dinlite_v5.yaml)
- [reports/experiment2_nyc_top500_to_top100_rescore_quality.json](/mnt/data/users/yyl/TMP/reports/experiment2_nyc_top500_to_top100_rescore_quality.json)
- [reports/experiment2_nyc_top500_to_top100_rescore_quality_train_val_test.json](/mnt/data/users/yyl/TMP/reports/experiment2_nyc_top500_to_top100_rescore_quality_train_val_test.json)
- [docs/clsprec_dataset_pipeline.md](/mnt/data/users/yyl/TMP/docs/clsprec_dataset_pipeline.md)
- [docs/experiment3_改进规划.md](/mnt/data/users/yyl/TMP/docs/experiment3_改进规划.md)

## 给其他 agent 的一句话摘要

```text
TMP experiment3 当前最新主线是 BERT v5 DIN-lite online pool：从 exp2 GraphRAG Top500 候选池在线压缩到 Top100，BERT-base-uncased 用匿名 3-LoRA 和最后 2 层解冻编码用户轨迹，候选侧使用结构化 anchor、增强 user-candidate relation、candidate-token attention、DIN-lite 历史行为注意力和 DCNv2 融合打分；NYC full test best(step600) 为 top5=0.360860、top10=0.451357、ndcg5=0.267823、ndcg10=0.296921、mrr=0.258366、candidate_hit=0.640271，较 BERT v3 小幅提升但仍低于离线 learned_pool 原始顺序(mrr=0.280377, candidate_hit=0.720588)。
```
