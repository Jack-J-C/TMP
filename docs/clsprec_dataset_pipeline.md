# CLSPRec/GETNext 数据在 TMP 中的预处理与训练样本构建规则

本文档说明当前 TMP 项目中 `dataset_clsprec/{NYC,SIN,PHO}` 与 `dataset_getnext_clsprec/TKY` 的来源、过滤规则、train/val/test 构建方式，以及后续进入 TeamLoRA 训练 parquet 的流程。

用户消息中两次写到 `/mnt/data/users/yyl/TMP/dataset_clsprec`。本文按当前仓库实际存在的数据说明：`NYC/SIN/PHO` 来自 CLSPRec raw data，`TKY` 来自 GETNext raw data，但 TKY 也沿用 CLSPRec-style 的轨迹有效性过滤规则。

## 结论

当前仓库此前没有一份完整、中文、与当前版本一致的说明文档。相关信息分散在：

- `src/data_build/convert_clsprec_raw_to_tmp_csv.py`
- `src/data_build/convert_getnext_tky_to_tmp_csv.py`
- `src/data_build/build_all_evidence.py`
- `src/data_build/build_semantic_poi_sft_data.py`
- `src/graphrag/build_semantic_poi_ids.py`
- `src/graphrag/build_graphrag_topk_candidates.py`
- `src/data_build/build_joined_poi_classification_data.py`
- 各目录下的 `metadata.json` / `build_summary_top100.json`

因此本文档作为当前版本的准说明。

## 目录关系

当前 CLSPRec/GETNext 数据在 TMP 中分为三层：

```text
/mnt/data/users/yyl/CLSPRec/raw_data
        |
        | convert_clsprec_raw_to_tmp_csv.py
        v
/mnt/data/users/yyl/TMP/raw_data/{NYC,PHO,SIN}

/mnt/data/users/yyl/GETNext/dataset/TKY
        |
        | convert_getnext_tky_to_tmp_csv.py
        v
/mnt/data/users/yyl/TMP/raw_data/TKY
        |
        | build_all_evidence.py
        v
/mnt/data/users/yyl/TMP/dataset_clsprec/{NYC,PHO,SIN}
/mnt/data/users/yyl/TMP/dataset_getnext_clsprec/TKY
/mnt/data/users/yyl/TMP/retrieval_assets_clsprec/{NYC,PHO,SIN}
/mnt/data/users/yyl/TMP/retrieval_assets_getnext_clsprec/TKY
        |
        | semantic_id / GraphRAG candidates / joined parquet / compact
        v
TeamLoRA reranker train/val/test parquet
```

其中：

- `raw_data/{city}` 是从外部原始 check-in 转成 TMP 标准 CSV 后的中间层。
- `dataset_clsprec/{city}` / `dataset_getnext_clsprec/TKY` 是经过 TMP evidence 构建阶段再次过滤后的标准训练 CSV 层。
- `retrieval_assets_clsprec/{city}` / `retrieval_assets_getnext_clsprec/TKY` 是 evidence、semantic prompt、GraphRAG 候选池和 joined parquet 的训练资产层。

## 第一步：外部原始数据转 TMP 标准 CSV

### NYC/SIN/PHO: CLSPRec raw data

脚本：

```text
src/data_build/convert_clsprec_raw_to_tmp_csv.py
```

输入：

```text
/mnt/data/users/yyl/CLSPRec/raw_data/{CITY}_checkin_with_active_regionId.csv
```

输出：

```text
raw_data/{CITY}/{CITY}_train.csv
raw_data/{CITY}/{CITY}_val.csv
raw_data/{CITY}/{CITY}_test.csv
raw_data/{CITY}/graph_X.csv
raw_data/{CITY}/metadata.json
raw_data/conversion_summary.json
```

当前使用的是 `clsprec-static7` 过滤策略。该策略只复用 CLSPRec 的有效样本过滤思想，不复用 CLSPRec 的 pickle 预处理、特征重编号和随机样本切分。

过滤规则：

| 规则 | 当前值 |
|---|---:|
| 轨迹单位 | 同一 `user_id` + 同一本地日期 |
| 有效日轨迹最小长度 | `min_seq_len=3` |
| 用户最少有效日轨迹数 | `min_seq_num=3` |
| 当前目标日轨迹最小长度 | `min_short_term_len=5` |
| 长期历史窗口 | `pre_seq_window_days=7` |
| 窗口内最少历史日轨迹数 | `min_long_term_count=2` |
| 连续重复 POI 去重 | 不做 |
| POI 频次过滤 | 不做 |
| user check-in 频次过滤 | 不做 |

字段映射：

| TMP 字段 | 来源或规则 |
|---|---|
| `user_id` | CLSPRec `UserId` |
| `POI_id` | 对原始 `VenueId` 排序后确定性映射为 `v{index}` |
| `POI_catid` | CLSPRec `L1_Category` |
| `POI_catid_code` | 对细粒度 `Category` 排序后确定性整数编码 |
| `POI_catname` | CLSPRec 细粒度 `Category` |
| `trajectory_id` | `{CITY}_{user_id}_{local_date}` |
| `local_time` | CLSPRec `Local_Time_True` |
| `UTC_time` | `local_time - timezone` |
| `norm_in_day_time` | 当天分钟数 / 1440 |

split 规则：

- 按用户内部时间顺序对日轨迹切分。
- 默认比例为 train/val/test = `0.8/0.1/0.1`。
- 当用户轨迹数较少时，脚本保证尽量保留测试轨迹：
  - `n=1` 或 `n=2`：只进入 train。
  - `n<10`：前 `n-2` 条为 train，倒数第二条为 val，最后一条为 test。
  - `n>=10`：按比例切分，若 train+val 覆盖全部，则回退确保至少 1 条 test。

当前 `raw_data` 层规模：

| city | rows after CLSPRec filter | users | POIs | trajectories | train traj | val traj | test traj |
|---|---:|---:|---:|---:|---:|---:|---:|
| NYC | 48162 | 820 | 8382 | 5991 | 4502 | 584 | 905 |
| PHO | 2806 | 43 | 534 | 384 | 289 | 40 | 55 |
| SIN | 49072 | 800 | 5883 | 6320 | 4755 | 627 | 938 |

### TKY: GETNext raw data

脚本：

```text
src/data_build/convert_getnext_tky_to_tmp_csv.py
```

输入：

```text
/mnt/data/users/yyl/GETNext/dataset/TKY/{TKY_train,TKY_val,TKY_test}.csv
```

输出：

```text
raw_data/TKY/TKY_train.csv
raw_data/TKY/TKY_val.csv
raw_data/TKY/TKY_test.csv
raw_data/TKY/graph_X.csv
raw_data/TKY/metadata.json
raw_data/TKY/conversion_summary.json
```

TKY 的特殊点：

- GETNext 原始 train/val/test split 不复用。
- 脚本先合并 GETNext 的三份 CSV，再按 `local_time` 重建同用户同日期日轨迹。
- 过滤规则仍使用 CLSPRec-style static7，有效轨迹规则与 NYC/SIN/PHO 一致。
- 过滤后重新按 TMP 的用户内时间顺序 train/val/test = `0.8/0.1/0.1` 切分。
- `POI_id` 对原始 GETNext `POI_id` 排序后确定性映射为 `v{index}`。

当前 TKY `raw_data` 层规模：

| city | rows after filter | users | POIs | trajectories | train traj | val traj | test traj |
|---|---:|---:|---:|---:|---:|---:|---:|
| TKY | 128769 | 1159 | 6386 | 16123 | 12319 | 1568 | 2236 |

## 第二步：构建 `dataset_clsprec` 与 evidence

脚本：

```text
src/data_build/build_all_evidence.py
```

输入：

```text
raw_data/{CITY}/{CITY}_train.csv
raw_data/{CITY}/{CITY}_val.csv
raw_data/{CITY}/{CITY}_test.csv
```

输出：

```text
dataset_clsprec/{CITY}/{CITY}_train.csv
dataset_clsprec/{CITY}/{CITY}_val.csv
dataset_clsprec/{CITY}/{CITY}_test.csv
dataset_clsprec/{CITY}/graph_X.csv
dataset_clsprec/{CITY}/metadata.json

retrieval_assets_clsprec/{CITY}/evidence/*.jsonl
retrieval_assets_clsprec/{CITY}/poi_sft/*.jsonl

dataset_getnext_clsprec/TKY/TKY_train.csv
dataset_getnext_clsprec/TKY/TKY_val.csv
dataset_getnext_clsprec/TKY/TKY_test.csv
dataset_getnext_clsprec/TKY/graph_X.csv
dataset_getnext_clsprec/TKY/metadata.json

retrieval_assets_getnext_clsprec/TKY/evidence/*.jsonl
retrieval_assets_getnext_clsprec/TKY/poi_sft/*.jsonl
```

当前 `dataset_clsprec` / `dataset_getnext_clsprec` 使用 TMP 的训练词表防泄露规则：

- 只用 train split 建立 train POI 集合和 train user 集合。
- 当前三城没有额外频次阈值过滤，等价于 `poi_threshold=0`、`user_threshold=0`。
- train 保留全部 train 用户和 train POI。
- val/test 中不在 train POI 集合或 train user 集合中的行会被过滤。
- 过滤后，长度小于 `min_traj_len=2` 的轨迹会被丢弃。
- `graph_X.csv` 只由过滤后的 train split 构建。

当前 `dataset_clsprec` 层规模：

| city | split | rows before vocab filter | rows after vocab filter | rows after traj filter | users | POIs | trajectories |
|---|---|---:|---:|---:|---:|---:|---:|
| NYC | train | 37468 | 37468 | 37468 | 820 | 7230 | 4502 |
| NYC | val | 4352 | 3827 | 3822 | 323 | 1943 | 579 |
| NYC | test | 6342 | 5357 | 5338 | 464 | 2431 | 884 |
| PHO | train | 2133 | 2133 | 2133 | 43 | 463 | 289 |
| PHO | val | 284 | 252 | 252 | 23 | 133 | 40 |
| PHO | test | 389 | 338 | 338 | 27 | 164 | 55 |
| SIN | train | 37974 | 37974 | 37974 | 800 | 5273 | 4755 |
| SIN | val | 4612 | 4319 | 4319 | 373 | 1806 | 627 |
| SIN | test | 6486 | 6006 | 6003 | 509 | 2137 | 935 |
| TKY | train | 101124 | 101124 | 101124 | 1159 | 6070 | 12319 |
| TKY | val | 11425 | 11264 | 11264 | 795 | 3008 | 1568 |
| TKY | test | 16220 | 15922 | 15922 | 932 | 3431 | 2236 |

样本构建规则：

- 每条长度 `>=2` 的日轨迹生成 1 个 next-POI 样本。
- 目标 `target_poi_id` 是该轨迹最后一个 POI。
- 输入当前轨迹 `current_trajectory` 是目标之前的前缀。
- 同用户历史 `history` 只来自 train split，且时间早于当前轨迹 cutoff。
- transition index、geo index、用户历史、revisit 统计都只由过滤后的 train split 构建。
- 对 train split 的 transition candidates，如果目标 POI 已出现在 train transition 计数中，会对当前目标做一次减计数，避免把当前样本标签直接泄露到 transition evidence。

因此当前流程不是随机打散样本再切分，而是：

```text
外部原始 check-in
  -> CLSPRec-style 有效目标日轨迹过滤
  -> 每用户日轨迹按时间切分 train/val/test
  -> TMP train-vocab 防泄露过滤
  -> 每条保留轨迹取最后一个 POI 作为标签
```

## 第三步：semantic_id 与 RAW semantic prompt

semantic_id 生成脚本：

```text
src/graphrag/build_semantic_poi_ids.py
```

semantic_id 是 TMP 当前自定义的可解释规则编码，不是 learned semantic ID。格式：

```text
{CITY}::{CATEGORY_TOKEN}::{GEO_CELL}::P{LOCAL_INDEX}
```

例如：

```text
NYC::COFFEE_SHOP::LAT40_74_LON_73_99::P0009
```

RAW semantic prompt 构建脚本：

```text
src/data_build/build_semantic_poi_sft_data.py
```

它把 `retrieval_assets_clsprec/{CITY}/poi_sft/stage1_{split}_raw.jsonl` 改写为更紧凑的 semantic-id prompt：

```text
retrieval_assets_clsprec/{CITY}/semantic_poi_sft/stage1_{split}_raw_semantic.jsonl
```

当前 prompt 版本是：

```text
raw_semantic_id_v1
```

## 第四步：GraphRAG 候选池

原始 v5 多城使用 GraphRAG Top100：

```text
retrieval_assets_clsprec/{CITY}/double_llm/graphrag_semantic_edges_v2_top100_{split}_candidates.jsonl
retrieval_assets_getnext_clsprec/TKY/double_llm/graphrag_semantic_edges_v2_top100_{split}_candidates.jsonl
```

候选池生成脚本：

```text
src/graphrag/build_graphrag_topk_candidates.py
```

GraphRAG 的 index 只由 train split 构建，主要信号包括：

- last POI / last category transition
- last2 POI / category transition
- user-specific transition
- category popular
- semantic_id / semantic category / semantic geo cell edges
- semantic covisit
- time slot / weekday slot popular
- global popular

当前 NYC `experiment2` 不再直接使用原始 GraphRAG Top100，而是：

```text
GraphRAG Top500 wide pool
  -> coverage-only linear scorer
  -> learned fixed Top100
```

对应候选池目录：

```text
retrieval_assets_clsprec/NYC/double_llm/experiment2_top500_best_rescored/
```

当前 NYC `experiment2` learned Top100 召回：

| split | rows | target in learned Top100 | hit ratio |
|---|---:|---:|---:|
| train | 4502 | 3378 | 0.750333 |
| val | 579 | 433 | 0.747841 |
| test | 884 | 637 | 0.720588 |

原始 GraphRAG Top100 召回：

| city | split | rows | target in GraphRAG Top100 | hit ratio |
|---|---|---:|---:|---:|
| NYC | train | 4502 | 3120 | 0.693025 |
| NYC | val | 579 | 396 | 0.683938 |
| NYC | test | 884 | 573 | 0.648190 |
| SIN | train | 4755 | 3337 | 0.701788 |
| SIN | val | 627 | 457 | 0.728868 |
| SIN | test | 935 | 641 | 0.685561 |
| TKY | train | 12319 | 10203 | 0.828233 |
| TKY | val | 1568 | 1268 | 0.808673 |
| TKY | test | 2236 | 1727 | 0.772361 |

PHO 当前没有作为主线训练城市继续推进。

## 第五步：joined parquet 与 compact

joined parquet 构建脚本：

```text
src/data_build/build_joined_poi_classification_data.py
```

它把以下内容按 `sample_id` 对齐合并：

- `semantic_poi_sft/stage1_{split}_raw_semantic.jsonl`
- `evidence/preference_evidence_{split}.jsonl`
- refined prompt 输出；当前 CLSPRec 版本允许缺失，`refined_text` 为空
- GraphRAG 或 learned Top100 candidates
- `semantic_poi_ids.jsonl`

当前 input_text 包含：

- `[VIEW=RAW_SEM]`
- `[VIEW=REFINED source=full]`
- `[VIEW=USER_SEMANTIC_PROFILE source=preference_evidence]`
- 条件触发的 `[VIEW=SIMILAR_USER_SEMANTIC_PROFILE source=bge_m3_user_knn]`
- `[VIEW=GRAPH_RAG]`

其中 similar user profile 的构建规则：

- 对全部样本建立相似用户画像候选。
- BGE-M3 query source 当前为 `recent_profile`。
- `k=20`，`min_sim=0.35`。
- 只有当自身画像不足时，训练输入中才保留 `SIMILAR_USER_SEMANTIC_PROFILE`；否则该字段为空。

compact 脚本：

```text
src/data_build/compress_joined_raw_context.py
```

当前 compact 规则：

| 字段 | 当前值 |
|---|---:|
| `trajectory_lines` | 12 |
| `transitions` | 6 |
| `nearby` | 6 |
| `top_categories` | 4 |
| `revisited_pois` | 4 |

当前 NYC `experiment2` 训练用 parquet：

```text
retrieval_assets_clsprec/NYC/joined_poi_classification_exp2_top500_rescored/
  train_joined_top100.parquet
  val_joined_top100.parquet
  test_joined_top100.parquet
```

当前 TKY v5-style 训练用 parquet：

```text
retrieval_assets_getnext_clsprec/TKY/joined_poi_classification/
  train_joined_top100.parquet
  val_joined_top100.parquet
  test_joined_top100.parquet
```

## 当前训练评估 split 口径

训练配置：

```text
config/train_nyc_exp2_top500_rescored_semprofile_simuser_v2.yaml
```

训练阶段：

- `train_hit_only=true`：只训练 target 已进入候选 Top100 的样本。
- `val_hit_only=true`：训练过程中的 val 只评估候选命中样本，观察纯 reranking 能力。
- 当前可通过 `eval_at_steps` 指定只在某些 step 评估。

最终 test：

- 不加 `--val-hit-only`。
- 使用全量 test 样本。
- 全量指标会受到候选池覆盖率上限 `candidate_hit` 影响。

## 防数据泄露要点

当前 CLSPRec -> TMP 流程的主要防泄露点：

1. train/val/test 是用户内按时间顺序切分，不是随机混洗样本切分。
2. `dataset_clsprec` 的 `graph_X.csv` 只由 train split 构建。
3. val/test 行会过滤到 train user 与 train POI 词表内，避免评估时出现训练词表外的 POI。
4. transition、geo、history、revisit、GraphRAG index 均只由 train split 构建。
5. evidence 中同用户历史只取当前轨迹之前的 train 历史。
6. train split 的 transition candidate 对当前样本目标做减计数，降低标签直接泄露。
7. joined parquet 使用 `sample_id` 对齐 raw/preference/graph 等文件，若 target mismatch 会跳过该样本。

## 复现入口命令

以下命令是当前流程的入口示例，具体城市和输出目录按实验需要调整。

从 CLSPRec 原始数据转 TMP `raw_data`：

```bash
cd /mnt/data/users/yyl/TMP
PY=/mnt/data/users/yyl/miniconda3/envs/poi_data/bin/python

$PY src/data_build/convert_clsprec_raw_to_tmp_csv.py \
  --clsprec-raw-dir /mnt/data/users/yyl/CLSPRec/raw_data \
  --output-dir raw_data \
  --cities NYC PHO SIN \
  --filter-policy clsprec-static7 \
  --overwrite
```

从 GETNext TKY 原始数据转 TMP `raw_data/TKY`：

```bash
$PY src/data_build/convert_getnext_tky_to_tmp_csv.py \
  --getnext-dir /mnt/data/users/yyl/GETNext/dataset/TKY \
  --output-dir raw_data/TKY \
  --city TKY \
  --overwrite
```

构建 `dataset_clsprec` 与 evidence：

```bash
$PY src/data_build/build_all_evidence.py \
  --cities NYC PHO SIN \
  --dataset-dir dataset_clsprec \
  --output-root retrieval_assets_clsprec \
  --poi-threshold 0 \
  --user-threshold 0 \
  --min-traj-len 2 \
  --overwrite
```

构建 TKY 的 `dataset_getnext_clsprec` 与 evidence：

```bash
$PY src/data_build/build_all_evidence.py \
  --cities TKY \
  --dataset-dir dataset_getnext_clsprec \
  --output-root retrieval_assets_getnext_clsprec \
  --poi-threshold 0 \
  --user-threshold 0 \
  --min-traj-len 2 \
  --overwrite
```

构建 joined parquet 的当前 NYC experiment2 示例见根目录 `README.md`。
