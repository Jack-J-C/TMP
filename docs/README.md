# LLMMRA 当前流程

当前项目主线是 **LoRA-A Prompt Refiner + 8B POI Predictor + Selector/Router** 的三阶段训练流程：

- `Stage1`: Raw-only 8B warmup，使用 `raw_v2_no_user_no_date`。
- `Stage2`: Raw + Refined mixed training，加入 LoRA-A 生成的 refined prompt。
- `Stage3`: Hard-focused finetune，只针对困难样本补强。

详细设计见 [adaptive_refiner_selector_summary.md](adaptive_refiner_selector_summary.md)。

## 当前目录

核心数据与模型路径：

```text
dataset/NewYork/
├── NY_train.csv
├── NY_val.csv
├── NY_test.csv
├── graph_X.csv
└── metadata.json

retrieval_assets/NewYork/evidence/
├── sequence_evidence_train.jsonl
├── sequence_evidence_val.jsonl
├── sequence_evidence_test.jsonl
├── geo_evidence_train.jsonl
├── geo_evidence_val.jsonl
├── geo_evidence_test.jsonl
├── preference_evidence_train.jsonl
├── preference_evidence_val.jsonl
├── preference_evidence_test.jsonl
├── teacher_distill_inputs_train.jsonl
├── teacher_distill_inputs_val.jsonl
├── teacher_distill_inputs_test.jsonl
├── teacher_distill_labels_train.jsonl
├── teacher_distill_labels_val.jsonl
├── teacher_distill_labels_test.jsonl
└── stats.json

models/
├── Llama-3.2-1B-Instruct/
├── prompt-refiner-lora-llama32-1b/
└── Llama-3.1-8B-Instruct/
```

当前流程不依赖历史 `stage2_llm_evidence_*.jsonl`。如果旧脚本或旧统计里出现 `stage2` 字样，视为历史 merged evidence 命名，不作为当前 Stage1/2/3 训练阶段命名。

### JSONL 归档策略

为避免 `retrieval_assets/NewYork/` 下生成文件过多，现阶段不再直接使用的 JSONL 已移动到：

```text
retrieval_assets/NewYork/archive_jsonl/
├── legacy_summary_distill/          # 旧 summary-style LoRA-A teacher 蒸馏样本
├── legacy_summary_refined/          # 旧 summary-style refined prompt 输出
├── decision_debug/                  # LoRA-A-v2 50 条试跑、临时输出、removed 输出
├── decision_intermediate/           # decision 蒸馏未清洗、removed、部分中断输出
├── eval_generations_legacy/         # 旧 Stage1/Stage2/Stage2b 评估 generations
└── smoke_debug/                     # smoke/小样本调试输出
```

当前主线仍保留在原目录的文件包括：

- `evidence/teacher_distill_inputs_*.jsonl` 与 `evidence/teacher_distill_labels_*.jsonl`：基础无标签输入和监督标签。
- `distill_decision/*_combined*.jsonl`、`distill_decision/*_clean*.jsonl`：LoRA-A-v2 训练、验证和当前生成输出。
- `poi_sft/stage2c_*_lora_a_decision_288.jsonl` 与对应 Stage2c generations：当前 LoRA-A-v2 refined prompt 的配对评估样本。
- `poi_sft/stage1_*_raw.jsonl`、`stage2*`、`stage2b*`：仍可用于基线复现或对照实验的大样本 SFT 数据。

如需复查历史结果，只读 `archive_jsonl/`；如需重新跑旧流程，建议重新生成到主目录，不要直接把归档文件混回当前主线。

## 数据准备

生成 NewYork 标准 CSV、POI 词表和三类 evidence：

```bash
cd /mnt/data/yyl/LLMMRA

/mnt/data/yyl/miniconda3/envs/poi_data/bin/python scripts/evidence/build_all_evidence.py \
  --cities NewYork \
  --tcpp-root /mnt/data/yyl/TCPP \
  --dataset-dir dataset \
  --output-root retrieval_assets \
  --overwrite
```

过滤协议：

- 只用 train split 建立用户和 POI 词表，避免 val/test 信息泄漏。
- `--poi-threshold 9`：保留 train 中出现次数不少于 9 的 POI。
- `--user-threshold 9`：保留 train 中签到次数不少于 9 的用户。
- `--min-traj-len 2`：过滤后轨迹长度至少为 2，最后一个点作为预测目标。
- val/test 只保留 train 词表内的用户和 POI。
- 缺失经纬度不删除，标准 CSV 中用 `coord_available=0` 标记。

## Teacher 输入

把三类 evidence 转成无标签 teacher/refiner 输入，同时把监督标签单独保存：

```bash
cd /mnt/data/yyl/LLMMRA

/mnt/data/yyl/miniconda3/envs/poi_data/bin/python scripts/evidence/build_teacher_distill_inputs.py \
  --cities NewYork \
  --evidence-root retrieval_assets \
  --splits train val test \
  --candidate-limit 64 \
  --overwrite
```

`teacher_distill_inputs_*.jsonl` 可用于 teacher API 或 LoRA-A refiner，不含 `target_poi_id`、`target_category` 等标签字段。

`teacher_distill_labels_*.jsonl` 只用于监督训练、评估、hard mining 和 SFT 样本拼接，不能传给 teacher/refiner。

## LoRA-A Prompt Refiner

### 历史 Summary-Style LoRA-A

这一节记录的是旧 summary-style LoRA-A 复现命令。对应已生成 JSONL 已归档到 `retrieval_assets/NewYork/archive_jsonl/legacy_summary_distill/` 和 `retrieval_assets/NewYork/archive_jsonl/legacy_summary_refined/`；当前主线优先使用下面的 `Decision-Style LoRA-A-v2`。

如需重新复现旧 summary-style 流程，先采样一小批 teacher 蒸馏数据：

```bash
cd /mnt/data/yyl/LLMMRA

/mnt/data/yyl/miniconda3/envs/poi_data/bin/python scripts/evidence/sample_teacher_distill_subset.py \
  --city NewYork \
  --split train \
  --size 3000 \
  --input retrieval_assets/NewYork/evidence/teacher_distill_inputs_train.jsonl \
  --labels retrieval_assets/NewYork/evidence/teacher_distill_labels_train.jsonl \
  --output-dir retrieval_assets/NewYork/distill \
  --name train_3000 \
  --overwrite

/mnt/data/yyl/miniconda3/envs/poi_data/bin/python scripts/evidence/sample_teacher_distill_subset.py \
  --city NewYork \
  --split val \
  --size 500 \
  --input retrieval_assets/NewYork/evidence/teacher_distill_inputs_val.jsonl \
  --labels retrieval_assets/NewYork/evidence/teacher_distill_labels_val.jsonl \
  --output-dir retrieval_assets/NewYork/distill \
  --name val_500 \
  --overwrite
```

调用 OpenAI-compatible teacher API：

```bash
export DASHSCOPE_API_KEY="你的百炼 key"

/mnt/data/yyl/miniconda3/envs/poi_data/bin/python scripts/evidence/call_teacher_prompt_distill_api.py \
  --input retrieval_assets/NewYork/distill/teacher_distill_inputs_train_3000.jsonl \
  --output retrieval_assets/NewYork/distill/teacher_prompt_outputs_train_3000.jsonl \
  --provider aliyun \
  --model deepseek-v4-flash \
  --limit -1 \
  --workers 6 \
  --max-output-tokens 512 \
  --temperature 0.0 \
  --resume
```

验证集同理，把 input/output 改成 `val_500`。

### 训练 LoRA-A

```bash
cd /mnt/data/yyl/LLMMRA

NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
/mnt/data/yyl/miniconda3/envs/poi_data/bin/python scripts/evidence/train_prompt_refiner_lora.py \
  --train-inputs retrieval_assets/NewYork/distill/teacher_distill_inputs_train_3000.jsonl \
  --train-outputs retrieval_assets/NewYork/distill/teacher_prompt_outputs_train_3000.jsonl \
  --val-inputs retrieval_assets/NewYork/distill/teacher_distill_inputs_val_500.jsonl \
  --val-outputs retrieval_assets/NewYork/distill/teacher_prompt_outputs_val_500.jsonl \
  --base-model /mnt/data/yyl/LLMMRA/models/Llama-3.2-1B-Instruct \
  --output-dir models/prompt-refiner-lora-llama32-1b \
  --max-source-length 1536 \
  --max-target-length 384 \
  --batch-size 2 \
  --grad-accum 8 \
  --epochs 3 \
  --lr 2e-4 \
  --bf16 \
  --gradient-checkpointing
```

### Decision-Style LoRA-A-v2

当前 summary-style refined prompt 在 Stage2 qsample 上没有稳定提升。下一轮 LoRA-A 建议改为 decision-style：不再生成长摘要，而是输出优先假设、证据来源、弱噪声线索和最终偏向。先做小规模 teacher 蒸馏验证，不需要全量重来。

信号分工：

- `confidence` / `useful_for_refinement` 由 DeepSeek teacher 蒸馏给 LoRA-A 学习，表示 refiner 对“本条 refined prompt 是否可靠、是否值得注入”的自评。它可以作为 prompt 内软提示或 selector 的辅助特征，但不能直接当作最终选择标签。
- `evidence_quality` / `difficulty_factors` 由结构化 evidence 的确定性统计规则生成，例如轨迹长度、历史是否缺失、geo 是否缺失、transition 是否稀疏、候选数量和转移分散度。它们不使用 teacher 判断，避免 teacher 幻觉影响采样、hard mining 和 selector 训练。
- 学习型 selector 的强标签应来自 paired evaluation，即比较同一样本在 RAW 和 REFINED 输入下的 8B 结果，而不是来自 LoRA-A 自己的 confidence。

LoRA-A-v2 本地生成 decision-style refined prompt 时，`generate_refined_prompts.py` 会从文本中解析并额外写出：

```json
{
  "refiner_confidence": "high|medium|low|unknown",
  "useful_for_refinement": "yes|no|unknown"
}
```

`refiner_confidence` 是后续规则 selector 判断 refined prompt 是否可信的主字段；`useful_for_refinement` 保留为辅助诊断字段。

当前 LoRA-A-v2 训练集由两部分 teacher 数据合并而成：

```text
train_2000 clean       = 1919
train_hard_1453 clean  = 1384
combined train clean   = 3303
val_300 clean matched  = 289
```

最终训练文件：

```text
retrieval_assets/NewYork/distill_decision/teacher_distill_inputs_train_decision_v2_combined.jsonl
retrieval_assets/NewYork/distill_decision/teacher_prompt_outputs_train_decision_v2_combined_clean.jsonl
retrieval_assets/NewYork/distill_decision/teacher_distill_inputs_val_decision_v2_clean_matched.jsonl
retrieval_assets/NewYork/distill_decision/teacher_prompt_outputs_val_300_decision_v2_clean.jsonl
```

如需从头复现，先采样基础 teacher 数据：

```bash
cd /mnt/data/yyl/LLMMRA

/mnt/data/yyl/miniconda3/envs/poi_data/bin/python scripts/evidence/sample_teacher_distill_subset.py \
  --city NewYork \
  --split train \
  --size 2000 \
  --input retrieval_assets/NewYork/evidence/teacher_distill_inputs_train.jsonl \
  --labels retrieval_assets/NewYork/evidence/teacher_distill_labels_train.jsonl \
  --output-dir retrieval_assets/NewYork/distill_decision \
  --name train_2000 \
  --overwrite

/mnt/data/yyl/miniconda3/envs/poi_data/bin/python scripts/evidence/sample_teacher_distill_subset.py \
  --city NewYork \
  --split val \
  --size 300 \
  --input retrieval_assets/NewYork/evidence/teacher_distill_inputs_val.jsonl \
  --labels retrieval_assets/NewYork/evidence/teacher_distill_labels_val.jsonl \
  --output-dir retrieval_assets/NewYork/distill_decision \
  --name val_300 \
  --overwrite
```

调用 DeepSeek/DashScope teacher 生成 decision-style 输出：

```bash
export DASHSCOPE_API_KEY="你的百炼 key"

/mnt/data/yyl/miniconda3/envs/poi_data/bin/python scripts/evidence/call_teacher_prompt_distill_api.py \
  --input retrieval_assets/NewYork/distill_decision/teacher_distill_inputs_train_2000.jsonl \
  --output retrieval_assets/NewYork/distill_decision/teacher_prompt_outputs_train_2000_decision_v2.jsonl \
  --provider aliyun \
  --model deepseek-v4-flash \
  --refiner-style decision \
  --limit -1 \
  --workers 6 \
  --max-output-tokens 360 \
  --temperature 0.0 \
  --resume

/mnt/data/yyl/miniconda3/envs/poi_data/bin/python scripts/evidence/call_teacher_prompt_distill_api.py \
  --input retrieval_assets/NewYork/distill_decision/teacher_distill_inputs_val_300.jsonl \
  --output retrieval_assets/NewYork/distill_decision/teacher_prompt_outputs_val_300_decision_v2.jsonl \
  --provider aliyun \
  --model deepseek-v4-flash \
  --refiner-style decision \
  --limit -1 \
  --workers 6 \
  --max-output-tokens 360 \
  --temperature 0.0 \
  --resume
```

补充 hard/boundary teacher 样本时，使用统计规则采样，排除已有 `train_2000`，不让 teacher 判断样本难度：

```bash
/mnt/data/yyl/miniconda3/envs/poi_data/bin/python scripts/evidence/sample_teacher_hard_boundary_subset.py \
  --city NewYork \
  --split train \
  --size 2000 \
  --input retrieval_assets/NewYork/evidence/teacher_distill_inputs_train.jsonl \
  --labels retrieval_assets/NewYork/evidence/teacher_distill_labels_train.jsonl \
  --exclude-inputs retrieval_assets/NewYork/distill_decision/teacher_distill_inputs_train_2000.jsonl \
  --output-dir retrieval_assets/NewYork/distill_decision \
  --name train_hard_2000 \
  --seed 43 \
  --overwrite

/mnt/data/yyl/miniconda3/envs/poi_data/bin/python scripts/evidence/call_teacher_prompt_distill_api.py \
  --input retrieval_assets/NewYork/distill_decision/teacher_distill_inputs_train_hard_2000.jsonl \
  --output retrieval_assets/NewYork/distill_decision/teacher_prompt_outputs_train_hard_2000_decision_v2.jsonl \
  --provider aliyun \
  --model deepseek-v4-flash \
  --refiner-style decision \
  --limit 1000 \
  --workers 4 \
  --max-output-tokens 360 \
  --temperature 0.0 \
  --resume
```

说明：如果该命令是在已有部分输出后用 `--resume --limit 1000` 继续跑，`--limit` 表示新增最多 1000 条，不是输出总量上限。本项目当前 hard 输出实际为 1453 条，清洗后 1384 条。

验证 decision-style 输出是否合格：

```bash
/mnt/data/yyl/miniconda3/envs/poi_data/bin/python scripts/evidence/validate_decision_refined_prompts.py \
  --input retrieval_assets/NewYork/distill_decision/teacher_prompt_outputs_train_2000_decision_v2.jsonl \
  --output-clean retrieval_assets/NewYork/distill_decision/teacher_prompt_outputs_train_2000_decision_v2_clean.jsonl \
  --output-removed retrieval_assets/NewYork/distill_decision/teacher_prompt_outputs_train_2000_decision_v2_removed.jsonl

/mnt/data/yyl/miniconda3/envs/poi_data/bin/python scripts/evidence/validate_decision_refined_prompts.py \
  --input retrieval_assets/NewYork/distill_decision/teacher_prompt_outputs_val_300_decision_v2.jsonl \
  --output-clean retrieval_assets/NewYork/distill_decision/teacher_prompt_outputs_val_300_decision_v2_clean.jsonl \
  --output-removed retrieval_assets/NewYork/distill_decision/teacher_prompt_outputs_val_300_decision_v2_removed.jsonl

/mnt/data/yyl/miniconda3/envs/poi_data/bin/python scripts/evidence/validate_decision_refined_prompts.py \
  --input retrieval_assets/NewYork/distill_decision/teacher_prompt_outputs_train_hard_2000_decision_v2.jsonl \
  --output-clean retrieval_assets/NewYork/distill_decision/teacher_prompt_outputs_train_hard_1453_decision_v2_clean.jsonl \
  --output-removed retrieval_assets/NewYork/distill_decision/teacher_prompt_outputs_train_hard_1453_decision_v2_removed.jsonl
```

训练 LoRA-A-v2：

```bash
CUDA_VISIBLE_DEVICES=0 \
NCCL_P2P_DISABLE=1 \
NCCL_IB_DISABLE=1 \
nohup /mnt/data/yyl/miniconda3/envs/poi_data/bin/python scripts/evidence/train_prompt_refiner_lora.py \
  --train-inputs retrieval_assets/NewYork/distill_decision/teacher_distill_inputs_train_decision_v2_combined.jsonl \
  --train-outputs retrieval_assets/NewYork/distill_decision/teacher_prompt_outputs_train_decision_v2_combined_clean.jsonl \
  --val-inputs retrieval_assets/NewYork/distill_decision/teacher_distill_inputs_val_decision_v2_clean_matched.jsonl \
  --val-outputs retrieval_assets/NewYork/distill_decision/teacher_prompt_outputs_val_300_decision_v2_clean.jsonl \
  --refiner-style decision \
  --base-model /mnt/data/yyl/LLMMRA/models/Llama-3.2-1B-Instruct \
  --output-dir models/prompt-refiner-lora-llama32-1b-decision-v2 \
  --max-source-length 1536 \
  --max-target-length 256 \
  --batch-size 2 \
  --grad-accum 8 \
  --epochs 3 \
  --lr 2e-4 \
  --bf16 \
  --gradient-checkpointing \
> train_lora_a_decision_v2.log 2>&1 &
```

用 LoRA-A-v2 生成 qsample decision refined prompt：

```bash
CUDA_VISIBLE_DEVICES=0 \
NCCL_P2P_DISABLE=1 \
NCCL_IB_DISABLE=1 \
/mnt/data/yyl/miniconda3/envs/poi_data/bin/python scripts/evidence/generate_refined_prompts.py \
  --inputs retrieval_assets/NewYork/refined_prompts/refiner_inputs_train_q6000.jsonl \
  --base-model models/Llama-3.2-1B-Instruct \
  --lora-path models/prompt-refiner-lora-llama32-1b-decision-v2/final \
  --output retrieval_assets/NewYork/refined_prompts_decision/refiner_outputs_train_q6000_decision.jsonl \
  --report retrieval_assets/NewYork/refined_prompts_decision/refiner_eval_train_q6000_decision.json \
  --refiner-style decision \
  --limit -1 \
  --max-source-length 1536 \
  --max-new-tokens 256 \
  --batch-size 8 \
  --device cuda:0 \
  --attn-implementation sdpa \
  --resume
```

LoRA-A-v2 生成端已内置 decision-style 修复逻辑：截断过长 `Weak or noisy cues`，清理重复段，并保证输出包含 `Final hint`。50 条验证样本上修复后 validator 通过率为 `50/50`。因此大规模生成建议使用普通贪心解码，不启用慢速 `no_repeat_ngram_size`：

```bash
CUDA_VISIBLE_DEVICES=0 \
NCCL_P2P_DISABLE=1 \
NCCL_IB_DISABLE=1 \
/mnt/data/yyl/miniconda3/envs/poi_data/bin/python scripts/evidence/generate_refined_prompts.py \
  --inputs retrieval_assets/NewYork/distill_decision/teacher_distill_inputs_val_decision_v2_clean_matched.jsonl \
  --base-model models/Llama-3.2-1B-Instruct \
  --lora-path models/prompt-refiner-lora-llama32-1b-decision-v2/final \
  --output retrieval_assets/NewYork/distill_decision/lora_a_decision_v2_val_outputs.jsonl \
  --report retrieval_assets/NewYork/distill_decision/lora_a_decision_v2_val_report.json \
  --refiner-style decision \
  --limit -1 \
  --max-source-length 1536 \
  --max-new-tokens 256 \
  --batch-size 8 \
  --device cuda:0 \
  --attn-implementation sdpa \
  --overwrite \
  --no-progress
```

## Refined Prompt 生成

采样脚本只使用结构化 evidence 启发式判断难度，不使用 teacher 的 `evidence_quality`，避免 teacher 幻觉影响样本选择。

普通 qsample，用于 Stage2 mixed training：

```bash
cd /mnt/data/yyl/LLMMRA

/mnt/data/yyl/miniconda3/envs/poi_data/bin/python scripts/evidence/sample_refiner_inputs_by_quality.py \
  --input retrieval_assets/NewYork/evidence/teacher_distill_inputs_train.jsonl \
  --labels retrieval_assets/NewYork/evidence/teacher_distill_labels_train.jsonl \
  --output retrieval_assets/NewYork/refined_prompts/refiner_inputs_train_q6000.jsonl \
  --size 6000 \
  --good-ratio 0.6 \
  --partial-ratio 0.3 \
  --weak-ratio 0.1 \
  --overwrite
```

可选的冷启动 hard qsample：如果 Stage2 还没有训练完成、暂时没有错误样本评估结果，可以先用启发式 hard factors 采一批 refined prompt 做预备数据。真正的 Stage3 hard patch 仍应以 Stage2 预测错或 rank 差的样本为准，见后文 Stage3。

```bash
/mnt/data/yyl/miniconda3/envs/poi_data/bin/python scripts/evidence/sample_refiner_inputs_by_quality.py \
  --input retrieval_assets/NewYork/evidence/teacher_distill_inputs_train.jsonl \
  --labels retrieval_assets/NewYork/evidence/teacher_distill_labels_train.jsonl \
  --output retrieval_assets/NewYork/refined_prompts/refiner_inputs_train_hard_q6000.jsonl \
  --size 6000 \
  --good-ratio 0.2 \
  --partial-ratio 0.5 \
  --weak-ratio 0.3 \
  --hard-ratio 0.8 \
  --overwrite
```

用 LoRA-A 生成 refined prompt：

```bash
CUDA_VISIBLE_DEVICES=0 \
NCCL_P2P_DISABLE=1 \
NCCL_IB_DISABLE=1 \
/mnt/data/yyl/miniconda3/envs/poi_data/bin/python scripts/evidence/generate_refined_prompts.py \
  --inputs retrieval_assets/NewYork/refined_prompts/refiner_inputs_train_q6000.jsonl \
  --base-model models/Llama-3.2-1B-Instruct \
  --lora-path models/prompt-refiner-lora-llama32-1b/final \
  --output retrieval_assets/NewYork/refined_prompts/refiner_outputs_train_q6000.jsonl \
  --report retrieval_assets/NewYork/refined_prompts/refiner_eval_train_q6000.json \
  --limit -1 \
  --max-source-length 1536 \
  --max-new-tokens 384 \
  --batch-size 8 \
  --device cuda:0 \
  --attn-implementation sdpa \
  --resume
```

## Stage1: Raw-Only Warmup

Stage1 使用 `raw_v2_no_user_no_date`。Prompt 内不包含 `user_id`、`sample_id`、`trajectory_id`、真实标签、`target_in_candidates`、原始经纬度或绝对日期。

构造 RAW SFT 数据：

```bash
cd /mnt/data/yyl/LLMMRA

/mnt/data/yyl/miniconda3/envs/poi_data/bin/python scripts/evidence/build_poi_sft_data.py \
  --inputs retrieval_assets/NewYork/evidence/teacher_distill_inputs_train.jsonl \
  --labels retrieval_assets/NewYork/evidence/teacher_distill_labels_train.jsonl \
  --output retrieval_assets/NewYork/poi_sft/stage1_train_raw.jsonl \
  --route RAW \
  --overwrite

/mnt/data/yyl/miniconda3/envs/poi_data/bin/python scripts/evidence/build_poi_sft_data.py \
  --inputs retrieval_assets/NewYork/evidence/teacher_distill_inputs_val.jsonl \
  --labels retrieval_assets/NewYork/evidence/teacher_distill_labels_val.jsonl \
  --output retrieval_assets/NewYork/poi_sft/stage1_val_raw.jsonl \
  --route RAW \
  --overwrite

/mnt/data/yyl/miniconda3/envs/poi_data/bin/python scripts/evidence/build_poi_sft_data.py \
  --inputs retrieval_assets/NewYork/evidence/teacher_distill_inputs_test.jsonl \
  --labels retrieval_assets/NewYork/evidence/teacher_distill_labels_test.jsonl \
  --output retrieval_assets/NewYork/poi_sft/stage1_test_raw.jsonl \
  --route RAW \
  --overwrite
```

训练 8B QLoRA：

当前 `poi_data` 环境中 `torch` 可能没有 `nn.Module.set_submodule`，而新版 `transformers/bitsandbytes` 的 4-bit 加载会用到该方法。`train_poi_lora.py` 和 `evaluate_poi_lora.py` 已内置兼容补丁，继续使用 `--load-in-4bit` 即可。

```bash
CUDA_VISIBLE_DEVICES=0 \
NCCL_P2P_DISABLE=1 \
NCCL_IB_DISABLE=1 \
/mnt/data/yyl/miniconda3/envs/poi_data/bin/python scripts/evidence/train_poi_lora.py \
  --train-data retrieval_assets/NewYork/poi_sft/stage1_train_raw.jsonl \
  --val-data retrieval_assets/NewYork/poi_sft/stage1_val_raw.jsonl \
  --base-model /mnt/data/yyl/LLMMRA/models/Llama-3.1-8B-Instruct \
  --output-dir models/stage1-raw-poi-lora-llama31-8b \
  --max-source-length 3072 \
  --max-target-length 64 \
  --batch-size 1 \
  --grad-accum 16 \
  --epochs 1 \
  --lr 2e-4 \
  --bf16 \
  --load-in-4bit \
  --gradient-checkpointing \
  --device-map auto \
  --attn-implementation sdpa
```

评估：

```bash
CUDA_VISIBLE_DEVICES=0 \
NCCL_P2P_DISABLE=1 \
NCCL_IB_DISABLE=1 \
/mnt/data/yyl/miniconda3/envs/poi_data/bin/python scripts/evidence/evaluate_poi_lora.py \
  --data retrieval_assets/NewYork/poi_sft/stage1_val_raw.jsonl \
  --base-model /mnt/data/yyl/LLMMRA/models/Llama-3.1-8B-Instruct \
  --lora-path models/stage1-raw-poi-lora-llama31-8b/final \
  --output retrieval_assets/NewYork/poi_sft/eval_val_stage1_raw_generations.jsonl \
  --report retrieval_assets/NewYork/poi_sft/eval_val_stage1_raw_report.json \
  --max-source-length 3072 \
  --max-new-tokens 32 \
  --batch-size 1 \
  --device cuda:0 \
  --bf16 \
  --load-in-4bit \
  --attn-implementation sdpa \
  --overwrite
```

候选排序评估可额外加：

```bash
--rank-candidates --rank-batch-size 16
```

候选排序指标只在 `target_in_candidates=true` 的样本上计算，因为当前候选集不是闭合集。

## Stage2 和 Stage3

Stage2/Stage3 复用 `build_poi_sft_data.py`、`train_poi_lora.py` 和 `evaluate_poi_lora.py`。

Stage2 构造 `REFINED` 样本时加入 LoRA-A 输出。当前优先使用 `clean_refined_prompts` 下剔除异常后的 refined prompt。`build_poi_sft_data.py` 会把 `[Refined Evidence]` 放在证据区，并把唯一的输出格式约束放到 prompt 末尾，避免 refined 自然语言弱化 JSON 输出约束。

```bash
/mnt/data/yyl/miniconda3/envs/poi_data/bin/python scripts/evidence/build_poi_sft_data.py \
  --inputs retrieval_assets/NewYork/refined_prompts/refiner_inputs_train_q6000.jsonl \
  --labels retrieval_assets/NewYork/evidence/teacher_distill_labels_train.jsonl \
  --refined-prompts retrieval_assets/NewYork/clean_refined_prompts/refiner_outputs_train_q6000_clean.jsonl \
  --output retrieval_assets/NewYork/poi_sft/stage2_train_refined_q6000.jsonl \
  --route REFINED \
  --overwrite

/mnt/data/yyl/miniconda3/envs/poi_data/bin/python scripts/evidence/build_poi_sft_data.py \
  --inputs retrieval_assets/NewYork/refined_prompts/refiner_inputs_val_q1000.jsonl \
  --labels retrieval_assets/NewYork/evidence/teacher_distill_labels_val.jsonl \
  --refined-prompts retrieval_assets/NewYork/clean_refined_prompts/refiner_outputs_val_q1000_clean.jsonl \
  --output retrieval_assets/NewYork/poi_sft/stage2_val_refined_q1000.jsonl \
  --route REFINED \
  --overwrite

/mnt/data/yyl/miniconda3/envs/poi_data/bin/python scripts/evidence/build_poi_sft_data.py \
  --inputs retrieval_assets/NewYork/refined_prompts/refiner_inputs_test_q2000.jsonl \
  --labels retrieval_assets/NewYork/evidence/teacher_distill_labels_test.jsonl \
  --refined-prompts retrieval_assets/NewYork/clean_refined_prompts/refiner_outputs_test_q2000_clean.jsonl \
  --output retrieval_assets/NewYork/poi_sft/stage2_test_refined_q2000.jsonl \
  --route REFINED \
  --overwrite
```

`REFINED` 可见 prompt 格式：

```text
[ROUTE=REFINED]
<raw_v2_no_user_no_date prompt body without the RAW route marker and without the old output block>

[Refined Evidence]
<LoRA-A refined prompt>

Output format:
{"next_poi_id":"<poi_id>"}
Return only this JSON object and no extra text.
```

Stage2 mixed train 数据由 `stage1_train_raw.jsonl` 加上 `stage2_train_refined_q6000.jsonl` 组成。可以直接拼接：

```bash
cat \
  retrieval_assets/NewYork/poi_sft/stage1_train_raw.jsonl \
  retrieval_assets/NewYork/poi_sft/stage2_train_refined_q6000.jsonl \
  > retrieval_assets/NewYork/poi_sft/stage2_train_mixed.jsonl
```

Stage2 应从 Stage1 adapter 继续训练，而不是重新从 8B base 新建 LoRA：

```bash
CUDA_VISIBLE_DEVICES=0 \
NCCL_P2P_DISABLE=1 \
NCCL_IB_DISABLE=1 \
/mnt/data/yyl/miniconda3/envs/poi_data/bin/python scripts/evidence/train_poi_lora.py \
  --train-data retrieval_assets/NewYork/poi_sft/stage2_train_mixed.jsonl \
  --val-data retrieval_assets/NewYork/poi_sft/stage1_val_raw.jsonl \
  --base-model /mnt/data/yyl/LLMMRA/models/Llama-3.1-8B-Instruct \
  --init-lora-path models/stage1-raw-poi-lora-llama31-8b/final \
  --output-dir models/stage2-mixed-poi-lora-llama31-8b \
  --max-source-length 3072 \
  --max-target-length 64 \
  --batch-size 1 \
  --grad-accum 16 \
  --epochs 1 \
  --lr 1e-4 \
  --bf16 \
  --load-in-4bit \
  --gradient-checkpointing \
  --device-map auto \
  --attn-implementation sdpa
```

Stage2 训练后先评估 RAW val，和 Stage1 baseline 直接对比：

```bash
CUDA_VISIBLE_DEVICES=0 \
NCCL_P2P_DISABLE=1 \
NCCL_IB_DISABLE=1 \
/mnt/data/yyl/miniconda3/envs/poi_data/bin/python scripts/evidence/evaluate_poi_lora.py \
  --data retrieval_assets/NewYork/poi_sft/stage1_val_raw.jsonl \
  --base-model /mnt/data/yyl/LLMMRA/models/Llama-3.1-8B-Instruct \
  --lora-path models/stage2-mixed-poi-lora-llama31-8b/final \
  --output retrieval_assets/NewYork/poi_sft/eval_val_stage2_raw_generations.jsonl \
  --report retrieval_assets/NewYork/poi_sft/eval_val_stage2_raw_report.json \
  --max-source-length 3072 \
  --max-new-tokens 32 \
  --batch-size 1 \
  --device cuda:0 \
  --bf16 \
  --load-in-4bit \
  --attn-implementation sdpa \
  --overwrite
```

再评估 refined val，判断 LoRA-A 辅助证据是否带来增益：

```bash
CUDA_VISIBLE_DEVICES=0 \
NCCL_P2P_DISABLE=1 \
NCCL_IB_DISABLE=1 \
/mnt/data/yyl/miniconda3/envs/poi_data/bin/python scripts/evidence/evaluate_poi_lora.py \
  --data retrieval_assets/NewYork/poi_sft/stage2_val_refined_q1000.jsonl \
  --base-model /mnt/data/yyl/LLMMRA/models/Llama-3.1-8B-Instruct \
  --lora-path models/stage2-mixed-poi-lora-llama31-8b/final \
  --output retrieval_assets/NewYork/poi_sft/eval_val_stage2_refined_q1000_generations.jsonl \
  --report retrieval_assets/NewYork/poi_sft/eval_val_stage2_refined_q1000_report.json \
  --max-source-length 3072 \
  --max-new-tokens 32 \
  --batch-size 1 \
  --device cuda:0 \
  --bf16 \
  --load-in-4bit \
  --attn-implementation sdpa \
  --overwrite
```

## Stage3: Hard Patch Finetune

Stage3 不是全量再生成 refined prompt，而是做 hard patch：先用 Stage2 模型在 train/val 上评估，揪出预测错、解析失败或候选 rank 很差的样本，再只对这些 hard samples 调 LoRA-A 补生成 refined prompt，最后小学习率继续训练。

1. 用 Stage2 模型评估 train 或 val，得到逐样本结果：

```bash
CUDA_VISIBLE_DEVICES=0 \
NCCL_P2P_DISABLE=1 \
NCCL_IB_DISABLE=1 \
/mnt/data/yyl/miniconda3/envs/poi_data/bin/python scripts/evidence/evaluate_poi_lora.py \
  --data retrieval_assets/NewYork/poi_sft/stage2_train_mixed.jsonl \
  --base-model /mnt/data/yyl/LLMMRA/models/Llama-3.1-8B-Instruct \
  --lora-path models/stage2-mixed-poi-lora-llama31-8b/final \
  --output retrieval_assets/NewYork/poi_sft/eval_train_stage2_mixed.jsonl \
  --report retrieval_assets/NewYork/poi_sft/eval_train_stage2_mixed_report.json \
  --max-source-length 3072 \
  --max-new-tokens 32 \
  --batch-size 1 \
  --rank-candidates \
  --rank-batch-size 16 \
  --device cuda:0 \
  --bf16 \
  --load-in-4bit \
  --attn-implementation sdpa \
  --overwrite
```

2. 从 Stage2 评估结果中挖 hard sample，只输出 label-free refiner inputs：

```bash
/mnt/data/yyl/miniconda3/envs/poi_data/bin/python scripts/evidence/mine_stage3_hard_samples.py \
  --inputs retrieval_assets/NewYork/evidence/teacher_distill_inputs_train.jsonl \
  --eval-outputs retrieval_assets/NewYork/poi_sft/eval_train_stage2_mixed.jsonl \
  --exclude-refined retrieval_assets/NewYork/refined_prompts/refiner_outputs_train_q6000.jsonl \
  --output retrieval_assets/NewYork/refined_prompts/refiner_inputs_train_stage3_hard_patch.jsonl \
  --rank-threshold 20 \
  --max-samples 6000 \
  --overwrite
```

默认 hard 条件：

- `correct=false`
- `pred_poi_id` 解析失败
- `candidate_rank > --rank-threshold`

如果还想把启发式困难样本也纳入补丁，可以额外加 `--include-heuristic-hard`。注意这仍然只用于补生成和训练采样，不写入模型可见 prompt。

3. 只对 hard patch 样本生成 refined prompt：

```bash
CUDA_VISIBLE_DEVICES=0 \
NCCL_P2P_DISABLE=1 \
NCCL_IB_DISABLE=1 \
/mnt/data/yyl/miniconda3/envs/poi_data/bin/python scripts/evidence/generate_refined_prompts.py \
  --inputs retrieval_assets/NewYork/refined_prompts/refiner_inputs_train_stage3_hard_patch.jsonl \
  --base-model models/Llama-3.2-1B-Instruct \
  --lora-path models/prompt-refiner-lora-llama32-1b/final \
  --output retrieval_assets/NewYork/refined_prompts/refiner_outputs_train_stage3_hard_patch.jsonl \
  --report retrieval_assets/NewYork/refined_prompts/refiner_eval_train_stage3_hard_patch.json \
  --limit -1 \
  --max-source-length 1536 \
  --max-new-tokens 384 \
  --batch-size 8 \
  --device cuda:0 \
  --attn-implementation sdpa \
  --resume
```

4. 构造 Stage3 refined patch SFT 数据：

```bash
/mnt/data/yyl/miniconda3/envs/poi_data/bin/python scripts/evidence/build_poi_sft_data.py \
  --inputs retrieval_assets/NewYork/refined_prompts/refiner_inputs_train_stage3_hard_patch.jsonl \
  --labels retrieval_assets/NewYork/evidence/teacher_distill_labels_train.jsonl \
  --refined-prompts retrieval_assets/NewYork/refined_prompts/refiner_outputs_train_stage3_hard_patch.jsonl \
  --output retrieval_assets/NewYork/poi_sft/stage3_train_hard_refined_patch.jsonl \
  --route REFINED \
  --require-refined \
  --overwrite
```

5. Stage3 小学习率继续训练。训练数据应以 hard patch 为主，可以按需要拼接少量 `stage1_train_raw.jsonl` 防止遗忘：

```bash
cat \
  retrieval_assets/NewYork/poi_sft/stage3_train_hard_refined_patch.jsonl \
  > retrieval_assets/NewYork/poi_sft/stage3_train_hard_patch.jsonl
```

然后继续训练：

```bash
CUDA_VISIBLE_DEVICES=0 \
NCCL_P2P_DISABLE=1 \
NCCL_IB_DISABLE=1 \
/mnt/data/yyl/miniconda3/envs/poi_data/bin/python scripts/evidence/train_poi_lora.py \
  --train-data retrieval_assets/NewYork/poi_sft/stage3_train_hard_patch.jsonl \
  --val-data retrieval_assets/NewYork/poi_sft/stage1_val_raw.jsonl \
  --base-model /mnt/data/yyl/LLMMRA/models/Llama-3.1-8B-Instruct \
  --output-dir models/stage3-hard-patch-poi-lora-llama31-8b \
  --max-source-length 3072 \
  --max-target-length 64 \
  --batch-size 1 \
  --grad-accum 16 \
  --epochs 0.5 \
  --lr 5e-5 \
  --bf16 \
  --load-in-4bit \
  --gradient-checkpointing \
  --device-map auto \
  --attn-implementation sdpa
```

## 当前脚本

```text
scripts/evidence/build_all_evidence.py
scripts/evidence/build_teacher_distill_inputs.py
scripts/evidence/sample_teacher_distill_subset.py
scripts/evidence/call_teacher_prompt_distill_api.py
scripts/evidence/train_prompt_refiner_lora.py
scripts/evidence/sample_refiner_inputs_by_quality.py
scripts/evidence/generate_refined_prompts.py
scripts/evidence/mine_stage3_hard_samples.py
scripts/evidence/build_poi_sft_data.py
scripts/evidence/train_poi_lora.py
scripts/evidence/evaluate_poi_lora.py
```
