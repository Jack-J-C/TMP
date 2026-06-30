# Adaptive Refiner Selector 当前方案摘要

本文档只记录当前有效方案，不记录每一步实验流水账。若后续主线方案变化，直接覆盖对应章节，避免保留过时判断。

## 当前主线

当前项目采用两模型解耦路线：

- `LoRA-A-v2`: 基于 `Llama-3.2-1B-Instruct` 的 decision-style prompt refiner，负责把三类 evidence 压缩为 label-free refined prompt，并输出自评信号。
- `LoRA-B`: 基于 `Llama-3.1-8B-Instruct` 的 next-POI predictor，负责最终 POI 预测。
- `Selector/Router`: 推理时决定是否使用 LoRA-A refined prompt。当前以离线 paired evaluation 验证为主，暂不急于训练学习型 selector。

训练和评估采用离线生成 refined prompt，不在 8B 训练 loop 中动态加载 1B LoRA-A。这样可以降低显存和工程复杂度。

## 信号分工

LoRA-A-v2 输出的自评字段：

- `refiner_confidence`: `high / medium / low / unknown`，表示 LoRA-A 对当前 refined prompt 可靠性的自评。
- `useful_for_refinement`: `yes / no / unknown`，表示该 refined prompt 是否可能值得注入 8B。

这两个字段来自 teacher 蒸馏和 LoRA-A 生成，是 refined prompt 的软自评。它们可以用于诊断、规则 selector 或 prompt 内提示，但不能直接当成样本真实难度。

样本难度字段：

- `evidence_quality`
- `difficulty_factors`

这两个字段必须由结构化 evidence 的统计规则生成，例如轨迹长度、历史是否缺失、geo 是否缺失、transition 是否稀疏、候选数量、转移分散度等。难度不交给 teacher/LoRA-A 判断，避免 teacher 幻觉或过度自信污染 hard mining 与 selector 训练。

## LoRA-A-v2 状态

LoRA-A-v2 已完成 teacher 蒸馏和训练。

模型路径：

```text
models/prompt-refiner-lora-llama32-1b-decision-v2/final
```

训练数据：

```text
retrieval_assets/NewYork/distill_decision/teacher_distill_inputs_train_decision_v2_combined.jsonl
retrieval_assets/NewYork/distill_decision/teacher_prompt_outputs_train_decision_v2_combined_clean.jsonl
retrieval_assets/NewYork/distill_decision/teacher_distill_inputs_val_decision_v2_clean_matched.jsonl
retrieval_assets/NewYork/distill_decision/teacher_prompt_outputs_val_300_decision_v2_clean.jsonl
```

数据规模：

```text
train_2000 clean       = 1919
train_hard_1453 clean  = 1384
combined train clean   = 3303
val_300 clean matched  = 289
```

LoRA-A-v2 生成端已修复 decision-style 解码问题，包括重复段落截断、`Weak or noisy cues` 过长截断、保证 `Final hint` 存在，以及解析顶层 `refiner_confidence` / `useful_for_refinement`。

当前 full val 生成结果：

```text
retrieval_assets/NewYork/distill_decision/lora_a_decision_v2_val_outputs.jsonl
retrieval_assets/NewYork/distill_decision/lora_a_decision_v2_val_outputs_clean.jsonl
```

质量结论：

- clean 后 `288 / 289` 条可用。
- confidence 分布约为 `medium=168, high=119/120, low=1`。
- `useful_for_refinement` 基本为 `yes`。
- 结构化输出整体可用，可以进入 8B 配对评估。

## Stage2c 当前评估结论

当前用于验证 LoRA-A-v2 的配对样本：

```text
retrieval_assets/NewYork/poi_sft/stage2c_val_raw_lora_a_decision_288.jsonl
retrieval_assets/NewYork/poi_sft/stage2c_val_refined_lora_a_decision_288.jsonl
```

评估使用的 8B LoRA-B：

```text
models/stage2b-teacher-decision-poi-lora-llama31-8b/final
```

评估报告：

```text
retrieval_assets/NewYork/poi_sft/eval_val_stage2c_lora_a_raw_288_report.json
retrieval_assets/NewYork/poi_sft/eval_val_stage2c_lora_a_refined_288_report.json
retrieval_assets/NewYork/poi_sft/eval_val_stage2c_lora_a_selector_288_report.json
```

核心结果：

```text
RAW top1                 = 0.305556
All REFINED top1         = 0.319444
high-only selector top1  = 0.312500

paired both_correct      = 85
paired both_wrong        = 193
refined_fix              = 7
refined_hurt             = 3
```

按当前结果，LoRA-A-v2 refined prompt 在 val_288 上有正收益：全量 refined 比 raw 高 `+0.013888` top1。`high-only` selector 也优于 raw，但低于 all-refined，说明 `medium` confidence 样本中也有有效增益。因此当前不能简单使用 `confidence=high` 作为唯一注入阈值。

当前结论：

- LoRA-A-v2 的 refined prompt 是有效的。
- 当前 8B `stage2b` 并不是用 LoRA-A-v2 生成的 refined prompt 训练的，因此 Stage2c 结果只是迁移验证，不是最终上限。
- 下一步如果显存允许，应基于 LoRA-A-v2 生成的 refined prompt 构建新的 8B 训练集，再训练新的 LoRA-B。

## Selector 当前策略

当前 selector 不应直接面向全部样本训练上线，原因是 paired 样本只有 288 条，且 `refined_fix=7`、`refined_hurt=3` 的有效监督样本太少。更合理的做法是先把 LoRA-A-v2 的 confidence 用作分流规则，只让 selector 处理中间不确定区间。

当前默认路由规则：

```text
refiner_confidence=high   -> 默认注入 refined prompt
refiner_confidence=low    -> 默认不注入，走 RAW
refiner_confidence=medium -> 交给 selector 判断是否注入
```

这个设计基于当前 Stage2c 结果：`high-only` 优于 RAW，但低于 `all-refined`，说明 `medium` 样本中也存在有效增益；因此不能简单丢弃 `medium`，也不应让 selector 重复判断已经较确定的 `high/low` 样本。

短期策略：

- 离线评估阶段同时跑 RAW 和 REFINED，继续积累 paired evaluation。
- 在小样本阶段优先比较 `all-refined`、`high-only`、`high+medium`、`high + medium-selector`、统计 hard-only 等规则。
- 不把 LoRA-A 的 `confidence` 当成 selector 监督标签，只作为 routing 分流特征。
- selector 的监督标签仍必须来自 paired evaluation：比较同一样本下 `8B(REFINED)` 是否优于 `8B(RAW)`。

当前默认实验策略：

```text
如果只是验证 LoRA-A-v2 是否有用：优先使用 all-refined。
如果构建下一版 8B 训练集：默认注入 high + medium，剔除 low refined。
如果关注推理成本或噪声控制：固定 high 注入、low 不注入，只对 medium 训练/评估 selector。
如果训练学习型 selector：优先只在 medium 子集上训练，并等待更多 paired evaluation 样本。
```

## 8B LoRA-B 后续路线

已有模型：

```text
models/stage1-raw-poi-lora-llama31-8b/final
models/stage2-mixed-poi-lora-llama31-8b/final
models/stage2b-teacher-decision-poi-lora-llama31-8b/final
```

当前问题是：`stage2b` 使用的是 teacher decision/refined 数据，不是 LoRA-A-v2 全量生成的数据。因此如果要验证最终闭环，需要重新构建 LoRA-A-v2 refined train/val，并训练新的 8B LoRA-B。

推荐下一步：

1. 使用 NYC train 全量 LoRA-A-v2 refined prompt 的 clean 版本。
2. 构建 `RAW + LoRA-A-v2 REFINED` mixed SFT 数据，默认只注入 `refiner_confidence in {high, medium}`，剔除 `low` refined。
3. 从 `stage1-raw-poi-lora-llama31-8b/final` 继续训练新的 Stage2c/Stage2d 8B LoRA-B。
4. 在 val 上做 RAW / high-only / high+medium / all-refined / high+medium-selector paired evaluation。
5. 如果 high+medium 的 refined 增益稳定，再只针对 medium 子集训练 selector 或进入 hard-focused finetune。

## 数据与归档策略

当前活跃 JSONL 保留在原目录：

- `retrieval_assets/NewYork/evidence/`
- `retrieval_assets/NewYork/distill_decision/`
- `retrieval_assets/NewYork/poi_sft/`
- `retrieval_assets/NewYork/refined_prompts/`
- `retrieval_assets/NewYork/clean_refined_prompts/`

不再直接使用的旧 JSONL 已归档：

```text
retrieval_assets/NewYork/archive_jsonl/
```

归档目录只用于复查历史，不作为当前命令默认输入。若要重新跑旧流程，应重新生成到主目录，避免把历史中间产物混入当前主线。

## 当前判断

当前主线已经从“summary-style refined prompt 是否有用”更新为：

```text
Decision-style LoRA-A-v2 可以生成结构稳定、带 confidence 自评的 refined prompt；
在 stage2c val_288 paired evaluation 中，全量 refined 优于 raw；
high 和 medium 都可能贡献 refined 增益，因此低置信度样本剔除，medium 样本交给 selector 做二次判断；
但 selector 监督样本仍不足，不能过早面向全样本训练学习型 selector；
最终需要用 LoRA-A-v2 生成的新 refined prompt 重新训练 8B LoRA-B，才能验证闭环收益。
```
