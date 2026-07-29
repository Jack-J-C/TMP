# experiment3 改进规划

本文档面向 TMP 当前 `experiment2` 主线之后的下一阶段工作，目标是给出一个可落地、可评估、可复用的 `experiment3` 方向。

先说明结论：

- `nextstepplan.md` 里的 LMRE / trajectory encoder / POI embedding 双塔路线，和当前 TMP 主线不是同一条实验线。
- 这条路线可以作为后续研究方向，但不适合作为当前 `experiment3` 的直接落地规划。
- 当前 experiment3 更应该建立在已有证据之上，继续围绕 `固定候选池 + 匿名三专家轨迹表示学习 + 轻量对齐模块 + semantic profile` 做增强和诊断。

## 1. `nextstepplan.md` 中不合理的地方

### 1.1 研究目标切换过大

文档把当前问题从“固定 Top100 候选池上的 candidate-wise reranking”直接切到“轨迹表征学习 + POI embedding 检索/排序”。

这会带来两个问题：

- 现有 experiment2 的结果无法直接继承，因为训练目标已经变了。
- 你已经积累的 `learned Top100`、`semantic_profile_simuser`、`RAAT`、`graph prior` 这些实验结论会失去可比性。

### 1.2 模型结构变化过重

文档引入了：

- Llama trajectory encoder
- POI graph encoder
- projection head
- user embedding / poi embedding 双塔 ranker

这已经不是“小改进”，而是新架构。

对 TMP 现在的状态来说，这等于把项目从：

```text
candidate-wise reranker
```

切成：

```text
retrieval / embedding ranking system
```

两者的评估方式、训练信号、失败模式都不同。

### 1.3 缺少与当前证据的一致性

当前 TMP 已经得到的核心事实是：

- 候选池压缩带来的提升很明显。
- TeamLoRA reranker 的增益存在，但弱于候选池提升。
- `GraphRAG Top500 -> learned Top100` 是当前最稳的候选池方案。

而 `nextstepplan.md` 没有承认这个前提，直接跳去做 embedding learning，容易把实验资源花在不对应当前瓶颈的地方。

### 1.4 训练目标不够贴合现有任务

文档提出 `InfoNCE + BPR`，更像检索模型目标。

但当前 TMP 的实际任务是：

- 输入固定 Top100 候选
- 进行逐候选/列表式重排
- 输出排序指标 `Top5 / Top10 / NDCG / MRR`

如果改成双塔 embedding，容易出现：

- 优化检索召回，但排序不一定更好
- 与已有 GraphRAG candidate pool 的边界不清
- 结果很难和 experiment2 主线直接比较

### 1.5 POI encoder 方案风险偏高

文档建议单独训练 POI graph encoder。

问题是：

- POI 数量大，训练复杂度上升；
- graph encoder 的好坏很难和 reranker 的收益直接分离；
- 一旦引入新的 POI embedding 层，很多收益来源会不清晰。

对当前阶段来说，这会增加分析负担，不利于快速形成论文主线。

## 2. experiment3 的正确定位

experiment3 不应该推翻 experiment2，而应该回答一个更窄、更有价值的问题：

> 在固定 Top100 候选池已经较稳定的前提下，怎样让匿名三专家 LLM 真正学到用户轨迹表示，并通过一个轻量、可学习的对齐模块与候选池语义结构对齐？

因此，experiment3 建议保持以下大框架不变：

```text
Semantic-ID GraphRAG Top100 / learned Top100
        ↓
fixed candidate pool anchors
        ↓
anonymous routed TeamLoRA trajectory encoder
        ↓
lightweight alignment module
        ↓
mobility representation / pool alignment objective
```

## 3. experiment3 的改进目标

### 3.1 目标一：降低显式 graph prior 的支配性

当前结果显示，candidate pool 的贡献很大，reranker 只提供小幅增益。

experiment3 需要验证：

- 去掉 `graph prior` 后，模型是否真正依赖语义和上下文；
- 如果去掉后掉点可控，说明 reranker 学到了更强的文本/画像信号；
- 如果掉点很大，说明当前模型仍只是“图排序修正器”。

### 3.2 目标二：提高 LLM 部分的实际贡献

当前要重点判断的是：

- `USER_SEMANTIC_PROFILE` 是否足够；
- `SIMILAR_USER_SEMANTIC_PROFILE` 是否真能改善画像不足样本；
- 匿名 routed LoRA experts 是否真的学到不同轨迹模式；
- RAAT 是否在训练中提供有效扰动。

### 3.3 目标三：判断是否需要轻量对齐模块

三专家如果只做轨迹编码，仍然会有一个问题：

- 轨迹表示学出来了，但它是否真的和固定候选池中的语义结构对齐？

因此 experiment3 建议考虑一个**轻量可学习对齐模块**，但不引入独立 POI encoder，也不改成双塔检索。

建议的对齐模块形态：

```text
trajectory embedding u
        ↓
linear / bilinear / low-rank alignment head
        ↓
score against fixed candidate pool anchors
```

这个模块的约束是：

- 只能是轻量模块；
- 不引入额外 POI 编码塔；
- 候选池保持固定；
- 候选侧尽量只使用 `semantic_id` / `category` / `geo_cell` / `graph_features` 这类结构化锚点，不再直接塞长文本 `POI_HYPOTHESIS`；
- 仍然依赖现有 GraphRAG / learned pool 的候选内容作为监督锚点。

### 3.4 目标四：保持评估口径稳定

experiment3 仍然使用：

- NYC 作为主实验城市；
- `Top100` 全量 test 评估；
- `top5 / top10 / ndcg5 / ndcg10 / mrr` 作为主指标；
- learned candidate pool 的 Hit@100 作为前置指标。

## 4. experiment3 建议实验主线

### 4.1 主线 A：匿名三专家轨迹编码

这是 experiment3 最重要的一条线。

实验内容：

- 保持匿名 routed TeamLoRA 不变；
- 取消 candidate-wise reranking 作为主目标；
- 输入仍保留轨迹、semantic profile、similar-user profile、graph context；
- 主输入不包含 `POI_HYPOTHESIS`；
- 输出改为轨迹表示或 mobility embedding；
- 候选池只作为对齐锚点，不作为逐候选打分对象；

目的：

- 判断匿名三专家是否能学到稳定的轨迹表示。

### 4.2 主线 B：轻量对齐模块

在匿名三专家轨迹编码之外，单独评估一个轻量对齐模块是否必要。

候选形态：

- `linear alignment head`
- `bilinear head`
- `low-rank projection + cosine scoring`

候选侧输入只保留结构化锚点，不使用长文本 `POI_HYPOTHESIS`。

目的：

- 把轨迹表示和固定候选池锚点对齐；
- 不引入额外 POI encoder；
- 避免直接退化成 full dual-tower retrieval。

### 4.3 主线 C：更强的负样本与 RAAT 对齐

在不改架构的情况下，优先试这些：

- 更合理的 hard negative 采样；
- 继续使用 `target_mask_2view`；
- 只在必要时引入更强扰动；
- 检查 RAAT 是否真的帮助模型摆脱候选池顺序惯性。

### 4.4 主线 D：输入模板微调

不是重做表示学习，而是优化现有输入：

- `USER_SEMANTIC_PROFILE` 是否只保留高置信长期画像；
- `SIMILAR_USER_SEMANTIC_PROFILE` 触发条件是否足够严格；
- raw context 是否还可以进一步压缩；
- graph evidence 是否保持结构化而不过度文本化；
- `GRAPH_CONTEXT` 是否应完全移出 LLM 主输入，仅保留在对齐模块侧；
- 对齐模块的候选锚点是否仅使用结构化字段而非长文本。

### 4.5 主线 E：LoRA expert 行为分析

当前 routed TeamLoRA 的问题不是“有没有三专家”，而是“匿名专家是否真的学到不同的轨迹模式”。

experiment3 应该补：

- expert gate 分布统计；
- expert 使用频率；
- 不同样本类型下的 expert 路由差异；
- 对 tail / mid / head 样本的专家行为分桶；
- 对 `user_profile_insufficient=true` 样本的 expert 路由差异。

## 5. experiment3 不建议做的事

以下内容不建议作为 experiment3 主干：

- 把主线直接改成双塔 embedding 检索；
- 重新训练 POI graph encoder 作为主线；
- 放弃固定候选池；
- 再去训练 8B 或 selector；
- 让实验重新回到 full POI vocab classifier。

这些方向不是不能做，而是当前不应该抢占 experiment3 的资源。

## 6. experiment3 推荐实验顺序

1. 匿名三专家轨迹编码基线。
2. 加轻量对齐模块。
3. 加 RAAT / hard negatives。
4. 加输入模板清理。
5. 做 expert 路由分析。
6. 迁移到 SIN / TKY 做稳定性验证。

## 7. experiment3 预期结论

如果 experiment3 成功，应该得到下面这种结论：

- learned candidate pool 继续保持较高覆盖率；
- 匿名三专家能学到稳定的轨迹表示；
- 轻量对齐模块确实能把轨迹表示拉向固定候选池语义结构；
- routed LoRA expert 的分工可以被统计分析出来；
- RAAT 和 semantic profile 不是装饰项，而是能解释模型改进的关键因素。

如果 experiment3 不成功，至少也能得到：

- 当前 TMP 的主要收益来源是否仍然是候选池；
- reranker 是否需要更强的结构约束；
- 是否值得在后续分支再考虑 embedding/ranking 的新架构。

## 8. 与当前仓库的关系

当前仓库已经有：

- `README.md`：experiment2 主线说明；
- `docs/clsprec_dataset_pipeline.md`：数据构建与过滤规则；
- `docs/experiment_branch_training.md`：训练与跨机器运行说明；
- `docs/adaptive_refiner_selector_summary.md`：旧路线已废弃的说明。

因此 experiment3 只需要在当前 experiment2 基础上继续推进，不要重新定义整个项目。
