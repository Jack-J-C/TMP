可以。先不改代码，等这轮 top100 + graph prior 训练结果出来后再决定。预改进方案如下。

**目标**
把当前模型从：

```text
GraphRAG top100 prior + MLP full-class fallback
```

升级成：

```text
RAAT-style candidate-pool robustness training
```

核心目的不是提高 GraphRAG recall 本身，而是让模型在候选池不完整、排序错误、含噪时不要被 prior 绑死。

**方案一：候选池噪声增强**
对每个训练样本构造 4 类候选池版本：

```text
o: original
原始 GraphRAG top100。

p: partial / masked
随机截断 top10/top30/top50，或随机删除 10%-20% 候选。

f: counterfactual
如果 target 在候选池中，以一定概率移除 target，并加入同 category / 相近 geo-cell 的错误 POI。

c: irrelevant
混入其他样本的热门 POI 或完全无关 POI，模拟错误召回。
```

评估时仍使用干净 top100。

**方案二：hardest candidate-pool selection**
每个 batch 内，对同一个样本的多个候选池版本分别计算 loss：

```text
loss_o, loss_p, loss_f, loss_c
```

选择当前最难版本反传：

```text
loss_adv = max(loss_o, loss_p, loss_f, loss_c)
```

这对应 RAAT 的 adaptive adversarial training，但我们用分类 CE 代替生成式 token likelihood。

**方案三：graph reliability 辅助头**
在 pooled hidden state 后加一个小 head：

```text
graph_reliability_head: hidden -> 4 classes
```

预测当前候选池类型：

```text
0 = original
1 = partial/masked
2 = counterfactual
3 = irrelevant
```

或者更贴合 POI：

```text
0 = target_in_top10
1 = target_in_top30
2 = target_in_top100
3 = target_not_in_top100
```

总损失：

```text
loss = poi_ce_loss
     + lambda_noise * graph_noise_cls_loss
```

建议 `lambda_noise=0.05 ~ 0.2`，不要太大，避免辅助任务压过 POI 主任务。

**优先级**
如果这轮训练结果显示：

```text
top20 接近 GraphRAG hit@20，但 top1/MRR 低
```

优先做重排序能力增强，候选池噪声可以轻量加入。

如果结果显示：

```text
top20/top100 内指标仍很低，接近 popularity baseline
```

优先检查 graph prior/eval/logits 融合，不急着上 RAAT。

如果结果显示：

```text
top1/top5 提升明显，但 top100 未命中样本表现差
```

再上 RAAT-style candidate-pool robustness，重点做 target-mask 和 counterfactual 候选。

**建议第一版参数**
后续若要改，先用轻量配置：

```text
graph_adv_training = true
graph_adv_variants = original,masked,counterfactual,irrelevant
graph_target_mask_prob = 0.25
graph_candidate_drop_prob = 0.10
graph_irrelevant_mix_prob = 0.10
graph_noise_loss_weight = 0.1
graph_adv_mode = max_loss
```

不要一开始噪声太强。GraphRAG top100 是目前最强信号，目标是提升鲁棒性，不是破坏召回。

---

## 已实现的第一版改进

当前先采用轻量 RAAT 融合，而不是完整 4 倍 forward 的 hardest-selection。原因是完整 RAAT 会显著增加显存和训练时间；当前首要问题是 step400 后 MLP logits 开始扰乱 GraphRAG prior 排序，因此先做可控融合和候选池鲁棒训练。

### 已加入机制

1. MLP logits 缩放

```text
final_logits = graph_prior + mlp_logit_scale * mlp_logits
```

用于避免后期 MLP full-class logits 覆盖 GraphRAG 排序。建议先试 `mlp_logit_scale=0.3` 或 `0.5`。

2. RAAT-style 候选池噪声

训练集动态构造候选池扰动，评估集保持干净 top100。

```text
original: 原始 GraphRAG top100
partial: dropout / cutoff / candidate drop
counterfactual: target-mask，模拟 GraphRAG 未命中
irrelevant: 混入随机错误 POI
rank-noise: 打乱候选排名
```

3. Graph noise auxiliary head

在 pooled hidden state 后增加 4 类辅助头，预测当前候选池噪声类型。

```text
loss = poi_ce_loss + graph_noise_loss_weight * graph_noise_cls_loss
```

建议 `graph_noise_loss_weight=0.05`，最多先不要超过 `0.1`。

4. Best checkpoint selection

支持按 `eval_mrr` 或 `eval_top1` 自动加载最佳模型，避免出现 step400 最好但 final 变差。

### 推荐下一版训练命令

```bash
cd /mnt/data/yyl/TMP
mkdir -p logs
nohup bash -c '
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/mnt/data/yyl/miniconda3/envs/poi_data/bin/python \
src/poi_classification/train_teamlora_poi_classifier.py \
  --output-dir models/poi-teamlora-classifier-llama32-1b-graphprior-raftlite-v1 \
  --pooling last_mean \
  --graph-prior-mode rank \
  --graph-prior-alpha 4.0 \
  --mlp-logit-scale 0.3 \
  --graph-prior-dropout 0.10 \
  --graph-prior-random-cutoffs 10,30,100 \
  --graph-candidate-drop-prob 0.10 \
  --graph-target-mask-prob 0.20 \
  --graph-irrelevant-mix-prob 0.10 \
  --graph-rank-noise-prob 0.10 \
  --graph-noise-loss-weight 0.05 \
  --max-length 3072 \
  --batch-size 4 \
  --eval-batch-size 2 \
  --grad-accum 4 \
  --epochs 1 \
  --lr 1e-4 \
  --bf16 \
  --gradient-checkpointing \
  --load-best-model-at-end \
  --metric-for-best-model mrr
' > logs/poi-teamlora-graphprior-raftlite-v1.log 2>&1 &
```

### 观察重点

```text
如果 step200/400 的 top1/MRR 高，但后续仍回落：
  继续降低 lr 或 mlp_logit_scale。

如果 top20 提升但 top1/MRR 不提升：
  说明 recall/fusion 有效，重排序仍弱；再考虑 full RAAT hardest-selection。

如果全部指标下降：
  噪声过强，先关闭 irrelevant/rank-noise，只保留 mlp_logit_scale + target-mask。
```




对，当前版本是：

```text
每个样本在每次 __getitem__ 时，动态随机生成一个候选池版本。
```

它可能是：

```text
original
partial/cutoff/dropout
target-mask/counterfactual
irrelevant mix
rank-noise
```

但一次 forward 只看一个版本，不会同时看四个版本。

后续可以上完整 RAAT，但建议等这版结果出来后再决定。完整 RAAT 需要这样改：

```text
同一个样本构造 4 个候选池版本：
o = original
p = partial/masked
f = counterfactual target-mask
c = irrelevant mix

同一批样本复制成 4 份 forward：
logits_o, logits_p, logits_f, logits_c

分别计算 CE：
loss_o, loss_p, loss_f, loss_c

选择最难版本：
loss_adv = max(loss_o, loss_p, loss_f, loss_c)

总损失：
loss = loss_adv + lambda_noise * noise_cls_loss
```

优点：

```text
更接近 RAAT；
明确训练模型处理最难候选池噪声；
比随机扰动更强。
```

代价：

```text
训练显存/时间接近 3-4 倍；
max-length=3072 时压力很大；
batch_size=4 可能要降到 1 或 2；
实现要改 Trainer.compute_loss，不能只在 Dataset 里做。
```

所以我建议分阶段：

```text
第一阶段：当前轻量 RAAT
看 step200/400/600 是否稳定优于上一版。

第二阶段：如果仍出现后期退化或候选外 fallback 弱
再做 full RAAT hardest-selection。

第三阶段：如果 full RAAT 显存太大
做折中版：每个样本只采样 2 个版本 original + hard_noise，再取 max loss。
```

最现实的下一版可能不是完整 4 版本 RAAT，而是：

```text
2-view RAAT:
original + target-mask
loss = max(loss_original, loss_target_mask)
```

这个对你的 POI 任务最有效，因为最大问题是 GraphRAG top100 未命中或被 prior 绑死。