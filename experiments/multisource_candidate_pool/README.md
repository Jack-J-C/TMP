# Multi-Source Candidate Pool

This experiment builds a high-recall Top500 candidate pool by fusing GraphRAG with source-specific candidate lists. It is separate from the existing GraphRAG and ranker outputs.

The goal is to improve the Top500 coverage ceiling before applying the learned candidate ranker.

## Val Build

```bash
cd /mnt/data/yyl/TMP

/mnt/data/yyl/miniconda3/envs/poi_data/bin/python \
  experiments/multisource_candidate_pool/scripts/build_multisource_top500_candidates.py \
  --inputs retrieval_assets/NewYork/evidence/teacher_distill_inputs_val.jsonl \
  --labels retrieval_assets/NewYork/evidence/teacher_distill_labels_val.jsonl \
  --index-inputs retrieval_assets/NewYork/evidence/teacher_distill_inputs_train.jsonl \
  --index-labels retrieval_assets/NewYork/evidence/teacher_distill_labels_train.jsonl \
  --semantic-map retrieval_assets/NewYork/double_llm/semantic_poi_ids.jsonl \
  --output experiments/multisource_candidate_pool/candidates/multisource_top500_val_candidates.jsonl \
  --report experiments/multisource_candidate_pool/reports/multisource_top500_val_report.json \
  --top-k 500 \
  --overwrite
```

## Train Build

```bash
cd /mnt/data/yyl/TMP

nohup /mnt/data/yyl/miniconda3/envs/poi_data/bin/python \
  experiments/multisource_candidate_pool/scripts/build_multisource_top500_candidates.py \
  --inputs retrieval_assets/NewYork/evidence/teacher_distill_inputs_train.jsonl \
  --labels retrieval_assets/NewYork/evidence/teacher_distill_labels_train.jsonl \
  --index-inputs retrieval_assets/NewYork/evidence/teacher_distill_inputs_train.jsonl \
  --index-labels retrieval_assets/NewYork/evidence/teacher_distill_labels_train.jsonl \
  --semantic-map retrieval_assets/NewYork/double_llm/semantic_poi_ids.jsonl \
  --output experiments/multisource_candidate_pool/candidates/multisource_top500_train_candidates.jsonl \
  --report experiments/multisource_candidate_pool/reports/multisource_top500_train_report.json \
  --top-k 500 \
  --overwrite \
  > experiments/multisource_candidate_pool/logs/build_multisource_top500_train.log 2>&1 &
```

The output JSONL keeps the same top-level structure as the GraphRAG candidates, with richer `candidate_details`: `sources`, `source_ranks`, `graph_rank`, and `graph_score`.

## Current Val Result

First source-quota + RRF fusion run:

```text
GraphRAG Top500 hit@500:      0.745390
Multi-source Top500 hit@500:  0.757515
Delta:                       +0.012125
```

Multi-source source-order coverage:

```text
hit@50   0.569083
hit@100  0.625158
hit@150  0.657742
hit@200  0.678959
hit@300  0.711543
hit@400  0.736044
hit@500  0.757515
```

Applying the existing GraphRAG-trained sklearn ranker with `alpha=0.05`:

```text
hit@50   0.583986
hit@100  0.644355
hit@150  0.673908
hit@200  0.694872
hit@300  0.722657
hit@500  0.757515
```

This is below the GraphRAG-only Top500 + sklearn ranker Top100 result (`0.652437` at `alpha=0.05`), even though the Top500 ceiling is higher. The likely reason is distribution shift: the sklearn ranker was trained on GraphRAG candidate order and GraphRAG-style source details, not on the multi-source fused candidate pool. The next step is to build `multisource_top500_train_candidates.jsonl` and train a new CPU/GPU ranker on that candidate distribution.

Practical decision after the first run:

```text
The Top500 gain is real but small (+1.21 pct). Do not spend a large full-train build on this exact source-quota recipe unless the goal is only to validate distribution-shift retraining. For meaningful coverage expansion, add genuinely new recall sources instead of reweighting mostly overlapping graph/evidence sources.
```

Candidate sources worth testing next:

```text
1. Dense semantic retrieval with BGE-M3 embeddings over POI semantic/category/cell text.
2. Geo-nearest POIs from last coordinate with a larger radius/k than current evidence_geo.
3. User-specific historical next-POI candidates from all historical trajectories, not only compact evidence fields.
4. Category-conditioned global popular pools with city/time-slot filters.
5. Union with existing legacy Top100 and any TeamLoRA/teacher candidate outputs if available.
```

Existing-ranker alpha sweep on multi-source val:

```text
alpha=0.00 hit@100=0.625158 hit@200=0.678959 hit@500=0.757515
alpha=0.01 hit@100=0.636019 hit@200=0.690326 hit@500=0.757515
alpha=0.02 hit@100=0.641576 hit@200=0.692094 hit@500=0.757515
alpha=0.03 hit@100=0.644102 hit@200=0.693609 hit@500=0.757515
alpha=0.05 hit@100=0.644355 hit@200=0.694872 hit@500=0.757515
alpha=0.07 hit@100=0.643597 hit@200=0.695883 hit@500=0.757515
alpha=0.10 hit@100=0.638798 hit@200=0.695125 hit@500=0.757515
alpha=0.15 hit@100=0.629704 hit@200=0.693862 hit@500=0.757515
alpha=0.20 hit@100=0.622632 hit@200=0.693609 hit@500=0.757515
```
