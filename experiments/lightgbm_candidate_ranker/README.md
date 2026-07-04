# Lightweight Candidate Ranker Probe

This experiment is isolated from the main TeamLoRA training. It tests whether a cheap CPU ranker can compress GraphRAG Top500 candidates into Top100/Top50 while retaining more targets than the original GraphRAG prefix.

`lightgbm` is not installed in the current `poi_data` environment, so the first probe uses `sklearn.ensemble.HistGradientBoostingClassifier`, which is a CPU gradient-boosted tree model with similar tabular-ranker purpose.

## Goal

```text
GraphRAG Top500
  -> CPU feature ranker
  -> Top100 / Top50
  -> measure target retention
```

Main comparison on val:

```text
GraphRAG Top500 hit@500 = 0.745390
GraphRAG Top100 hit@100 = 0.624653
GraphRAG Top50  hit@50  = 0.572114
```

Useful result threshold:

```text
learned Top100 >= 0.68
learned Top50  >  0.572
```

## Step 1: Build Train Top500 Candidates

Run only when CPU/IO pressure is acceptable. This does not use GPU.

```bash
cd /mnt/data/yyl/TMP

nohup /mnt/data/yyl/miniconda3/envs/poi_data/bin/python src/graphrag/build_graphrag_topk_candidates.py \
  --inputs retrieval_assets/NewYork/evidence/teacher_distill_inputs_train.jsonl \
  --labels retrieval_assets/NewYork/evidence/teacher_distill_labels_train.jsonl \
  --index-inputs retrieval_assets/NewYork/evidence/teacher_distill_inputs_train.jsonl \
  --index-labels retrieval_assets/NewYork/evidence/teacher_distill_labels_train.jsonl \
  --semantic-map retrieval_assets/NewYork/double_llm/semantic_poi_ids.jsonl \
  --output experiments/lightgbm_candidate_ranker/candidates/graphrag_semantic_edges_v2_top500_train_candidates.jsonl \
  --report experiments/lightgbm_candidate_ranker/reports/graphrag_top500_train_report.json \
  --top-k 500 \
  --overwrite \
  > experiments/lightgbm_candidate_ranker/logs/build_top500_train.log 2>&1 &
```

## Step 2: Train CPU Ranker

```bash
cd /mnt/data/yyl/TMP

nohup /mnt/data/yyl/miniconda3/envs/poi_data/bin/python experiments/lightgbm_candidate_ranker/scripts/train_sklearn_candidate_ranker.py \
  --train-candidates experiments/lightgbm_candidate_ranker/candidates/graphrag_semantic_edges_v2_top500_train_candidates.jsonl \
  --val-candidates experiments/graphrag_bge_rerank/candidates/graphrag_semantic_edges_v2_top500_val_candidates.jsonl \
  --model-output experiments/lightgbm_candidate_ranker/models/sklearn_hgb_top500_ranker.pkl \
  --report-output experiments/lightgbm_candidate_ranker/reports/sklearn_hgb_top500_ranker_report.json \
  --pred-output experiments/lightgbm_candidate_ranker/candidates/sklearn_hgb_top500_val_ranked.jsonl \
  --top-in 500 \
  --top-outs 50 100 150 200 300 500 \
  --negatives-per-positive 80 \
  --hard-top-n 60 \
  --near-target-window 30 \
  --max-iter 200 \
  --learning-rate 0.06 \
  --max-leaf-nodes 31 \
  --l2-regularization 0.05 \
  > experiments/lightgbm_candidate_ranker/logs/train_sklearn_hgb_top500_ranker.log 2>&1 &
```

For a quick sanity check, add:

```text
--max-train-groups 5000 --max-val-groups 1000
```

## Notes

This ranker currently uses only structure features:

- GraphRAG rank and score.
- Candidate semantic category/cell IDs.
- Source flags such as transition, geo, user, covisit, semantic edge, time, global.

It does not read long text and does not touch GPU.
