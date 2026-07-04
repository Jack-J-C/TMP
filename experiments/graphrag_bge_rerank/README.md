# GraphRAG Large-K + BGE Rerank Probe

This directory is isolated from the main training artifacts. It is used to test whether expanding GraphRAG candidates before reranking can improve candidate coverage.

## Files

- `candidates/`: validation candidate JSONL files generated with GraphRAG Top200/300/400/500.
- `reports/graphrag_val_top200_500_coverage_summary.json`: coverage summary computed from `target_rank`.
- `scripts/summarize_graphrag_coverage.py`: standalone coverage summarizer.

## Current Val Coverage

Rows: 3959

| Candidate K | Hit Ratio |
| --- | ---: |
| Top50 | 0.572114 |
| Top100 | 0.624653 |
| Top150 | 0.653448 |
| Top200 | 0.675676 |
| Top300 | 0.710533 |
| Top400 | 0.728719 |
| Top500 | 0.745390 |

## Interpretation

Expanding GraphRAG from Top100 to Top500 adds about `+12.07` absolute candidate-hit points on val. This makes the two-stage idea viable:

1. Use GraphRAG as a high-recall candidate generator, e.g. Top300 or Top500.
2. Use an external reranker such as `BAAI/bge-reranker-v2-m3` to compress the large candidate set back to Top100 or Top50.
3. Feed the compressed candidate list into TeamLoRA / downstream reranker.

The key next metric is retention after compression:

- `GraphRAG Top500 -> BGE Top100 hit@100`
- `GraphRAG Top500 -> BGE Top50 hit@50`

If BGE Top100 stays clearly above the original GraphRAG Top100 `0.624653`, it is worth integrating. If it falls back near `0.62`, BGE is not adding useful POI-specific ranking signal.

## Regenerate Coverage

```bash
cd /mnt/data/yyl/TMP

for K in 200 300 400 500; do
  /mnt/data/yyl/miniconda3/envs/poi_data/bin/python src/graphrag/build_graphrag_topk_candidates.py \
    --inputs retrieval_assets/NewYork/evidence/teacher_distill_inputs_val.jsonl \
    --labels retrieval_assets/NewYork/evidence/teacher_distill_labels_val.jsonl \
    --index-inputs retrieval_assets/NewYork/evidence/teacher_distill_inputs_train.jsonl \
    --index-labels retrieval_assets/NewYork/evidence/teacher_distill_labels_train.jsonl \
    --semantic-map retrieval_assets/NewYork/double_llm/semantic_poi_ids.jsonl \
    --output experiments/graphrag_bge_rerank/candidates/graphrag_semantic_edges_v2_top${K}_val_candidates.jsonl \
    --report experiments/graphrag_bge_rerank/reports/graphrag_top${K}_val_report.json \
    --top-k ${K} \
    --overwrite
done

/mnt/data/yyl/miniconda3/envs/poi_data/bin/python experiments/graphrag_bge_rerank/scripts/summarize_graphrag_coverage.py \
  --inputs \
    experiments/graphrag_bge_rerank/candidates/graphrag_semantic_edges_v2_top200_val_candidates.jsonl \
    experiments/graphrag_bge_rerank/candidates/graphrag_semantic_edges_v2_top300_val_candidates.jsonl \
    experiments/graphrag_bge_rerank/candidates/graphrag_semantic_edges_v2_top400_val_candidates.jsonl \
    experiments/graphrag_bge_rerank/candidates/graphrag_semantic_edges_v2_top500_val_candidates.jsonl \
  --ks 50 100 150 200 300 400 500 \
  --output experiments/graphrag_bge_rerank/reports/graphrag_val_top200_500_coverage_summary.json
```

## BGE Rerank Probe

Run this only when GPU is available. Start with `Top500 -> Top100`; if it improves over original GraphRAG Top100 `0.624653`, then test `Top500 -> Top50`.

The default recommended probe uses only compact trajectory + transition evidence as the query and only basic candidate identity/category as the passage:

- `--query-mode trajectory_transition`
- `--passage-mode basic`
- no `--include-refined`

This tests whether BGE can compress a large candidate pool using source-side behavioral evidence, instead of relying on LORA-A refined evidence or GraphRAG numeric scores.

```bash
cd /mnt/data/yyl/TMP

CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/mnt/data/yyl/miniconda3/envs/poi_data/bin/python experiments/graphrag_bge_rerank/scripts/rerank_with_bge.py \
  --candidates experiments/graphrag_bge_rerank/candidates/graphrag_semantic_edges_v2_top500_val_candidates.jsonl \
  --joined-parquet retrieval_assets/NewYork/joined_poi_classification/val_joined_top100.parquet \
  --model-name-or-path BAAI/bge-reranker-v2-m3 \
  --output experiments/graphrag_bge_rerank/candidates/bge_top500_to_top100_val.jsonl \
  --report experiments/graphrag_bge_rerank/reports/bge_top500_to_top100_val_report.json \
  --top-in 500 \
  --top-out 100 \
  --batch-size 8 \
  --max-length 512 \
  --query-mode trajectory_transition \
  --passage-mode basic
```

For Top50:

```bash
cd /mnt/data/yyl/TMP

CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/mnt/data/yyl/miniconda3/envs/poi_data/bin/python experiments/graphrag_bge_rerank/scripts/rerank_with_bge.py \
  --candidates experiments/graphrag_bge_rerank/candidates/graphrag_semantic_edges_v2_top500_val_candidates.jsonl \
  --joined-parquet retrieval_assets/NewYork/joined_poi_classification/val_joined_top100.parquet \
  --model-name-or-path BAAI/bge-reranker-v2-m3 \
  --output experiments/graphrag_bge_rerank/candidates/bge_top500_to_top50_val.jsonl \
  --report experiments/graphrag_bge_rerank/reports/bge_top500_to_top50_val_report.json \
  --top-in 500 \
  --top-out 50 \
  --batch-size 8 \
  --max-length 512 \
  --query-mode trajectory_transition \
  --passage-mode basic
```

Useful ablations:

```bash
# Add user preference, still no refine_prompt.
--query-mode trajectory_transition_preference --passage-mode basic

# Expose GraphRAG score/sources to BGE passage text.
--query-mode trajectory_transition --passage-mode graph

# Use full compact raw_text.
--query-mode raw --passage-mode basic
```
