# Semantic-ID GraphRAG Modules

Copied from:

`/mnt/data/yyl/LLMMRA/scripts/double_llm/`

## Files

- `build_semantic_poi_ids.py`
  - Builds deterministic POI semantic IDs:
    - `NYC::<CATEGORY_TOKEN>::<GEO_CELL>::P####`

- `build_graphrag_topk_candidates.py`
  - Builds GraphRAG-style TopK candidates.
  - Latest version includes explicit Semantic-ID graph edges:
    - `semantic_edge:last_semantic_id`
    - `semantic_edge:last2_semantic_id`
    - `semantic_edge:last_category_token`
    - `semantic_edge:last_geo_cell`
    - `semantic_edge:user_last_category_token`
    - `semantic_edge:user_last_geo_cell`
    - `semantic_edge:semantic_covisit`
  - Output is streamed line-by-line to avoid holding large JSONL outputs in memory.

- `build_c1_reranker_sft_data.py`
  - Builds old C1 Top100 -> Top50 SFT data.
  - Current recommendation: do not use the generative C1 path as the mainline.

- `train_c1_candidate_reranker_lora.py`
  - Old C1 generative LoRA trainer.

- `generate_eval_c1_candidate_reranker.py`
  - Old C1 generation/evaluation script.

## Rebuild Semantic IDs

```bash
python src/graphrag/build_semantic_poi_ids.py \
  --inputs retrieval_assets/NewYork/evidence/teacher_distill_inputs_train.jsonl retrieval_assets/NewYork/evidence/teacher_distill_inputs_val.jsonl \
  --output retrieval_assets/NewYork/double_llm/semantic_poi_ids.jsonl \
  --city-code NYC \
  --overwrite
```

## Build Semantic-ID GraphRAG Top100

Validation:

```bash
python src/graphrag/build_graphrag_topk_candidates.py \
  --inputs retrieval_assets/NewYork/evidence/teacher_distill_inputs_val.jsonl \
  --labels retrieval_assets/NewYork/evidence/teacher_distill_labels_val.jsonl \
  --index-inputs retrieval_assets/NewYork/evidence/teacher_distill_inputs_train.jsonl \
  --index-labels retrieval_assets/NewYork/evidence/teacher_distill_labels_train.jsonl \
  --semantic-map retrieval_assets/NewYork/double_llm/semantic_poi_ids.jsonl \
  --output retrieval_assets/NewYork/double_llm/graphrag_semantic_edges_v2_top100_val_candidates.jsonl \
  --top-k 100 \
  --overwrite
```

Train:

```bash
python src/graphrag/build_graphrag_topk_candidates.py \
  --inputs retrieval_assets/NewYork/evidence/teacher_distill_inputs_train.jsonl \
  --labels retrieval_assets/NewYork/evidence/teacher_distill_labels_train.jsonl \
  --index-inputs retrieval_assets/NewYork/evidence/teacher_distill_inputs_train.jsonl \
  --index-labels retrieval_assets/NewYork/evidence/teacher_distill_labels_train.jsonl \
  --semantic-map retrieval_assets/NewYork/double_llm/semantic_poi_ids.jsonl \
  --output retrieval_assets/NewYork/double_llm/graphrag_semantic_edges_v2_top100_train_candidates.jsonl \
  --top-k 100 \
  --overwrite
```

## Existing Copied Artifacts

The TMP package currently includes:

- `retrieval_assets/NewYork/double_llm/semantic_poi_ids.jsonl`
- `retrieval_assets/NewYork/double_llm/semantic_poi_ids.jsonl.stats.json`
- `retrieval_assets/NewYork/double_llm/graphrag_semantic_edges_v2_top100_train_candidates.jsonl.stats.json`
- `retrieval_assets/NewYork/double_llm/graphrag_semantic_edges_v2_top100_val_candidates.jsonl.stats.json`

Large GraphRAG candidate JSONL files are not copied by default to save disk:

- `graphrag_semantic_edges_v2_top100_train_candidates.jsonl`
- `graphrag_semantic_edges_v2_top100_val_candidates.jsonl`
