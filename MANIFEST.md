# TMP New Project Preparation Manifest

Created for the TeamLoRA / 1B-only POI project preparation.

## Contents

- `models/Llama-3.2-1B-Instruct/`
  - Source: `/mnt/data/yyl/LLMMRA/models/Llama-3.2-1B-Instruct/`
  - Use: shared 1B base model.

- `models/prompt-refiner-lora-llama32-1b-decision-v2-final/`
  - Source: `/mnt/data/yyl/LLMMRA/models/prompt-refiner-lora-llama32-1b-decision-v2/final/`
  - Use: distilled 1B LoRA-A-v2 weights for refine_prompt generation.

- `dataset/NewYork/`
  - Source: `/mnt/data/yyl/LLMMRA/dataset/NewYork/`
  - Use: original NYC full dataset files.
  - Includes: `NY_train.csv`, `NY_val.csv`, `NY_test.csv`, `graph_X.csv`, `metadata.json`.

- `retrieval_assets/NewYork/evidence/`
  - Source: `/mnt/data/yyl/LLMMRA/retrieval_assets/NewYork/evidence/`
  - Use: constructed evidence/schema-like inputs and labels.
  - Includes sequence, geo, preference evidence and `teacher_distill_inputs_*` / `teacher_distill_labels_*`.

- `retrieval_assets/NewYork/refined_prompts_decision/`
  - Source: `/mnt/data/yyl/LLMMRA/retrieval_assets/NewYork/refined_prompts_decision/`
  - Use: final full NYC refine_prompt outputs from LoRA-A-v2 and cleaned/high-medium variants.

- `retrieval_assets/NewYork/refined_prompts/`
  - Source: `/mnt/data/yyl/LLMMRA/retrieval_assets/NewYork/refined_prompts/`
  - Use: original refiner input JSONL examples for schema reference.

- `retrieval_assets/NewYork/distill_decision/`
  - Source: selected files from `/mnt/data/yyl/LLMMRA/retrieval_assets/NewYork/distill_decision/`
  - Use: LoRA-A-v2 decision-style refiner retraining data.
  - Includes combined clean train inputs/outputs and matched val clean inputs/outputs.

- `retrieval_assets/NewYork/poi_sft/`
  - Source: selected raw files from `/mnt/data/yyl/LLMMRA/retrieval_assets/NewYork/poi_sft/`
  - Use: original raw SFT train/val/test schema reference.
  - Includes: `stage1_train_raw.jsonl`, `stage1_val_raw.jsonl`, `stage1_test_raw.jsonl` and stats.

- `docs/`
  - Source: selected project docs from `/mnt/data/yyl/LLMMRA/`
  - Use: current project context and historical pipeline notes.

- `src/refine_prompt/`
  - Source: latest refine_prompt scripts from `/mnt/data/yyl/LLMMRA/scripts/evidence/`
  - Use: refiner LoRA training, generation, prompt style, validation.

- `src/data_build/`
  - Source: latest evidence/distillation/data construction scripts from `/mnt/data/yyl/LLMMRA/scripts/evidence/`
  - Use: rebuild teacher inputs, call teacher API, sample distill data, build POI SFT data.

- `src/graphrag/`
  - Source: latest Semantic-ID GraphRAG scripts from `/mnt/data/yyl/LLMMRA/scripts/double_llm/`
  - Use: build POI semantic IDs, construct Semantic-ID GraphRAG TopK candidates, and keep old C1 reference scripts.

- `retrieval_assets/NewYork/double_llm/`
  - Source: selected light artifacts from `/mnt/data/yyl/LLMMRA/retrieval_assets/NewYork/double_llm/`
  - Use: existing Semantic-ID map, Semantic-ID GraphRAG Top100 candidate files, and GraphRAG stats.
  - Includes:
    - `semantic_poi_ids.jsonl`
    - `graphrag_semantic_edges_v2_top100_train_candidates.jsonl`
    - `graphrag_semantic_edges_v2_top100_val_candidates.jsonl`
    - corresponding `.stats.json` files.

- `legacy_scripts_snapshot/`
  - Source: `/mnt/data/yyl/LLMMRA/scripts/`
  - Use: full original scripts snapshot. Prefer this for immediate execution because imports still expect the original `scripts.evidence.*` package layout.

## Notes

- No standalone file named `schema` or `shema` was found in the source repository.
- The schema reference is represented by the raw SFT JSONL files and evidence/distill input JSONL files.
- Files were copied, not moved.
