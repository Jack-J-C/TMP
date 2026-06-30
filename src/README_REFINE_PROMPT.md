# Refine Prompt Module Layout

This directory contains the latest LoRA-A-v2 decision-style refine_prompt code copied from:

`/mnt/data/yyl/LLMMRA/scripts/evidence/`

## Directories

- `refine_prompt/`
  - `train_prompt_refiner_lora.py`
  - `generate_refined_prompts.py`
  - `refiner_prompt_styles.py`
  - `validate_decision_refined_prompts.py`

- `data_build/`
  - `build_all_evidence.py`
  - `build_teacher_distill_inputs.py`
  - `sample_teacher_distill_subset.py`
  - `sample_teacher_hard_boundary_subset.py`
  - `call_teacher_prompt_distill_api.py`
  - `sample_refiner_inputs_by_quality.py`
  - `build_poi_sft_data.py`

- `../legacy_scripts_snapshot/`
  - Full scripts snapshot from the original LLMMRA repo for compatibility/reference.

## Latest Version

The current latest refine_prompt line is LoRA-A-v2 decision-style:

- base model: `models/Llama-3.2-1B-Instruct`
- LoRA weights: `models/prompt-refiner-lora-llama32-1b-decision-v2-final`
- style flag: `--refiner-style decision`

## Train LoRA-A-v2 Decision

Run from project root after wiring imports or using the legacy snapshot path:

```bash
CUDA_VISIBLE_DEVICES=0 \
NCCL_P2P_DISABLE=1 \
NCCL_IB_DISABLE=1 \
python legacy_scripts_snapshot/evidence/train_prompt_refiner_lora.py \
  --train-inputs retrieval_assets/NewYork/distill_decision/teacher_distill_inputs_train_decision_v2_combined.jsonl \
  --train-outputs retrieval_assets/NewYork/distill_decision/teacher_prompt_outputs_train_decision_v2_combined_clean.jsonl \
  --val-inputs retrieval_assets/NewYork/distill_decision/teacher_distill_inputs_val_decision_v2_clean_matched.jsonl \
  --val-outputs retrieval_assets/NewYork/distill_decision/teacher_prompt_outputs_val_300_decision_v2_clean.jsonl \
  --refiner-style decision \
  --base-model models/Llama-3.2-1B-Instruct \
  --output-dir models/prompt-refiner-lora-llama32-1b-decision-v2 \
  --max-source-length 1536 \
  --max-target-length 256 \
  --batch-size 2 \
  --grad-accum 8 \
  --epochs 3 \
  --lr 2e-4 \
  --bf16 \
  --gradient-checkpointing
```

## Generate Final Full NYC Refine Prompts

Using the copied final LoRA weights:

```bash
CUDA_VISIBLE_DEVICES=0 \
NCCL_P2P_DISABLE=1 \
NCCL_IB_DISABLE=1 \
python legacy_scripts_snapshot/evidence/generate_refined_prompts.py \
  --inputs retrieval_assets/NewYork/evidence/teacher_distill_inputs_train.jsonl \
  --base-model models/Llama-3.2-1B-Instruct \
  --lora-path models/prompt-refiner-lora-llama32-1b-decision-v2-final \
  --output retrieval_assets/NewYork/refined_prompts_decision/lora_a_decision_v2_train_full_outputs.jsonl \
  --report retrieval_assets/NewYork/refined_prompts_decision/lora_a_decision_v2_train_full_report.json \
  --refiner-style decision \
  --limit -1 \
  --max-source-length 1536 \
  --max-new-tokens 256 \
  --batch-size 8 \
  --device cuda:0 \
  --attn-implementation sdpa \
  --overwrite
```

Validation/cleaning:

```bash
python legacy_scripts_snapshot/evidence/validate_decision_refined_prompts.py \
  --input retrieval_assets/NewYork/refined_prompts_decision/lora_a_decision_v2_train_full_outputs.jsonl \
  --output-clean retrieval_assets/NewYork/refined_prompts_decision/lora_a_decision_v2_train_full_outputs_clean.jsonl \
  --output-removed retrieval_assets/NewYork/refined_prompts_decision/lora_a_decision_v2_train_full_outputs_removed.jsonl
```

## Import Note

The copied scripts still use imports like:

`from scripts.evidence.train_prompt_refiner_lora import read_jsonl`

For immediate execution, use `legacy_scripts_snapshot/evidence/...` from the TMP project root, or refactor imports in `src/refine_prompt/` when creating the new package.
