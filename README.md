# Local_LLAMA_Training

Blog creation and local QLoRA SFT training pipeline.

## Pipeline Steps

1. Preprocess raw blogs

```bash
python scripts/preprocess.py --tokenizer-model hf-internal-testing/llama-tokenizer
```

2. Build conversation SFT dataset

```bash
python scripts/build_sft_dataset.py
```

3. Validate conversation dataset

```bash
python scripts/validate_sft_dataset.py
```

Use strict mode when you want warnings to fail CI:

```bash
python scripts/validate_sft_dataset.py --strict
```

4. Train QLoRA SFT model

```bash
python scripts/train_qlora_sft.py \
	--model-name meta-llama/Llama-3.2-3B-Instruct \
	--train-file data/processed/sft/train_conversations.jsonl \
	--val-file data/processed/sft/val_conversations.jsonl \
	--output-root outputs/qlora_sft \
	--max-seq-length 2048 \
	--num-train-epochs 3 \
	--per-device-train-batch-size 1 \
	--gradient-accumulation-steps 16 \
	--learning-rate 2e-4 \
	--eval-steps 50 \
	--save-steps 50
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Outputs

- Conversation SFT files: `data/processed/sft/`
- SFT conversion run metadata: `data/meta/sft_runs/`
- SFT validation reports: `data/meta/validation_runs/`
- QLoRA checkpoints and adapter weights: `outputs/qlora_sft/`
- Training run metadata: `data/meta/training_runs/`
