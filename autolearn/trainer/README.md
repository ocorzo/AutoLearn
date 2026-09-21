# AutoLearn trainer environment

This image is isolated from the running Chaak inference service. It uses the Qwen3.5 BF16 LoRA path with CUDA, PyTorch, Transformers v5, PEFT, and TRL.

The actual MVP trainer intentionally uses the standard `Transformers + PEFT` LoRA implementation. The available Unsloth package was incompatible with the Transformers version required by Qwen3.5 in this environment; Unsloth is an optimization rather than a requirement for LoRA.

The first trial mounts these host directories under `/workspace`:

- `trial`: immutable inputs and evaluation artifacts for the trial;
- `models`: Hugging Face model checkpoint cache;
- `outputs`: candidate adapters, logs, and metrics;
- `hf-cache`: dependency and tokenizer cache.

The trainer must not bind to ports or replace the production `llama-server` container. It writes only to the mounted `outputs` directory. `train_lora.py` rejects an attempt to use the frozen evaluation file as training input and requires at least 75% reasoning anchors.

`run-training.sh` is deliberately an explicit launch command. Preparing this environment does not launch a training run or promote an adapter to production.
