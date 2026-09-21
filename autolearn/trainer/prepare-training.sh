#!/usr/bin/env bash
set -euo pipefail

python3 - <<'PY'
import torch
import transformers
import peft
import trl

print(f"torch={torch.__version__}")
print(f"transformers={transformers.__version__}")
print(f"peft={peft.__version__}")
print(f"trl={trl.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available inside the trainer container")
print(f"gpu={torch.cuda.get_device_name(0)}")
print(f"gpu_total_mib={torch.cuda.get_device_properties(0).total_memory // 1024 // 1024}")
print(f"bf16_supported={torch.cuda.is_bf16_supported()}")
PY
