| Model Type | VRAM used | Paramters | Quantizatior | MMLU Pro
|----------|----------|----------|----------|----------|
| gemma4:e4b | 7.8GB  | 8B  | Q4_K_M | 69.4%
| gemma4:26b  | 21.6GB  | 26B  |  Q4_K_M | 82.6%
| qwen3.5:27b | 21.7GN | 27.8B | Q4_K_M | 86.1%

Used qwen3.5 for best MMLU Pro, while keeping sufficient VRAM for multi calls (base GPU is 5090 with 32G VRAM)

Potentially consider qwen3.6, but all the benchmarks were for coding assistant, hence didn't try yet.