# microgpt
Experiments with Andrej Karpathy's MicroGPT on ESP32

Based on https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95

Character-level GPT for generating name-like strings. Train on PC, run inference in C (host or ESP32).

## Files

**input.txt** — One string per line (e.g. names, API names). The model learns to generate similar strings. Use `extract_lwip_apis.py` to build this from LWIP headers.

**train.py** — Trains on `input.txt`, exports weights to `data.c` and `data.h`. Run: `python train.py`

**infer.c** — Inference-only C program. Reads a prefix from stdin, generates a continuation. Compile: `gcc -O2 -o infer infer.c data.c -lm`

## Usage

```bash
python train.py                    # train and export data.c/data.h
gcc -O2 -o infer infer.c data.c -lm
echo "tcp_" | ./infer              # generate from prefix
./infer                            # empty input = random from BOS
```

Invalid characters in the prefix produce an error.


