# AQA

Audio quality assessment experiments, including baseline evaluation, supervised fine-tuning (SFT), and GRPO training.

## Contents

- `code/baselines/`: baseline evaluation scripts, requirements, and CSV results.
- `code/pilot_grpo/`: pilot SFT/GRPO scripts, reward functions, evaluation scripts, and CSV results. See [the experiment README](code/pilot_grpo/README.md).
- `pilot/data/`: JSONL training and validation manifests.
- `VQAThinker/`: source snapshot from [clh124/VQAThinker](https://github.com/clh124/VQAThinker). Original documentation and license files are retained.
- `MOS-Bench summary(1).xlsx`: benchmark summary workbook.

## Local setup

See the experiment-specific README files and requirements for setup and execution. Dataset and model paths may need to be adapted to your environment.

Model weights, checkpoints, training logs, local editor settings, and backups are excluded from version control. This repository does not include the complete local training environment or media datasets.
