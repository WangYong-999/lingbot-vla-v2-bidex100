# BiDex-100 · LingBot-VLA-v2

[Document](https://bidex-100.github.io/BiDex-100/document.html) · [Benchmark code](https://github.com/robbyant-research/BiDex-100)


### Create the environment

```bash
git clone --recurse-submodules https://github.com/WangYong-999/lingbot-vla-v2-bidex100.git
cd lingbot-vla-v2-bidex100
bash tools/create_train_env.sh --env-name bidex-lingbot
conda activate bidex-lingbot
```

The repository installer creates Python 3.12 and pins the PyTorch 2.8 stack.
It also installs FlashAttention and the depth packages. A matching local
FlashAttention wheel can be supplied with `--flash-attn-wheel /path/to/wheel.whl`.
Download the LingBot-VLA-v2-6B base checkpoint, Qwen3-VL tokenizer/processor,
MoGe weights, and the depth/video teacher files described in the upstream README.

### Prepare the 58-D checkpoint and data

BiDex-100 uses 58 action dimensions; the pretrained LingBot checkpoint uses 55.
Use the migrated projection-expansion utility before fine-tuning:

```bash
python scripts/bidex_create_58d_ckpt.py \
  --src /absolute/path/to/lingbot-vla-v2-6b \
  --dst /absolute/path/to/lingbot-vla-v2-6b-58d

python scripts/prepare_bidex100.py \
  --data-root /absolute/path/to/BiDex-100/downloads/BiDex-100/datasets/lerobotv30 \
  --model-path /absolute/path/to/lingbot-vla-v2-6b-58d \
  --tokenizer-path /absolute/path/to/Qwen3-VL-4B-Instruct \
  --teacher-root /absolute/path/to/lingbot-vla-v2-6b \
  --moge-path /absolute/path/to/moge-2-vitb-normal/model.pt \
  --output-dir /absolute/path/to/outputs/lingbot-bidex
```

This writes `assets/training_data/bidex_all_joint.txt` and
`configs/vla/bidex/local_joint.yaml`. It keeps the joint delta-action convention
and checkpoint milestones from the source integration, while supplying your
data, tokenizer, teacher and output paths. The released joint normalization
statistics are included at `assets/norm_stats/bidex_joint.json`.

### Train

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  bash train.sh tasks/vla/train_lingbotvla.py configs/vla/bidex/local_joint.yaml
```

The generated recipe trains for 60k steps with the source 58-D architecture and
alignment objectives. For fewer GPUs, adjust micro/global batch sizes and
gradient accumulation together. Evaluate the saved step directory (or its
`hf_ckpt` subdirectory) that contains the Hugging Face weights.

### Evaluate

From the **BiDex-100** repository, with the model environment activated:

```bash
export MODEL_REPO=/absolute/path/to/lingbot-vla-v2-bidex100
export MODEL_PY="$(command -v python)"
export MODEL_VLM_PATH=/absolute/path/to/Qwen3-VL-4B-Instruct
export MODEL_CKPT=/absolute/path/to/outputs/lingbot-bidex/global_step_60000
export BIDEX_LEADERBOARD_ROOT=/absolute/path/to/evaluation-output

MID_OVERRIDE=lingbot_smoke MODEL_GPUS=0 RENDER_GPUS=0 \
  bash scripts/evaluation/run_vec_models_eval.sh lingbot smoke
```
