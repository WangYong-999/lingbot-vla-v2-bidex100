#!/usr/bin/env python3
"""Create a 58D pretrained checkpoint from the 55D lingbot-vla-v2-6b.

Problem: BiDex-100 uses a 58D action space (arm14+dex44), but the pretrained
lingbot-vla-v2-6b was trained with 55D (arm14+ee14+eff2+hand12+waist4+head2+
base3+reserved4). The 3 projection layers (action_in_proj, action_out_proj,
state_proj) have width 55 in the pretrained ckpt but need width 58 in BiDex-100.

The BiDex-100 58D hand layout (BiDex-100/bidexbench/env.py: LEFT/RIGHT_SHARPA
_*ACTUATORS) is right-first [right(22), left(22)], each hand ordered
thumb(5)+index(4)+middle(4)+ring(4)+pinky(5). Mapping the pretrained [0:55]
onto the 58D dims and copying [52:55]->[55:58] (the previous "fix") made the
LAST 7 hand dims = model dims [51:58] source entirely from the pretrained
"reserved" block [51:55] (4 dims) + its copy [52:55] (3 dims):
  [51:53] = left ring PIP / left ring DIP  (verbatim reserved)
  [53:58] = left pinky CMC / MCP_FE / MCP_AA / PIP / DIP
          (CMC/MCP_FE verbatim reserved; MCP_AA/PIP/DIP copied from reserved)
The "reserved" block was NEVER trained (verified: its projection row/col
norms are ~0.55 vs ~2.9-3.9 for every trained dim). So [51:58] -- i.e. the
LEFT RING FINGERTIP + the ENTIRE LEFT PINKY -- started from essentially
untrained weights and stayed severely under-trained at low step counts
(verified at global_step_10000: action_out row norms ~1.0-1.85 and state_proj
col norms ~0.63-1.41, only ~30-60% of trained), so the distal left pinky JITTERS
violently during inference (amplified by bounds_99_woclip having no output
clamp and the Joint ControlAdapter doing pure passthrough).

The previous "fix" was ineffective: it only added [55:58] as a COPY of
reserved [52:55] (garbage -> garbage), and never touched the 4 verbatim
reserved dims [51:55] (left ring fingertip + pinky CMC/MCP_FE).

Solution: This script creates a 58D pretrained checkpoint OFFLINE by sourcing
the ENTIRE under-trained region [51:58] (7 dims) from a TRAINED pretrained
block -- the pretrained "hand12" flexion block [30:37] (right-hand
thumb/index/middle, norm ~3.8) -- instead of the reserved block:
1. Copy pretrained dims [0:51] verbatim (all TRAINED: arm/ee/eff/hand12/
   waist/head/base; the reserved [51:55] is SKIPPED, no longer placed anywhere).
2. Copy pretrained dims [30:37] (trained hand12, 7 dims) into [51:58].
This gives the left ring-fingertip + left pinky a well-scaled, structured
warm start -- the same kind of (semantically-mismatched but TRAINED) start
that let the other hand dims [14:51] converge within ~10k -- so the pinky
converges early instead of jittering. (Reusing pretrained [30:37] for both
58D [30:37] and [51:58] is fine: they're independent params and diverge
during training, exactly as the prior [52:55]->[55:58] copy already did.)

The resulting 58D checkpoint loads with post_training=True with no shape
mismatch, eliminating the need for the resize patch entirely.

Usage:
    python scripts/bidex_create_58d_ckpt.py \
        --src /path/to/lingbot-vla-v2-6b \
        --dst /path/to/lingbot-vla-v2-6b-58d
    # re-run after editing this script: add --force to replace an existing dst.

Note: Only the safetensors shards containing the 4 projection weights are
rewritten; the remaining shards are symlinked to save disk.
"""

import argparse
import json
import os
import shutil
from collections import OrderedDict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


# The 58D region that must be re-sourced from a TRAINED pretrained block
# (not the never-trained "reserved" block). In the BiDex-100 58D hand layout this
# is the LEFT RING FINGERTIP + the ENTIRE LEFT PINKY:
#   58D[51:53] = left ring PIP / ring DIP   (verbatim reserved before)
#   58D[53:58] = left pinky CMC/MCP_FE/MCP_AA/PIP/DIP
FILL_START = 51  # first 58D dim to source from the trained block
FILL_END = 58  # exclusive -> 7 dims (left ring fingertip + left pinky)
assert FILL_END - FILL_START == 7

# Pretrained source block for [51:58]: the "hand12" flexion block [30:37]
# (right-hand thumb/index/middle). It is TRAINED (norm ~3.8, vs ~0.55 for the
# reserved block) and the most finger-flexion-like pretrained quantity, giving
# [51:58] a structured, well-scaled warm start so the pinky converges early.
SRC_START = 30
SRC_END = 37  # exclusive -> 7 trained dims
assert SRC_END - SRC_START == FILL_END - FILL_START

# Tensors that need shape change (55→58) and the axis along which to expand.
RESIZE_TENSORS = {
    "model.action_out_proj.weight": 0,  # (55,768) -> (58,768), expand along dim 0 (rows)
    "model.action_out_proj.bias": 0,  # (55,)    -> (58,),    expand along dim 0
    "model.action_in_proj.weight": 1,  # (768,55) -> (768,58), expand along dim 1 (cols)
    "model.state_proj.weight": 1,  # (768,55) -> (768,58), expand along dim 1 (cols)
}


def expand_tensor(tensor: torch.Tensor, expand_dim: int) -> torch.Tensor:
    """Expand a 55D-axis pretrained tensor to 58D by re-sourcing dims [51:58].

    Result layout (length 58):
      [0:FILL_START)        = pretrained [0:FILL_START) verbatim  (all TRAINED dims)
      [FILL_START:FILL_END) = pretrained [SRC_START:SRC_END)     (trained hand12 block)
    The never-trained "reserved" pretrained block [51:55] is SKIPPED -- it is no
    longer placed at any 58D position (previously it occupied [51:55] and was
    duplicated at [55:58], which left the left ring-fingertip + left pinky
    initialized from untrained weights and jittering).
    """
    orig_size = tensor.shape[expand_dim]
    assert orig_size == 55, f"Expected dim {expand_dim} size 55, got {orig_size}"

    # Build index: [0..FILL_START-1]  +  [SRC_START..SRC_END-1]
    # (keep every TRAINED pretrained dim verbatim, then fill [51:58] from the
    # trained hand12 block instead of the reserved block).
    idx = list(range(0, FILL_START)) + list(range(SRC_START, SRC_END))
    index = torch.tensor(idx, device=tensor.device)
    assert len(idx) == 58, f"index length {len(idx)} != 58"

    result = tensor.index_select(expand_dim, index)

    # Verify
    new_size = result.shape[expand_dim]
    assert new_size == 58, f"Expected expanded dim to be 58, got {new_size}"
    # [0:FILL_START] identical to pretrained [0:FILL_START]
    leading = result.index_select(expand_dim, torch.arange(FILL_START, device=tensor.device))
    assert torch.equal(leading, tensor.index_select(expand_dim, torch.arange(FILL_START, device=tensor.device))), (
        "Leading [0:FILL_START] dims changed after expand!"
    )
    # [FILL_START:58] == pretrained [SRC_START:SRC_END]
    filled = result.index_select(expand_dim, torch.arange(FILL_START, FILL_END, device=tensor.device))
    source = tensor.index_select(expand_dim, torch.arange(SRC_START, SRC_END, device=tensor.device))
    assert torch.equal(filled, source), "Copy verification failed: [FILL_START:58] != [SRC_START:SRC_END]"

    return result


def process_shard(src_path: str, dst_path: str) -> dict:
    """Read a safetensors shard, expand resize tensors, write to dst_path.

    Returns a dict of {key: shape} for all tensors in this shard
    (with modified shapes for the expanded tensors).
    """
    tensors = {}
    shapes = {}

    with safe_open(src_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            if key in RESIZE_TENSORS:
                expand_dim = RESIZE_TENSORS[key]
                tensor = expand_tensor(tensor, expand_dim)
                print(f"  Expanded: {key} {list(tensor.shape)} (from dim {expand_dim})")
            tensors[key] = tensor
            shapes[key] = list(tensor.shape)

    # Save — use same dtype as source
    save_file(tensors, dst_path)
    print(f"  Saved: {dst_path} ({len(tensors)} keys)")
    return shapes


def main():
    parser = argparse.ArgumentParser(description="Create 58D pretrained checkpoint from 55D")
    parser.add_argument(
        "--src",
        required=True,
        help="Source 55D checkpoint directory",
    )
    parser.add_argument(
        "--dst",
        required=True,
        help="Destination 58D checkpoint directory",
    )
    parser.add_argument(
        "--force", action="store_true", help="Replace an existing destination directory (rm -rf then recreate)"
    )
    args = parser.parse_args()

    src_dir = Path(args.src)
    dst_dir = Path(args.dst)

    if src_dir.resolve() == dst_dir.resolve() or src_dir.resolve().is_relative_to(dst_dir.resolve()):
        raise ValueError("Destination must not be the source checkpoint or its parent directory")

    if not src_dir.exists():
        raise FileNotFoundError(f"Source checkpoint not found: {src_dir}")
    if dst_dir.exists():
        if not args.force:
            raise FileExistsError(
                f"Destination already exists: {dst_dir} (pass --force to replace, or remove it first if re-creating)"
            )
        # Recreate cleanly so no stale symlinks/files from a prior run linger.
        print(f"--force: removing existing destination {dst_dir}")
        shutil.rmtree(dst_dir)

    dst_dir.mkdir(parents=True, exist_ok=True)
    print(f"Creating 58D checkpoint: {src_dir} -> {dst_dir}")

    # Read index
    index_path = src_dir / "model.safetensors.index.json"
    with open(index_path) as f:
        index = json.load(f, object_pairs_hook=OrderedDict)

    weight_map = index["weight_map"]
    metadata = index.get("metadata", {})

    # Identify affected shards
    affected_shards = set()
    for key in RESIZE_TENSORS:
        shard = weight_map.get(key)
        if shard is None:
            raise KeyError(f"Resize tensor '{key}' not found in weight_map")
        affected_shards.add(shard)
        print(f"  {key} -> {shard}")

    print(f"\nAffected shards: {sorted(affected_shards)}")

    # Process each file in the checkpoint
    new_weight_map = OrderedDict()
    new_total_size = 0

    # Copy/symlink non-safetensor files and non-affected shards
    all_files = sorted(src_dir.iterdir())
    safetensors_files = set()

    for key, shard in weight_map.items():
        safetensors_files.add(shard)

    for fpath in all_files:
        fname = fpath.name
        src_file = src_dir / fname
        dst_file = dst_dir / fname

        if fname.endswith(".safetensors"):
            if fname in affected_shards:
                # Process this shard: expand resize tensors
                print(f"\nProcessing affected shard: {fname}")
                shapes = process_shard(str(src_file), str(dst_file))
                # Update weight_map and compute new total_size
                for key, shard_name in weight_map.items():
                    if shard_name == fname and key in shapes:
                        new_weight_map[key] = fname
                    elif shard_name == fname:
                        new_weight_map[key] = fname
            else:
                # Symlink unaffected shards
                os.symlink(str(src_file.resolve()), str(dst_file))
                print(f"  Symlinked: {fname}")
                # Copy weight_map entries for this shard
                for key, shard_name in weight_map.items():
                    if shard_name == fname:
                        new_weight_map[key] = fname
        elif fname == "model.safetensors.index.json":
            # Will be written later
            pass
        elif fname == "config.json":
            # Update config: change max_action_dim / max_state_dim from 55 to 58
            with open(src_file) as f:
                config = json.load(f, object_pairs_hook=OrderedDict)
            # The config is minimal for lingbot — the action dims are set by the
            # training YAML, not config.json. But update any relevant fields.
            # Note: the current lingbot config.json only has vlm_family.
            # The action dims are controlled by bidex_joint.yaml at training time.
            # Still, save the config for completeness.
            with open(dst_file, "w") as f:
                json.dump(config, f, indent=2)
            print(f"  Copied: {fname}")
        else:
            # Symlink everything else (tokenizer, etc.)
            if src_file.is_dir():
                shutil.copytree(src_file, dst_file, symlinks=True)
                print(f"  Copied dir: {fname}")
            else:
                os.symlink(str(src_file.resolve()), str(dst_file))
                print(f"  Symlinked: {fname}")

    # Compute new total size
    # We need to account for the extra parameters in the expanded tensors
    # Original: 4 tensors changed, each gaining 3 dims
    # action_out_proj.weight: (55,768)->(58,768): +3*768*4bytes = 9216
    # action_out_proj.bias:   (55,)->(58,):      +3*4bytes = 12
    # action_in_proj.weight:  (768,55)->(768,58): +768*3*4bytes = 9216
    # state_proj.weight:      (768,55)->(768,58): +768*3*4bytes = 9216
    extra_bytes = 3 * (768 * 4 + 1 * 4 + 768 * 4 + 768 * 4)
    new_total_size = metadata.get("total_size", 0) + extra_bytes
    print(f"\nOriginal total_size: {metadata.get('total_size', 0):,}")
    print(f"Extra params bytes: {extra_bytes:,}")
    print(f"New total_size: {new_total_size:,}")

    # Write new index
    new_index = OrderedDict()
    new_index["metadata"] = {"total_size": new_total_size}
    new_index["weight_map"] = new_weight_map

    with open(dst_dir / "model.safetensors.index.json", "w") as f:
        json.dump(new_index, f, indent=2, sort_keys=False)
    print(f"\nWritten: model.safetensors.index.json ({len(new_weight_map)} keys)")

    # Verify: read back the expanded tensors
    print("\n=== Verification ===")
    for shard_name in sorted(affected_shards):
        shard_path = dst_dir / shard_name
        with safe_open(str(shard_path), framework="pt", device="cpu") as f:
            for key in RESIZE_TENSORS:
                if key in f.keys():
                    t = f.get_tensor(key)
                    # Check specific shapes
                    if "weight" in key and t.shape[0] == 58:
                        assert t.shape[0] == 58
                        # [FILL_START:58] was sourced from pretrained [SRC_START:SRC_END],
                        # which is also present verbatim at 58D [SRC_START:SRC_END]
                        # (SRC_END <= FILL_START), so the two slices must be identical.
                        assert (t[FILL_START:FILL_END] == t[SRC_START:SRC_END]).all(), (
                            f"Copy check failed for {key}: [FILL_START:58] != [SRC_START:SRC_END]"
                        )
                    elif "weight" in key and t.shape[1] == 58:
                        assert t.shape[1] == 58
                        assert (t[:, FILL_START:FILL_END] == t[:, SRC_START:SRC_END]).all(), (
                            f"Copy check failed for {key}: [:,FILL_START:58] != [:,SRC_START:SRC_END]"
                        )
                    elif "bias" in key and t.shape[0] == 58:
                        assert t.shape[0] == 58
                        assert (t[FILL_START:FILL_END] == t[SRC_START:SRC_END]).all(), (
                            f"Copy check failed for {key}: [FILL_START:58] != [SRC_START:SRC_END]"
                        )
                    print(
                        f"  ✅ {key}: shape={tuple(t.shape)}, warm-start verified "
                        f"([FILL_START:58] <- pretrained [SRC_START:SRC_END])"
                    )

    print("\n58D checkpoint creation complete!")
    print(f"  Source: {src_dir}")
    print(f"  Destination: {dst_dir}")


if __name__ == "__main__":
    main()
