"""Generate portable LingBot-VLA-v2 BiDex-100 co-training inputs."""

import argparse
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True, help="Expanded 58-D checkpoint")
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--teacher-root", type=Path, required=True, help="Original lingbot-vla-v2-6b checkpoint")
    parser.add_argument("--moge-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config-out", type=Path, default=Path("configs/vla/bidex/local_joint.yaml"))
    args = parser.parse_args()
    root = args.data_root.resolve()
    suites = ("Generalization", "Instruction_Following", "Limited_Demonstrations", "Long_Horizon", "Memory")
    paths = [info.parent.parent for suite in suites for info in sorted((root / suite).glob("*/meta/info.json"))]
    if not paths:
        parser.error(f"No BiDex-100 task datasets found under {root}")
    train_list = Path("assets/training_data/bidex_all_joint.txt")
    train_list.parent.mkdir(parents=True, exist_ok=True)
    train_list.write_text("".join(f"bidex {p}\n" for p in paths))
    config = yaml.safe_load(Path("configs/vla/bidex/bidex_joint.yaml").read_text())
    config["model"]["model_path"] = str(args.model_path.resolve())
    config["model"]["tokenizer_path"] = str(args.tokenizer_path.resolve())
    config["data"]["train_path"] = str(train_list)
    config["train"]["output_dir"] = str(args.output_dir.resolve())
    config["train"]["max_steps"] = 60_000
    config["train"]["num_train_epochs"] = 60_000
    config["train"]["save_milestones"] = [5_000, 10_000, 30_000, 60_000]
    align = config["train"]["align_params"]
    align["depth"]["moge_path"] = str(args.moge_path.resolve())
    align["depth"]["morgbd_path"] = str(args.teacher_root.resolve() / "depth/model.pt")
    align["video"]["ckpt_path"] = str(args.teacher_root.resolve() / "dino_video/teacher_step_10000.pth")
    align["video"]["config_path"] = str(args.teacher_root.resolve() / "dino_video/config.yaml")
    args.config_out.parent.mkdir(parents=True, exist_ok=True)
    args.config_out.write_text(yaml.safe_dump(config, sort_keys=False))
    print(f"Prepared {len(paths)} task datasets; config: {args.config_out}")


if __name__ == "__main__":
    main()
