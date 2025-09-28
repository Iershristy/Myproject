import argparse
from pathlib import Path

from pd_gait.utils.config import load_config
from pd_gait.utils.seeding import seed_everything
from pd_gait.engine.trainer import Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train multimodal PD gait model")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg.get("seed", 42))

    save_dir = Path(cfg.get("eval", {}).get("save_dir", "experiments/default"))
    save_dir.mkdir(parents=True, exist_ok=True)

    trainer = Trainer(cfg)
    trainer.fit()


if __name__ == "__main__":
    main()

