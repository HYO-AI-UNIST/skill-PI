from __future__ import annotations

import argparse
import configparser
import os
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import pi05 as pi05_module
from pi05 import PI05


class RandomPi05Dataset(Dataset):
    """Tiny smoke-test dataset with the same batch structure PI05 expects."""

    def __init__(
        self,
        *,
        num_samples: int,
        num_images: int,
        image_size: int,
        action_horizon: int,
        action_dim: int,
        state_low: float,
        state_high: float,
    ) -> None:
        self.num_samples = num_samples
        self.num_images = num_images
        self.image_size = image_size
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        self.state_low = state_low
        self.state_high = state_high

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> dict[str, Any]:
        images = [
            torch.empty(3, self.image_size, self.image_size).uniform_(-1.0, 1.0)
            for _ in range(self.num_images)
        ]
        state = torch.empty(self.action_dim).uniform_(self.state_low, self.state_high)
        actions = torch.empty(self.action_horizon, self.action_dim).uniform_(-1.0, 1.0)
        return {
            "images": images,
            "images_mask": torch.ones(self.num_images, dtype=torch.bool),
            "task": f"smoke training sample {index}",
            "state": state,
            "actions": actions,
        }


class TensorDictPi05Dataset(Dataset):
    """Loads a torch-saved tensor dict/list.

    Supported .pt/.pth shapes:
      1. list[dict], each dict containing images, state, actions, and task or tokens.
      2. dict of tensors/lists with batch dimension on images/state/actions.

    Expected model batch keys after collation:
      observation = {
          "images": list[Tensor[B,3,H,W]],
          "images_mask": list[BoolTensor[B]],
          "task": list[str], "state": Tensor[B,Ad],
          # or already-tokenized mode:
          "tokens": LongTensor[B,L], "tokens_mask": BoolTensor[B,L],
      }
      actions = Tensor[B,Ah,Ad]
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)
        self.data = torch.load(self.path, map_location="cpu")
        if not isinstance(self.data, (list, tuple, dict)):
            raise TypeError(f"Unsupported dataset object: {type(self.data)!r}")

    def __len__(self) -> int:
        if isinstance(self.data, dict):
            if "actions" not in self.data:
                raise KeyError("Tensor dict dataset must contain an 'actions' key.")
            return len(self.data["actions"])
        return len(self.data)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if isinstance(self.data, dict):
            sample: dict[str, Any] = {}
            for key, value in self.data.items():
                if isinstance(value, torch.Tensor):
                    sample[key] = value[index]
                elif isinstance(value, (list, tuple)):
                    sample[key] = value[index]
                else:
                    sample[key] = value
            return sample
        return dict(self.data[index])


def collate_pi05(samples: list[dict[str, Any]]) -> tuple[dict[str, Any], torch.Tensor]:
    first = samples[0]
    actions = torch.stack([sample["actions"] for sample in samples], dim=0)

    images_by_view: list[torch.Tensor] = []
    image_masks_by_view: list[torch.Tensor] = []
    for view_idx in range(len(first["images"])):
        images_by_view.append(torch.stack([sample["images"][view_idx] for sample in samples], dim=0))
        if "images_mask" in first:
            mask_value = [
                sample["images_mask"][view_idx]
                if isinstance(sample["images_mask"], torch.Tensor)
                else sample["images_mask"][view_idx]
                for sample in samples
            ]
            image_masks_by_view.append(torch.as_tensor(mask_value, dtype=torch.bool))
        else:
            image_masks_by_view.append(torch.ones(len(samples), dtype=torch.bool))

    observation: dict[str, Any] = {
        "images": images_by_view,
        "images_mask": image_masks_by_view,
        "state": torch.stack([sample["state"] for sample in samples], dim=0),
    }

    if "tokens" in first:
        observation["tokens"] = torch.stack([sample["tokens"] for sample in samples], dim=0)
        observation["tokens_mask"] = torch.stack([sample["tokens_mask"] for sample in samples], dim=0)
    else:
        observation["task"] = [str(sample["task"]) for sample in samples]

    return observation, actions


def move_observation_to_device(observation: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in observation.items():
        if key in {"images", "images_mask"}:
            moved[key] = [item.to(device, non_blocking=True) for item in value]
        elif isinstance(value, torch.Tensor):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value
    return moved


def build_dataset(args: argparse.Namespace, model_cfg: configparser.ConfigParser) -> Dataset:
    action_cfg = model_cfg["ActionExpert"]
    model_section = model_cfg["PI05"]
    if args.data:
        return TensorDictPi05Dataset(args.data)
    return RandomPi05Dataset(
        num_samples=args.smoke_samples,
        num_images=args.num_images,
        image_size=int(model_section["image_width"]),
        action_horizon=int(action_cfg["action_horizontal"]),
        action_dim=int(action_cfg["action_dim"]),
        state_low=float(action_cfg["state_lowerbound"]),
        state_high=float(action_cfg["state_upperbound"]),
    )


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    epoch: int,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "epoch": epoch,
            "args": vars(args),
        },
        path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the local PI05 flow-matching model.")
    parser.add_argument("--config", default="config.cfg", help="Path to PI05 config.cfg.")
    parser.add_argument("--data", default=None, help="Optional .pt/.pth dataset path. Omit for random smoke data.")
    parser.add_argument("--output-dir", default="checkpoints/pi05", help="Directory for checkpoints.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--resume", default=None, help="Checkpoint to resume from.")
    parser.add_argument("--debug", type=int, default=0, help="Sets pi05.DEBUG. Use 0 for real training.")
    parser.add_argument("--num-images", type=int, default=3, help="Only used by random smoke dataset.")
    parser.add_argument("--smoke-samples", type=int, default=8, help="Only used without --data.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    pi05_module.DEBUG = args.debug

    config = configparser.ConfigParser()
    config_path = Path(args.config)
    if not config_path.is_absolute() and not config_path.exists():
        config_path = Path(__file__).resolve().parent / config_path
    if not config.read(config_path):
        raise FileNotFoundError(f"Could not read config: {config_path}")

    dataset = build_dataset(args, config)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_pi05,
    )

    model = PI05(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    start_epoch = 0
    global_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt.get("epoch", 0))
        global_step = int(ckpt.get("step", 0))

    model.train()
    output_dir = Path(args.output_dir)

    for epoch in range(start_epoch, args.epochs):
        for observation, actions in loader:
            observation = move_observation_to_device(observation, device)
            actions = actions.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            loss_per_dim = model.pi05_whole(observation, actions)
            loss = loss_per_dim.mean()
            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            global_step += 1
            if global_step % args.log_every == 0:
                print(
                    f"epoch={epoch + 1}/{args.epochs} "
                    f"step={global_step} "
                    f"loss={loss.item():.6f}"
                )
            if args.save_every > 0 and global_step % args.save_every == 0:
                save_checkpoint(
                    output_dir / f"step_{global_step}.pt",
                    model=model,
                    optimizer=optimizer,
                    step=global_step,
                    epoch=epoch,
                    args=args,
                )

    save_checkpoint(
        output_dir / "last.pt",
        model=model,
        optimizer=optimizer,
        step=global_step,
        epoch=args.epochs,
        args=args,
    )


if __name__ == "__main__":
    main()
