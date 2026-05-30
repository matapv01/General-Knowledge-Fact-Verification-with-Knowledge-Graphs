"""Run ablation variants of the training pipeline.

Each variant inherits from ``configs/default.yaml`` and applies overrides.
Output dirs are auto-suffixed so artifacts don't collide.
"""

import argparse
import copy
import os
import sys

import importlib.util

import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def _load_train_module():
    """Module filenames start with a digit, so importlib.util is needed."""
    path = os.path.join(os.path.dirname(__file__), "02_train_stage3.py")
    spec = importlib.util.spec_from_file_location("train_stage3", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


VARIANTS = [
    # name                 overrides
    ("default",            {}),
    ("concat_baseline",    {"model_type": "concat_baseline"}),  # fair baseline (#5)
    ("max_nodes_5",        {"max_nodes": 5}),
    ("max_nodes_15",       {"max_nodes": 15}),
    ("max_nodes_20",       {"max_nodes": 20}),
    ("format_separators",  {"triple_format": "separators"}),
    ("kernels_11",         {"num_kernels": 11}),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_config", default="configs/default.yaml")
    parser.add_argument("--only", nargs="*", default=None,
                        help="Run only the named variants (default: all)")
    args = parser.parse_args()

    with open(args.base_config) as f:
        base_cfg = yaml.safe_load(f)

    train_mod = _load_train_module()

    for name, overrides in VARIANTS:
        if args.only and name not in args.only:
            continue

        cfg = copy.deepcopy(base_cfg)
        cfg.update(overrides)
        cfg["output_dir"] = os.path.join(base_cfg["output_dir"], f"ablation_{name}")
        os.makedirs(cfg["output_dir"], exist_ok=True)

        print("\n" + "=" * 60)
        print(f"Running ablation: {name}")
        print(f"  Overrides: {overrides}")
        print(f"  Output: {cfg['output_dir']}")
        print("=" * 60)

        train_mod.train(cfg)


if __name__ == "__main__":
    main()
