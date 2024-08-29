import argparse

from training import train, test
from pck_aproximator import PCKAproximator


def main():

    # Example options:
    # train ./Configs/Base.yaml
    # test ./Configs/Base.yaml

    ap = argparse.ArgumentParser("Progressive Transformers")

    # Choose between Train and Test
    ap.add_argument("mode", choices=["train", "test"],
                    help="train a model or test")
    # Path to Config
    ap.add_argument("config_path", type=str,
                    help="path to YAML config file")

    # Optional path to checkpoint
    ap.add_argument("--ckpt", type=str,
                    help="path to model checkpoint")
    
    ap.add_argument("--save_output", type=bool,
                    help='saves output of predictions')

    args = ap.parse_args()

    # If Train
    if args.mode == "train":
        train(cfg_file=args.config_path, ckpt=args.ckpt)
    # If Test
    elif args.mode == "test":
        test(cfg_file=args.config_path, ckpt=args.ckpt, save_output=args.save_output)
    else:
        raise ValueError("Unknown mode")

if __name__ == "__main__":
    main()
