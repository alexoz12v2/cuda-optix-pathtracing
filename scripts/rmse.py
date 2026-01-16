#!/usr/bin/env python3

import sys
import argparse
import numpy as np
from PIL import Image


def load_image(path):
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.float64) / 255.0


def rmse_image(img, ref):
    diff = img - ref
    mse = np.mean(diff ** 2, axis=2)
    rmse = np.sqrt(mse)
    return rmse


def main():
    parser = argparse.ArgumentParser(
        description="Compute average RMSE from an RMSE image or from image+reference"
    )

    parser.add_argument(
        "--from-pair",
        action="store_true",
        help="Compute RMSE from (image, reference) instead of reading an RMSE image",
    )

    parser.add_argument(
        "--save-rmse",
        type=str,
        default=None,
        help="Optional output path for the RMSE image (only with --from-pair)",
    )

    args, paths = parser.parse_known_args()

    if args.from_pair:
        if len(paths) != 2:
            sys.exit("Expected: image reference")

        img = load_image(paths[0])
        ref = load_image(paths[1])

        if img.shape != ref.shape:
            sys.exit("Image and reference must have the same resolution")

        rmse = rmse_image(img, ref)
        avg_rmse = rmse.mean()

        if args.save_rmse:
            rmse_img = (rmse / rmse.max() if rmse.max() > 0 else rmse)
            rmse_img = (rmse_img * 255).astype(np.uint8)
            Image.fromarray(rmse_img, mode="L").save(args.save_rmse)

        print(avg_rmse)

    else:
        if len(paths) != 1:
            sys.exit("Expected: rmse_image")

        rmse_rgb = load_image(paths[0])

        # If RGB RMSE image, average channels
        rmse = rmse_rgb.mean(axis=2)
        avg_rmse = rmse.mean()

        print(avg_rmse)


if __name__ == "__main__":
    main()

