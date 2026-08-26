import os
from pathlib import Path
import csv
import cv2
from tqdm import tqdm


def tile_image(img, tile_size=256):
    h, w = img.shape[:2]

    # compute largest divisible region
    new_h = (h // tile_size) * tile_size
    new_w = (w // tile_size) * tile_size

    if new_h == 0 or new_w == 0:
        raise ValueError(f"Image too small: {h}x{w}")

    # center crop (better than top-left)
    y_offset = (h - new_h) // 2
    x_offset = (w - new_w) // 2

    img = img[y_offset:y_offset + new_h, x_offset:x_offset + new_w]

    tiles = {}
    nrows = new_h // tile_size
    ncols = new_w // tile_size

    for r in range(nrows):
        for c in range(ncols):
            y0 = r * tile_size
            y1 = (r + 1) * tile_size
            x0 = c * tile_size
            x1 = (c + 1) * tile_size
            tiles[f"r{r}_c{c}"] = img[y0:y1, x0:x1]

    return tiles


def tile_tif_dataset(
    input_dir,
    output_dir,
    dataset_name,
    image_ext=".tif",
    output_ext=".tif",
    tile_size=256,
):
    """
    Tiles each image into non-overlapping tile_size x tile_size tiles
    and saves to output_dir, writing metadata to csv.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = output_dir / "tiles_metadata.csv"
    image_paths = sorted(input_dir.rglob(f"*{image_ext}"))

    with open(metadata_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "dataset",
            "parent_path",
            "parent_name",
            "tile_name",
            "tile_path",
            "tile_id",
            "row",
            "col",
        ])

        for img_path in tqdm(image_paths, unit="image", desc=f"Tiling {dataset_name}"):
            img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"Skipping unreadable file: {img_path}")
                continue

            try:
                tiles = tile_image(img, tile_size=tile_size)
            except ValueError as e:
                print(f"Skipping {img_path}: {e}")
                continue

            stem = img_path.stem
            for rc, tile in tiles.items():
                # rc looks like "r0_c0"
                row = int(rc.split("_")[0][1:])
                col = int(rc.split("_")[1][1:])

                tile_name = f"{stem}_{rc}{output_ext}"
                tile_path = output_dir / tile_name

                ok = cv2.imwrite(str(tile_path), tile)
                if not ok:
                    raise RuntimeError(f"Failed to write tile: {tile_path}")

                writer.writerow([
                    dataset_name,
                    str(img_path),
                    img_path.name,
                    tile_name,
                    str(tile_path),
                    rc,
                    row,
                    col,
                ])

    print(f"Done. Wrote tiles to: {output_dir}")
    print(f"Metadata saved to: {metadata_path}")


def main():
    root_dir = Path("/cgi/data/nvesd/workspaces/logan/data/remote_sensing")

    # # -------------------------
    # # ( Sim ): SynRS3D
    # # -------------------------
    # synrs3d_dir = root_dir / "sim" / "synrs3d"
    
    # for data_dir in os.listdir(synrs3d_dir):
    #     # for data_type in ["opt", "pre_opt"]:
    #     data_type = "gt_ss_mask"

    #     input_dir = synrs3d_dir / data_dir / data_type
    #     output_dir = root_dir / "tiled" / "synrs3d" / data_dir / data_type
    #     dataset_name = f"synrs3d_{data_dir}_{data_type}"

    #     tile_tif_dataset(
    #         input_dir=input_dir,
    #         output_dir=output_dir,
    #         dataset_name=dataset_name,
    #     )
    

    # -------------------------
    # ( Real )
    # -------------------------
    real_dirs = {
        # "DFC18": root_dir / "real" / "DFC18" / "DFC18" / "opt",
        # "DFC19": root_dir / "real" / "DFC19" / "gt_ss_mask",
        # "GeoNRW": root_dir / "real" / "GeoNRW" / "opt",
        # "OGC_ARG": root_dir / "OGC_ARG" / "OGC_ARG" / "opt",
        # "OGC_ATL": root_dir / "OGC_ATL" / "OGC_ATL" / "opt",
        "OEM": root_dir / "full" / "real" / "OEM" / "gt_ss_mask",
    }

    for dataset, data_dir in real_dirs.items():
        output_dir = root_dir / "tiled" / "real" / dataset / "gt_ss_mask"
        dataset_name = f"{dataset}_gt_ss_mask"

        tile_tif_dataset(
            input_dir=data_dir,
            output_dir=output_dir,
            dataset_name=dataset_name,
        )

    print("Done!")


if __name__ == "__main__":
    main()