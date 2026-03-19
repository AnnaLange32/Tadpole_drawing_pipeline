import os
import cv2
import numpy as np
import random
import csv
import argparse
from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class PartEntry:
    filename: str
    part: str
    image_path: str
    cropped_h: int = 0
    cropped_w: int = 0


CANVAS_FILL_FRACTION = 0.85  # head+torso+legs fill this fraction of canvas
CANVAS_TOP_MARGIN_FRACTION = 0.08
DENSITY_REJECT_THRESHOLD = 0.25


def load_as_grayscale(img_path: str) -> Optional[np.ndarray]:
    """
    Format-aware grayscale loader that correctly handles RGBA PNGs, BGR JPEGs, and grayscale images.
    - RGBA PNGs: uses inverted alpha channel as grayscale mask (lines=dark, background=white)
    - BGR/JPEG: converts to grayscale, binarises JPEGs to remove compression artifacts
    - Grayscale: returns as-is

    Args:
        img_path: Path to the image file to load.

    Returns:
        img: Grayscale image as a 2D NumPy array where lines are dark (0) and background is white (255), or None if loading fails.
    """
    img_bgra = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img_bgra is None:
        return None
    if img_bgra.ndim == 3 and img_bgra.shape[2] == 4:
        return 255 - img_bgra[:, :, 3]
    elif img_bgra.ndim == 3 and img_bgra.shape[2] == 3:
        img = cv2.cvtColor(img_bgra, cv2.COLOR_BGR2GRAY)
        if img_path.lower().endswith((".jpg", ".jpeg")):
            _, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
            img = 255 - img
        return img
    return img_bgra


def crop_to_content(
    img: np.ndarray, threshold: int = 30, max_fill_frac: float = 0.95
) -> Optional[np.ndarray]:
    """
    Return the tightly-cropped region containing all pixels below `threshold` (dark lines).
    Returns None if the image is essentially white or the crop covers >max_fill_frac
    of the full image in BOTH dimensions.

    Args:
        img: Input grayscale image as a 2D NumPy array where lines are dark (0) and background is white (255).
        threshold: Pixel intensity threshold to consider as "content" (default=30).
        max_fill_frac: Maximum allowed fraction of the image that the crop can fill in both dimensions (default=0.95).

    Returns:
        cropped_img: Cropped image containing the content, or None if the image is essentially white or the crop is too large.
    """
    mask = img < (255 - threshold)
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return None
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    crop_h = rmax - rmin + 1
    crop_w = cmax - cmin + 1
    full_h, full_w = img.shape
    if (crop_h / full_h) > max_fill_frac and (crop_w / full_w) > max_fill_frac:
        return None

    return img[rmin : rmax + 1, cmin : cmax + 1]


def load_pool(
    output_folder: str, csv_path: str, csv_all_path: str
) -> Tuple[dict, dict]:
    """
    Load body part images from disk and filter them based on CSV metadata.
    Frequencies are calculated from csv, which includes all frames
    (also those with poor visibility), giving a more representative distribution.

    Args:
        output_folder: Folder containing subfolders for each body part (head, torso, arms, legs) with images.
        csv_path: Path to the CSV file containing visibility ratings for each image.
        csv_all_path: Path to the CSV file containing visibility ratings for all frames (used for frequency calculations).

    Returns:
        pool: Dictionary with keys "head", "torso", "arms", "legs" and values as lists of PartEntry objects for each part.
        frequencies: Dictionary with keys "head", "torso", "arms", "legs" and values as the frequency of visible parts calculated from csv_all_path.
    """
    pool = {"head": [], "torso": [], "arms": [], "legs": []}

    filename_filter_body = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row["filename"].strip()
            filename_filter_body[filename] = {
                "head": row["head_visible"].strip(),
                "torso": row["torso_visible"].strip(),
                "arms": row["arms_visible"].strip(),
                "legs": row["legs_visible"].strip(),
            }

    all_counts = {"head": 0, "torso": 0, "arms": 0, "legs": 0}
    total_all = 0
    with open(csv_all_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_all += 1
            all_counts["head"] += 1 if row["head_visible"].strip() == "1" else 0
            all_counts["torso"] += 1 if row["torso_visible"].strip() == "1" else 0
            all_counts["arms"] += 1 if row["arms_visible"].strip() == "1" else 0
            all_counts["legs"] += 1 if row["legs_visible"].strip() == "1" else 0

    frequencies = {}
    for part, count in all_counts.items():
        freq = count / total_all if total_all > 0 else 1.0
        frequencies[part] = min(freq, 1.0)
        print(f"  [freq from all] {part:<6} {count}/{total_all} = {freq:.0%}")

    for part in ["head", "torso", "arms", "legs"]:
        part_folder = os.path.join(output_folder, part)
        if not os.path.exists(part_folder):
            print(f"  [WARN] folder not found: {part_folder}")
            continue

        for fname in os.listdir(part_folder):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                continue

            # Strip the leading "partname_" prefix to recover original filename
            original_fname = "_".join(fname.split("_")[1:])
            if original_fname not in filename_filter_body:
                stem = os.path.splitext(original_fname)[0]
                original_fname = next(
                    (k for k in filename_filter_body if os.path.splitext(k)[0] == stem),
                    None,
                )
                if original_fname is None:
                    continue

            if filename_filter_body[original_fname][part] != "1":
                continue

            img_path = os.path.join(part_folder, fname)
            img = load_as_grayscale(img_path)
            if img is None:
                continue

            cropped = crop_to_content(img)
            if cropped is None:
                continue

            entry = PartEntry(
                filename=original_fname,
                part=part,
                image_path=img_path,
                cropped_h=cropped.shape[0],
                cropped_w=cropped.shape[1],
            )
            pool[part].append(entry)
    return pool, frequencies


def compute_proportions(pool: dict, canvas_size: int) -> dict:
    """
    Derive target_h (and target_w for arms/torso) from the median CROPPED sizes.
    Head + torso + legs should together fill ~85% of canvas height.

    Args:
        pool: Dictionary with keys "head", "torso", "arms", "legs" and values as lists of PartEntry objects for each part.
        canvas_size: Size of the square canvas in pixels (e.g., 512).

    Returns:
        proportions: Dictionary with keys "head", "torso", "arms", "legs" and values as dictionaries containing target heights and widths for each part, e.g.:
      {
        "head": {"target_h": 120, "target_w": None},
        "torso": {"target_h": 200, "target_w": 110},
        "arms": {"target_h": 180, "target_w": 90},
        "legs": {"target_h": 220, "target_w": None},
      }
    """
    median_h = {}
    median_w = {}
    for part, entries in pool.items():
        if entries:
            median_h[part] = int(np.median([e.cropped_h for e in entries]))
            median_w[part] = int(np.median([e.cropped_w for e in entries]))
        else:
            median_h[part] = 1
            median_w[part] = 1

    layout_parts = ["head", "torso", "legs"]
    total_raw = sum(median_h.get(p, 1) for p in layout_parts)
    usable_h = canvas_size * CANVAS_FILL_FRACTION

    proportions = {}
    for part in ["head", "torso", "arms", "legs"]:
        raw_h = median_h.get(part, 1)
        raw_w = median_w.get(part, 1)
        target_h = max(10, int((raw_h / total_raw) * usable_h))

        if part in ("arms", "torso"):
            prop_w = max(4, int(raw_w * (target_h / raw_h)))
            max_w = int(canvas_size * (0.70 if part == "arms" else 0.55))
            target_w = min(prop_w, max_w)
            if target_w < prop_w:
                target_h = max(4, int(raw_h * (target_w / raw_w)))
        else:
            target_w = None

        proportions[part] = {"target_h": target_h, "target_w": target_w}
        print(
            f"  [proportions] {part:<6} h={target_h}px"
            + (f"  w={target_w}px" if target_w else "")
        )

    return proportions


def sample_part(pool: dict, part: str) -> Optional[PartEntry]:
    """
    Randomly select a part entry from the pool.

    Args:
        pool: Dictionary with keys "head", "torso", "arms", "legs" and values as lists of PartEntry objects for each part.
        part: The body part to sample ("head", "torso", "arms", or "legs").

    Returns:
        entry: A randomly selected PartEntry object for the specified part, or None if no entries are available.
    """
    entries = pool.get(part, [])
    return random.choice(entries) if entries else None


def load_cropped(entry: PartEntry, verbose: bool = False) -> Optional[np.ndarray]:
    """
    Load and crop an image from a PartEntry using format-aware loading.
    Rejects crops with >25% black pixel density (solid blocks, not skeleton lines).

    Args:
        entry: A PartEntry object containing the image path and metadata for a body part.
        verbose: If True, prints debug information about loading and cropping decisions.

    Returns:
        cropped_img: Cropped grayscale image as a 2D NumPy array where lines are dark (0) and background is white (255), or None if loading fails or the crop is rejected due to high density.
    """
    img = load_as_grayscale(entry.image_path)
    if img is None:
        return None

    cropped = crop_to_content(img)
    if verbose:
        if cropped is None:
            print(
                f"  [CROP REJECTED] {entry.part} {os.path.basename(entry.image_path)}  full={img.shape}"
            )

    if cropped is not None:
        nz = int(np.count_nonzero(cropped < 225))
        density = nz / cropped.size if cropped.size > 0 else 0
        if density > DENSITY_REJECT_THRESHOLD:
            if verbose:
                print(f"  [DENSITY REJECT] {entry.part} density={density:.2%}")
            return None

    return cropped


def scale_part(
    img: np.ndarray, target_h: int, target_w: Optional[int] = None
) -> np.ndarray:
    """
    Scale an image to a target height, optionally with a target width.
    Width is computed proportionally if target_w is not specified.
    Args:
        img: Input grayscale image as a 2D NumPy array where lines are dark (0) and background is white (255).
        target_h: Desired height of the output image in pixels.
        target_w: Optional desired width of the output image in pixels. If None, width is scaled proportionally to the original aspect ratio.

    Returns:
        scaled_img: Scaled image as a 2D NumPy array with the specified target height and width.
    """
    h, w = img.shape
    if target_w is not None:
        new_h, new_w = target_h, target_w
    else:
        scale = target_h / h
        new_h = target_h
        new_w = max(1, int(w * scale))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)


def place_on_canvas(
    canvas: np.ndarray, part_img: np.ndarray, centre_x: int, top_y: int
) -> int:
    """
    Place a part image onto a canvas at a specified position with clipping.
    Returns the bottom y-coordinate of the placed image.

    Args:
        canvas: The canvas image as a 2D NumPy array where the part will be placed.
        part_img: The part image as a 2D NumPy array to be placed on the canvas.
        centre_x: The x-coordinate of the center position on the canvas where the part should be placed.
        top_y: The y-coordinate of the top edge where the part should be placed on the canvas.

    Returns:
        bottom_y: The y-coordinate of the bottom edge of the placed part image on the canvas after placement.
    """
    h, w = part_img.shape
    ch, cw = canvas.shape
    x0, x1 = centre_x - w // 2, centre_x - w // 2 + w
    y0, y1 = top_y, top_y + h

    cx0, cx1 = max(0, x0), min(cw, x1)
    cy0, cy1 = max(0, y0), min(ch, y1)
    px0 = cx0 - x0
    px1 = px0 + (cx1 - cx0)
    py0 = cy0 - y0
    py1 = py0 + (cy1 - cy0)

    if cx1 > cx0 and cy1 > cy0:
        roi = canvas[cy0:cy1, cx0:cx1]
        canvas[cy0:cy1, cx0:cx1] = cv2.bitwise_and(roi, part_img[py0:py1, px0:px1])

    return y1


def assemble_figure(
    pool: dict,
    proportions: dict,
    frequencies: dict,
    canvas_size: int = 512,
    joint_overlap_frac: float = 0.06,
) -> Optional[np.ndarray]:
    """
    Stack head → torso → legs top-to-bottom with a small overlap so joints touch.
    Each part is only included with probability = its normalised frequency.
    Arms anchor to torso midpoint, or best available fallback.

    Args:
        pool: Dictionary with keys "head", "torso", "arms", "legs" and values as lists of PartEntry objects for each part.
        proportions: Dictionary with keys "head", "torso", "arms", "legs" and values as dictionaries containing target heights and widths for each part.
        frequencies: Dictionary with keys "head", "torso", "arms", "legs" and values as the frequency of visible parts calculated from csv_all_path.
        canvas_size: Size of the square canvas in pixels (e.g., 512).
        joint_overlap_frac: Fraction of part height to overlap between joints (default=0.06 means 6% overlap).

    Returns:
        canvas: A 2D NumPy array representing the assembled figure on a white background, or None if the pool is too sparse to assemble a figure.
    """
    canvas = np.full((canvas_size, canvas_size), 255, dtype=np.uint8)
    cx = canvas_size // 2

    def get_scaled(part) -> Optional[np.ndarray]:
        max_freq = max(frequencies.values()) if frequencies else 1.0
        norm_freq = frequencies.get(part, 1.0) / max_freq if max_freq > 0 else 1.0
        if random.random() > norm_freq:
            return None
        entry = sample_part(pool, part)
        if entry is None:
            return None
        img = load_cropped(entry)
        if img is None:
            return None
        p = proportions[part]
        return scale_part(img, p["target_h"], p.get("target_w"))

    head_img = get_scaled("head")
    torso_img = get_scaled("torso")
    arms_img = get_scaled("arms")
    legs_img = get_scaled("legs")

    if all(x is None for x in [head_img, torso_img, legs_img]):
        return None

    current_y = int(canvas_size * CANVAS_TOP_MARGIN_FRACTION)
    head_bottom_y: Optional[int] = None
    torso_mid_y: Optional[int] = None
    legs_top_y: Optional[int] = None

    if head_img is not None:
        bottom = place_on_canvas(canvas, head_img, cx, current_y)
        head_bottom_y = bottom
        overlap = int(head_img.shape[0] * joint_overlap_frac)
        current_y = bottom - overlap

    if torso_img is not None:
        torso_top = current_y
        bottom = place_on_canvas(canvas, torso_img, cx, torso_top)
        torso_mid_y = torso_top + torso_img.shape[0] // 2
        overlap = int(torso_img.shape[0] * joint_overlap_frac)
        current_y = bottom - overlap

    if legs_img is not None:
        legs_top_y = current_y
        place_on_canvas(canvas, legs_img, cx, current_y)

    if arms_img is not None:
        if torso_mid_y is not None:
            anchor_y = torso_mid_y
        elif head_bottom_y is not None and legs_top_y is not None:
            anchor_y = (head_bottom_y + legs_top_y) // 2
        elif head_bottom_y is not None:
            anchor_y = head_bottom_y + arms_img.shape[0] // 4
        elif legs_top_y is not None:
            anchor_y = legs_top_y - arms_img.shape[0] // 4
        else:
            anchor_y = canvas_size // 2

        arms_top = anchor_y - arms_img.shape[0] // 2
        place_on_canvas(canvas, arms_img, cx, arms_top)

    return canvas


def generate_figures(
    output_folder: str,
    csv_path: str,
    csv_all_path: str,
    count: int = 20,
    canvas_size: int = 512,
    seed: Optional[int] = None,
    joint_overlap_frac: float = 0.06,
):
    """
    Loads body part images, computes proportions, and assembles complete figures.
    Generated figures are saved in a 'figures' folder.

    Args:
        output_folder: Folder containing subfolders for each body part (head, torso, arms,legs) with images.
        csv_path: Path to the CSV file containing visibility ratings for each image.
        csv_all_path: Path to the CSV file containing visibility ratings for all frames
            (used for frequency calculations).
        count: Number of figures to generate (default=20).
        canvas_size: Size of the square canvas in pixels (default=512).
        seed: Optional random seed for reproducibility (default=None).
        joint_overlap_frac: Fraction of part height to overlap between joints (default=0.06 means 6% overlap).
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    print(f"Loading pool from '{output_folder}' …")
    pool, frequencies = load_pool(output_folder, csv_path, csv_all_path)

    print("\nComputing proportions …")
    proportions = compute_proportions(pool, canvas_size)

    os.makedirs("figures", exist_ok=True)
    generated = 0
    for i in range(count):
        fig = assemble_figure(
            pool, proportions, frequencies, canvas_size, joint_overlap_frac
        )
        if fig is None:
            print(f"  [!] Figure {i+1}: pool too sparse to assemble.")
            continue
        out_path = os.path.join("figures", f"figure_{i+1:04d}.png")
        cv2.imwrite(out_path, fig)
        generated += 1

    print(f"\nDone — generated {generated}/{count} figures → ./figures/")


def main():
    parser = argparse.ArgumentParser(
        description="Assemble stick figures from pose_landmarks outputs"
    )
    parser.add_argument(
        "--output_folder",
        required=True,
        help="Folder produced by pose_landmarks.py, e.g. output_images_rated",
    )
    parser.add_argument(
        "--csv", required=True, help="CSV rating file used in pose_landmarks.py"
    )
    parser.add_argument(
        "--csv_all",
        required=True,
        help="CSV rating file all data (for percentage calculations)",
    )
    parser.add_argument(
        "--count", type=int, default=20, help="Number of figures to generate"
    )
    parser.add_argument(
        "--size", type=int, default=512, help="Canvas size in pixels (square)"
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.06,
        help="Joint overlap as fraction of part height (default 0.06)",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    generate_figures(
        args.output_folder,
        args.csv,
        args.csv_all,
        count=args.count,
        canvas_size=args.size,
        seed=args.seed,
        joint_overlap_frac=args.overlap,
    )


if __name__ == "__main__":
    main()
