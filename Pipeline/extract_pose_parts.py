import os
import numpy as np
import mediapipe as mp
import cv2
from pathlib import Path
import argparse
from mediapipe.tasks import python
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles
from mediapipe.tasks.python import vision
import csv

# Landmark IDs
HEAD = [
    vision.PoseLandmark.NOSE,
    vision.PoseLandmark.LEFT_EYE,
    vision.PoseLandmark.RIGHT_EYE,
    vision.PoseLandmark.LEFT_EAR,
    vision.PoseLandmark.RIGHT_EAR,
    vision.PoseLandmark.MOUTH_LEFT,
    vision.PoseLandmark.MOUTH_RIGHT,
]

TORSO = [
    vision.PoseLandmark.LEFT_SHOULDER,
    vision.PoseLandmark.RIGHT_SHOULDER,
    vision.PoseLandmark.LEFT_HIP,
    vision.PoseLandmark.RIGHT_HIP,
]

ARMS = [
    vision.PoseLandmark.LEFT_ELBOW,
    vision.PoseLandmark.LEFT_WRIST,
    vision.PoseLandmark.RIGHT_ELBOW,
    vision.PoseLandmark.RIGHT_WRIST,
]

LEGS = [
    vision.PoseLandmark.LEFT_KNEE,
    vision.PoseLandmark.LEFT_ANKLE,
    vision.PoseLandmark.RIGHT_KNEE,
    vision.PoseLandmark.RIGHT_ANKLE,
]

REGION_IDS = {"head": HEAD, "torso": TORSO, "arms": ARMS, "legs": LEGS}

# A connection belongs to arms/legs if:
# - at least one endpoint is in the region (elbow, wrist, knee, ankle)
# - OR it's a bridge connection (shoulder<->elbow, hip<->knee) where one end is torso
# - BUT NOT if both endpoints are torso-only (excludes hip<->hip, shoulder<->shoulder)

ALL_CONNECTIONS = vision.PoseLandmarksConnections.POSE_LANDMARKS

HEAD_CONNECTIONS = [c for c in ALL_CONNECTIONS if c.start in HEAD and c.end in HEAD]
TORSO_CONNECTIONS = [c for c in ALL_CONNECTIONS if c.start in TORSO and c.end in TORSO]
ARMS_CONNECTIONS = [
    c
    for c in ALL_CONNECTIONS
    if (c.start in ARMS and c.end in ARMS) or (c.start in ARMS and c.end in TORSO)
]
LEGS_CONNECTIONS = [
    c
    for c in ALL_CONNECTIONS
    if (c.start in LEGS and c.end in LEGS) or (c.start in LEGS and c.end in TORSO)
]

REGION_CONNECTIONS = {
    "head": HEAD_CONNECTIONS,
    "torso": TORSO_CONNECTIONS,
    "arms": ARMS_CONNECTIONS,
    "legs": LEGS_CONNECTIONS,
}


def to_png_name(filename: str) -> str:
    """
    Replace any image extension with .png to force lossless output.

    Args:
        filename: Original filename (e.g., "image1.jpg")

    Returns:
        png_name: Filename with .png extension (e.g., "image1.png")
    """
    return Path(filename).stem + ".png"


def to_transparent_png(img_bgr: np.ndarray) -> np.ndarray:
    """
    Convert a BGR image with white background to BGRA where white -> transparent.
    Any pixel that is white (all channels > 245) gets alpha=0, others alpha=255.

    Args:
        img_bgr: Input image in BGR format (height, width, 3) with white background.

    Returns:
        bgra: Output image in BGRA format (height, width, 4) where white background is transparent.
    """
    bgra = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2BGRA)
    # Mask: pixels where all BGR channels are near-zero = background
    white_mask = (
        (img_bgr[:, :, 0] > 245) & (img_bgr[:, :, 1] > 245) & (img_bgr[:, :, 2] > 245)
    )
    bgra[:, :, 3] = np.where(white_mask, 0, 255).astype(np.uint8)
    return bgra


def draw_head_outline(detection_result, rgb_image, padding=20):
    """
    Draw a head outline based on pose landmarks (Nose, Eyes, Ears, Mouth) detected in an image.
    The head center is calculated as the mean of head landmark positions and the radius is
    set to encompass all head landmarks with some padding. It also marks facial features including eyes, nose, and mouth.

    Args:
        detection_result: The result object from MediaPipe PoseLandmarker containing
            detected pose landmarks for the image.
        rgb_image: The original RGB image (np.ndarray of shape (height, width, 3)) on
            which to base the head outline drawing.
        padding (int): Additional pixels to add to the head radius beyond the outermost
            landmark for better head coverage (default=20).
    Returns:
        tuple: A tuple containing three elements:
            - img: Image with drawn head outline and facial features.
            - mask: Binary mask with the head circle region marked,
              shape (height, width).
            - outline_points: Array containing the rightmost point
              on the head outline circle.
    """
    img = np.full_like(rgb_image, 255)
    h, w, _ = rgb_image.shape

    if not detection_result.pose_landmarks:
        print(f"No pose landmarks detected in image of shape {rgb_image.shape}")
        return None, None, None

    lm = detection_result.pose_landmarks[0]

    head_points_idx = [
        vision.PoseLandmark.LEFT_EAR,
        vision.PoseLandmark.RIGHT_EAR,
        vision.PoseLandmark.LEFT_EYE,
        vision.PoseLandmark.RIGHT_EYE,
        vision.PoseLandmark.NOSE,
        vision.PoseLandmark.MOUTH_LEFT,
        vision.PoseLandmark.MOUTH_RIGHT,
    ]
    approx_points = np.array(
        [[int(lm[idx].x * w), int(lm[idx].y * h)] for idx in head_points_idx]
    )
    center = np.mean(approx_points, axis=0).astype(int)
    radius = int(np.max(np.linalg.norm(approx_points - center, axis=1))) + padding

    cv2.circle(img, tuple(center), radius, (0, 0, 0), 2)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, tuple(center), radius, 1, -1)
    outline_points = np.array([[center[0] + radius, center[1]]])

    for idx in [vision.PoseLandmark.LEFT_EYE, vision.PoseLandmark.RIGHT_EYE]:
        p = lm[idx]
        cv2.circle(img, (int(p.x * w), int(p.y * h)), 4, (0, 0, 0), -1)

    p = lm[vision.PoseLandmark.NOSE]
    cv2.circle(img, (int(p.x * w), int(p.y * h)), 4, (0, 0, 0), -1)

    mouth_pts = []
    for idx in [vision.PoseLandmark.MOUTH_LEFT, vision.PoseLandmark.MOUTH_RIGHT]:
        p = lm[idx]
        x, y = int(p.x * w), int(p.y * h)
        mouth_pts.append((x, y))
        cv2.circle(img, (x, y), 4, (255, 0, 0), -1)
    if len(mouth_pts) == 2:
        cv2.line(img, mouth_pts[0], mouth_pts[1], (255, 0, 0), 2)
    return img, mask, outline_points


def render_region_skeleton(detection_result, image_shape, connections):
    """
    Draw pose landmark connections on a white image.
    This function creates a new image with pose landmarks and their connections
    drawn as black lines.

    Args:
        detection_result: The result object containing pose landmarks detected by MediaPipe.
        image_shape: Tuple (height, width, channels) defining the shape of the output image.
        connections: List of landmark connections to draw (e.g., head, torso, arms or legs connections)
    Returns:
        img: np.ndarray of shape `image_shape` with pose landmark connections drawn
            as black lines on a white background.
    """
    img = np.full(image_shape, 255, dtype=np.uint8)
    if not detection_result.pose_landmarks:
        return img

    connection_style = drawing_utils.DrawingSpec(color=(0, 0, 0), thickness=2)
    for landmarks in detection_result.pose_landmarks:
        drawing_utils.draw_landmarks(
            image=img,
            landmark_list=landmarks,
            connections=connections,
            landmark_drawing_spec=None,
            connection_drawing_spec=connection_style,
        )
    return img


def extract_region_landmarks(detection_result, region_ids, img_shape):
    """
    Extract pose landmarks for specified regions from detection results.
    Returns array of shape (len(region_ids), 3) containing [x, y, z] coordinates
                    scaled to image dimensions, or None if no pose landmarks detected.

    Args:
        detection_result: The result object containing pose landmarks detected by MediaPipe.
        region_ids: List of landmark IDs corresponding to the body region of interest (e.g., head, torso, arms, legs).
        img_shape: Tuple (height, width, channels) defining the shape of the original image for scaling coordinates.

    Returns:
        coords: NumPy array of shape (len(region_ids), 3) containing [x, y, z] coordinates of the specified landmarks scaled to image dimensions, or None if no pose landmarks detected.
    """
    if not detection_result.pose_landmarks:
        return None
    h, w, _ = img_shape
    lm = detection_result.pose_landmarks[0]
    coords = [[lm[idx].x * w, lm[idx].y * h, lm[idx].z] for idx in region_ids]
    return np.array(coords, dtype=np.float32)


def draw_landmarks_on_image(rgb_image, detection_result):
    """
    Draw pose landmarks and their connections on an image (annotated image).
    Connections between landmarks are drawn as black lines with thickness of 2.

    Args:
        rgb_image: The original RGB image on which to draw the landmarks.
        detection_result: The result object containing pose landmarks detected by MediaPipe.

    Returns:
        annotated_image: A copy of `rgb_image` (np.ndarray) with pose landmarks drawn
            as styled dots and black connection lines overlaid.
    """
    annotated_image = np.copy(rgb_image)
    pose_landmark_style = drawing_styles.get_default_pose_landmarks_style()
    connection_style = drawing_utils.DrawingSpec(color=(0, 0, 0), thickness=2)
    for pose_landmarks in detection_result.pose_landmarks:
        drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=pose_landmarks,
            connections=vision.PoseLandmarksConnections.POSE_LANDMARKS,
            landmark_drawing_spec=pose_landmark_style,
            connection_drawing_spec=connection_style,
        )
    return annotated_image


def main():
    """
    Process images to detect and extract pose landmarks using MediaPipe PoseLandmarker.
    Reads images from input folder, detects pose landmarks using pre-trained
    pose landmarker model from MediaPipe, and extracts various body regions (head, torso, arms, legs)
    based on human-made visibility ratings provided in CSV file.
    Detected landmarks are saved as annotated images, transparent PNGs for each body region,
    and NumPy arrays containing landmark coordinates.
    """
    parser = argparse.ArgumentParser(
        description="Extract pose landmark regions from images using MediaPipe PoseLandmarker."
    )
    parser.add_argument(
        "--input_folder",
        default="traindata",
        help="Folder containing input images and a single CSV rating file (default: traindata)",
    )
    parser.add_argument(
        "--output_folder",
        default="output_images_all",
        help="Folder where extracted part images and .npy files will be saved (default: output_images_all)",
    )
    parser.add_argument(
        "--model",
        default="pose_landmarker.task",
        help="Path to the MediaPipe PoseLandmarker .task model file (default: pose_landmarker.task)",
    )
    args = parser.parse_args()

    input_folder = Path(args.input_folder)
    output_folder = Path(args.output_folder)

    # Create directory structure
    (output_folder / "annotated").mkdir(parents=True, exist_ok=True)
    (output_folder / "head").mkdir(parents=True, exist_ok=True)
    (output_folder / "torso").mkdir(parents=True, exist_ok=True)
    (output_folder / "arms").mkdir(parents=True, exist_ok=True)
    (output_folder / "legs").mkdir(parents=True, exist_ok=True)
    (output_folder / "npy").mkdir(parents=True, exist_ok=True)
    (output_folder / "failed").mkdir(parents=True, exist_ok=True)

    base_options = python.BaseOptions(model_asset_path=args.model)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options, output_segmentation_masks=True
    )
    detector = vision.PoseLandmarker.create_from_options(options)

    failed_images = []
    csv_files = [f for f in os.listdir(input_folder) if f.lower().endswith(".csv")]
    if len(csv_files) != 1:
        raise ValueError("Expected exactly one CSV file in input folder.")

    csv_path = input_folder / csv_files[0]
    filename_filter_good = {}
    filename_filter_body = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row["filename"].strip()
            filename_filter_good[filename] = row["body_good"].strip()
            filename_filter_body[filename] = {
                "head": row["head_visible"].strip(),
                "torso": row["torso_visible"].strip(),
                "arms": row["arms_visible"].strip(),
                "legs": row["legs_visible"].strip(),
            }

    for img_file in os.listdir(input_folder):
        if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        if img_file not in filename_filter_good:
            print(f"[SKIP - NOT RATED:] {img_file}")
            continue
        if filename_filter_good[img_file] != "1":
            print(f"[SKIP - NOT GOOD VISIBILITY] {img_file}")
            continue

        img_path = os.path.join(input_folder, img_file)
        image = mp.Image.create_from_file(img_path)
        rgb = image.numpy_view()
        detection_result = detector.detect(image)

        # Use PNG stem for all part output filenames (avoids JPEG compression artifacts)
        png_name = to_png_name(img_file)

        if detection_result.pose_landmarks:
            annotated = draw_landmarks_on_image(rgb, detection_result)
            cv2.imwrite(
                str(output_folder / "annotated" / f"annotated_{png_name}"),
                cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR),
            )
            print("Detected pose landmarks for", img_file)
        else:
            print(f"[NO POSE DETECTED] {img_file}")
            failed_images.append(img_file)
            cv2.imwrite(
                str(output_folder / "failed" / png_name),
                cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
            )
            continue

        # Head
        if filename_filter_body[img_file]["head"] != "1":
            print(f"[SKIP HEAD EXTRACTION - NOT GOOD VISIBILITY] {img_file}")
        else:
            head_img, head_mask, head_hull = draw_head_outline(detection_result, rgb)
            cv2.imwrite(
                str(output_folder / "head" / f"head_{png_name}"),
                to_transparent_png(cv2.cvtColor(head_img, cv2.COLOR_RGB2BGR)),
            )
            stem = Path(img_file).stem
            if head_mask is not None:
                np.save(str(output_folder / "npy" / f"head_mask_{stem}.npy"), head_mask)
            if head_hull is not None:
                np.save(str(output_folder / "npy" / f"head_hull_{stem}.npy"), head_hull)

        # Other regions
        for region_name, connections in REGION_CONNECTIONS.items():
            if region_name == "head":
                continue
            if filename_filter_body[img_file][region_name] != "1":
                print(
                    f"[SKIP {region_name.upper()} EXTRACTION - NOT GOOD VISIBILITY] {img_file}"
                )
            else:
                region_img = render_region_skeleton(
                    detection_result, rgb.shape, connections
                )
                cv2.imwrite(
                    str(output_folder / region_name / f"{region_name}_{png_name}"),
                    to_transparent_png(cv2.cvtColor(region_img, cv2.COLOR_RGB2BGR)),
                )
                coords = extract_region_landmarks(
                    detection_result, REGION_IDS[region_name], rgb.shape
                )
                if coords is not None:
                    stem = Path(img_file).stem
                    np.save(
                        str(output_folder / "npy" / f"{region_name}_{stem}.npy"), coords
                    )

        print(f"Processed {img_file}")

    print("\nDetection Summary")
    print("-----------------")
    print(f"Failed detections: {len(failed_images)}")
    for f in failed_images:
        print(f" - {f}")


if __name__ == "__main__":
    main()
