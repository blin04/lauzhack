from transformers import Sam3Processor, Sam3Model
from PIL import Image, ImageDraw
import torch
import numpy as np
import matplotlib

def compute_orientation(mask, threshold=0.5):
    """
    Estimate orientation using PCA on mask pixels.
    Returns (angle_deg, centroid_xy, major_len, minor_len) or None if empty.
    angle_deg: degrees from image +x axis (right) positive CCW.
    centroid_xy: (x, y) in image coordinates.
    """
    if torch.is_tensor(mask):
        arr = mask.cpu().numpy()
    else:
        arr = np.array(mask)

    # normalize to boolean
    if arr.dtype == np.uint8 or arr.max() > 1:
        bw = arr > 127
    else:
        bw = arr > threshold

    coords = np.column_stack(np.where(bw))  # rows (y), cols (x)
    if coords.size == 0:
        return None

    centroid = coords.mean(axis=0)  # (y, x)
    centered = coords - centroid
    if centered.shape[0] < 2:
        return None

    cov = np.cov(centered, rowvar=False)  # 2x2 cov of [y, x]
    eigvals, eigvecs = np.linalg.eigh(cov)
    # largest eigenvalue -> principal axis
    idx = np.argmax(eigvals)
    major_vec = eigvecs[:, idx]  # [vy, vx]
    # convert to (vx, vy) and normalize
    vx, vy = float(major_vec[1]), float(major_vec[0])
    norm = np.hypot(vx, vy) or 1.0
    vx /= norm; vy /= norm

    angle_rad = np.arctan2(vy, vx)
    angle_deg = np.degrees(angle_rad)

    # approximate axis lengths from eigenvalues (scale factor arbitrary)
    major_len = 2.0 * np.sqrt(max(eigvals[idx], 0.0))
    minor_len = 2.0 * np.sqrt(max(eigvals[1-idx], 0.0))

    # centroid in (x, y)
    centroid_xy = (float(centroid[1]), float(centroid[0]))
    return angle_deg, centroid_xy, major_len, minor_len

def overlay_masks(image, masks, draw_orientations=True):
    image = image.convert("RGBA")
    masks_np = 255 * masks.cpu().numpy().astype(np.uint8)
    
    n_masks = masks_np.shape[0]
    cmap = matplotlib.colormaps.get_cmap("rainbow").resampled(n_masks)
    colors = [
        tuple(int(c * 255) for c in cmap(i)[:3])
        for i in range(n_masks)
    ]

    orientations = []
    for mask, color in zip(masks_np, colors):
        mask_img = Image.fromarray(mask)
        overlay = Image.new("RGBA", image.size, color + (0,))
        alpha = mask_img.point(lambda v: int(v * 0.5))
        overlay.putalpha(alpha)
        image = Image.alpha_composite(image, overlay)

        # compute orientation for this mask
        orient = compute_orientation(mask)
        orientations.append((orient, color))

    if draw_orientations:
        draw = ImageDraw.Draw(image)
        for item in orientations:
            orient, color = item
            if orient is None:
                continue
            angle_deg, (cx, cy), major_len, minor_len = orient
            # draw a line for the major axis (scaled for visibility)
            L = max(20, major_len * 4)
            # direction vector
            ang = np.radians(angle_deg)
            dx = np.cos(ang) * L
            dy = np.sin(ang) * L
            # image coordinates: x right, y down (PIL uses same)
            start = (cx - dx, cy - dy)
            end = (cx + dx, cy + dy)
            draw.line([start, end], fill=(255,255,255,255), width=3)
            # small circle at centroid
            r = 4
            draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=(255,255,255,255))

    return image, orientations


device = "cpu"
model = Sam3Model.from_pretrained("facebook/sam3").to(device)
processor = Sam3Processor.from_pretrained("facebook/sam3")


image_name = input("Enter image filename (in images/): ")
image = Image.open(f"images/{image_name}.png")

# detect cars and estimate orientations
print("\nDetecting cars and estimating their orientations...")

inputs_cars = processor(images=image, text="car", return_tensors="pt").to(device)

with torch.no_grad():
    outputs_cars = model(**inputs_cars)

results_cars = processor.post_process_instance_segmentation(
    outputs_cars,
    threshold=0.5,
    mask_threshold=0.5,
    target_sizes=inputs_cars.get("original_sizes").tolist()
)[0]

print(f"Found {len(results_cars['masks'])} car objects")

cars_image, car_orientations = overlay_masks(image, results_cars["masks"], draw_orientations=False)
for i, (orient, _) in enumerate(car_orientations):
    if orient is None:
        print(f"Car {i}: no mask pixels")
    else:
        angle_deg, centroid_xy, major_len, minor_len = orient
        print(f"Car {i}: angle={angle_deg:.1f}°, centroid={centroid_xy}, major={major_len:.1f}, minor={minor_len:.1f}")

cars_image.save(f"images/{image_name}_cars.png")
# cars_image.show()

# --- roads detection (existing) --- #
print("\nDetecting roads and estimating their directions...")

# prepare inputs for 'roads'
inputs_roads = processor(images=image, text="street road", return_tensors="pt").to(device)

with torch.no_grad():
    outputs_roads = model(**inputs_roads)

results_roads = processor.post_process_instance_segmentation(
    outputs_roads,
    threshold=0.5,
    mask_threshold=0.5,
    target_sizes=inputs_roads.get("original_sizes").tolist()
)[0]

print(f"Found {len(results_roads['masks'])} road objects")

road_image, road_orientations = overlay_masks(image, results_roads["masks"])
# print road orientations
for i, (orient, _) in enumerate(road_orientations):
    if orient is None:
        print(f"Road {i}: no mask pixels")
    else:
        angle_deg, centroid_xy, major_len, minor_len = orient
        print(f"Road {i}: angle={angle_deg:.1f}°, centroid={centroid_xy}, major={major_len:.1f}, minor={minor_len:.1f}")

road_image.save(f"images/{image_name}_roads.png")
# road_image.show()

# --- compute angle differences between each car and nearest road --- #
def minimal_axis_angle_diff(a_deg, b_deg):
    """
    Return smallest absolute angle difference between two axial angles (degrees).
    Accounts for 180-degree axis ambiguity and returns value in [0, 90].
    """
    # minimal difference on circle
    raw = abs(((a_deg - b_deg + 180) % 360) - 180)  # in [0,180]
    if raw > 90:
        raw = 180 - raw
    return raw

if len(road_orientations) == 0:
    print("\nNo roads found — cannot compute car vs road angle differences.")
else:
    # build list of road centroids and angles (skip None)
    roads = [(ang, cen) for (ang, cen, *_), _ in road_orientations if ang is not None]
    if len(roads) == 0:
        print("\nNo valid road orientations found.")
    else:
        print("\nCar vs nearest-road orientation differences:")
        for i, (car_orient, _) in enumerate(car_orientations):
            if car_orient is None:
                print(f"Car {i}: no mask pixels")
                continue
            car_ang, car_centroid, *_ = car_orient
            # find nearest road by centroid distance
            best_j = None
            best_dist = float("inf")
            best_road_ang = None
            for j, (road_ang, road_centroid) in enumerate(roads):
                dx = car_centroid[0] - road_centroid[0]
                dy = car_centroid[1] - road_centroid[1]
                d = (dx*dx + dy*dy) ** 0.5
                if d < best_dist:
                    best_dist = d
                    best_j = j
                    best_road_ang = road_ang
            diff_deg = minimal_axis_angle_diff(car_ang, best_road_ang)
            print(f"Car {i}: angle={car_ang:.1f}°, nearest road angle={best_road_ang:.1f}° (road #{best_j}), diff={diff_deg:.1f}° , distance={best_dist:.1f}px")

# --- generate image with only parked cars --- #
angle_thresh = 20.0  # degrees: if car vs road diff <= this -> considered aligned
default_dist_thresh = 50.0  # px: minimum distance to consider "on road" if road major_len is tiny

# build roads info (ang, centroid, major_len)
roads_info = []
for (orient, _color) in road_orientations:
    if orient is None:
        continue
    ang, cen, major_len, _ = orient
    roads_info.append((ang, cen, max(major_len, 1.0)))

parked_indices = []
for i, (car_orient, _color) in enumerate(car_orientations):
    if car_orient is None:
        continue
    car_ang, car_centroid, car_major, _ = car_orient

    if len(roads_info) == 0:
        # no roads -> can't be "on road and aligned", so treat as parked
        parked = True
    else:
        # find nearest road
        best_d = float("inf")
        best_road_ang = None
        best_road_major = None
        for (road_ang, road_cent, road_major) in roads_info:
            dx = car_centroid[0] - road_cent[0]
            dy = car_centroid[1] - road_cent[1]
            d = (dx*dx + dy*dy) ** 0.5
            if d < best_d:
                best_d = d
                best_road_ang = road_ang
                best_road_major = road_major

        dist_thresh = max(best_road_major * 1.5, default_dist_thresh)
        on_road = best_d <= dist_thresh
        aligned = minimal_axis_angle_diff(car_ang, best_road_ang) <= angle_thresh
        # car is NOT parked if it's on the road AND aligned with the road
        parked = not (on_road and aligned)

    if parked:
        parked_indices.append(i)

print(f"\nParked car indices: {parked_indices}")

if len(parked_indices) == 0:
    print("No parked cars found.")
else:
    # extract parked car masks and render them into a separate image
    parked_masks = results_cars["masks"][parked_indices]  # tensor indexing with list
    parked_image, parked_orientations = overlay_masks(image, parked_masks, draw_orientations=False)
    parked_image.save(f"images/{image_name}_parked_cars.png")
    parked_image.show()
    print(f"Saved parked cars image to images/{image_name}_parked_cars.png")