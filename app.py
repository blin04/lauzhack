from flask import Flask, render_template, request, jsonify
import os
import torch
import numpy as np
from PIL import Image, ImageDraw
import matplotlib
from io import BytesIO
import base64
from transformers import Sam3Processor, Sam3Model
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull
from scipy.ndimage import distance_transform_edt
import traceback

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global model variables (loaded once)
model = None
processor = None
device = None

def load_model():
    global model, processor, device
    if model is None:
        print("Loading SAM3 model...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        processor = Sam3Processor.from_pretrained("facebook/sam3")
        model = Sam3Model.from_pretrained("facebook/sam3")
        model = model.to(device)
        model.eval()
        print(f"Model loaded successfully on {device}")

def compute_orientation(mask, threshold=0.5):
    """
    Estimate orientation using PCA on mask pixels.
    Returns (angle_deg, centroid_xy, major_len, minor_len) or None if empty.
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

    # approximate axis lengths from eigenvalues
    major_len = 2.0 * np.sqrt(max(eigvals[idx], 0.0))
    minor_len = 2.0 * np.sqrt(max(eigvals[1-idx], 0.0))

    # centroid in (x, y)
    centroid_xy = (float(centroid[1]), float(centroid[0]))
    return angle_deg, centroid_xy, major_len, minor_len

def overlay_masks(image, masks, draw_orientations=True):
    """Overlay masks on image with optional orientation lines."""
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

        orient = compute_orientation(mask)
        orientations.append((orient, color))

    if draw_orientations:
        draw = ImageDraw.Draw(image)
        for item in orientations:
            orient, color = item
            if orient is None:
                continue
            angle_deg, (cx, cy), major_len, minor_len = orient
            L = max(20, major_len * 4)
            ang = np.radians(angle_deg)
            dx = np.cos(ang) * L
            dy = np.sin(ang) * L
            start = (cx - dx, cy - dy)
            end = (cx + dx, cy + dy)
            draw.line([start, end], fill=(255,255,255,255), width=3)
            r = 4
            draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=(255,255,255,255))

    return image, orientations

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'ok', 'model_loaded': model is not None})

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        print("Received analyze request")
        load_model()
        
        if 'image' not in request.files:
            print("No image in request")
            return jsonify({'error': 'No image uploaded'}), 400
        
        file = request.files['image']
        if file.filename == '':
            print("Empty filename")
            return jsonify({'error': 'No file selected'}), 400
        
        print(f"Processing file: {file.filename}")
        
        # Read image
        image = Image.open(file.stream).convert("RGB")
        print(f"Image loaded: {image.size}")
        
        # ===== DETECT CARS ===== #
        print("\nDetecting cars and estimating their orientations...")
        inputs_cars = processor(images=image, text="car body", return_tensors="pt").to(device)

        with torch.no_grad():
            outputs_cars = model(**inputs_cars)

        results_cars = processor.post_process_instance_segmentation(
            outputs_cars,
            threshold=0.5,
            mask_threshold=0.5,
            target_sizes=inputs_cars.get("original_sizes").tolist()
        )[0]

        print(f"Initially found {len(results_cars['masks'])} car objects. Filtering by aspect ratio and area...")
        
        # Filter by aspect ratio and area
        keep_indices = []
        min_aspect_ratio = 1.3
        max_aspect_ratio = 2.8
        min_area = 800
        max_area = 2000
        
        for i, mask in enumerate(results_cars["masks"]):
            if torch.is_tensor(mask):
                mask_np = mask.cpu().numpy()
            else:
                mask_np = np.array(mask)
            
            if mask_np.dtype == bool:
                area = np.sum(mask_np)
            else:
                area = np.sum(mask_np > 0)
                
            if area < min_area or area > max_area:
                continue

            orient = compute_orientation(mask)
            if orient is None:
                continue
            _, _, major_len, minor_len = orient
            if minor_len < 1.0:
                continue
            
            aspect_ratio = major_len / minor_len
            if min_aspect_ratio <= aspect_ratio <= max_aspect_ratio:
                keep_indices.append(i)
        
        # Apply filter
        results_cars["masks"] = results_cars["masks"][keep_indices]
        results_cars["scores"] = results_cars["scores"][keep_indices]
        if "labels" in results_cars:
            results_cars["labels"] = results_cars["labels"][keep_indices]
        
        print(f"Found {len(results_cars['masks'])} car objects after filtering")
        
        # Get car orientations
        cars_image, car_orientations = overlay_masks(image, results_cars["masks"], draw_orientations=False)
        
        # ===== DETECT ROADS ===== #
        print("\nDetecting roads...")
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
        
        road_masks = results_roads["masks"]
        if torch.is_tensor(road_masks):
            road_masks_np = road_masks.cpu().numpy().astype(bool)
        else:
            road_masks_np = np.array(road_masks).astype(bool)
        
        # ===== FILTER CARS ON ROADS ===== #
        print("\n=== Filtering out cars that are on roads ===")
        
        car_masks = results_cars["masks"]
        on_road_flags = []
        
        if road_masks_np.size == 0:
            union_road = None
            dist_map = None
            print("No road masks available, skipping road-based filtering.")
            on_road_flags = [False] * len(car_masks)
        else:
            union_road = road_masks_np.any(axis=0)
            dist_map = distance_transform_edt(~union_road)
            
            road_dist_thresh = 40.0
            
            for i, mask in enumerate(car_masks):
                if torch.is_tensor(mask):
                    car_mask_np = mask.cpu().numpy()
                else:
                    car_mask_np = np.array(mask)
                
                if car_mask_np.dtype == bool:
                    car_bin = car_mask_np
                else:
                    car_bin = car_mask_np > 0
                
                car_area = car_bin.sum()
                if car_area == 0:
                    on_road_flags.append(False)
                    continue
                
                ys, xs = np.where(car_bin)
                cy = int(round(ys.mean()))
                cx = int(round(xs.mean()))
                
                h, w = dist_map.shape
                cy = max(0, min(h - 1, cy))
                cx = max(0, min(w - 1, cx))
                
                dist_to_road = dist_map[cy, cx]
                on_road = dist_to_road <= road_dist_thresh
                on_road_flags.append(on_road)
        
        n_on_road = sum(on_road_flags)
        print(f"Cars near roads (filtered out): {n_on_road}")
        print(f"Cars off roads (kept for clustering): {len(on_road_flags) - n_on_road}")
        
        # ===== CLUSTER OFF-ROAD CARS ===== #
        print("\n=== Clustering OFF-ROAD cars into groups ===")
        
        offroad_centroids = []
        offroad_indices = []
        
        for i, (orient, _) in enumerate(car_orientations):
            if on_road_flags[i]:
                continue
            
            if orient is not None:
                _, centroid, _, _ = orient
                offroad_centroids.append(centroid)
                offroad_indices.append(i)
            else:
                mask = results_cars["masks"][i]
                if torch.is_tensor(mask):
                    mask_np = mask.cpu().numpy()
                else:
                    mask_np = np.array(mask)
                ys, xs = np.where(mask_np > 0)
                if len(xs) == 0:
                    continue
                offroad_centroids.append((float(xs.mean()), float(ys.mean())))
                offroad_indices.append(i)
        
        if len(offroad_centroids) == 0:
            print("No off-road cars to cluster.")
            return jsonify({
                'success': True,
                'total_cars': int(len(car_masks)),
                'offroad_cars': 0,
                'num_clusters': 0,
                'detection_image': '',
                'cluster_image': '',
                'cluster_details': [],
                'message': 'No parked cars detected (all cars are on roads).'
            })
        
        offroad_centroids = np.array(offroad_centroids)
        
        # DBSCAN clustering
        eps = 150
        min_samples = 1
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(offroad_centroids)
        labels = clustering.labels_
        
        unique_labels = sorted(set(labels))
        print(f"Total off-road cars: {len(offroad_centroids)}")
        print(f"Number of clusters: {len(unique_labels)}")
        
        # ===== VISUALIZATION ===== #
        # Show roads in background
        clustered_image = image.copy().convert("RGBA")
        roads_vis, _ = overlay_masks(image.copy(), results_roads["masks"], draw_orientations=False)
        clustered_image = roads_vis.convert("RGBA")
        
        # Create transparent overlay for convex hulls
        overlay = Image.new('RGBA', clustered_image.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        
        base_colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (255, 128, 0), (128, 0, 255),
            (0, 128, 255), (128, 255, 0)
        ]
        
        cluster_details = []
        
        for lbl in unique_labels:
            idxs = np.where(labels == lbl)[0]
            if len(idxs) == 0:
                continue
            
            color = base_colors[lbl % len(base_colors)] if lbl != -1 else (200, 200, 200)
            cluster_points = offroad_centroids[idxs]
            
            # Calculate average angle for this cluster
            cluster_angles = []
            for idx in idxs:
                original_idx = offroad_indices[idx]
                orient, _ = car_orientations[original_idx]
                if orient is not None:
                    angle_deg, _, _, _ = orient
                    cluster_angles.append(angle_deg)
            
            avg_angle = float(np.mean(cluster_angles)) if cluster_angles else 0.0
            
            # Draw convex hull
            if len(cluster_points) >= 3:
                hull = ConvexHull(cluster_points)
                hull_points = cluster_points[hull.vertices]
                polygon = [tuple(point) for point in hull_points]
                overlay_draw.polygon(polygon, fill=color + (50,), outline=color + (255,), width=3)
            elif len(cluster_points) == 2:
                p1, p2 = cluster_points
                padding = 20
                overlay_draw.line([tuple(p1), tuple(p2)], fill=color + (255,), width=3)
                for p in cluster_points:
                    overlay_draw.ellipse([p[0]-padding, p[1]-padding, p[0]+padding, p[1]+padding],
                                        outline=color + (255,), width=3)
            else:
                p = cluster_points[0]
                padding = 30
                overlay_draw.ellipse([p[0]-padding, p[1]-padding, p[0]+padding, p[1]+padding],
                                    fill=color + (50,), outline=color + (255,), width=3)
            
            cluster_details.append({
                'cluster_id': int(lbl) if lbl >= 0 else -1,
                'count': int(len(idxs)),
                'angle': avg_angle,
                'centroid': [float(x) for x in cluster_points.mean(axis=0)]
            })
        
        # Composite overlay
        clustered_image = Image.alpha_composite(clustered_image, overlay)
        draw = ImageDraw.Draw(clustered_image)
        
        # Draw cluster labels
        for lbl in unique_labels:
            idxs = np.where(labels == lbl)[0]
            if len(idxs) == 0:
                continue
            cluster_center = offroad_centroids[idxs].mean(axis=0)
            label_text = f"C{lbl if lbl >= 0 else 'N'} ({len(idxs)})"
            draw.text((cluster_center[0], cluster_center[1]),
                     label_text,
                     fill=(255, 255, 255, 255))
        
        # Convert images to base64
        def pil_to_base64(img):
            buffered = BytesIO()
            img.convert("RGB").save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode()
        
        print(f"Analysis complete: {len(offroad_centroids)} parked cars, {len(unique_labels)} clusters")
        
        return jsonify({
            'success': True,
            'total_cars': int(len(car_masks)),
            'offroad_cars': int(len(offroad_centroids)),
            'onroad_cars': int(n_on_road),
            'num_clusters': int(len(unique_labels)),
            'detection_image': pil_to_base64(cars_image),
            'cluster_image': pil_to_base64(clustered_image),
            'cluster_details': cluster_details
        })
        
    except Exception as e:
        print(f"Error in analyze: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🚀 sPARKing Server Starting...")
    print("="*50)
    print(f"\n📍 Server will be available at:")
    print(f"   • http://localhost:5000")
    print(f"   • http://127.0.0.1:5000")
    print(f"\n⚡ Press Ctrl+C to stop\n")
    print("="*50 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)