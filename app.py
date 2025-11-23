from flask import Flask, render_template, request, jsonify, send_file
import os
import torch
import numpy as np
from PIL import Image, ImageDraw
import matplotlib
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
import cv2
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
        print("Loading model...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        processor = OneFormerProcessor.from_pretrained("shi-labs/oneformer_coco_swin_large")
        model = OneFormerForUniversalSegmentation.from_pretrained("shi-labs/oneformer_coco_swin_large")
        model = model.to(device)
        model.eval()
        print(f"Model loaded successfully on {device}")

def compute_orientation(mask):
    """Compute orientation angle and centroid for a binary mask."""
    try:
        if torch.is_tensor(mask):
            mask_np = mask.cpu().numpy()
        else:
            mask_np = np.array(mask)
        
        if mask_np.dtype == bool:
            mask_uint8 = mask_np.astype(np.uint8) * 255
        else:
            mask_uint8 = (mask_np > 0).astype(np.uint8) * 255
        
        coords = np.column_stack(np.where(mask_uint8 > 0))
        if len(coords) < 5:
            return None
        
        points = coords[:, [1, 0]].astype(np.float32)
        (cx, cy), (w, h), angle = cv2.fitEllipse(points)
        
        major_len = max(w, h) / 2.0
        minor_len = min(w, h) / 2.0
        
        if w > h:
            angle_deg = angle
        else:
            angle_deg = (angle + 90) % 180
        
        centroid = np.mean(coords, axis=0)
        centroid_xy = (float(centroid[1]), float(centroid[0]))
        return angle_deg, centroid_xy, major_len, minor_len
    except Exception as e:
        print(f"Error in compute_orientation: {e}")
        return None

def minimal_axis_angle_diff(a_deg, b_deg):
    """Return smallest absolute angle difference between two axial angles."""
    raw = abs(((a_deg - b_deg + 180) % 360) - 180)
    if raw > 90:
        raw = 180 - raw
    return raw

def cluster_orientations(orientations, angle_threshold=15.0):
    """Cluster car orientations based on angle similarity."""
    if not orientations:
        return []
    
    clusters = []
    for item in orientations:
        orient, color = item
        if orient is None:
            continue
        angle_deg, centroid, major_len, minor_len = orient
        
        placed = False
        for cluster in clusters:
            cluster_angle = cluster['representative_angle']
            if minimal_axis_angle_diff(angle_deg, cluster_angle) < angle_threshold:
                cluster['items'].append((orient, color))
                placed = True
                break
        
        if not placed:
            clusters.append({
                'representative_angle': angle_deg,
                'items': [(orient, color)]
            })
    
    return clusters

def overlay_masks(image, masks, draw_orientations=True):
    """Overlay detected car masks on the image."""
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

def derive_instance_masks_from_panoptic(segmentation, segments_info, target_label_id=None):
    """
    Convert panoptic segmentation into per-object binary masks.
    Optionally filter by COCO class ID.
    """
    seg_np = segmentation.cpu().numpy()
    masks = []

    for seg in segments_info:
        if target_label_id is not None and seg["label_id"] != target_label_id:
            continue
        
        mask = (seg_np == seg["id"])
        if mask.sum() > 10:  # remove noise
            masks.append(torch.tensor(mask, dtype=torch.float32))
    
    if len(masks) == 0:
        return torch.empty((0, seg_np.shape[0], seg_np.shape[1]))
    
    return torch.stack(masks)

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
        
        # Detect cars
        width, height = image.size
        inputs_cars = processor(
            images=image,
            text="car body",
            task_inputs=["panoptic"],
            return_tensors="pt"
        )

        # Step 1: prepare inputs
        inputs_cars = processor(
            images=image,
            text="car body",
            task_inputs=["panoptic"],
            return_tensors="pt"
        ).to(device)

        # Step 2: inference
        with torch.no_grad():
            outputs_cars = model(**inputs_cars)

        # Step 3: required fix — provide original resolution manually
        height, width = image.size[1], image.size[0]
        target_sizes = torch.tensor([[height, width]])

        # Step 4: post-processing
        results = processor.post_process_panoptic_segmentation(
            outputs_cars,
            target_sizes=target_sizes.tolist()
        )[0]

        # Extract masks manually (car label_id = 2 in COCO)
        masks = derive_instance_masks_from_panoptic(
            results["segmentation"],
            results["segments_info"],
            target_label_id=2
        )

        results["masks"] = masks


        print(f"Initially found {len(results['masks'])} car objects. Filtering by aspect ratio and area...")
        
        # Filter by aspect ratio and area (EXACT code from notebook)
        keep_indices = []
        min_aspect_ratio = 1.3
        max_aspect_ratio = 2.8
        min_area = 800  # Minimum area in pixels to be considered a car
        max_area = 2000  # Maximum area in pixels to exclude trucks
        
        for i, mask in enumerate(results["masks"]):
            # Calculate area
            if torch.is_tensor(mask):
                mask_np = mask.cpu().numpy()
            else:
                mask_np = np.array(mask)
            
            # Check if mask is boolean or 0-1 or 0-255
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
            if minor_len < 1.0:  # Avoid division by zero
                continue
            
            aspect_ratio = major_len / minor_len
            if min_aspect_ratio <= aspect_ratio <= max_aspect_ratio:
                keep_indices.append(i)
        
        # Apply filter
        filtered_masks = results["masks"][keep_indices]
        
        print(f"Found {len(filtered_masks)} car objects after filtering")
        
        if len(filtered_masks) == 0:
            return jsonify({
                'success': True,
                'total_cars': 0,
                'num_clusters': 0,
                'detection_image': '',
                'cluster_image': '',
                'cluster_details': [],
                'message': 'No cars detected. Try a different image or adjust detection parameters.'
            })
        
        # Overlay masks and get orientations
        result_image, orientations = overlay_masks(image, filtered_masks, draw_orientations=True)
        
        # Cluster orientations
        clusters = cluster_orientations(orientations, angle_threshold=15.0)
        
        # Create cluster visualization
        clustered_image = image.convert("RGBA")
        cluster_colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 128, 0)
        ]
        
        draw = ImageDraw.Draw(clustered_image)
        for cluster_idx, cluster in enumerate(clusters):
            cluster_color = cluster_colors[cluster_idx % len(cluster_colors)]
            for orient, _ in cluster['items']:
                if orient is None:
                    continue
                angle_deg, (cx, cy), major_len, minor_len = orient
                L = max(20, major_len * 4)
                ang = np.radians(angle_deg)
                dx = np.cos(ang) * L
                dy = np.sin(ang) * L
                start = (cx - dx, cy - dy)
                end = (cx + dx, cy + dy)
                draw.line([start, end], fill=cluster_color + (255,), width=4)
                r = 5
                draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=cluster_color + (255,))
        
        # Convert images to base64
        def pil_to_base64(img):
            buffered = BytesIO()
            img.convert("RGB").save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode()
        
        print(f"Analysis complete: {len(filtered_masks)} cars, {len(clusters)} clusters")
        
        return jsonify({
            'success': True,
            'total_cars': len(filtered_masks),
            'num_clusters': len(clusters),
            'detection_image': pil_to_base64(result_image),
            'cluster_image': pil_to_base64(clustered_image),
            'cluster_details': [{
                'cluster_id': idx,
                'angle': cluster['representative_angle'],
                'count': len(cluster['items'])
            } for idx, cluster in enumerate(clusters)]
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