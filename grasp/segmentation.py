import os
import argparse
import cv2
import torch
import sys
import numpy as np
import pyrealsense2 as rs
from PIL import Image



# Add the *inner package* directory to PYTHONPATH
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SAM2_PATH = os.path.join(THIS_DIR, "sam2")  # this contains the second sam2/ inside it
sys.path.insert(0, SAM2_PATH)


# SAM2 Imports
# from sam2.sam2.build_sam import build_sam2_hf
# from sam2.sam2_image_predictor import SAM2ImagePredictor

from sam2.build_sam import build_sam2_hf
from sam2.sam2_image_predictor import SAM2ImagePredictor


# Grounding DINO Imports
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

# path to sam2 if needed (keeping your original path logic)
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../sam2')))

def load_models(device="cuda"):
    print(f"Loading Grounding DINO on {device}...")
    dino_id = "IDEA-Research/grounding-dino-tiny"
    processor = AutoProcessor.from_pretrained(dino_id)
    dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(dino_id).to(device)

    print(f"Loading SAM 2 on {device}...")
    # You can switch to "facebook/sam2-hiera-large" for better accuracy but slower speed
    sam_model = build_sam2_hf("facebook/sam2-hiera-tiny", device=device)
    predictor = SAM2ImagePredictor(sam_model)
    
    return processor, dino_model, predictor

def overlay_mask(image, mask, color=(0, 255, 0), alpha=0.5):
    """
    Overlays mask on BGR image correctly.
    mask: (H, W) boolean or 0/1 array
    image: (H, W, 3)
    """
    mask = mask.astype(bool)

    # Colored mask same shape as image
    colored = np.zeros_like(image, dtype=np.uint8)
    colored[mask] = color

    # Use cv2.addWeighted only on full image
    blended = cv2.addWeighted(image, 1 - alpha, colored, alpha, 0)

    # Replace masked region with blended region
    image[mask] = blended[mask]

    return image


def run_realsense_loop(text_prompt, box_threshold=0.35, device="cuda"):
    # 1. Load Models
    processor, dino_model, predictor = load_models(device)
    
    # 2. Setup RealSense Pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    
    # Enable RGB stream (Depth is not strictly needed for DINO/SAM, but usually available)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    print(f"\n[INFO] Starting RealSense Stream...")
    print(f"[INFO] Looking for: '{text_prompt}'")
    print(f"[INFO] Press 'q' to exit.")
    
    profile = pipeline.start(config)

    try:
        while True:
            # 3. Get Frames
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            
            if not color_frame:
                continue

            # Convert to numpy
            color_image = np.asanyarray(color_frame.get_data())
            
            # 4. Pre-process for DINO (Needs PIL or RGB format)
            # RealSense is BGR, Convert to RGB for models
            image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)

            # 5. Grounding DINO Inference (Detect Boxes)
            inputs = processor(images=pil_image, text=text_prompt, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = dino_model(**inputs)

            results = processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=box_threshold,
                
                target_sizes=[pil_image.size[::-1]]
            )

            dino_boxes = results[0]["boxes"].cpu().numpy()
            print(f"DINO Shape:{dino_boxes.shape}\n")
            print(f"DINO BOXES:{dino_boxes}\n")
            phrases = results[0]["labels"]

            # 6. SAM2 Inference (Segment based on Boxes)
            if len(dino_boxes) > 0:
                predictor.set_image(image_rgb)
                
                # SAM2 prediction
                masks, scores, _ = predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=dino_boxes,
                    multimask_output=False
                )
                
                # Squeeze dimensions if needed (N, 1, H, W) -> (N, H, W)
                if masks.ndim == 4:
                    masks = masks.squeeze(1)
                
                # print(masks)
                # 7. Visualization
                # Draw masks
                for i, mask in enumerate(masks):
                    # Generate a random color or cycle colors
                    # color = ((i * 50) % 255, (i * 100) % 255, (i * 150) % 255)
                    color = (255,255,255)
                    color_image = overlay_mask(color_image, mask, color=color, alpha=0.5)

                # Draw boxes
                for box, phrase in zip(dino_boxes, phrases):
                    x1, y1, x2, y2 = map(int, box)
                    # Draw Rectangle
                    cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # Draw Label
                    cv2.putText(color_image, phrase, (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # 8. Display Result
            cv2.imshow('RealSense SAM2 + DINO', color_image)

            # Exit condition
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RealSense Live SAM2 + Grounding DINO")
    parser.add_argument("--prompt", required=True, type=str, help="Text prompt for object detection (e.g., 'cup. laptop.')")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threshold", default=0.35, type=float, help="Detection threshold")
    
    args = parser.parse_args()

    # Format prompt with period if missing (DINO quirk)
    clean_prompt = args.prompt if args.prompt.endswith(".") else args.prompt + "."

    run_realsense_loop(clean_prompt, args.threshold, args.device)