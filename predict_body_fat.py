import torch
import os
import numpy as np
import sys
import cv2
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet50
from torch import nn

# --- COMPATIBILITY PATCH ---
try:
    import numpy._core
except ImportError:
    sys.modules['numpy._core'] = np.core
    sys.modules['numpy._core.multiarray'] = np.core.multiarray

# ============================================================
# STEP 1: MEDIAPIPE CROP — tightly crop person from image
# ============================================================
def crop_to_person(image_path, padding=30):
    """
    Uses MediaPipe Pose to detect body landmarks and crop tightly
    around the person, removing background clutter.
    Falls back to the original image if detection fails.
    """
    try:
        import mediapipe as mp
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise ValueError(f"Could not read image: {image_path}")
        h, w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        mp_pose = mp.solutions.pose
        with mp_pose.Pose(static_image_mode=True, model_complexity=2) as pose:
            results = pose.process(img_rgb)

        if not results.pose_landmarks:
            print(f"[WARN] No pose landmarks found in {image_path}, using full image.")
            return Image.fromarray(img_rgb)

        landmarks = results.pose_landmarks.landmark
        xs = [lm.x * w for lm in landmarks]
        ys = [lm.y * h for lm in landmarks]

        x1 = max(0, int(min(xs)) - padding)
        x2 = min(w, int(max(xs)) + padding)
        y1 = max(0, int(min(ys)) - padding)
        y2 = min(h, int(max(ys)) + padding)

        cropped = img_rgb[y1:y2, x1:x2]
        return Image.fromarray(cropped)

    except ImportError:
        print("[WARN] mediapipe not installed. pip install mediapipe. Using full image.")
        img = Image.open(image_path).convert("RGB")
        return img
    except Exception as e:
        print(f"[WARN] Crop failed ({e}), using full image.")
        img = Image.open(image_path).convert("RGB")
        return img


# ============================================================
# STEP 2: BACKGROUND REMOVAL — isolate body pixels only
# ============================================================
def remove_background(pil_image):
    """
    Uses rembg to strip the background, leaving only the person.
    Returns an RGB PIL image with background replaced by white.
    Falls back to original image if rembg is not available.
    """
    try:
        from rembg import remove
        # rembg returns RGBA
        result_rgba = remove(pil_image)
        # Composite onto white background
        background = Image.new("RGB", result_rgba.size, (255, 255, 255))
        background.paste(result_rgba, mask=result_rgba.split()[3])
        return background
    except ImportError:
        print("[WARN] rembg not installed. pip install rembg. Skipping BG removal.")
        return pil_image.convert("RGB")
    except Exception as e:
        print(f"[WARN] BG removal failed ({e}), skipping.")
        return pil_image.convert("RGB")


# ============================================================
# STEP 3: POSE MEASUREMENTS — extract body proportion features
# ============================================================
def extract_pose_measurements(image_path):
    """
    Extracts normalized body proportion ratios from MediaPipe landmarks.
    These are strongly correlated with body fat and supplement raw pixels.

    Returns a 6-element tensor:
        [shoulder_width, hip_width, torso_length, waist_est,
         shoulder_hip_ratio, torso_height_ratio]
    All values normalized to [0, 1] range.
    Returns zeros if detection fails (graceful degradation).
    """
    try:
        import mediapipe as mp
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise ValueError(f"Could not read image: {image_path}")
        h, w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        mp_pose = mp.solutions.pose
        with mp_pose.Pose(static_image_mode=True, model_complexity=2) as pose:
            results = pose.process(img_rgb)

        if not results.pose_landmarks:
            return torch.zeros(6)

        lm = results.pose_landmarks.landmark

        # Key landmark indices (MediaPipe Pose)
        LEFT_SHOULDER   = 11
        RIGHT_SHOULDER  = 12
        LEFT_HIP        = 23
        RIGHT_HIP       = 24
        LEFT_EAR        = 7
        RIGHT_EAR       = 8
        LEFT_ANKLE      = 27
        RIGHT_ANKLE     = 28
        LEFT_ELBOW      = 13
        RIGHT_ELBOW     = 14

        def dist(a, b):
            return ((lm[a].x - lm[b].x)**2 + (lm[a].y - lm[b].y)**2) ** 0.5

        shoulder_width  = dist(LEFT_SHOULDER, RIGHT_SHOULDER)
        hip_width       = dist(LEFT_HIP, RIGHT_HIP)
        torso_length    = (abs(lm[LEFT_SHOULDER].y - lm[LEFT_HIP].y) +
                           abs(lm[RIGHT_SHOULDER].y - lm[RIGHT_HIP].y)) / 2
        body_height     = abs(
            ((lm[LEFT_EAR].y + lm[RIGHT_EAR].y) / 2) -
            ((lm[LEFT_ANKLE].y + lm[RIGHT_ANKLE].y) / 2)
        ) + 1e-6

        # Waist estimated as horizontal spread of elbows (proxy when no waist landmark)
        waist_est       = dist(LEFT_ELBOW, RIGHT_ELBOW)
        shoulder_hip_ratio  = shoulder_width / (hip_width + 1e-6)
        torso_height_ratio  = torso_length / body_height

        # Clamp all to [0, 2] then divide by 2 → [0, 1]
        features = [
            min(shoulder_width, 2.0)  / 2.0,
            min(hip_width, 2.0)       / 2.0,
            min(torso_length, 2.0)    / 2.0,
            min(waist_est, 2.0)       / 2.0,
            min(shoulder_hip_ratio, 2.0) / 2.0,
            min(torso_height_ratio, 2.0) / 2.0,
        ]
        return torch.tensor([features], dtype=torch.float32)  # shape [1, 6]

    except ImportError:
        print("[WARN] mediapipe not installed. Pose measurements disabled.")
        return torch.zeros(1, 6)
    except Exception as e:
        print(f"[WARN] Pose measurement failed ({e}), using zeros.")
        return torch.zeros(1, 6)


# ============================================================
# MODEL ARCHITECTURE
# Input: image features (2048) + stats (2) + pose (6) = 2056
# ============================================================
class BodyFatRegressor(nn.Module):
    def __init__(self, use_pose_features=True):
        super().__init__()
        extra_features = 6 if use_pose_features else 0
        combined_size  = 2048 + 2 + extra_features  # 2056 or 2050

        base_model = resnet50(weights=None)
        self.features = nn.Sequential(
            *list(base_model.children())[:-2],
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.regressor = nn.Sequential(
            nn.Linear(combined_size, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.use_pose_features = use_pose_features

    def forward(self, img, stats, pose_feats=None):
        x = self.features(img)
        x = x.view(x.size(0), -1)                     # [B, 2048]
        parts = [x, stats]
        if self.use_pose_features and pose_feats is not None:
            parts.append(pose_feats)
        combined = torch.cat(parts, dim=1)
        return self.regressor(combined)


# ============================================================
# TRAINING TRANSFORMS — with augmentation
# ============================================================
def get_train_transforms():
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomAffine(degrees=5, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def get_inference_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


# ============================================================
# PREDICTION FUNCTION — full improved pipeline
# ============================================================
def predict_body_fat(front_path, side_path, height_cm, weight_kg,
                     model_path, norm_params_path, use_pose=False):
    """
    Full pipeline:
      1. MediaPipe crop (remove background context)
      2. rembg background removal (isolate body pixels)
      3. Pose measurement extraction (body proportions)
      4. ResNet50 + regressor inference
      5. Average front + side predictions

    Args:
        front_path      : path to front-view image
        side_path       : path to side-view image
        height_cm       : person's height in cm
        weight_kg       : person's weight in kg
        model_path      : path to saved model .pt file
        norm_params_path: path to normalization .npz file
        use_pose        : whether to use pose measurement features

    Returns:
        (float, str): body fat percentage, category label
    """
    device = torch.device("cpu")

    # --- Load normalization params ---
    if not os.path.exists(norm_params_path):
        return 0.0, "Error: Missing Params File"
    try:
        params  = np.load(norm_params_path)
        t_mean  = float(params['mean'])
        t_std   = float(params['std'])
    except Exception as e:
        return 0.0, f"Stats Error: {e}"

    # --- Load model ---
    model = BodyFatRegressor(use_pose_features=use_pose)
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint)
        model.to(device)
        model.eval()
    except Exception as e:
        return 0.0, f"Model Error: {e}"

    transform = get_inference_transforms()

    def prepare_input(path):
        """Full preprocessing: crop → bg remove → transform"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image missing: {path}")
        # Step 1: Crop tightly to person
        cropped = crop_to_person(path)
        # Step 2: Remove background
        clean   = remove_background(cropped)
        # Step 3: Transform to tensor
        return transform(clean).unsqueeze(0).to(device)

    try:
        # Prepare image tensors
        front_tensor = prepare_input(front_path)
        side_tensor  = prepare_input(side_path)

        # Physical stats (normalized)
        stats = torch.tensor(
            [[height_cm / 250.0, weight_kg / 200.0]],
            dtype=torch.float32
        ).to(device)

        # Pose measurements
        pose_front = extract_pose_measurements(front_path).to(device) if use_pose else None
        pose_side  = extract_pose_measurements(side_path).to(device)  if use_pose else None

        # Inference
        with torch.no_grad():
            res_f_norm = model(front_tensor, stats, pose_front).item()
            res_s_norm = model(side_tensor,  stats, pose_side).item()

        # Denormalize
        bf_front = (res_f_norm * t_std) + t_mean
        bf_side  = (res_s_norm * t_std) + t_mean
        final_bf = (bf_front + bf_side) / 2

        # Sanity clamp + category
        if final_bf < 4.0:
            final_bf, cat = 4.0, "Error / Very Lean"
        elif final_bf > 60.0:
            final_bf, cat = 60.0, "Error / High"
        else:
            if   final_bf < 6:  cat = "Essential Fat"
            elif final_bf < 14: cat = "Athletic"
            elif final_bf < 18: cat = "Fitness"
            elif final_bf < 25: cat = "Average"
            else:               cat = "Obese"

        return round(final_bf, 2), cat

    except Exception as e:
        print(f"Prediction Error: {e}")
        return 0.0, f"Inference Error: {e}"