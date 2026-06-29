import torch
import os
import numpy as np
import sys
import cv2
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet50
from torch import nn

try:
    import numpy._core
except ImportError:
    sys.modules['numpy._core'] = np.core
    sys.modules['numpy._core.multiarray'] = np.core.multiarray

def crop_to_person(image_path, padding=30):
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
        return Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"[WARN] Crop failed ({e}), using full image.")
        return Image.open(image_path).convert("RGB")

def remove_background(pil_image):
    try:
        from rembg import remove
        result_rgba = remove(pil_image)
        background = Image.new("RGB", result_rgba.size, (255, 255, 255))
        background.paste(result_rgba, mask=result_rgba.split()[3])
        return background
    except ImportError:
        print("[WARN] rembg not installed. pip install rembg. Skipping BG removal.")
        return pil_image.convert("RGB")
    except Exception as e:
        print(f"[WARN] BG removal failed ({e}), skipping.")
        return pil_image.convert("RGB")

def extract_pose_measurements(image_path):
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
            return torch.zeros(1, 6)

        lm = results.pose_landmarks.landmark

        LEFT_SHOULDER  = 11
        RIGHT_SHOULDER = 12
        LEFT_HIP       = 23
        RIGHT_HIP      = 24
        LEFT_EAR       = 7
        RIGHT_EAR      = 8
        LEFT_ANKLE     = 27
        RIGHT_ANKLE    = 28
        LEFT_ELBOW     = 13
        RIGHT_ELBOW    = 14

        def dist(a, b):
            return ((lm[a].x - lm[b].x)**2 + (lm[a].y - lm[b].y)**2) ** 0.5

        shoulder_width     = dist(LEFT_SHOULDER, RIGHT_SHOULDER)
        hip_width          = dist(LEFT_HIP, RIGHT_HIP)
        torso_length       = (abs(lm[LEFT_SHOULDER].y - lm[LEFT_HIP].y) +
                               abs(lm[RIGHT_SHOULDER].y - lm[RIGHT_HIP].y)) / 2
        body_height        = abs(
            ((lm[LEFT_EAR].y + lm[RIGHT_EAR].y) / 2) -
            ((lm[LEFT_ANKLE].y + lm[RIGHT_ANKLE].y) / 2)
        ) + 1e-6
        waist_est          = dist(LEFT_ELBOW, RIGHT_ELBOW)
        shoulder_hip_ratio = shoulder_width / (hip_width + 1e-6)
        torso_height_ratio = torso_length / body_height

        features = [
            min(shoulder_width, 2.0)       / 2.0,
            min(hip_width, 2.0)            / 2.0,
            min(torso_length, 2.0)         / 2.0,
            min(waist_est, 2.0)            / 2.0,
            min(shoulder_hip_ratio, 2.0)   / 2.0,
            min(torso_height_ratio, 2.0)   / 2.0,
        ]
        return torch.tensor([features], dtype=torch.float32)

    except ImportError:
        print("[WARN] mediapipe not installed. Pose measurements disabled.")
        return torch.zeros(1, 6)
    except Exception as e:
        print(f"[WARN] Pose measurement failed ({e}), using zeros.")
        return torch.zeros(1, 6)

class BodyFatRegressor(nn.Module):
    """
    Dual-encoder model: separate ResNet-50 branches for front and side images.
    Feature dim: 2048 (front) + 2048 (side) + 2 (stats) = 4098
    """
    def __init__(self):
        super().__init__()

        def make_encoder():
            base = resnet50(weights=None)
            return nn.Sequential(
                *list(base.children())[:-2],
                nn.AdaptiveAvgPool2d((1, 1))
            )

        self.front_encoder = make_encoder()
        self.side_encoder  = make_encoder()

        self.regressor = nn.Sequential(
            nn.Linear(2048 * 2 + 2, 1024),
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

    def forward(self, front_img, side_img, stats):
        f = self.front_encoder(front_img).view(front_img.size(0), -1)  # (B, 2048)
        s = self.side_encoder(side_img).view(side_img.size(0), -1)     # (B, 2048)
        combined = torch.cat([f, s, stats], dim=1)                     # (B, 4098)
        return self.regressor(combined)

def get_inference_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def predict_body_fat(front_path, side_path, height_cm, weight_kg,
                     model_path, norm_params_path):

    device = torch.device("cpu")

    if not os.path.exists(norm_params_path):
        return 0.0, "Error: Missing Params File"
    try:
        params = np.load(norm_params_path)
        t_mean = float(params['mean'])
        t_std  = float(params['std'])
    except Exception as e:
        return 0.0, f"Stats Error: {e}"

    model = BodyFatRegressor()
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint)
        model.to(device)
        model.eval()
    except Exception as e:
        return 0.0, f"Model Error: {e}"

    transform = get_inference_transforms()

    def prepare_input(path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image missing: {path}")
        cropped = crop_to_person(path)
        clean   = remove_background(cropped)
        return transform(clean).unsqueeze(0).to(device)

    try:
        front_tensor = prepare_input(front_path)
        side_tensor  = prepare_input(side_path)

        stats = torch.tensor(
            [[height_cm / 250.0, weight_kg / 200.0]],
            dtype=torch.float32
        ).to(device)

        with torch.no_grad():
            pred_norm = model(front_tensor, side_tensor, stats).item()

        final_bf = (pred_norm * t_std) + t_mean

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
