import inspect
import collections
import os
import torch
import smplx
import trimesh
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter
import pyrender
from tqdm import tqdm
import torch_directml
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights

if not hasattr(inspect, 'getargspec'):
    ArgSpec = collections.namedtuple('ArgSpec', ['args', 'varargs', 'keywords', 'defaults'])

    def getargspec(func):
        spec = inspect.getfullargspec(func)
        return ArgSpec(spec.args, spec.varargs, spec.varkw, spec.defaults)

    inspect.getargspec = getargspec
    inspect.ArgSpec = ArgSpec

np.bool = bool
np.int = int
np.float = float
np.complex = complex
np.object = object
np.unicode = str
np.str = str

#1: REALISTIC DATA GENERATION

MALE_MODEL_FILE = r'C:\Users\mihne\OneDrive\Desktop\sarpili\IOT\models\SMPL_MALE.pkl'
OUTPUT_DIR = r'C:\Users\mihne\OneDrive\Desktop\sarpili\IOT\synthetic_dataset_v2'
NUM_SAMPLES = 3000

os.makedirs(os.path.join(OUTPUT_DIR, 'images'), exist_ok=True)

model = smplx.create(MALE_MODEL_FILE, model_type='smpl', gender='male', num_betas=10)

def get_mesh_volume(vertices, faces):
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    return abs(mesh.volume)

def render_realistic_body(vertices, faces, angle=0):
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    bg_colors = [
        [0.95, 0.95, 0.95],
        [0.9, 0.92, 0.94],
        [0.92, 0.90, 0.88],
        [0.88, 0.90, 0.92],
        [1.0, 1.0, 1.0]
    ]
    bg = bg_colors[np.random.randint(0, len(bg_colors))]
    scene = pyrender.Scene(bg_color=bg)

    angle_var = angle + np.random.uniform(-15, 15)
    rot = trimesh.transformations.rotation_matrix(np.radians(angle_var), [0, 1, 0])
    mesh.apply_transform(rot)

    skin_tones = [
        [0.92, 0.80, 0.70],
        [0.85, 0.72, 0.62],
        [0.78, 0.65, 0.55],
        [0.70, 0.58, 0.48],
        [0.60, 0.50, 0.42],
    ]
    skin = skin_tones[np.random.randint(0, len(skin_tones))]

    material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=skin + [1.0],
        roughnessFactor=np.random.uniform(0.6, 0.9),
        metallicFactor=0.0
    )

    render_mesh = pyrender.Mesh.from_trimesh(mesh, material=material)
    scene.add(render_mesh)

    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    cam_pose = np.eye(4)
    cam_pose[:3, 3] = [
        np.random.uniform(-0.15, 0.15),
        np.random.uniform(-0.1, 0.1),
        np.random.uniform(2.2, 2.8)
    ]
    scene.add(camera, pose=cam_pose)

    main_intensity = np.random.uniform(2.5, 4.0)
    main_light = pyrender.DirectionalLight(
        color=[1.0, 0.98, 0.95],
        intensity=main_intensity
    )
    scene.add(main_light, pose=cam_pose)

    if np.random.rand() > 0.3:
        fill_pose = cam_pose.copy()
        fill_pose[:3, 3] = [1.5, 0.5, 2.0]
        fill_light = pyrender.DirectionalLight(
            color=[0.95, 0.97, 1.0],
            intensity=main_intensity * 0.4
        )
        scene.add(fill_light, pose=fill_pose)

    r = pyrender.OffscreenRenderer(224, 224)
    color, _ = r.render(scene)
    img = Image.fromarray(color)

    augmentations = np.random.rand()

    if augmentations > 0.3:
        img = ImageEnhance.Brightness(img).enhance(np.random.uniform(0.85, 1.15))

    if augmentations > 0.4:
        img = ImageEnhance.Contrast(img).enhance(np.random.uniform(0.9, 1.1))

    if augmentations > 0.6:
        img = img.filter(ImageFilter.GaussianBlur(radius=np.random.uniform(0, 0.8)))

    if augmentations > 0.7:
        img = ImageEnhance.Color(img).enhance(np.random.uniform(0.95, 1.05))

    return img

def generate_realistic_dataset():
    data_log = []
    print(f"Generating {NUM_SAMPLES} REALISTIC synthetic bodies...")

    for i in tqdm(range(NUM_SAMPLES), desc="Generating bodies"):
        if i < NUM_SAMPLES * 0.15:
            beta_scale = np.random.uniform(1.0, 1.5)
        elif i < NUM_SAMPLES * 0.40:
            beta_scale = np.random.uniform(1.5, 2.2)
        elif i < NUM_SAMPLES * 0.75:
            beta_scale = np.random.uniform(2.0, 3.0)
        else:
            beta_scale = np.random.uniform(2.8, 4.5)

        betas = torch.randn([1, 10]) * beta_scale

        output = model(betas=betas, return_verts=True)
        verts = output.vertices.detach().cpu().numpy().squeeze()
        faces = model.faces

        volume_m3 = get_mesh_volume(verts, faces)
        volume_liters = volume_m3 * 1000.0
        density = np.random.uniform(1.005, 1.095)
        weight_kg = volume_liters * density
        bf_percent = np.clip((495 / density) - 450, 3.0, 50.0)
        height_cm = (verts[:, 1].max() - verts[:, 1].min()) * 100.0

        f_img = render_realistic_body(verts, faces, angle=0)
        s_img = render_realistic_body(verts, faces, angle=90)

        f_name = f"body_{i:04d}_front.png"
        s_name = f"body_{i:04d}_side.png"
        f_img.save(os.path.join(OUTPUT_DIR, 'images', f_name))
        s_img.save(os.path.join(OUTPUT_DIR, 'images', s_name))

        data_log.append({
            'front': f_name, 'side': s_name,
            'height': height_cm, 'weight': weight_kg, 'bf': bf_percent
        })

    df = pd.DataFrame(data_log)
    df.to_csv(os.path.join(OUTPUT_DIR, 'realistic_labels.csv'), index=False)

    print(f"\n✅ Dataset generated: {OUTPUT_DIR}")
    print(f"Body Fat % Stats:")
    print(f"  Mean: {df['bf'].mean():.2f}%")
    print(f"  Std:  {df['bf'].std():.2f}%")
    print(f"  Range: {df['bf'].min():.2f}% - {df['bf'].max():.2f}%")

    return df

# PART 2: TRAINING ON REALISTIC DATA

class BodyFatRegressor(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        base_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        self.features = nn.Sequential(*list(base_model.children())[:-2],
                                      nn.AdaptiveAvgPool2d((1, 1)))

        self.regressor = nn.Sequential(
            nn.Linear(2048 + 2, 1024),
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

    def forward(self, img, stats):
        x = self.features(img)
        x = x.view(x.size(0), -1)
        combined = torch.cat((x, stats), dim=1)
        return self.regressor(combined)


class RealisticBodyFatDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

        self.target_mean = self.data['bf'].mean()
        self.target_std = self.data['bf'].std()

        print(f"Dataset: {len(self.data)} samples")
        print(f"BF% - Mean: {self.target_mean:.2f}%, Std: {self.target_std:.2f}%")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.img_dir, 'images', row['front'])
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        stats = torch.tensor([row['height'] / 250.0, row['weight'] / 200.0],
                             dtype=torch.float32)
        target_norm = (row['bf'] - self.target_mean) / self.target_std

        return image, stats, torch.tensor([target_norm], dtype=torch.float32)


def train_on_realistic_data():
    device = torch_directml.device()
    print(f"\n{'=' * 60}")
    print(f"Training on AMD GPU: {device}")
    print(f"{'=' * 60}\n")

    csv_file = os.path.join(OUTPUT_DIR, "realistic_labels.csv")
    output_dir = os.path.join(OUTPUT_DIR, "model_outputs")
    os.makedirs(output_dir, exist_ok=True)

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = RealisticBodyFatDataset(csv_file, OUTPUT_DIR, transform=train_transform)

    np.savez(os.path.join(output_dir, "bodyfat_norm_params.npz"),
             mean=dataset.target_mean, std=dataset.target_std)

    loader = DataLoader(dataset, batch_size=32, shuffle=True,
                        num_workers=0, pin_memory=True)

    model = BodyFatRegressor(pretrained=True).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
    criterion = nn.SmoothL1Loss()

    print("Starting training for 50 epochs")
    best_loss = float('inf')

    for epoch in range(50):
        model.train()
        total_loss = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/50")

        for imgs, stats, targets in pbar:
            imgs = imgs.to(device)
            stats = stats.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(imgs, stats)
            loss = criterion(outputs, targets)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}",
                             lr=f"{optimizer.param_groups[0]['lr']:.6f}")

        avg_loss = total_loss / len(loader)
        scheduler.step()

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(),
                       os.path.join(output_dir, "bodyfat_model_BEST.pth"))
            print(f" New best model saved! Loss: {best_loss:.4f}")

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}: Loss={avg_loss:.4f}, Best={best_loss:.4f}")

    torch.save(model.state_dict(),
               os.path.join(output_dir, "bodyfat_model_final.pth"))

    print(f"\n{'=' * 60}")
    print(f" Training complete!")
    print(f"Best model: {output_dir}/bodyfat_model_BEST.pth")
    print(f"{'=' * 60}\n")

# MAIN EXECUTION

if __name__ == "__main__":
    print("=" * 60)
    print("COMPLETE RETRAINING PIPELINE")
    print("=" * 60)
    print("\nThis will:")
    print("Generate 3000 realistic synthetic bodies")

    response = input("Continue? (y/n): ")
    if response.lower() != 'y':
        print("Cancelled.")
        exit()

    print("\n" + "=" * 60)
    print("STEP 1: Generating Realistic Synthetic Dataset")
    print("=" * 60 + "\n")

    df = generate_realistic_dataset()

    print("\n" + "=" * 60)
    print("STEP 2: Training Model on Realistic Data")
    print("=" * 60 + "\n")

    train_on_realistic_data()

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE!")
    print("=" * 60)
    print(f"\nNew model location:")
    print(f"  {OUTPUT_DIR}/model_outputs/bodyfat_model_BEST.pth")
    print(f"  model_path = r'{OUTPUT_DIR}\\model_outputs\\bodyfat_model_BEST.pth'")
