"""
PERSON 2 OWNS THIS FILE.
Loads images and applies the "real-world transformations" (our secret sauce).

Key idea: during TRAINING we randomly mangle images (JPEG, blur, noise, ...) so
the model learns fingerprints that survive mangling. During EVALUATION we apply
ONE fixed transformation at a time so Person 3 can measure robustness per-corruption.

The corruptions below match the OFFICIAL spec table (5.2) exactly.
"""
import io, os, glob, random
import numpy as np
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import config


# ---------- Custom corruptions (operate on PIL images) ----------
class RandomJPEG:
    """Re-save as JPEG to fake social-media / messaging re-encode."""
    def __init__(self, quality_range=(30, 90), p=0.5):
        self.quality_range, self.p = quality_range, p
    def __call__(self, img):
        if random.random() < self.p:
            q = random.randint(*self.quality_range)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=q)
            buf.seek(0)
            img = Image.open(buf).convert("RGB")
        return img


class Downscale:
    """Shrink then re-enlarge -> thumbnail generation. factor 2 = 0.5x, 4 = 0.25x."""
    def __init__(self, factor=2):
        self.factor = factor
    def __call__(self, img):
        w, h = img.size
        small = img.resize((max(1, w // self.factor), max(1, h // self.factor)), Image.BILINEAR)
        return small.resize((w, h), Image.BILINEAR)


class CenterCropResize:
    """Crop the middle then scale back up -> profile-picture cropping."""
    def __init__(self, frac=0.8):
        self.frac = frac
    def __call__(self, img):
        w, h = img.size
        cw, ch = int(w * self.frac), int(h * self.frac)
        left, top = (w - cw) // 2, (h - ch) // 2
        return img.crop((left, top, left + cw, top + ch)).resize((w, h), Image.BILINEAR)


class GaussianNoise:
    """Add pixel noise -> low-light sensor noise. sigma is on a 0..1 scale."""
    def __init__(self, sigma=0.05):
        self.sigma = sigma
    def __call__(self, img):
        arr = np.asarray(img).astype(np.float32) / 255.0
        arr = np.clip(arr + np.random.randn(*arr.shape) * self.sigma, 0, 1)
        return Image.fromarray((arr * 255).astype(np.uint8))


def _blur(sigma):
    """Fixed-strength Gaussian blur (kernel 5)."""
    return T.GaussianBlur(kernel_size=5, sigma=(sigma, sigma))


# ---------- Transforms ----------
def train_transform():
    """Heavy random augmentation = robustness. This is the heart of the project.
    We randomly apply the same families of corruptions the model is tested on."""
    return T.Compose([
        T.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        RandomJPEG(quality_range=(30, 90), p=0.5),
        T.RandomApply([T.GaussianBlur(3, sigma=(0.1, 2.0))], p=0.3),
        T.RandomApply([Downscale(random.choice([2, 4]))], p=0.3),
        T.RandomApply([GaussianNoise(sigma=random.choice([0.02, 0.05, 0.10]))], p=0.3),
        T.RandomApply([T.ColorJitter(0.2, 0.2, 0.2, 0.0)], p=0.3),   # +-20%
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(config.MEAN, config.STD),
    ])


def eval_transform(corruption=None):
    """Clean by default; pass one corruption to test robustness against it."""
    ops = [T.Resize((config.IMG_SIZE, config.IMG_SIZE))]
    if corruption is not None:
        ops.append(corruption)
    ops += [T.ToTensor(), T.Normalize(config.MEAN, config.STD)]
    return T.Compose(ops)


def get_corruptions():
    """Named single corruptions for the robustness table (Person 3 uses this).
    Matches the official spec table 5.2."""
    return {
        "clean":          None,
        "jpeg_q90":       RandomJPEG((90, 90), p=1.0),
        "jpeg_q70":       RandomJPEG((70, 70), p=1.0),
        "jpeg_q50":       RandomJPEG((50, 50), p=1.0),
        "jpeg_q30":       RandomJPEG((30, 30), p=1.0),
        "blur_0.5":       _blur(0.5),
        "blur_1.0":       _blur(1.0),
        "blur_2.0":       _blur(2.0),
        "resize_0.5":     Downscale(2),
        "resize_0.25":    Downscale(4),
        "noise_0.02":     GaussianNoise(0.02),
        "noise_0.05":     GaussianNoise(0.05),
        "noise_0.10":     GaussianNoise(0.10),
        "color_jitter":   T.ColorJitter(0.2, 0.2, 0.2, 0.0),
        "center_crop_80": CenterCropResize(0.8),
    }


# ---------- Dataset ----------
class ImageFolderDataset(Dataset):
    def __init__(self, roots, transform):
        """roots: one folder (str) or several (list) to combine multiple datasets."""
        if isinstance(roots, str):
            roots = [roots]
        self.samples, self.transform = [], transform
        for root in roots:
            for label, name in enumerate(config.CLASS_NAMES):  # real=0, fake=1
                folder = os.path.join(root, name)
                if not os.path.isdir(folder):                  # accept REAL/FAKE too
                    folder = os.path.join(root, name.upper())
                for path in glob.glob(os.path.join(folder, "*")):
                    self.samples.append((path, label))
        if not self.samples:
            print(f"[data] WARNING: no images found under {roots} — download the dataset first.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


def get_dataloaders():
    train_ds = ImageFolderDataset(config.TRAIN_DIRS, train_transform())
    test_ds  = ImageFolderDataset(config.TEST_DIRS,  eval_transform())
    nw, pin = config.get_num_workers(), config.use_pin_memory()
    train_dl = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True,  num_workers=nw, pin_memory=pin)
    test_dl  = DataLoader(test_ds,  batch_size=config.BATCH_SIZE, shuffle=False, num_workers=nw, pin_memory=pin)
    return train_dl, test_dl


if __name__ == "__main__":
    tr = ImageFolderDataset(config.TRAIN_DIRS, train_transform())
    te = ImageFolderDataset(config.TEST_DIRS, eval_transform())
    print(f"datasets: {config.DATA_ROOTS}")
    print(f"train images: {len(tr)}  |  test images: {len(te)}")
    print("corruptions available:", list(get_corruptions().keys()))
