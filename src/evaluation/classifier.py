import time

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.models import ResNet18_Weights, resnet18

# the pretrained weights expect inputs normalized with these
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# same augmentation as in the diffusion training (MedicalImageDataset)
FLIP_PROBABILITY = 0.5
MAX_ROTATION_DEGREES = 10.0


def build_classifier(num_classes: int, pretrained: bool = True) -> torch.nn.Module:
    # the ImageNet weights (~45 MB) are downloaded into ~/.cache/torch on first use
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model



def _augment_single(image: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    if torch.rand(1, generator=generator).item() < FLIP_PROBABILITY:
        image = TF.hflip(image)

    angle = (torch.rand(1, generator=generator).item() * 2 - 1) * MAX_ROTATION_DEGREES
    # black corners after rotation, like RandomRotation in the Dataset
    return TF.rotate(image, angle, fill=0.0)



def prepare_batch(
    images_uint8: np.ndarray, 
    augment: bool,
    generator: torch.Generator | None = None
) -> torch.Tensor:
    """uint8 [B, H, W] grayscale -> normalized float [B, 3, H, W].
    The images stay at their own size (128x128), ResNet accepts any input size."""
    images = torch.from_numpy(images_uint8).float().div(255.0).unsqueeze(1)

    if augment:
        if generator is None:
            raise ValueError("Augmentation needs a random generator.")
        augmented = []
        for index in range(images.shape[0]):
            augmented.append(_augment_single(images[index], generator))
        images = torch.stack(augmented)

    images = images.repeat(1, 3, 1, 1)
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (images - mean) / std



def class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    """Inverse frequency loss weights with a mean of 1, so rare classes are not ignored."""
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    weights = np.zeros(num_classes)
    present = counts > 0
    weights[present] = counts.sum() / (present.sum() * counts[present])
    return torch.tensor(weights, dtype=torch.float32)



def train_classifier(
    images: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    device: torch.device,
    seed: int,
    pretrained: bool = True,
    log_prefix: str = "",
) -> torch.nn.Module:
    """Trains for a fixed number of epochs without early stopping: picking the best epoch
    would need real validation images, and then the TSTR classifier would see real data too."""
    torch.manual_seed(seed)
    model = build_classifier(num_classes, pretrained=pretrained)
    model.to(device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_weights = class_weights(labels, num_classes).to(device)

    shuffle_rng = np.random.RandomState(seed)
    augment_generator = torch.Generator()
    augment_generator.manual_seed(seed)

    sample_count = images.shape[0]
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        order = shuffle_rng.permutation(sample_count)
        total_loss = 0.0
        batch_count = 0

        for start in range(0, sample_count, batch_size):
            batch_indices = order[start:start + batch_size]
            # BatchNorm fails in training mode on a batch of one
            if batch_indices.shape[0] < 2:
                continue

            inputs = prepare_batch(images[batch_indices], augment=True,
                                   generator=augment_generator).to(device)
            targets = torch.from_numpy(labels[batch_indices]).long().to(device)

            loss = F.cross_entropy(model(inputs), targets, weight=loss_weights)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1

        average_loss = total_loss / max(batch_count, 1)
        print(f"{log_prefix}epoch {epoch:2d}/{epochs}  loss: {average_loss:.4f}  "
              f"({time.time() - start_time:.0f}s)")

    return model



@torch.no_grad()
def predict(model: torch.nn.Module, images: np.ndarray, batch_size: int,
            device: torch.device) -> np.ndarray:
    model.eval()
    predictions = []
    for start in range(0, images.shape[0], batch_size):
        inputs = prepare_batch(images[start:start + batch_size], augment=False).to(device)
        predictions.append(model(inputs).argmax(dim=1).cpu().numpy())
    return np.concatenate(predictions).astype(np.int64)