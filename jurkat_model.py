import torch
from torch import nn
from torch.nn import functional as F


class JurkatCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 4)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def predict_example(checkpoint_path, example_path):
    from pathlib import Path
    from PIL import Image
    import numpy as np

    checkpoint_path = Path(checkpoint_path)
    stages = ["G1", "S", "G2", "Mitosis"]
    model = JurkatCNN().eval()

    # If a saved trained checkpoint is provided, perform real inference.
    if checkpoint_path.is_file() and Path(example_path).is_file():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        state = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state, strict=True)
        is_trained = True
    else:
        torch.manual_seed(2026)
        model = JurkatCNN().eval()
        is_trained = False

    if is_trained:
        # Use the exported held-out BBBC048 Ch3 cell (with metadata).
        image = Image.open(example_path).convert("L").resize((64, 64))
        arr = np.asarray(image, dtype=np.float32) / 255.0
        x = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
    else:
        x = torch.zeros((1, 1, 64, 64), dtype=torch.float32)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0)
    return is_trained, tuple(x.shape), tuple(logits.shape), [round(float(p), 4) for p in probs], stages[int(probs.argmax())]
