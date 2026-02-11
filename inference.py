import torch
from models.generator import Generator
from torchvision.utils import save_image

device = "cuda" if torch.cuda.is_available() else "cpu"
G = Generator().to(device)

noise = torch.randn(1, 100, 1, 1).to(device)
image = G(noise)

save_image(image, "output.png", normalize=True)
print("Image generated and saved.")
