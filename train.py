import torch
from models.generator import Generator
from models.discriminator import Discriminator

device = "cuda" if torch.cuda.is_available() else "cpu"

G = Generator().to(device)
D = Discriminator().to(device)

noise = torch.randn(1, 100, 1, 1).to(device)
fake_image = G(noise)

print("Generated image shape:", fake_image.shape)
