# Attention-Based Text-to-Image GAN

This project enhances a Generative Adversarial Network (GAN) by incorporating
self-attention mechanisms to improve image quality in text-to-image generation tasks.

## Key Idea
Self-attention allows the model to focus on important spatial regions, improving
global coherence and fine details in generated images.

## Architecture
- Generator with Self-Attention
- Discriminator with Self-Attention
- GAN framework using PyTorch

## How to Run
```bash
pip install -r requirements.txt
python train.py
python inference.py
