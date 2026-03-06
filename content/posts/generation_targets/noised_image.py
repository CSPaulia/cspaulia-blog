from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

original_image_path = 'content/posts/generation_targets/Bao.jpg'
original_image = np.array(Image.open(original_image_path))

def add_noise(image, noise_level):
    """Add Gaussian noise to an image."""
    noisy_image = image + noise_level * np.random.randn(*image.shape) * 255
    noisy_image = np.clip(noisy_image, 0, 255)
    return noisy_image

if __name__ == "__main__":
    noise_level = 0.1
    save_steps = 20

    noisy_image = original_image.copy()
    for i in range(100):
        noisy_image = add_noise(noisy_image, noise_level)
        if i % save_steps == 0:
            Image.fromarray(noisy_image.astype(np.uint8)).save(f'content/posts/generation_targets/noisy_Bao_{i}.png')