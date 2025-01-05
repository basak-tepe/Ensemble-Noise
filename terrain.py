import numpy as np
import matplotlib.pyplot as plt
from noise import pnoise2
from scipy.ndimage import gaussian_filter

'''
Acknowledgment
This project benefited from the use of AI tools, including ChatGPT by OpenAI, for code generation, debugging.
The AI provided assistance in designing certain functions.
'''

def generate_perlin_noise(width, height, scale):
    noise = np.zeros((width, height))
    for x in range(width):
        for y in range(height):
            noise[x][y] = pnoise2(x / scale, y / scale, repeatx=width, repeaty=height)
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    return noise


def generate_random_noise(width, height):
    noise = np.random.rand(width, height)
    return noise


def generate_value_noise(width, height, scale):
    grid_width = width // scale
    grid_height = height // scale
    grid = np.random.rand(grid_width, grid_height)
    noise = np.zeros((width, height))
    
    for x in range(width):
        for y in range(height):
            x0, y0 = int(x / scale), int(y / scale)
            x1, y1 = min(x0 + 1, grid_width - 1), min(y0 + 1, grid_height - 1)       
            sx, sy = x / scale - x0, y / scale - y0
            t = sx * sx * (3.0 - 2.0 * sx)
            u = sy * sy * (3.0 - 2.0 * sy)
            noise[x, y] = (1 - t) * (1 - u) * grid[x0-1, y0-1] + t * (1 - u) * grid[x1-1, y0-1] + (1 - t) * u * grid[x0-1, y1-1] + t * u * grid[x1-1, y1-1]
    
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    return noise


def generate_wavelet_noise(width, height, scale):
    base_noise = np.random.rand(width, height)
    wavelet_noise = gaussian_filter(base_noise, sigma=scale)
    wavelet_noise = (wavelet_noise - wavelet_noise.min()) / (wavelet_noise.max() - wavelet_noise.min())
    return wavelet_noise


def generate_worley_noise(width, height, num_points):
    feature_points = np.random.rand(num_points, 2) * [width, height]
    noise = np.zeros((height, width))
    
    for i in range(height):
        for j in range(width):
            distances = np.linalg.norm(feature_points - np.array([j, i]), axis=1)
            noise[i, j] = np.min(distances)
    
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    
    return noise


def combine_noises(noise1, noise2, weight1, weight2):
    combined = weight1 * noise1 + weight2 * noise2
    combined = (combined - combined.min()) / (combined.max() - combined.min())
    return combined

# Parameters
width, height = 256, 256
scale_perlin1 = 20.0
scale_perlin2 = 40.0
scale_value = 10
scale_wavelet = 3.0
weight1, weight2 = 0.6, 0.4


original_perlin = generate_perlin_noise(width, height, scale_perlin1)
value_noise = generate_value_noise(width, height, scale_value)
wavelet_noise = generate_wavelet_noise(width, height, scale_wavelet)
plain_worley_noise = generate_worley_noise(width, height,120)
worley_perlin_combined = combine_noises(plain_worley_noise, original_perlin, weight1, weight2)

combined_perlin = combine_noises(
    generate_perlin_noise(width, height, scale_perlin1),
    generate_perlin_noise(width, height, scale_perlin2),
    weight1, weight2
)

original_random = generate_random_noise(width, height)
combined_perlin_random = combine_noises(
    generate_perlin_noise(width, height, scale_perlin1),
    generate_random_noise(width, height),
    weight1, weight2
)


perlin_value_combined = combine_noises(original_perlin, value_noise, weight1, weight2)
perlin_wavelet_combined = combine_noises(original_perlin, wavelet_noise, weight1, weight2)
fig, axs = plt.subplots(2, 5, figsize=(18, 8))


axs[0, 0].imshow(original_perlin, cmap="terrain", origin="upper")
axs[0, 0].set_title("Perlin Noise")
axs[0, 0].axis("off")

axs[0, 1].imshow(original_random, cmap="terrain", origin="upper")
axs[0, 1].set_title("Random Noise")
axs[0, 1].axis("off")

axs[0, 2].imshow(value_noise, cmap="terrain", origin="upper")
axs[0, 2].set_title("Value Noise")
axs[0, 2].axis("off")

axs[0, 3].imshow(wavelet_noise, cmap="terrain", origin="upper")
axs[0, 3].set_title("Wavelet Noise")
axs[0, 3].axis("off")

axs[0, 4].imshow(plain_worley_noise, cmap="terrain", origin="upper")
axs[0, 4].set_title("Worley Noise")
axs[0, 4].axis("off")


axs[1, 0].imshow(combined_perlin, cmap="terrain", origin="upper")
axs[1, 0].set_title("Perlin + Perlin")
axs[1, 0].axis("off")

axs[1, 1].imshow(combined_perlin_random, cmap="terrain", origin="upper")
axs[1, 1].set_title("Perlin + Random")
axs[1, 1].axis("off")

axs[1, 2].imshow(perlin_value_combined, cmap="terrain", origin="upper")
axs[1, 2].set_title("Perlin + Value")
axs[1, 2].axis("off")

axs[1, 3].imshow(perlin_wavelet_combined, cmap="terrain", origin="upper")
axs[1, 3].set_title("Perlin + Wavelet")
axs[1, 3].axis("off")

axs[1, 4].imshow(worley_perlin_combined, cmap="terrain", origin="upper")
axs[1, 4].set_title("Perlin + Worley")
axs[1, 4].axis("off")

plt.tight_layout()
plt.show()
