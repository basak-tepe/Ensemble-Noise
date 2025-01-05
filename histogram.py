import numpy as np
import matplotlib.pyplot as plt
from noise import pnoise2
from scipy.stats import entropy
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

def calculate_metrics(noise):
    histogram, _ = np.histogram(noise, bins=256, range=(0, 1), density=True)
    randomness = entropy(histogram)
    autocorrelation = np.corrcoef(noise[:-1, :].flatten(), noise[1:, :].flatten())[0, 1]
    mean_intensity = np.mean(noise)
    std_deviation = np.std(noise)
    return randomness, autocorrelation, mean_intensity, std_deviation

# Parameters
width, height = 256, 256
scale_perlin1, scale_perlin2 = 20.0, 100.0
scale_value = 10
scale_wavelet = 3.0
weight1, weight2 = 0.7, 0.3


original_perlin = generate_perlin_noise(width, height, scale_perlin1)
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
value_noise = generate_value_noise(width, height, scale_value)
wavelet_noise = generate_wavelet_noise(width, height, scale_wavelet)


plain_worley_noise = generate_worley_noise(width, height,120)
worley_perlin_combined = combine_noises(plain_worley_noise, original_perlin, weight1, weight2)
value_perlin_combined = combine_noises(value_noise, original_perlin, weight1, weight2)
perlin_wavelet_combined = combine_noises(original_perlin, wavelet_noise, weight1, weight2)


metrics = {
    "Original Perlin": calculate_metrics(original_perlin),
    "Weighted Perlin": calculate_metrics(combined_perlin),
    "Original Random": calculate_metrics(original_random),
    "Perlin + Random": calculate_metrics(combined_perlin_random),
    "Value Noise": calculate_metrics(value_noise),
    "Value + Perlin": calculate_metrics(value_perlin_combined),
    "Wavelet Noise": calculate_metrics(wavelet_noise),
    "Perlin + Wavelet": calculate_metrics(perlin_wavelet_combined),
    "Worley Noise": calculate_metrics(plain_worley_noise),
    "Perlin + Worley": calculate_metrics(worley_perlin_combined),
}


metric_names = ["Randomness", "Autocorrelation", "Mean Intensity", "Std Deviation"]
x = np.arange(len(metric_names))
bar_width = 0.05 
colors = [
    "#FF6347", "#8B0000",
    "#87CEFA", "#1E90FF", 
    "#98FB98", "#32CD32",
    "#FFD700", "#FF8C00",
    "#DA70D6", "#8A2BE2",  
]

fig, ax = plt.subplots(figsize=(12, 8))

for i, (name, values) in enumerate(metrics.items()):
    ax.barh(x + i * bar_width, values, bar_width, label=name, color=colors[i % len(colors)])


ax.set_yticks(x + bar_width * 3.5)
ax.set_yticklabels(metric_names, fontsize=12)
ax.set_xlabel("Metric Values", fontsize=14)
ax.set_title("Noise Map Metrics Comparison", fontsize=16, fontweight="bold")
ax.legend(fontsize=10, title="Noise Types")
ax.grid(axis="x", linestyle="--", alpha=0.7)


for i, (name, values) in enumerate(metrics.items()):
    for j, value in enumerate(values):
        ax.text(value + 0.01, x[j] + i * bar_width, f"{value:.2f}", va="center", fontsize=9)

plt.tight_layout()
plt.show()
