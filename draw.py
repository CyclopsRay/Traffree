

# import matplotlib.pyplot as plt
# import numpy as np

# # Defining the bar data
# categories = ['Rain', 'DirtyLens', 'Haze', 'GaussianBlur', 'Snow']
# accuracies = [0.543, 0.584, 0.562, 0.563, 0.488]
# read_accu = [0.682, 0.604, 0.596, 0.587, 0.493]
# focus_accu = [0.758, 0.739, 0.644, 0.635, 0.598]

# x = np.arange(len(categories))  # the label locations
# width = 0.2  # the width of the bars, adjusted for 3 bars

# fig, ax = plt.subplots(figsize=(10, 8))
# rects1 = ax.bar(x - width, accuracies, width, label='Backend')
# rects2 = ax.bar(x, read_accu, width, label='With Read')
# rects3 = ax.bar(x + width, focus_accu, width, label='With Read and Focus')

# # Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_ylabel('Accuracy')
# ax.set_title('Accuracy by Different Attack Scheme')
# ax.set_xticks(x)
# ax.set_xticklabels(categories)
# ax.legend()

# # Label with the percentage values
# def autolabel(rects, heights):
#     """Attach a text label above each bar in *rects*, displaying its height."""
#     for rect, height in zip(rects, heights):
#         ax.annotate(f'{height:.3f}',
#                     xy=(rect.get_x() + rect.get_width() / 2, height),
#                     xytext=(0, 3),  # 3 points vertical offset
#                     textcoords="offset points",
#                     ha='center', va='bottom')

# autolabel(rects1, accuracies)
# autolabel(rects2, read_accu)
# autolabel(rects3, focus_accu)

# fig.tight_layout()
# plt.show()

# ---------------------------

# import matplotlib.pyplot as plt
# import numpy as np

# # Data for plotting
# attack_strength = [1, 2, 3, 4, 5]
# overall_accuracy = [0.843, 0.843, 0.805, 0.784, 0.73]
# backend_read = [0.852, 0.76, 0.633, 0.589, 0.527]
# backend_read_rt = [0.838, 0.707, 0.562, 0.498, 0.447]

# fig, ax = plt.subplots(figsize=(10, 8))
# ax.plot(attack_strength, overall_accuracy, marker='o', label='Backend + F + R')
# ax.plot(attack_strength, backend_read, marker='s', label='Backend + Read')
# ax.plot(attack_strength, backend_read_rt, marker='^', label='Baseline')

# # Labels and Title
# ax.set_xlabel('Attack Strength')
# ax.set_ylabel('Overall Accuracy')
# ax.set_title('Performance under Different Attack Strengths')
# ax.set_xticks(attack_strength)
# ax.set_xticklabels(['1', '2', '3', '4', '5'])

# # Adding labels to each point on the lines
# for strength, oa, br, brt in zip(attack_strength, overall_accuracy, backend_read, backend_read_rt):
#     ax.text(strength, oa, f'{oa:.2f}', color='blue', ha='right')
#     ax.text(strength, br, f'{br:.2f}', color='green', ha='left')
#     ax.text(strength, brt, f'{brt:.2f}', color='red', ha='left')

# ax.legend()

# plt.show()

# --------------------------


import pywt
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load the image
original_image_path = "stop.bmp"  # Update this path to your image location
image = Image.open(original_image_path).convert('L')  # Convert to grayscale

# Convert image to numpy array
image_np = np.array(image)

# Perform 2D Discrete Wavelet Transform (DWT)
coeffs2 = pywt.dwt2(image_np, 'haar')
LL, (LH, HL, HH) = coeffs2

# Plot the results
fig, axes = plt.subplots(1, 4, figsize=(12, 3))
fig.suptitle('Wavelet Decomposition')

axes[0].imshow(LL, cmap='gray')
axes[0].set_title('LL (Low-Low)')
axes[0].axis('off')

axes[1].imshow(LH, cmap='gray')
axes[1].set_title('LH (Low-High)')
axes[1].axis('off')

axes[2].imshow(HL, cmap='gray')
axes[2].set_title('HL (High-Low)')
axes[2].axis('off')

axes[3].imshow(HH, cmap='gray')
axes[3].set_title('HH (High-High)')
axes[3].axis('off')

# Save the plot
output_path = "wavelet_decomposition.png"  # Update this path to your desired output location
plt.savefig(output_path)
plt.show()

