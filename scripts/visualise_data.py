import numpy as np
import matplotlib.pyplot as plt
from fairlearn.metrics import demographic_parity_difference

from utils.dataset_utils import create_dataset_male_female_synth

n_samples = 1000
X, y, sensitive_feature = create_dataset_male_female_synth(n_samples)
# Check demographic parity difference
dp_diff = demographic_parity_difference(y, y, sensitive_features=sensitive_feature)

print("Demographic parity difference: ", dp_diff)

# Plot the data
plt.figure(figsize=(10, 7))
plt.scatter(X[sensitive_feature == 0, 0], X[sensitive_feature == 0, 1], label="male", alpha=0.4)
plt.scatter(X[sensitive_feature == 1, 0], X[sensitive_feature == 1, 1], label="female", alpha=0.4)

threshold = 4
# Add decision boundary
plt.axvline(x=threshold, color="k", linestyle="--", label="Decision boundary")

plt.title("Binary Classification with Sensitive Feature")
plt.legend(loc="best")
plt.savefig("./plots/dataset-male-female-synth.png")
