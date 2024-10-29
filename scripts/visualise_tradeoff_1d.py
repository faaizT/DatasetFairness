import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def compute_accuracy(c, x_male, x_female, y_male, y_female):
    y_pred_male = (x_male > c).astype(int)
    y_pred_female = (x_female > c).astype(int)

    correct_male = (y_pred_male == y_male).sum()
    correct_female = (y_pred_female == y_female).sum()

    total_correct = correct_male + correct_female
    total_predictions = len(y_male) + len(y_female)

    return total_correct / total_predictions


def compute_equalized_odds(c, x_male, x_female, y_male, y_female, eop):
    # Classifier decisions
    y_pred_male = (x_male > c).astype(int)
    y_pred_female = (x_female > c).astype(int)

    # True positives and negatives for males
    TP_male = np.sum((y_pred_male == 1) & (y_male == 1))
    TN_male = np.sum((y_pred_male == 0) & (y_male == 0))

    # True positives and negatives for females
    TP_female = np.sum((y_pred_female == 1) & (y_female == 1))
    TN_female = np.sum((y_pred_female == 0) & (y_female == 0))

    # False positives and negatives for males
    FP_male = np.sum((y_pred_male == 1) & (y_male == 0))
    FN_male = np.sum((y_pred_male == 0) & (y_male == 1))

    # False positives and negatives for females
    FP_female = np.sum((y_pred_female == 1) & (y_female == 0))
    FN_female = np.sum((y_pred_female == 0) & (y_female == 1))

    # Compute true positive rates (TPR) and false positive rates (FPR) for both groups
    TPR_male = TP_male / (TP_male + FN_male)
    TPR_female = TP_female / (TP_female + FN_female)
    FPR_male = FP_male / (FP_male + TN_male)
    FPR_female = FP_female / (FP_female + TN_female)

    # Equalized odds is the maximum difference between TPRs and FPRs of the groups
    if eop:
        fairness_metric = abs(TPR_male - TPR_female)
    else:
        fairness_metric = (
            1 / 2 * (abs(TPR_male - TPR_female) + abs(FPR_male - FPR_female))
        )

    return fairness_metric


def compute_demographic_parity(c, male_mean, male_std_dev, female_mean, female_std_dev):
    pred_male = 1 - norm.cdf(c, loc=male_mean, scale=male_std_dev)
    pred_female = 1 - norm.cdf(c, loc=female_mean, scale=female_std_dev)

    return abs(pred_male - pred_female)


def plot_tradeoff(
    x_male,
    x_female,
    y_male,
    y_female,
    male_mean,
    male_std_dev,
    female_mean,
    female_std_dev,
    fairness_metric="dp",
    ax=None,
):
    c_values = np.linspace(-3, 3, 100)  # Change this range depending on your data
    accuracies = []
    fairness_metrics = []

    for c in c_values:
        accuracy = compute_accuracy(c, x_male, x_female, y_male, y_female)
        if fairness_metric == "dp":
            fairness = compute_demographic_parity(
                c, male_mean, male_std_dev, female_mean, female_std_dev
            )
        elif fairness_metric in ["eo", "eop"]:
            fairness = compute_equalized_odds(
                c, x_male, x_female, y_male, y_female, eop=fairness_metric == "eop"
            )
        accuracies.append(accuracy)
        fairness_metrics.append(fairness)

    # Assume fairness_metrics and accuracies are your two lists
    combined = list(zip(accuracies, fairness_metrics))

    # Sort combined list
    combined.sort()

    # Unzip the sorted combined list into accuracies and fairness_metrics
    accuracies_sorted, fairness_metrics_sorted = zip(*combined)

    if ax is None:
        plt.figure(figsize=(10, 5))
        plt.plot(
            accuracies_sorted,
            fairness_metrics_sorted,
            marker="o",
            label=f"GT {fairness_metric}",
        )
        plt.ylabel(fairness_metric.capitalize())
        plt.xlabel("Accuracy")
        plt.title(f"Trade-off between Accuracy and {fairness_metric.capitalize()} (GT)")
        plt.grid(True)
        plt.savefig(f"../plots/gt_tradeoff_1d_{fairness_metric}.png")
        print(
            f"saved ground truth tradeoff at ../plots/gt_tradeoff_1d_{fairness_metric}.png"
        )
    else:
        ax.plot(
            accuracies_sorted,
            fairness_metrics_sorted,
            linestyle="--",
            color="lime",
            label=f"GT {fairness_metric}",
            linewidth=3,
        )
        return ax
