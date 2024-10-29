import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
import matplotlib.animation as animation
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import os
import re
from tqdm import tqdm

from utils.confidence_intervals import (
    get_ci1_plus_ci2_series,
    get_fairness_bounds,
    get_hoeffding_bounds,
)
from utils.dataset_utils import create_dataset_male_female_synth_1d

# Plot labels
plot_labels = {
    "dp": "Demographic Parity",
    "eo": "Equalized Odds",
    "eop": "Equalized Opportunity",
    "eo_fpr": "Equalized odds (FPR)",
    "eo_tpr": "Equalized odds (TPR)",
}


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def create_tag(args, ignore_args=[]):
    tag_parts = []
    for arg, value in vars(args).items():
        if arg not in ignore_args:
            tag_parts.append(f"{arg}_{value}")
    tag = "_".join(tag_parts)
    return tag


def plot_dp_versus_lambda(model, test_loader, lambda_reg_values):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dp_vals = []
    with torch.no_grad():
        for lambda_reg in lambda_reg_values:
            dp = 0.0
            count = 0
            for i, (data, labels, sensitive_attr) in enumerate(test_loader):
                output = model(
                    data.to(device),
                    torch.FloatTensor([lambda_reg]).reshape(-1, 1).to(device),
                )
                dp += torch.abs(
                    (
                        output[sensitive_attr.to(device) == 0].mean()
                        - output[sensitive_attr.to(device) == 1].mean()
                    )
                ).item()
                count += 1
            dp_vals.append(dp / count)

    fig = plt.figure(figsize=(10, 6))
    plt.plot(lambda_reg_values, dp_vals, marker="o")
    plt.xlabel("Lambda Regularization")
    plt.ylabel("Demographic Parity")
    plt.title("Demographic Parity vs Lambda Regularization")
    plt.grid(True)
    return fig


def plot_metrics_versus_lambda_yoto(
    model, test_loader, lambda_reg_values, fig=None, axs=None
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dp_vals = []
    acc_vals = []
    with torch.no_grad():
        for lambda_reg in tqdm(lambda_reg_values):
            dp_0 = 0.0
            dp_1 = 0.0
            sensitive_0 = 0
            sensitive_1 = 0
            acc = 0
            total_preds = 0
            for i, (data, labels, sensitive_attr) in enumerate(test_loader):
                data = data.to(device)
                labels = labels.to(device)
                output = model(
                    data, torch.FloatTensor([lambda_reg]).reshape(-1, 1).to(device)
                )
                acc += (
                    (
                        (output > 0.5).reshape(-1) * labels.reshape(-1)
                        + ((output <= 0.5).reshape(-1) * (1 - labels).reshape(-1))
                    )
                    .sum()
                    .item()
                )
                total_preds += labels.shape[0]
                dp_0 += output[sensitive_attr.to(device) == 0].sum()
                dp_1 += output[sensitive_attr.to(device) == 1].sum()
                sensitive_0 += (sensitive_attr.to(device) == 0).sum()
                sensitive_1 += (sensitive_attr.to(device) == 1).sum()
            dp = torch.abs(dp_0 / sensitive_0 - dp_1 / sensitive_1).item()
            dp_vals.append(dp)
            acc_vals.append(acc / total_preds)

    if fig is None and axs is None:
        fig, axs = plt.subplots(3, figsize=(10, 18))

    axs[0].plot(lambda_reg_values, dp_vals, marker="o", label="yoto")
    axs[0].set_xlabel("Lambda Regularization")
    axs[0].set_ylabel("Demographic Parity")
    axs[0].set_title("Demographic Parity vs Lambda Regularization")
    axs[0].grid(True)

    axs[1].plot(lambda_reg_values, acc_vals, marker="x", label="yoto")
    axs[1].set_xlabel("Lambda Regularization")
    axs[1].set_ylabel("Accuracy")
    axs[1].set_title("Accuracy vs Lambda Regularization")
    axs[1].grid(True)

    axs[2].plot(acc_vals, dp_vals, marker="o", label="yoto")
    axs[2].set_ylabel("Demographic Parity")
    axs[2].set_xlabel("Accuracy")
    axs[2].set_title("Demographic Parity vs Accuracy")
    axs[2].grid(True)

    plt.tight_layout()
    return fig, axs


def plot_metrics_versus_lambda_eo(
    model, test_loader, lambda_reg_values, fig=None, axs=None
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    eo_vals = []
    eo_vals_2 = []
    acc_vals = []
    with torch.no_grad():
        for lambda_reg in lambda_reg_values:
            tpr0 = 0.0
            tpr1 = 0.0
            fpr0 = 0.0
            fpr1 = 0.0
            y_1_sensitive_0, y_0_sensitive_0, y_1_sensitive_1, y_0_sensitive_1 = (
                0,
                0,
                0,
                0,
            )
            acc = 0
            total_preds = 0
            for i, values in enumerate(test_loader):
                data, labels, sensitive_attr = values[0], values[1], values[2]
                data = data.to(device)
                labels = labels.to(device)
                output = model(
                    data, torch.FloatTensor([lambda_reg]).reshape(-1, 1).to(device)
                )
                acc += (
                    (
                        (output > 0.5).reshape(-1) * labels.reshape(-1)
                        + ((output <= 0.5).reshape(-1) * (1 - labels).reshape(-1))
                    )
                    .sum()
                    .item()
                )
                total_preds += labels.shape[0]
                tpr0 += (
                    output * (labels == 1) * (sensitive_attr.to(device) == 0)
                ).sum()
                tpr1 += (
                    output * (labels == 1) * (sensitive_attr.to(device) == 1)
                ).sum()
                fpr0 += (
                    output * (labels == 0) * (sensitive_attr.to(device) == 0)
                ).sum()
                fpr1 += (
                    output * (labels == 0) * (sensitive_attr.to(device) == 1)
                ).sum()
                y_1_sensitive_0 += (
                    (labels == 1) * (sensitive_attr.to(device) == 0)
                ).sum()
                y_1_sensitive_1 += (
                    (labels == 1) * (sensitive_attr.to(device) == 1)
                ).sum()
                y_0_sensitive_0 += (
                    (labels == 0) * (sensitive_attr.to(device) == 0)
                ).sum()
                y_0_sensitive_1 += (
                    (labels == 0) * (sensitive_attr.to(device) == 1)
                ).sum()
            eo = torch.abs(tpr0 / y_1_sensitive_0 - tpr1 / y_1_sensitive_1) + torch.abs(
                fpr0 / y_0_sensitive_0 - fpr1 / y_0_sensitive_1
            )
            eo_vals.append(eo.item())
            acc_vals.append(acc / total_preds)

    if fig is None and axs is None:
        fig, axs = plt.subplots(3, figsize=(10, 18))

    axs[0].plot(lambda_reg_values, eo_vals, marker="o", label="yoto")
    # axs[0].plot(lambda_reg_values, eo_vals_2, marker="x", label="yoto_resampled")
    axs[0].set_xlabel("Lambda Regularization")
    axs[0].set_ylabel("Equalized Odds")
    axs[0].set_title("Equalized Odds vs Lambda Regularization")
    axs[0].grid(True)

    axs[1].plot(lambda_reg_values, acc_vals, marker="x", label="yoto")
    axs[1].set_xlabel("Lambda Regularization")
    axs[1].set_ylabel("Accuracy")
    axs[1].set_title("Accuracy vs Lambda Regularization")
    axs[1].grid(True)

    axs[2].plot(acc_vals, eo_vals, marker="o", label="yoto")
    # axs[2].plot(acc_vals, eo_vals_2, marker="x", label="yoto_resampled")
    axs[2].set_ylabel("Equalized Odds")
    axs[2].set_xlabel("Accuracy")
    axs[2].set_title("Equalized Odds vs Accuracy")
    axs[2].grid(True)
    if data.shape[1] == 1:
        create_dataset_male_female_synth_1d(
            n_samples=50000, plot_tradeoff_=True, fairness_metric="eo", ax=axs[2]
        )

    plt.tight_layout()
    return fig, axs


def plot_metrics_versus_lambda_eop(
    model, test_loader, lambda_reg_values, fig=None, axs=None
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    eop_vals = []
    acc_vals = []
    with torch.no_grad():
        for lambda_reg in lambda_reg_values:
            tpr0 = 0.0
            tpr1 = 0.0
            y_1_sensitive_0, y_0_sensitive_0, y_1_sensitive_1, y_0_sensitive_1 = (
                0,
                0,
                0,
                0,
            )
            acc = 0
            total_preds = 0
            for i, values in enumerate(test_loader):
                data, labels, sensitive_attr = values[0], values[1], values[2]
                data = data.to(device)
                labels = labels.to(device)
                output = model(
                    data, torch.FloatTensor([lambda_reg]).reshape(-1, 1).to(device)
                )
                acc += (
                    (
                        (output > 0.5).reshape(-1) * labels.reshape(-1)
                        + ((output <= 0.5).reshape(-1) * (1 - labels).reshape(-1))
                    )
                    .sum()
                    .item()
                )
                total_preds += labels.shape[0]
                tpr0 += (
                    output * (labels == 1) * (sensitive_attr.to(device) == 0)
                ).sum()
                tpr1 += (
                    output * (labels == 1) * (sensitive_attr.to(device) == 1)
                ).sum()
                y_1_sensitive_0 += (
                    (labels == 1) * (sensitive_attr.to(device) == 0)
                ).sum()
                y_1_sensitive_1 += (
                    (labels == 1) * (sensitive_attr.to(device) == 1)
                ).sum()
            eop = torch.abs(tpr0 / y_1_sensitive_0 - tpr1 / y_1_sensitive_1)
            eop_vals.append(eop.item())
            acc_vals.append(acc / total_preds)

    if fig is None and axs is None:
        fig, axs = plt.subplots(3, figsize=(10, 18))

    axs[0].plot(lambda_reg_values, eop_vals, marker="o", label="yoto")
    axs[0].set_xlabel("Lambda Regularization")
    axs[0].set_ylabel("Equalized Opportunity")
    axs[0].set_title("Equalized Opportunity vs Lambda Regularization")
    axs[0].grid(True)

    axs[1].plot(lambda_reg_values, acc_vals, marker="x", label="yoto")
    axs[1].set_xlabel("Lambda Regularization")
    axs[1].set_ylabel("Accuracy")
    axs[1].set_title("Accuracy vs Lambda Regularization")
    axs[1].grid(True)

    axs[2].plot(acc_vals, eop_vals, marker="o", label="yoto")
    axs[2].set_ylabel("Equalized Opportunity")
    axs[2].set_xlabel("Accuracy")
    axs[2].set_title("Equalized Opportunity vs Accuracy")
    axs[2].grid(True)
    if data.shape[1] == 1:
        create_dataset_male_female_synth_1d(
            n_samples=50000, plot_tradeoff_=True, fairness_metric="eop", ax=axs[2]
        )

    plt.tight_layout()
    return fig, axs


def animate_decision_boundary(model, data_loader, lambda_reg_values):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define the domain of the plot
    x_range = np.linspace(0, 10, 100)
    y_range = np.linspace(0, 10, 100)

    xx, yy = np.meshgrid(x_range, y_range)
    zz = np.zeros(xx.shape)

    fig, ax = plt.subplots()

    def animate(i):
        ax.cla()  # clear the previous plot
        lambda_val = lambda_reg_values[i]
        print(f"on frame {i}")

        # Perform inference for each input in the grid
        for i in range(xx.shape[0]):
            for j in range(xx.shape[1]):
                x = np.array([xx[i, j], yy[i, j]], dtype=np.float32)
                output = model(
                    torch.from_numpy(x).unsqueeze(0).to(device),
                    torch.FloatTensor([lambda_val]).reshape(-1, 1).to(device),
                )
                zz[i, j] = output > 0.5  # converting output to boolean values

        dp = 0.0
        count = 0
        acc = 0
        total_preds = 0

        data_points = []
        sensitive_attr_points = []

        # calculate demographic parity and accuracy
        for i, (data, labels, sensitive_attr) in enumerate(data_loader):
            data = data.to(device)
            labels = labels.to(device)
            output = model(
                data, torch.FloatTensor([lambda_val]).reshape(-1, 1).to(device)
            )
            acc += (
                (
                    (output > 0.5).reshape(-1) * labels.reshape(-1)
                    + ((output <= 0.5).reshape(-1) * (1 - labels).reshape(-1))
                )
                .sum()
                .item()
            )
            total_preds += labels.shape[0]
            dp += torch.abs(
                (
                    output[sensitive_attr.to(device) == 0].mean()
                    - output[sensitive_attr.to(device) == 1].mean()
                )
            ).item()
            count += 1

            data_points.append(data.cpu().numpy())
            sensitive_attr_points.append(sensitive_attr.cpu().numpy())

        dp = dp / count
        acc = acc / total_preds

        data_points = np.concatenate(data_points)
        sensitive_attr_points = np.concatenate(sensitive_attr_points).reshape(-1)

        # Plot the decision boundary and data points
        ax.contourf(xx, yy, zz, levels=1, colors=["red", "blue"], alpha=0.5)
        ax.scatter(
            data_points[sensitive_attr_points == 0, 0],
            data_points[sensitive_attr_points == 0, 1],
            color="red",
        )
        ax.scatter(
            data_points[sensitive_attr_points == 1, 0],
            data_points[sensitive_attr_points == 1, 1],
            color="blue",
        )
        ax.set_title(f"Lambda = {lambda_val:.2f}, DP = {dp:.2f}, Acc = {acc:.2f}")
        ax.set_xlim([0, 10])
        ax.set_ylim([0, 10])

    ani = animation.FuncAnimation(
        fig, animate, frames=len(lambda_reg_values), repeat=False
    )
    return ani


def average_dataframes(df1, df2):
    if len(df1) == 0:
        return df2
    if len(df2) == 0:
        return df1
    # Merge the dataframes on the "Lambda" column
    merged_df = pd.merge(df1, df2, on="Lambda", suffixes=("_df1", "_df2"))

    result_df = pd.DataFrame()
    result_df["Lambda"] = merged_df["Lambda"]

    # We do not process the "Lambda" column
    columns = set(df1.columns) - set(["Lambda"])

    for column in columns:
        col_df1 = column + "_df1"
        col_df2 = column + "_df2"

        if df1[column].dtype == object:
            # For object data types, we assume these are lists
            result_df[column] = [
                (np.array(a) + np.array(b)) / 2
                for a, b in zip(merged_df[col_df1], merged_df[col_df2])
            ]
        else:
            # For non-object (numerical) data types, we simply take the mean
            result_df[column] = (merged_df[col_df1] + merged_df[col_df2]) / 2

    return result_df


def get_metrics_for_model_yoto(
    model, test_loader, lambda_reg_values, fairness="dp", accumulate=False
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []

    with torch.no_grad():
        for lambda_reg in lambda_reg_values:
            fairness_0 = np.array([])
            fairness_1 = np.array([])
            tpr0 = np.array([])
            tpr1 = np.array([])
            fpr0 = np.array([])
            fpr1 = np.array([])
            acc = np.array([])

            for i, (data, labels, sensitive_attr) in enumerate(test_loader):
                data = data.to(device)
                labels = labels.to(device)
                output = model(
                    data, torch.FloatTensor([lambda_reg]).reshape(-1, 1).to(device)
                )
                acc_list = (
                    (
                        (output > 0.5).reshape(-1) * labels.reshape(-1)
                        + ((output <= 0.5).reshape(-1) * (1 - labels).reshape(-1))
                    )
                    .reshape(-1)
                    .cpu()
                    .numpy()
                )
                acc = np.concatenate([acc, acc_list])
                if fairness == "dp":
                    fairness_0 = np.concatenate(
                        [
                            fairness_0,
                            output[sensitive_attr.to(device) == 0]
                            .reshape(-1)
                            .cpu()
                            .numpy(),
                        ]
                    )
                    fairness_1 = np.concatenate(
                        [
                            fairness_1,
                            output[sensitive_attr.to(device) == 1]
                            .reshape(-1)
                            .cpu()
                            .numpy(),
                        ]
                    )
                elif fairness.startswith("eo"):
                    fpr0 = np.concatenate(
                        [
                            fpr0,
                            output[(labels == 0) * (sensitive_attr.to(device) == 0)]
                            .reshape(-1)
                            .cpu()
                            .numpy(),
                        ]
                    )
                    fpr1 = np.concatenate(
                        [
                            fpr1,
                            output[(labels == 0) * (sensitive_attr.to(device) == 1)]
                            .reshape(-1)
                            .cpu()
                            .numpy(),
                        ]
                    )
                    tpr0 = np.concatenate(
                        [
                            tpr0,
                            output[(labels == 1) * (sensitive_attr.to(device) == 0)]
                            .reshape(-1)
                            .cpu()
                            .numpy(),
                        ]
                    )
                    tpr1 = np.concatenate(
                        [
                            tpr1,
                            output[(labels == 1) * (sensitive_attr.to(device) == 1)]
                            .reshape(-1)
                            .cpu()
                            .numpy(),
                        ]
                    )

            accuracy_value = np.mean(acc)
            if fairness == "dp":
                dp = np.abs(np.mean(fairness_0) - np.mean(fairness_1))
                if not accumulate:
                    results.append((lambda_reg, dp, accuracy_value))
                else:
                    results.append(
                        (lambda_reg, dp, accuracy_value, fairness_0, fairness_1, acc)
                    )
            elif fairness == "eo":
                fairness_metric = np.abs(np.mean(tpr0) - np.mean(tpr1)) + np.abs(
                    np.mean(fpr0) - np.mean(fpr1)
                )
            else:
                fairness_metric = np.abs(np.mean(tpr0) - np.mean(tpr1))
            if fairness.startswith("eo"):
                if not accumulate:
                    results.append((lambda_reg, fairness_metric, accuracy_value))
                else:
                    results.append(
                        (
                            lambda_reg,
                            fairness_metric,
                            accuracy_value,
                            tpr0,
                            tpr1,
                            fpr0,
                            fpr1,
                            acc,
                        )
                    )

    if accumulate and fairness == "dp":
        return pd.DataFrame(
            results,
            columns=[
                "Lambda",
                "Fairness metric",
                "Accuracy",
                "Fairness_0",
                "Fairness_1",
                "acc_vec",
            ],
        )
    if accumulate and fairness.startswith("eo"):
        df = pd.DataFrame(
            results,
            columns=[
                "Lambda",
                "Fairness metric",
                "Accuracy",
                "Tpr0",
                "Tpr1",
                "Fpr0",
                "Fpr1",
                "acc_vec",
            ],
        )
        if fairness == "eo":
            df["Tpr0"] /= 2
            df["Tpr1"] /= 2
            df["Fpr0"] /= 2
            df["Fpr1"] /= 2
            df["Fairness metric"] /= 2
        return df
    df = pd.DataFrame(results, columns=["Lambda", "Fairness metric", "Accuracy"])
    if fairness == "eo":
        df["Fairness metric"] /= 2
    return df


def plot_metrics_for_all_models_yoto(
    model_dir,
    test_loader,
    lambda_reg_values,
    input_dim,
    max_epochs,
    stoppage_epoch,
    final_epoch,
    n_samples,
    lr,
    hidden_layer_size,
    film_hidden_size,
    n_layers_film,
    batch_size,
    square_dp,
    accumulate,
    fairness="dp",
    yoto_results=None,
    fig=None,
    axs=None,
):
    all_results = pd.DataFrame()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Iterate over all saved models
    CI_data_retained = False
    model_num = 0
    if yoto_results is None:
        for filename in tqdm(os.listdir(model_dir)):
            # Check if this file corresponds to a model with the given parameters
            if final_epoch:
                regex1 = rf"model_input_dim_{input_dim}_max_epochs_{max_epochs}_n_samples_{n_samples}_lr_{lr}_hidden_layer_size_{hidden_layer_size}_film_hidden_size_{film_hidden_size}_n_layers_film_{n_layers_film}_seed_(\d+)_batch_size_{batch_size}_lambda_lb_1e-05_lambda_ub_(?:2.0|5.0|10.0)_per_batch_lambda_True_(?:square_dp|sq_fl)_{square_dp}(?:_fairness_{fairness})?.pth"
                # The following regex is only for the adult dataset
                regex2 = rf"model_input_dim_{input_dim}_max_epochs_{max_epochs}_n_samples_{n_samples}_lr_{lr}_hidden_layer_size_{hidden_layer_size}_film_hidden_size_{film_hidden_size}_n_layers_film_{n_layers_film}_seed_(\d+)_lambda_lb_1e-05_lambda_ub_(?:2.0|5.0|10.0)_per_batch_lambda_True.pth"
                regex3 = rf"model_fe(\d+)_input_dim_{input_dim}_max_epochs_{max_epochs}_n_samples_{n_samples}_lr_{lr}_hidden_layer_size_{hidden_layer_size}_film_hidden_size_{film_hidden_size}_n_layers_film_{n_layers_film}_seed_(\d+)_batch_size_{batch_size}_lambda_lb_1e-05_lambda_ub_(?:5.0|10.0)_per_batch_lambda_True_(?:square_dp|sq_fl)_{square_dp}(?:_fairness_{fairness})?.pth"
                regexes = np.array([regex1, regex2, regex3])
                for regex in regexes:
                    if re.match(regex, filename):
                        break
            else:
                regex = rf"model_e{stoppage_epoch}_input_dim_{input_dim}_max_epochs_{max_epochs}_n_samples_{n_samples}_lr_{lr}_hidden_layer_size_{hidden_layer_size}_film_hidden_size_{film_hidden_size}_n_layers_film_{n_layers_film}_seed_(\d+)_batch_size_{batch_size}_lambda_lb_1e-05_lambda_ub_5.0_per_batch_lambda_True_(?:square_dp|sq_fl)_{square_dp}(?:_fairness_{fairness})?.pth"
            # regex = rf"model_input_dim_{input_dim}_max_epochs_{max_epochs}_n_samples_{n_samples}_lr_{lr}_hidden_layer_size_{hidden_layer_size}_film_hidden_size_{film_hidden_size}_n_layers_film_{n_layers_film}_seed_(\d+)_lambda_lb_1e-05_lambda_ub_5.0_per_batch_lambda_True.pth"
            match = re.search(regex, filename)
            if match is not None and model_num < 4:
                model_path = os.path.join(model_dir, filename)
                model = torch.load(model_path, map_location=device)
                model_results = get_metrics_for_model_yoto(
                    model,
                    test_loader,
                    lambda_reg_values,
                    fairness=fairness,
                    accumulate=accumulate,
                )
                if accumulate:
                    all_results = average_dataframes(all_results, model_results)
                else:
                    all_results = pd.concat(
                        [all_results, model_results], ignore_index=True
                    )
                model_num += 1
    else:
        all_results = yoto_results

    # Plot results
    if fig is None and axs is None:
        fig, axs = plt.subplots(3, figsize=(10, 18))

    if len(all_results) > 0:
        all_results_grouped = (
            all_results.groupby(by="Lambda").mean().sort_values(by="Accuracy")
        )
        fairness_values = all_results_grouped["Fairness metric"].values
        min_fairness = [
            np.min(fairness_values[i:]) for i in range(len(fairness_values))
        ]
        if len(axs) == 3:
            sns.lineplot(
                data=all_results,
                x="Lambda",
                y="Fairness metric",
                ax=axs[0],
                label="yoto",
            )
            sns.lineplot(
                data=all_results, x="Lambda", y="Accuracy", ax=axs[1], label="yoto"
            )
            sns.lineplot(
                data=all_results_grouped,
                x="Accuracy",
                y=min_fairness,
                ax=axs[2],
                label="yoto",
                marker="o",
            )
        elif len(axs) == 1:
            sns.lineplot(
                data=all_results_grouped,
                x="Accuracy",
                y=min_fairness,
                ax=axs[0],
                label="YOTO (Ours)",
                marker="o",
                linewidth=4,
                markersize=10,
            )
    else:
        return None, None, None
    return fig, axs, all_results


def delta_senstive_analysis(df_yoto, df_sep):
    delta = 0
    if len(df_sep) == 0:
        return delta
    for index, row in df_yoto.iterrows():
        if (df_sep["accuracy"] >= row["Accuracy"]).sum() > 0:
            sep_dp = df_sep.loc[
                df_sep["accuracy"] >= row["Accuracy"], "fairness_metric"
            ].min()
            delta = max(delta, row["Fairness metric"] - sep_dp)
    return delta


def get_sep_model_stats(test_loader, model_filepath, device, accumulate=False):
    # Load the saved model
    model = torch.load(model_filepath, map_location=device).to(device)

    # Evaluate model on the test set and compute metrics
    accuracies = np.array([])
    dp_males = np.array([])
    dp_females = np.array([])
    with torch.no_grad():
        for i, (data, labels, sensitive_attr) in tqdm(enumerate(test_loader)):
            data, labels, sensitive_attr = (
                data.to(device),
                labels.to(device),
                sensitive_attr.to(device),
            )
            outputs = model(data)
            accuracy = (
                (
                    (outputs > 0.5).reshape(-1) * labels.reshape(-1)
                    + (outputs <= 0.5).reshape(-1) * (1 - labels).reshape(-1)
                )
                .reshape(-1)
                .cpu()
                .numpy()
            )
            dp_0 = outputs[sensitive_attr == 0].reshape(-1).cpu().numpy()
            dp_1 = outputs[sensitive_attr == 1].reshape(-1).cpu().numpy()

            accuracies = np.concatenate([accuracies, accuracy])
            dp_males = np.concatenate([dp_males, dp_0])
            dp_females = np.concatenate([dp_females, dp_1])

    if accumulate:
        return accuracies, dp_males, dp_females

    accuracy = np.mean(accuracies)
    dp = np.abs(np.mean(dp_males) - np.mean(dp_females))

    return accuracy, dp


def get_sep_model_stats_eo(test_loader, model_filepath, device, accumulate=False):
    # Load the saved model
    model = torch.load(model_filepath, map_location=device).to(device)

    # Evaluate model on the test set and compute metrics
    accuracies = np.array([])
    tpr0, tpr1, fpr0, fpr1 = np.array([]), np.array([]), np.array([]), np.array([])
    with torch.no_grad():
        for i, (data, labels, sensitive_attr) in tqdm(enumerate(test_loader)):
            data, labels, sensitive_attr = (
                data.to(device),
                labels.to(device),
                sensitive_attr.to(device),
            )
            outputs = model(data)
            accuracy = (
                (
                    (outputs > 0.5).reshape(-1) * labels.reshape(-1)
                    + ((outputs <= 0.5).reshape(-1) * (1 - labels).reshape(-1))
                )
                .reshape(-1)
                .cpu()
                .numpy()
            )
            tpr0 = np.concatenate(
                [
                    tpr0,
                    outputs[(labels == 1) * (sensitive_attr.to(device) == 0)]
                    .reshape(-1)
                    .cpu()
                    .numpy(),
                ]
            )
            tpr1 = np.concatenate(
                [
                    tpr1,
                    outputs[(labels == 1) * (sensitive_attr.to(device) == 1)]
                    .reshape(-1)
                    .cpu()
                    .numpy(),
                ]
            )
            fpr0 = np.concatenate(
                [
                    fpr0,
                    outputs[(labels == 0) * (sensitive_attr.to(device) == 0)]
                    .reshape(-1)
                    .cpu()
                    .numpy(),
                ]
            )
            fpr1 = np.concatenate(
                [
                    fpr1,
                    outputs[(labels == 0) * (sensitive_attr.to(device) == 1)]
                    .reshape(-1)
                    .cpu()
                    .numpy(),
                ]
            )
            accuracies = np.concatenate([accuracies, accuracy])

    if accumulate:
        return accuracies, tpr0, tpr1, fpr0, fpr1

    accuracy = np.mean(accuracies)
    eo_tpr = np.abs(np.mean(tpr0) - np.mean(tpr1))
    eo_fpr = np.abs(np.mean(fpr0) - np.mean(fpr1))
    return accuracy, eo_tpr, eo_fpr


def get_eo_bounds(accuracies, tpr0, tpr1, fpr0, fpr1, alpha):
    accuracy, accuracy_lb, accuracy_ub, tpr, tpr_lb, tpr_ub = get_fairness_bounds(
        accuracies, tpr0, tpr1, alpha / 2
    )
    accuracy, accuracy_lb, accuracy_ub, fpr, fpr_lb, fpr_ub = get_fairness_bounds(
        accuracies, fpr0, fpr1, alpha / 2
    )
    eo = tpr + fpr
    eo_lb, eo_ub = get_ci1_plus_ci2_series((tpr_lb, tpr_ub), (fpr_lb, fpr_ub))
    return accuracy, accuracy_lb, accuracy_ub, eo, eo_lb, eo_ub


def get_fairness_bounds_eo(acc_vec, tpr0, tpr1, fpr0, fpr1, alpha, bound_type):
    accuracy, accuracy_lb, accuracy_ub, tpr, tpr_lb, tpr_ub = get_fairness_bounds(
        acc_vec,
        tpr0,
        tpr1,
        alpha=alpha,
        bound_type=bound_type,
    )
    accuracy, accuracy_lb, accuracy_ub, fpr, fpr_lb, fpr_ub = get_fairness_bounds(
        acc_vec,
        fpr0,
        fpr1,
        alpha=alpha,
        bound_type=bound_type,
    )
    eo_lb, eo_ub = get_ci1_plus_ci2_series((tpr_lb, tpr_ub), (fpr_lb, fpr_ub))
    return (
        accuracy,
        accuracy_lb,
        accuracy_ub,
        tpr + fpr,
        eo_lb,
        eo_ub,
    )


def get_fairness_bounds_separate(
    model_filepath, dataloader, device, alpha, output_CIs, fairness="dp"
):
    if output_CIs and fairness == "dp":
        accuracies, dp_males_arr, dp_females_arr = get_sep_model_stats(
            dataloader, model_filepath, device, accumulate=True
        )
        return get_fairness_bounds(accuracies, dp_males_arr, dp_females_arr, alpha)
    if output_CIs and fairness.startswith("eo"):
        accuracies, tpr0, tpr1, fpr0, fpr1 = get_sep_model_stats_eo(
            dataloader, model_filepath, device, accumulate=True
        )
        if fairness == "eo":
            return get_eo_bounds(
                accuracies, tpr0 / 2, tpr1 / 2, fpr0 / 2, fpr1 / 2, alpha
            )
        # compute bounds for eop
        return get_fairness_bounds(accuracies, tpr0, tpr1, alpha)
    accuracy_lb, accuracy_ub, fairness_lb, fairness_ub = (
        None,
        None,
        None,
        None,
    )
    if fairness == "dp":
        accuracy, dp = get_sep_model_stats(
            dataloader, model_filepath, device, accumulate=False
        )
        return accuracy, accuracy_lb, accuracy_ub, dp, fairness_lb, fairness_ub
    accuracy, eo_tpr, eo_fpr = get_sep_model_stats_eo(
        dataloader, model_filepath, device, accumulate=False
    )
    if fairness == "eo":
        return (
            accuracy,
            accuracy_lb,
            accuracy_ub,
            (eo_tpr + eo_fpr) / 2,
            fairness_lb,
            fairness_ub,
        )
    # Output EOP
    return accuracy, accuracy_lb, accuracy_ub, eo_tpr, fairness_lb, fairness_ub


def plot_confidence_intervals(
    df, color, bound_type, L=1, delta=0, fig=None, axs=None, fairness="dp"
):
    """
    Plot the confidence intervals of demographic parity.

    Args:
        acc_vals (list or numpy array): Values of accuracy.
        dp_lb (list or numpy array): Lower bounds of demographic parity.
        dp_ub (list or numpy array): Upper bounds of demographic parity.
    """

    # accuracy, accuracy_lb, accuracy_ub, dp, dp_lb, dp_ub
    if fairness == "dp":
        bounds_df = df.apply(
            lambda x: get_fairness_bounds(
                x["acc_vec"],
                x["Fairness_0"],
                x["Fairness_1"],
                alpha=0.05,
                bound_type=bound_type,
            ),
            axis=1,
        )
    elif fairness == "eo":
        bounds_df = df.apply(
            lambda x: get_fairness_bounds_eo(
                x["acc_vec"],
                x["Tpr0"],
                x["Tpr1"],
                x["Fpr0"],
                x["Fpr1"],
                alpha=0.05,
                bound_type=bound_type,
            ),
            axis=1,
        )
    elif fairness == "eop":
        bounds_df = df.apply(
            lambda x: get_fairness_bounds(
                x["acc_vec"],
                x["Tpr0"],
                x["Tpr1"],
                alpha=0.05,
                bound_type=bound_type,
            ),
            axis=1,
        )
    # unpacking
    (
        accuracy,
        accuracy_lb,
        accuracy_ub,
        fairness_metric,
        fairness_lb,
        fairness_ub,
    ) = tuple(list(x) for x in zip(*bounds_df))

    # Create a new figure
    if fig is None:
        fig, axs = plt.subplots(1, figsize=(10, 6))

    accuracy_lb.sort()
    accuracy_ub.sort()
    fairness_lb.sort()
    fairness_ub.sort()
    # Plot the colored region
    accuracy_lb = np.concatenate((np.linspace(0.5, accuracy_lb[0], 10), accuracy_lb))
    accuracy_ub = np.concatenate((np.linspace(0.5, accuracy_ub[0], 10), accuracy_ub))

    fairness_lb = np.concatenate(
        (
            np.linspace(fairness_lb[0], fairness_lb[0], 10),
            fairness_lb,
        )
    )
    fairness_ub = np.concatenate(
        (
            np.linspace(fairness_ub[0], fairness_ub[0], 10),
            fairness_ub,
        )
    )

    accuracy_lb = np.concatenate(
        (accuracy_lb, np.linspace(accuracy_lb[-1], accuracy_ub[-1], 10))
    )
    accuracy_ub = np.concatenate((accuracy_ub, np.linspace(accuracy_ub[-1], 1, 10)))
    fairness_lb = np.concatenate((fairness_lb, np.ones(10) * fairness_lb[-1]))
    fairness_ub = np.concatenate((fairness_ub, np.linspace(fairness_ub[-1], 1, 10)))

    x = np.concatenate(
        (accuracy_lb, accuracy_ub[::-1])
    )  # note the order: we go forward on x1, then backward on x2
    fairness_lb_w_delta = np.where(
        np.array(fairness_lb) - L * delta > 0,
        np.array(fairness_lb) - L * delta,
        0,
    )
    y = np.concatenate((fairness_ub, fairness_lb_w_delta[::-1]))  # same here
    axs.fill(x, y, alpha=0.15, label=f"{bound_type.split('_')[0]} CIs", color=color)
    axs.plot(accuracy_lb, fairness_ub, color=color)
    axs.plot(accuracy_ub, fairness_lb_w_delta, color=color)

    return fig, axs
