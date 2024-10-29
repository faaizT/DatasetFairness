import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import re
from utils.confidence_intervals import (
    get_asymptotic_bounds,
    get_bentkus_bounds,
    get_bernstein_bounds,
    get_ci1_over_ci2_series,
    get_ci1_plus_ci2_series,
    get_hoeffding_bounds,
    get_bootstrap_bounds,
)
from torch.utils.data import DataLoader

from utils.dataset_utils import CustomDataset
from utils.helper_functions import average_dataframes
from utils.confidence_intervals import get_fairness_bounds


def create_surrogate_sensatts(sensitive_attr, prob=1.0):
    # Step 1: create a binary mask
    mask = torch.bernoulli(torch.full(sensitive_attr.shape, prob)).bool()

    # Step 2: create surrogate tensor
    sensitive_attr_surrogates = sensitive_attr.clone()

    # Create a permutation of sensitive_attr
    perm = torch.randperm(len(sensitive_attr))

    # Replace ~mask elements with the permuted sensitive_attr elements
    sensitive_attr_surrogates[~mask] = sensitive_attr[perm][~mask]
    return sensitive_attr_surrogates


def get_dataloader_w_surrogates(dataloader, prob=1.0):
    dataset = dataloader.dataset
    surrogate_probs = dataset.sensitive_attr * prob + (1 - dataset.sensitive_attr) * (
        1 - prob
    )
    surrogate_sensatts = torch.bernoulli(torch.FloatTensor(surrogate_probs))
    return DataLoader(
        CustomDataset(
            dataset.X,
            dataset.y,
            dataset.sensitive_attr,
            (surrogate_probs, surrogate_sensatts),
        ),
        batch_size=32,
    )


def get_metrics_for_model_yoto_w_sensatt(
    model, dataloader_w_sensatt, lambda_reg_values
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []

    with torch.no_grad():
        for lambda_reg in lambda_reg_values:
            dp_0_err = np.array([])
            dp_1_err = np.array([])
            acc = np.array([])
            output_vec = np.array([])
            sensitive_attr_vec = np.array([])
            surrogate_vec = np.array([])

            for i, (
                data,
                labels,
                sensitive_attr,
                surrogate_probs,
                surrogate_sensatts,
            ) in enumerate(dataloader_w_sensatt):
                data = data.to(device)
                labels = labels.to(device)
                sensitive_attr = sensitive_attr.to(device)
                surrogate_probs = surrogate_probs.to(device)
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
                output_vec = np.concatenate(
                    [output_vec, output.reshape(-1).cpu().numpy()]
                )
                sensitive_attr_vec = np.concatenate(
                    [sensitive_attr_vec, sensitive_attr.reshape(-1).cpu().numpy()]
                )
                surrogate_vec = np.concatenate(
                    [surrogate_vec, surrogate_probs.reshape(-1).cpu().numpy()]
                )
                dp_0_err = np.concatenate(
                    [
                        dp_0_err,
                        (
                            output * (sensitive_attr.to(device) == 0)
                            - output * (1 - surrogate_probs)
                        )
                        .reshape(-1)
                        .cpu()
                        .numpy(),
                    ]
                )
                dp_1_err = np.concatenate(
                    [
                        dp_1_err,
                        (
                            output * (sensitive_attr.to(device) == 1)
                            - output * surrogate_probs
                        )
                        .reshape(-1)
                        .cpu()
                        .numpy(),
                    ]
                )

            dp_err = np.abs(np.mean(dp_0_err) - np.mean(dp_1_err))
            accuracy_value = np.mean(acc)
            results.append(
                (
                    lambda_reg,
                    dp_err,
                    accuracy_value,
                    dp_0_err,
                    dp_1_err,
                    acc,
                    output_vec,
                    sensitive_attr_vec,
                    surrogate_vec,
                )
            )

    return pd.DataFrame(
        results,
        columns=[
            "Lambda",
            "DP_error",
            "Accuracy",
            "DP_0_error",
            "DP_1_error",
            "acc_vec",
            "output_vec",
            "sens_att_vec",
            "surrogate_vec",
        ],
    )


def get_metrics_for_model_yoto_wo_sensatt(
    model, dataloader_wo_sensatt, lambda_reg_values
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []

    with torch.no_grad():
        for lambda_reg in lambda_reg_values:
            dp_0 = np.array([])
            dp_1 = np.array([])
            dp_0_gt = np.array([])
            dp_1_gt = np.array([])
            acc = np.array([])

            for i, (
                data,
                labels,
                sensitive_attr,
                surrogate_probs,
                surrogate_sensatt,
            ) in enumerate(dataloader_wo_sensatt):
                data = data.to(device)
                labels = labels.to(device)
                sensitive_attr = sensitive_attr.to(device)
                surrogate_sensatt = surrogate_sensatt.to(device)
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
                dp_0 = np.concatenate(
                    [
                        dp_0,
                        output[surrogate_sensatt.reshape(-1).to(device) == 0]
                        .reshape(-1)
                        .cpu()
                        .numpy(),
                    ]
                )
                dp_1 = np.concatenate(
                    [
                        dp_1,
                        output[surrogate_sensatt.reshape(-1).to(device) == 1]
                        .reshape(-1)
                        .cpu()
                        .numpy(),
                    ]
                )
                dp_0_gt = np.concatenate(
                    [
                        dp_0_gt,
                        output[sensitive_attr.reshape(-1).to(device) == 0]
                        .reshape(-1)
                        .cpu()
                        .numpy(),
                    ]
                )
                dp_1_gt = np.concatenate(
                    [
                        dp_1_gt,
                        output[sensitive_attr.reshape(-1).to(device) == 1]
                        .reshape(-1)
                        .cpu()
                        .numpy(),
                    ]
                )

            dp = np.abs(np.mean(dp_0) - np.mean(dp_1))
            dp_gt = np.abs(np.mean(dp_0_gt) - np.mean(dp_1_gt))
            accuracy_value = np.mean(acc)
            results.append((lambda_reg, dp, dp_gt, accuracy_value, dp_0, dp_1, acc))

    return (
        pd.DataFrame(
            results,
            columns=[
                "Lambda",
                "Demographic Parity",
                "Demographic Parity (GT)",
                "Accuracy",
                "DP_0",
                "DP_1",
                "acc_vec",
            ],
        ),
        dataloader_wo_sensatt.dataset.surrogates[0].reshape(-1).cpu().numpy(),
    )


def get_bootstrap_intervals_for_dp_errors(output_arr, A_arr, surrog_arr, alpha, B=100):
    data_df = pd.DataFrame({"output": output_arr, "A": A_arr, "surrog": surrog_arr})
    bootstrap_values = []
    for i in range(B):
        # bootstrap sample
        sample_df = data_df.sample(n=len(data_df), replace=True)
        # compute statistic for this bootstrap sample
        stat_a1 = sample_df[sample_df["A"] == 1]["output"].mean()
        stat_a0 = sample_df[sample_df["A"] == 0]["output"].mean()
        # stat_s1 = sample_df[sample_df["surrog"] == 1]["output"].mean()
        # stat_s0 = sample_df[sample_df["surrog"] == 0]["output"].mean()
        stat_s1 = (sample_df["output"] * sample_df["surrog"]).mean() / sample_df[
            "surrog"
        ].mean()
        stat_s0 = (sample_df["output"] * (1 - sample_df["surrog"])).mean() / (
            1 - sample_df["surrog"]
        ).mean()
        bootstrap_stat = (stat_a1 - stat_s1) - (stat_a0 - stat_s0)
        bootstrap_values.append(bootstrap_stat)
    # compute lower and upper percentiles
    lower_bound = np.percentile(bootstrap_values, 100 * alpha / 2)
    upper_bound = np.percentile(bootstrap_values, 100 * (1 - alpha / 2))
    return lower_bound, upper_bound


def get_intervals_on_dp_error(
    df_metrics_w_sensatt, senstive_att_surrogates, get_bounds, alpha
):
    # obtain intervals for E[Y [1(A=0)-1(f(X)=0)] ]
    (
        dp0_err_wsensatt_lb,
        dp0_err_wsensatt,
        dp0_err_wsensatt_ub,
    ) = get_cis_elementwise_from_series(
        df_metrics_w_sensatt["DP_0_error"], get_bounds, alpha / 2
    )
    dp0_err_wsensatt_lb = np.clip(dp0_err_wsensatt_lb, -1, 1)
    dp0_err_wsensatt_ub = np.clip(dp0_err_wsensatt_ub, -1, 1)

    # obtain intervals for P(f(X) = 1) and P(f(X) = 0)
    p_fx_1_lb, _, p_fx_1_ub = get_bounds(senstive_att_surrogates, alpha / 2)
    p_fx_0_lb, p_fx_0_ub = 1 - p_fx_1_ub, 1 - p_fx_1_lb
    p_fx_1_lb = np.clip(p_fx_1_lb, 0, 1)
    p_fx_1_ub = np.clip(p_fx_1_ub, 0, 1)
    p_fx_0_lb = np.clip(p_fx_0_lb, 0, 1)
    p_fx_0_ub = np.clip(p_fx_0_ub, 0, 1)

    # obtain intervals for E[Y [1(A=0)-1(f(X)=0)] ]/P(f(X)=0)
    dp0_err_wsensatt_lb, dp0_err_wsensatt_ub = get_ci1_over_ci2_series(
        (dp0_err_wsensatt_lb, dp0_err_wsensatt_ub), (p_fx_0_lb, p_fx_0_ub)
    )

    # obtain intervals for E[Y [1(A=1)-1(f(X)=1)] ]
    (
        dp1_err_wsensatt_lb,
        dp1_err_wsensatt,
        dp1_err_wsensatt_ub,
    ) = get_cis_elementwise_from_series(
        df_metrics_w_sensatt["DP_1_error"], get_bounds, alpha / 2
    )
    dp0_err_wsensatt_lb = np.clip(dp0_err_wsensatt_lb, -1, 1)
    dp0_err_wsensatt_ub = np.clip(dp0_err_wsensatt_ub, -1, 1)

    # obtain intervals for E[Y [1(A=1)-1(f(X)=1)] ]/P(f(X)=1)
    dp1_err_wsensatt_lb, dp1_err_wsensatt_ub = get_ci1_over_ci2_series(
        (dp1_err_wsensatt_lb, dp1_err_wsensatt_ub), (p_fx_1_lb, p_fx_1_ub)
    )

    # Confidence intervals for (E[Y [1(A=1)-1(f(X)=1)] ]/P(f(X)=1) - E[Y [1(A=0)-1(f(X)=0)] ]/P(f(X)=0))
    dp_err_wsensatt_lb, dp_err_wsensatt_ub = get_ci1_plus_ci2_series(
        (dp1_err_wsensatt_lb, dp1_err_wsensatt_ub),
        (-dp0_err_wsensatt_ub, -dp0_err_wsensatt_lb),
    )
    return dp_err_wsensatt_lb, dp_err_wsensatt_ub


def get_cis_elementwise_from_series(series, get_bounds_func, alpha):
    """
    Get confidence intervals from series of arrays elementwise
    """
    bounds_df = series.apply(lambda x: get_bounds_func(x, alpha / 2))
    lb, mean, ub = tuple(pd.Series(x) for x in zip(*bounds_df))
    return lb, mean, ub


def get_bounds_by_combining_datasets_with_and_without_sensatt(
    df_metrics_w_sensatt,
    df_metrics_wo_sensatt,
    senstive_att_surrogates,
    alpha,
    bound_type,
    use_bootstrap_for_dp_error=True,
):
    if bound_type == "hoeffdings":
        get_bounds = get_hoeffding_bounds
    elif bound_type == "asymptotic":
        get_bounds = get_asymptotic_bounds
    elif bound_type == "bernstein":
        get_bounds = get_bernstein_bounds
    elif bound_type == "bentkus":
        get_bounds = get_bentkus_bounds
    elif bound_type == "bootstrap":
        get_bounds = get_bootstrap_bounds
    else:
        raise ValueError(
            "bound_type must be one of ['hoeffdings', 'asymptotic', 'bernstein', 'bentkus']"
        )

    accuracy_vectors = pd.merge(
        left=df_metrics_w_sensatt.loc[:, ["Lambda", "acc_vec"]],
        right=df_metrics_wo_sensatt.loc[:, ["Lambda", "acc_vec"]],
        on=["Lambda"],
    )

    # obtain intervals for accuracy (here we can combine the two datasets as accuracy computation doesn't involve sensitive attributes)
    accuracy_vectors["acc_vec"] = accuracy_vectors.apply(
        lambda x: np.concatenate((x["acc_vec_x"], x["acc_vec_y"])), axis=1
    )

    accuracy_lb, accuracy, accuracy_ub = get_cis_elementwise_from_series(
        accuracy_vectors["acc_vec"], get_bounds, alpha / 2
    )

    if use_bootstrap_for_dp_error:
        # use bootstrap to obtain intervals on dp error
        bootstrap_intervals = df_metrics_w_sensatt.apply(
            lambda x: get_bootstrap_intervals_for_dp_errors(
                x["output_vec"],
                x["sens_att_vec"],
                x["surrogate_vec"],
                alpha=alpha / 4,
                B=100,
            ),
            axis=1,
        )
        dp_err_wsensatt_lb, dp_err_wsensatt_ub = tuple(
            pd.Series(x) for x in zip(*bootstrap_intervals)
        )
    else:
        # Confidence intervals for (E[Y [1(A=1)-1(f(X)=1)] ]/P(f(X)=1) - E[Y [1(A=0)-1(f(X)=0)] ]/P(f(X)=0))
        dp_err_wsensatt_lb, dp_err_wsensatt_ub = get_intervals_on_dp_error(
            df_metrics_w_sensatt, senstive_att_surrogates, get_bounds, alpha / 4
        )

    # CIs for E[Y | f(X) = 0]
    bounds_df = df_metrics_wo_sensatt.apply(
        lambda x: get_fairness_bounds(
            x["acc_vec"],
            x["DP_0"],
            x["DP_1"],
            alpha=0.05,
            bound_type=f"{bound_type}_diff",
        ),
        axis=1,
    )
    # unpacking
    (
        _,
        _,
        _,
        fairness_metric,
        dp_wo_sensatt_lb,
        dp_wo_sensatt_ub,
    ) = tuple(list(x) for x in zip(*bounds_df))

    # Finally, CIs for  E[Y | A = 1] - E[Y | A = 0]
    dp_lb, dp_ub = get_ci1_plus_ci2_series(
        (dp_err_wsensatt_lb, dp_err_wsensatt_ub), (dp_wo_sensatt_lb, dp_wo_sensatt_ub)
    )

    # turn into CIs for | E[Y | A = 1] - E[Y | A = 0] |
    dp_lb_abs = np.where(
        (dp_lb <= 0) & (dp_ub >= 0), 0, np.minimum(dp_lb.abs(), dp_ub.abs())
    )
    dp_ub_abs = np.maximum(dp_lb.abs(), dp_ub.abs())

    # Clip values between 0 and 1
    accuracy_lb = np.clip(accuracy_lb, 0, 1)
    accuracy_ub = np.clip(accuracy_ub, 0, 1)
    dp_lb_abs = np.clip(dp_lb_abs, 0, 1)
    dp_ub_abs = np.clip(dp_ub_abs, 0, 1)
    return accuracy_lb, accuracy_ub, dp_lb_abs, dp_ub_abs


def get_dp_fairness_bounds_combined_datasets(
    yoto_model,
    dataloader_w_sensatt,
    dataloader_wo_sensatt,
    bound_type,
    alpha,
    surrogate_sensatt_overlap_prob,
):
    lambda_reg_values = list(np.linspace(0.0, 5, num=101))
    df_0 = get_metrics_for_model_yoto_w_sensatt(
        yoto_model,
        dataloader_w_sensatt,
        lambda_reg_values,
    )
    df_1, senstive_att_surrogates = get_metrics_for_model_yoto_wo_sensatt(
        yoto_model,
        dataloader_wo_sensatt,
        lambda_reg_values,
    )
    (
        accuracy_lb,
        accuracy_ub,
        dp_lb,
        dp_ub,
    ) = get_bounds_by_combining_datasets_with_and_without_sensatt(
        df_metrics_w_sensatt=df_0,
        df_metrics_wo_sensatt=df_1,
        senstive_att_surrogates=senstive_att_surrogates,
        alpha=alpha,
        bound_type=bound_type,
    )
    return accuracy_lb, accuracy_ub, dp_lb, dp_ub


def get_dp_fairness_metrics_with_surrogate_sensatts_across_multiple_yoto_models(
    model_dir,
    dataloader_w_sensatt,
    dataloader_wo_sensatt,
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
):
    all_results_w_sensatt = pd.DataFrame()
    all_results_wo_sensatt = pd.DataFrame()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Iterate over all saved models
    model_num = 0
    for filename in tqdm(os.listdir(model_dir)):
        # Check if this file corresponds to a model with the given parameters
        if final_epoch:
            regex1 = rf"model_fe(\d+)_input_dim_{input_dim}_max_epochs_{max_epochs}_n_samples_{n_samples}_lr_{lr}_hidden_layer_size_{hidden_layer_size}_film_hidden_size_{film_hidden_size}_n_layers_film_{n_layers_film}_seed_(\d+)_batch_size_{batch_size}_lambda_lb_1e-05_lambda_ub_5.0_per_batch_lambda_True_(?:square_dp|sq_fl)_{square_dp}.pth"
            regex2 = rf"model_input_dim_{input_dim}_max_epochs_{max_epochs}_n_samples_{n_samples}_lr_{lr}_hidden_layer_size_{hidden_layer_size}_film_hidden_size_{film_hidden_size}_n_layers_film_{n_layers_film}_seed_(\d+)_batch_size_{batch_size}_lambda_lb_1e-05_lambda_ub_5.0_per_batch_lambda_True_(?:square_dp|sq_fl)_{square_dp}(?:_fairness_dp)?.pth"
            # The following regex is only for the adult dataset
            regex3 = rf"model_input_dim_{input_dim}_max_epochs_{max_epochs}_n_samples_{n_samples}_lr_{lr}_hidden_layer_size_{hidden_layer_size}_film_hidden_size_{film_hidden_size}_n_layers_film_{n_layers_film}_seed_(\d+)_lambda_lb_1e-05_lambda_ub_5.0_per_batch_lambda_True.pth"
            regexes = np.array([regex1, regex2, regex3])
            for regex in regexes:
                if re.match(regex, filename):
                    break
        else:
            regex = rf"model_e{stoppage_epoch}_input_dim_{input_dim}_max_epochs_{max_epochs}_n_samples_{n_samples}_lr_{lr}_hidden_layer_size_{hidden_layer_size}_film_hidden_size_{film_hidden_size}_n_layers_film_{n_layers_film}_seed_(\d+)_batch_size_{batch_size}_lambda_lb_1e-05_lambda_ub_5.0_per_batch_lambda_True_(?:square_dp|sq_fl)_{square_dp}.pth"
        # regex = rf"model_input_dim_{input_dim}_max_epochs_{max_epochs}_n_samples_{n_samples}_lr_{lr}_hidden_layer_size_{hidden_layer_size}_film_hidden_size_{film_hidden_size}_n_layers_film_{n_layers_film}_seed_(\d+)_lambda_lb_1e-05_lambda_ub_5.0_per_batch_lambda_True.pth"
        match = re.search(regex, filename)
        if match is not None and model_num < 4:
            model_path = os.path.join(model_dir, filename)
            model = torch.load(model_path, map_location=device)
            model_results_w_sensatt = get_metrics_for_model_yoto_w_sensatt(
                model,
                dataloader_w_sensatt,
                lambda_reg_values,
            )
            (
                model_results_wo_sensatt,
                senstive_att_surrogates,
            ) = get_metrics_for_model_yoto_wo_sensatt(
                model,
                dataloader_wo_sensatt,
                lambda_reg_values,
            )
            all_results_w_sensatt = average_dataframes(
                all_results_w_sensatt, model_results_w_sensatt
            )
            all_results_wo_sensatt = average_dataframes(
                all_results_wo_sensatt, model_results_wo_sensatt
            )
            model_num += 1

    return all_results_w_sensatt, all_results_wo_sensatt, senstive_att_surrogates


def plot_CI(
    df_metrics_w_sensatt,
    df_metrics_wo_sensatt,
    senstive_att_surrogates,
    alpha,
    color,
    bound_type,
    fig,
    axs,
):
    (
        accuracy_lb,
        accuracy_ub,
        dp_lb,
        dp_ub,
    ) = get_bounds_by_combining_datasets_with_and_without_sensatt(
        df_metrics_w_sensatt,
        df_metrics_wo_sensatt,
        senstive_att_surrogates=senstive_att_surrogates,
        alpha=alpha,
        bound_type=bound_type,
    )

    accuracies_lb_sorted = list(accuracy_lb)
    accuracies_ub_sorted = list(accuracy_ub)
    dp_lb_sorted = list(dp_lb)
    dp_ub_sorted = list(dp_ub)
    accuracies_lb_sorted.sort()
    accuracies_ub_sorted.sort()
    dp_lb_sorted.sort()
    dp_ub_sorted.sort()

    # Plot the colored region
    accuracies_lb_sorted = np.concatenate(
        (np.linspace(0.5, accuracies_lb_sorted[0], 10), accuracies_lb_sorted)
    )
    accuracies_ub_sorted = np.concatenate(
        (np.linspace(0.5, accuracies_ub_sorted[0], 10), accuracies_ub_sorted)
    )

    dp_lb_sorted = np.concatenate(
        (
            np.linspace(dp_lb_sorted[0], dp_lb_sorted[0], 10),
            dp_lb_sorted,
        )
    )
    dp_ub_sorted = np.concatenate(
        (
            np.linspace(dp_ub_sorted[0], dp_ub_sorted[0], 10),
            dp_ub_sorted,
        )
    )
    accuracies_lb_sorted = np.concatenate((accuracies_lb_sorted, np.linspace(accuracies_lb_sorted[-1], accuracies_ub_sorted[-1], 10)))
    accuracies_ub_sorted = np.concatenate((accuracies_ub_sorted, np.linspace(accuracies_ub_sorted[-1], 1, 10)))
    dp_lb_sorted = np.concatenate((dp_lb_sorted, np.ones(10)*dp_lb_sorted[-1]))
    dp_ub_sorted = np.concatenate((dp_ub_sorted, np.linspace(dp_ub_sorted[-1], 1, 10)))


    x = np.concatenate(
        (accuracies_lb_sorted, accuracies_ub_sorted[::-1])
    )  # note the order: we go forward on x1, then backward on x2
    y = np.concatenate((dp_ub_sorted, dp_lb_sorted[::-1]))  # same here

    axs.fill(x, y, alpha=0.1, label=f"{bound_type.split('_')[0]} CIs", color=color)
    axs.plot(accuracies_lb_sorted, dp_ub_sorted, color=color)
    axs.plot(accuracies_ub_sorted, dp_lb_sorted, color=color)
