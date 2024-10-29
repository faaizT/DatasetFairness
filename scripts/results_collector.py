import json
import pandas as pd
import torch
import os
import re

from utils.dataset_utils import (
    create_dataset_male_female_synth_1d,
    get_data_loaders_without_validation,
)
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from utils.helper_functions import (
    delta_senstive_analysis,
    plot_confidence_intervals,
    plot_metrics_for_all_models_yoto,
    animate_decision_boundary,
    get_fairness_bounds_separate,
    plot_labels,
)


def extract_results_as_df(
    results_directory,
    test_loader,
    input_dim,
    max_epochs,
    n_samples,
    lr,
    wd,
    hidden_layer_size,
    batch_size,
    sq_fl,
    fairness="dp",
):
    # Checking if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initializing the list to store the data
    data_list = []

    # Loop over all the models in the directory
    for filename in os.listdir(results_directory):
        if filename.endswith(".pth"):
            # Extract parameters from filename using regex
            # The following regex is for adult dataset models
            regex1 = rf"model_input_dim_{input_dim}_max_epochs_{max_epochs}_n_samples_{n_samples}_lambda_reg_([\d.]+)_lr_{lr}_wd_{wd}_hidden_layer_size_{hidden_layer_size}_seed_(\d+).pth"
            regex2 = rf"model_input_dim_{input_dim}_max_epochs_{max_epochs}_n_samples_{n_samples}_lambda_reg_([\d.]+)_lr_{lr}_wd_{wd}_hidden_layer_size_{hidden_layer_size}_batch_size_{batch_size}(?:_fairness_{fairness})?_seed_(\d+)_(?:square_dp|sq_fl)_{sq_fl}.pth"
            regex = regex1 if re.search(regex1, filename) else regex2
            match = re.search(regex, filename)
            if match is not None:
                lambda_reg, seed = map(float, match.groups())

                # Get model statistics
                model_filepath = os.path.join(results_directory, filename)
                (
                    accuracy,
                    accuracy_lb,
                    accuracy_ub,
                    mean_fairness,
                    fairness_lb,
                    fairness_ub,
                ) = get_fairness_bounds_separate(
                    model_filepath,
                    test_loader,
                    device,
                    alpha=0.05,
                    output_CIs=False,
                    fairness=fairness,
                )

                # Append the extracted data to the list
                data_list.append(
                    [
                        lambda_reg,
                        seed,
                        max_epochs,
                        n_samples,
                        input_dim,
                        accuracy,
                        mean_fairness,
                        accuracy_lb,
                        fairness_lb,
                        fairness_ub,
                    ]
                )

    # Convert the list into a DataFrame
    df = pd.DataFrame(
        data_list,
        columns=[
            "lambda",
            "seed",
            "max_epochs",
            "num_samples",
            "input_dim",
            "accuracy",
            "fairness_metric",
            "accuracy_lb",
            "fairness_metric_lb",
            "fairness_metric_ub",
        ],
    )

    # Saving the dataframe as a CSV file
    return df


def plot_metrics_vs_lambda_from_saved_results(df, fairness, fig=None, ax=None):
    # Create a new figure with three subplots
    if len(df) == 0:
        return
    if fig is None:
        fig, ax = plt.subplots(3, 1, figsize=(12, 18))

    if len(ax) == 3:
        # ax[0].set_title("Demographic Parity vs Lambda Regularization", fontsize=18)
        ax[0].set_xlabel("Lambda Regularization", fontsize=18)
        ax[0].set_ylabel(plot_labels[fairness], fontsize=18)

        # ax[1].set_title("Accuracy vs Lambda Regularization", fontsize=18)
        ax[1].set_xlabel("Lambda Regularization", fontsize=18)
        ax[1].set_ylabel("Accuracy", fontsize=18)

        # ax[2].set_title("Demographic Parity vs Accuracy", fontsize=18)
        ax[2].set_ylabel(plot_labels[fairness], fontsize=18)
        ax[2].set_xlabel("Accuracy", fontsize=18)

        # Demographic Parity vs Lambda Regularization
        sns.lineplot(
            data=df,
            x="lambda",
            y="fairness_metric",
            ax=ax[0],
            label="separate",
            estimator="median",
        )

        # Accuracy vs Lambda Regularization
        sns.lineplot(
            data=df,
            x="lambda",
            y="accuracy",
            ax=ax[1],
            label="separate",
            estimator="median",
        )

        # Accuracy vs Demographic Parity
        sns.scatterplot(
            data=df,
            x="accuracy",
            y="fairness_metric",
            ax=ax[2],
            label="separate",
            s=120,
            color="orange",
        )
    else:
        ax[0].set_ylabel(plot_labels[fairness], fontsize=18)
        ax[0].set_xlabel("Accuracy", fontsize=18)
        sns.scatterplot(
            data=df,
            x="accuracy",
            y="fairness_metric",
            ax=ax[0],
            label="separate",
            s=120,
            color="orange",
        )
    return fig, ax


def plot_different_cis(
    model_results_for_CIs, df_sep, fig, axs, delta=None, fairness="dp", diff_method=True
):
    if delta is None:
        delta = delta_senstive_analysis(model_results_for_CIs, df_sep)

    diff_str = "_diff" if diff_method else ""
    plot_confidence_intervals(
        model_results_for_CIs,
        color="skyblue",
        bound_type=f"hoeffdings{diff_str}",
        fig=fig,
        axs=axs,
        delta=delta,
        fairness=fairness,
    )
    plot_confidence_intervals(
        model_results_for_CIs,
        color="purple",
        bound_type=f"asymptotic{diff_str}",
        fig=fig,
        axs=axs,
        delta=delta,
        fairness=fairness,
    )
    plot_confidence_intervals(
        model_results_for_CIs,
        color="pink",
        bound_type=f"bernstein{diff_str}",
        fig=fig,
        axs=axs,
        delta=delta,
        fairness=fairness,
    )
    plot_confidence_intervals(
        model_results_for_CIs,
        color="red",
        bound_type=f"bootstrap{diff_str}",
        fig=fig,
        axs=axs,
        delta=delta,
        fairness=fairness,
    )
    return delta


def load_baseline_results(input_dim, fairness, axs):
    if input_dim == 9:
        filename = "compas"
    elif input_dim == 102:
        filename = "adult"
    elif input_dim == 512:
        filename = "jigsaw"
    elif input_dim == 12288:
        filename = "celeba"
    if os.path.exists(f"../FairSurrogates/{filename}_logistic_{fairness}_test.csv"):
        df = pd.read_csv(f"../FairSurrogates/{filename}_logistic_{fairness}_test.csv")
        df.sort_values(by="Accuracy", inplace=True)
        sns.scatterplot(
            data=df,
            x="Accuracy",
            y="Unfairness",
            ax=axs,
            label="logsig",
            color="purple",
            marker="s",
            s=80,
        )
    if os.path.exists(f"../FairSurrogates/{filename}_linear_{fairness}_test.csv"):
        df = pd.read_csv(f"../FairSurrogates/{filename}_linear_{fairness}_test.csv")
        df.sort_values(by="Accuracy", inplace=True)
        sns.scatterplot(
            data=df,
            x="Accuracy",
            y="Unfairness",
            ax=axs,
            label="linear",
            color="green",
            marker="d",
            s=80,
        )
    if os.path.exists(f"./result_files/{filename}_reductions_{fairness}.csv"):
        df = pd.read_csv(f"./result_files/{filename}_reductions_{fairness}.csv")
        sns.scatterplot(
            data=df,
            x="Acc",
            y="DP",
            ax=axs,
            label="reductions",
            color="midnightblue",
            marker="p",
            s=80,
        )
    if os.path.exists(f"./result_files/input_dim_{input_dim}_{fairness}_adversary.csv"):
        df = pd.read_csv(
            f"./result_files/input_dim_{input_dim}_{fairness}_adversary.csv"
        )
        df.sort_values(by="Accuracy", inplace=True)
        sns.scatterplot(
            data=df,
            x="Accuracy",
            y="Unfairness",
            ax=axs,
            label="adversary",
            color="maroon",
            marker="*",
            s=320,
        )


def generate_plots(
    input_dim,
    n_samples,
    n_test_samples,
    results_dir,
    yoto_model_dir,
    max_epochs_yoto,
    yoto_stopping_epoch,
    final_epoch_yoto,
    max_epochs_sep,
    lr_separate,
    wd_separate,
    hidden_layer_size_separate,
    lr_yoto,
    hidden_layer_size_yoto,
    film_hidden_size,
    n_layers_film,
    batch_size,
    square_dp,
    create_animation=False,
    fairness="dp",
    xlims=None,
    ylims=None,
):
    yoto_model_tag = f"eps_{yoto_stopping_epoch}_me_{max_epochs_yoto}_lr_yoto_{lr_yoto}_hls_yoto_{hidden_layer_size_yoto}_fhs_{film_hidden_size}_nlfilm_{n_layers_film}"
    separate_model_tag = f"inpdim_{input_dim}_maxepochs_{max_epochs_sep}_n_samples_{n_samples}_lr_{lr_separate}_hls_{hidden_layer_size_separate}"
    global_tag = f"batch_size_{batch_size}_square_dp_{square_dp}_testsize_{n_test_samples}_f{fairness}"
    filename = f"./result_files_final/plots_{separate_model_tag}_{yoto_model_tag}_{global_tag}.pdf"
    if os.path.exists(filename):
        print(f"{filename} already exists. Returning")
        return
    (
        cal_data_loader,
        cal_dataset,
    ) = get_data_loaders_without_validation(
        input_dim=input_dim,
        n_samples=int(n_test_samples // 0.60),
        split=False,
        for_eo=fairness.startswith("eo"),
        fairness=fairness,
    )

    # Results for separately trained NNs
    if os.path.exists(f"./result_files/results_sep_{input_dim}_{fairness}.csv"):
        df_sep = pd.read_csv(f"./result_files/results_sep_{input_dim}_{fairness}.csv")
    else:
        df_sep = extract_results_as_df(
            results_dir,
            cal_data_loader,
            input_dim=input_dim,
            max_epochs=max_epochs_sep,
            n_samples=n_samples,
            lr=lr_separate,
            wd=wd_separate,
            hidden_layer_size=hidden_layer_size_separate,
            batch_size=batch_size,
            sq_fl=square_dp,
            fairness=fairness,
        )
        df_sep.to_csv(
            f"./result_files/results_sep_{input_dim}_{fairness}.csv", index=False
        )
    if len(df_sep) == 0:
        print("No separate models found")
        return "No separate models found"

    # YOTO results
    lambda_reg_values = list(np.linspace(0.0, 10, num=21))
    if os.path.exists(
        f"./result_files/results_yoto_cis_{input_dim}_{fairness}_{n_test_samples}.pkl"
    ):
        model_results_for_CIs = pd.read_pickle(
            f"./result_files/results_yoto_cis_{input_dim}_{fairness}_{n_test_samples}.pkl"
        )
    else:
        _, _, model_results_for_CIs = plot_metrics_for_all_models_yoto(
            yoto_model_dir,
            cal_data_loader,
            lambda_reg_values,
            input_dim,
            max_epochs_yoto,
            yoto_stopping_epoch,
            final_epoch_yoto,
            n_samples,
            lr_yoto,
            hidden_layer_size_yoto,
            film_hidden_size,
            n_layers_film,
            batch_size,
            square_dp,
            accumulate=True,
            fairness=fairness,
        )
        if len(model_results_for_CIs) > 0:
            model_results_for_CIs.to_pickle(
                f"./result_files/results_yoto_cis_{input_dim}_{fairness}_{n_test_samples}.pkl"
            )
    if model_results_for_CIs is None or len(model_results_for_CIs) == 0:
        print("No yoto models found")
        return "No yoto models found"

    fig, axs = plt.subplots(1, 1, figsize=(7, 4.5))
    axs = [axs]

    tradeoff_ax = axs[-1]
    delta = plot_different_cis(
        model_results_for_CIs.sort_values(by="Accuracy"),
        df_sep.sample(frac=1).head(n=2),
        fig,
        tradeoff_ax,
        fairness=fairness,
        delta=0,
    )

    if len(axs) == 3:
        axs[0].set_title(
            f"{plot_labels[fairness]} vs Lambda Regularization", fontsize=18
        )
        axs[1].set_title("Accuracy vs Lambda Regularization", fontsize=18)
        axs[2].set_title(
            f"{plot_labels[fairness]} vs Accuracy. Test data size:{n_test_samples}, $\Delta={round(delta, 2)}$",
            fontsize=18,
        )
    fig, axs, _ = plot_metrics_for_all_models_yoto(
        yoto_model_dir,
        cal_data_loader,
        lambda_reg_values,
        input_dim,
        max_epochs_yoto,
        yoto_stopping_epoch,
        final_epoch_yoto,
        n_samples,
        lr_yoto,
        hidden_layer_size_yoto,
        film_hidden_size,
        n_layers_film,
        batch_size,
        square_dp,
        accumulate=False,
        fairness=fairness,
        yoto_results=model_results_for_CIs,
        fig=fig,
        axs=axs,
    )

    fig, axs = plot_metrics_vs_lambda_from_saved_results(df_sep, fairness, fig, axs)
    load_baseline_results(input_dim, fairness, tradeoff_ax)

    if input_dim == 1:
        create_dataset_male_female_synth_1d(
            n_samples=50000,
            plot_tradeoff_=True,
            ax=tradeoff_ax,
            fairness_metric=fairness,
        )

    # save figure
    for i in range(len(axs)):
        axs[i].legend(fontsize=18)
        axs[i].grid(True)
        axs[i].tick_params(axis="both", which="major", labelsize=18)
    if xlims is not None:
        tradeoff_ax.set_xlim(xlims)
    if ylims is not None:
        tradeoff_ax.set_ylim(ylims)
    plt.legend().remove()

    from matplotlib.ticker import FormatStrFormatter

    plt.gca().yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    plt.ylabel(plot_labels[fairness], fontsize=18)
    plt.xlabel("Accuracy", fontsize=18)
    plt.tight_layout()
    fig.savefig(f"{filename}")
    print("saved the file at", filename)
    plt.close()

    # create and save animation
    if create_animation:
        model_yoto = torch.load(f"{yoto_model_dir}/model_{yoto_model_tag}.pth")
        ani = animate_decision_boundary(model_yoto, cal_data_loader, lambda_reg_values)
        ani.save(f"./animation_{yoto_model_tag}.gif", writer="pillow")


if __name__ == "__main__":
    input_dim = 102
    fairness = "dp"
    legreg = False
    # Load default YOTO parameters
    suffix = "_legreg" if legreg else ""
    with open(
        f"./configs/config_yoto_input_dim_{input_dim}_{fairness[:2]}{suffix}.json",
        "r",
    ) as f:
        config_yoto = json.load(f)
    n_samples = config_yoto["n_samples"]
    max_epochs_yoto = config_yoto["max_epochs"]
    lr_yoto = config_yoto["lr"]
    hidden_layer_size_yoto = config_yoto["hidden_layer_size"]
    n_layers_film = config_yoto["n_layers_film"]
    film_hidden_size = config_yoto["film_hidden_size"]
    yoto_model_dir = (
        config_yoto["model_dir_eop"] if fairness == "eop" else config_yoto["model_dir"]
    )

    # Load default sep_model parameters
    with open(
        f"./configs/config_sep_input_dim_{input_dim}_{fairness[:2]}{suffix}.json",
        "r",
    ) as f:
        config_sep = json.load(f)
    max_epochs_sep = config_sep["max_epochs"]
    lr_separate = config_sep["lr"]
    wd_separate = config_sep["wd"]
    hidden_layer_size_separate = config_sep["hidden_layer_size"]
    results_dir = (
        config_sep["model_dir_eop"] if fairness == "eop" else config_sep["model_dir"]
    )

    n_test_samples = 2000
    create_animation = False
    batch_size = 32
    square_dp = False
    yoto_stoppage_epoch = 50
    final_epoch_yoto = True

    if "ax_lims" in config_sep:
        xlims = config_sep["ax_lims"]
    else:
        xlims = None
    if fairness == "eop" and input_dim == 12288:
        xlims = [0.88, 0.925]

    ylims = None
    if fairness in ["dp", "eo"] and "ax_lims_y" in config_sep:
        ylims = config_sep["ax_lims_y"]
    elif "ax_lims_y_eop" in config_sep:
        ylims = config_sep["ax_lims_y_eop"]

    return_message = generate_plots(
        input_dim,
        n_samples,
        n_test_samples,
        results_dir,
        yoto_model_dir,
        max_epochs_yoto,
        yoto_stoppage_epoch,
        final_epoch_yoto,
        max_epochs_sep,
        lr_separate,
        wd_separate,
        hidden_layer_size_separate,
        lr_yoto,
        hidden_layer_size_yoto,
        film_hidden_size,
        n_layers_film,
        batch_size,
        square_dp,
        create_animation,
        fairness,
        xlims=xlims,
        ylims=ylims,
    )
