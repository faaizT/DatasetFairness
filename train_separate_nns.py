import json
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
import os
from jigsaw.model import get_jigsaw_model_with_optimiser

import matplotlib.pyplot as plt

from utils.dataset_utils import (
    CustomDataset,
    create_dataset,
    get_data_loaders_with_validation,
)
from utils.helper_functions import create_tag, str2bool
from utils.models import CelebAClassifier, Classifier

import argparse
import wandb

from utils.sep_losses import get_loss_function

# Login into your wandb account
# wandb.login()


def load_default_configs(
    args,
    ignore_keys=[
        "model_dir",
        "model_dir_eop",
        "model_dir_eo",
        "ax_lims",
        "ax_lims_y",
        "ax_lims_y_eop",
    ],
):
    fairness = "eo" if args.fairness.startswith("eo") else args.fairness
    with open(
        f"./configs/config_sep_input_dim_{args.input_dim}_{fairness}.json", "r"
    ) as f:
        config = json.load(f)
    args_dict = vars(args)
    for k in config:
        if k not in ignore_keys:
            args_dict[k] = config[k]
    return argparse.Namespace(**args_dict)


def set_args():
    parser = argparse.ArgumentParser(description="Arguments for training the model.")

    parser.add_argument(
        "--input_dim",
        type=int,
        default=9,
        help="Dimension of input features for the model.",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=3,
        help="Number of max epochs to train the model.",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=190000,
        help="Number of samples in the dataset.",
    )
    parser.add_argument(
        "--lambda_reg",
        type=float,
        default=0.0,
        help="Regularization hyperparameter for fairness loss.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate.",
    )
    parser.add_argument(
        "--wd",
        type=float,
        default=0.0,
        help="Weight decay.",
    )
    parser.add_argument(
        "--hidden_layer_size",
        type=int,
        default=1,
        help="Hidden layer size of classifier.",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../plots",
        help="Directory to save output files.",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="../models",
        help="Directory to save trained models.",
    )
    parser.add_argument(
        "--fairness",
        type=str,
        default="eop",
        help="Fairness metric to use. Must be one of ['dp', 'eo', 'eop'].",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed to use for random initialization."
    )
    parser.add_argument(
        "--sq_fl",
        type=str2bool,
        default="False",
        help="Use square fairness loss.",
    )
    parser.add_argument(
        "--fairsurrogates",
        type=str2bool,
        default="False",
        help="Use fair surrogates fairness loss.",
    )
    parser.add_argument(
        "--form",
        type=str,
        default="linear",
        help="Which fair surrogates fairness loss to use.",
    )
    parser.add_argument(
        "--load_default_config",
        type=str2bool,
        default="True",
        help="Load default config from config.",
    )

    args = parser.parse_args()

    if args.load_default_config:
        args = load_default_configs(args)

    return args


def get_validation_loss(model, val_loader, device, loss_function, lambda_reg, sq_fl):
    val_loss = 0.0
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            fairness_loss, combined_loss, acc = loss_function(
                model, batch, device, lambda_reg, sq_fl
            )
            if fairness_loss is None:
                continue
            val_loss += combined_loss.item()

    avg_val_loss = val_loss / len(val_loader)
    return avg_val_loss


def main(args):
    # Initialize a new wandb run
    run = wandb.init(project="regularized-fairness-sep")

    # Define the config
    config = wandb.config
    config.input_dim = args.input_dim
    config.max_epochs = args.max_epochs
    config.n_samples = args.n_samples
    config.lambda_reg = args.lambda_reg
    config.output_dir = args.output_dir
    config.model_dir = args.model_dir
    config.seed = args.seed
    config.lr = args.lr
    config.hidden_layer_size = args.hidden_layer_size
    config.wd = args.wd
    config.sq_fl = args.sq_fl
    config.batch_size = args.batch_size
    config.fairness = args.fairness
    config.fairsurrogates = args.fairsurrogates
    config.form = args.form
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    tag = create_tag(
        args,
        ignore_args=[
            "output_dir",
            "model_dir",
            "load_default_config",
            "fairsurrogates",
            "form",
        ],
    )

    # Checking if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    (
        train_loader,
        val_loader,
        test_loader,
        train_dataset,
        val_dataset,
        test_dataset,
    ) = get_data_loaders_with_validation(
        config.input_dim,
        config.n_samples,
        config.batch_size,
        for_eo=config.fairness in ["eo", "eop"],
        fairness=config.fairness,
    )

    if config.input_dim == 512:
        model, optimizer = get_jigsaw_model_with_optimiser(device, yoto=False)
    else:
        if config.input_dim == 64 * 64 * 3:
            model = CelebAClassifier(hidden_layer_size=config.hidden_layer_size).to(
                device
            )
        else:
            model = Classifier(
                config.input_dim, hidden_layer_size=config.hidden_layer_size
            ).to(device)
        optimizer = Adam(model.parameters(), lr=config.lr, weight_decay=config.wd)

    loss_function = get_loss_function(
        config.input_dim,
        config.fairness,
        config.fairsurrogates,
        train_dataset if config.input_dim == 512 else None,
        config.form,
    )
    # Lists for storing metrics
    train_losses = []
    test_losses = []
    fairness_losses = []
    validation_losses = []
    test_accuracies = []
    train_accuracies = []

    patience = 10  # set your patience level
    counter = 0
    best_val_loss = float("inf")

    # Create directory for saving models
    if not os.path.exists(config.model_dir):
        os.makedirs(config.model_dir)

    for epoch in range(config.max_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_accuracy = 0.0
        fairness_loss_sum = 0.0
        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()
            fairness_loss, combined_loss, acc = loss_function(
                model,
                batch,
                device,
                config.lambda_reg,
                config.sq_fl,
            )
            if fairness_loss is None:
                print("skipping iteration as both classes not in the minibatch")
                continue
            train_accuracy += acc
            combined_loss.backward()
            optimizer.step()
            train_loss += combined_loss.item()
            fairness_loss_sum += fairness_loss.item()
            if i > 0 and i % 200 == 0:
                # Average metrics over all batches
                avg_train_accuracy = train_accuracy / i / args.batch_size
                avg_train_loss = train_loss / i / args.batch_size
                avg_fairness_loss = fairness_loss_sum / i
                metrics = {
                    "step": i,
                    "train_loss": avg_train_loss,
                    "fairness_loss": avg_fairness_loss,
                    "train_accuracy": avg_train_accuracy,
                }
                wandb.log(metrics)
                print(metrics)

        # Average metrics over all batches
        avg_train_accuracy = train_accuracy / len(train_dataset)
        avg_train_loss = train_loss / len(train_loader)
        avg_fairness_loss = fairness_loss_sum / len(train_loader)
        train_losses.append(avg_train_loss)
        fairness_losses.append(avg_fairness_loss)
        train_accuracies.append(avg_train_accuracy)

        # Testing
        model.eval()
        test_loss = 0.0
        test_accuracy = 0.0
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                (
                    fairness_loss,
                    combined_loss,
                    acc,
                ) = loss_function(
                    model,
                    batch,
                    device,
                    config.lambda_reg,
                    config.sq_fl,
                )
                if fairness_loss is None:
                    continue
                test_loss += combined_loss.item()
                test_accuracy += acc

        avg_test_accuracy = test_accuracy / len(test_dataset)
        avg_test_loss = test_loss / len(test_loader)
        test_losses.append(avg_test_loss)
        test_accuracies.append(avg_test_accuracy)

        # Print some info
        print(f"Epoch {epoch+1}/{config.max_epochs}")
        print(
            f"Train loss: {avg_train_loss}, Test loss: {avg_test_loss}, Fairness loss: {avg_fairness_loss}"
        )

        # Compute validation loss
        val_loss = get_validation_loss(
            model,
            val_loader,
            device,
            loss_function,
            config.lambda_reg,
            config.sq_fl,
        )
        validation_losses.append(val_loss)

        wandb.log(
            {
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "test_loss": avg_test_loss,
                "fairness_loss": avg_fairness_loss,
                "test_accuracy": avg_test_accuracy,
                "train_accuracy": avg_train_accuracy,
                "validation_loss": val_loss,
            }
        )

        # Check if validation loss didn't improve
        if val_loss >= best_val_loss:
            counter += 1
            print(f"EarlyStopping counter: {counter} out of {patience}")
            if counter >= patience:
                print("Early stopping")
                break
        else:
            best_val_loss = val_loss
            counter = 0  # reset counter if validation loss improved
            print("saving the model at:", f"{config.model_dir}/model_{tag}.pth")
            torch.save(model, f"{config.model_dir}/model_{tag}.pth")
            torch.save(
                optimizer.state_dict(), f"{config.model_dir}/optimizer_{tag}.pth"
            )

    # Plotting
    plt.figure(figsize=(18, 4))

    plt.subplot(1, 3, 1)
    plt.plot(np.arange(len(train_losses)), train_losses, label="Train Loss")
    plt.plot(np.arange(len(test_losses)), test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train and Test Loss over Time")
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(np.arange(len(train_accuracies)), train_accuracies, label="Train Accuracy")
    plt.plot(np.arange(len(test_accuracies)), test_accuracies, label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Train and Test Accuracy over Time")
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(np.arange(len(fairness_losses)), fairness_losses, label="Fairness Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Fairness Loss over Time")
    plt.legend()

    plt.tight_layout()
    plt.suptitle(f"$\lambda={config.lambda_reg}$", fontsize=14)

    # Adjust the layout to create space between subplots and main title
    plt.subplots_adjust(top=0.85)
    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)
    plt.savefig(f"{config.output_dir}/sep_training_curves_{tag}.png")
    print("Saved plots")

    run.finish()


if __name__ == "__main__":
    args = set_args()
    main(args)
