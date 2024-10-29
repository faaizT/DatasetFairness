import json
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
import os
from jigsaw.model import get_jigsaw_model_with_optimiser
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt

from utils.dataset_utils import (
    CustomDataset,
    create_dataset,
    get_data_loaders_with_validation,
)
from utils.helper_functions import create_tag, get_fairness_bounds_separate, str2bool
from utils.models import CelebAClassifier, Classifier

import argparse
import wandb

# Login into your wandb account
wandb.login()


def load_default_configs(args, ignore_keys=["model_dir"]):
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
        default=512,
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
        default=1,
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
        default=4,
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
        "--load_default_config",
        type=str2bool,
        default="False",
        help="Load default config from config.",
    )

    args = parser.parse_args()

    if args.load_default_config:
        args = load_default_configs(args)

    return args


# Checking if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BERT_Adversarial_Wrapper(nn.Module):
    def __init__(self, model_path):
        super(BERT_Adversarial_Wrapper, self).__init__()

        model = torch.load(model_path, map_location=device).to(device)

        # Initialize BERT base model
        self.bert = model.bert

        # Classification head
        self.classifier = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, data):
        input_ids, attention_mask = data[:, :, 0], data[:, :, 1]
        # Forward pass through BERT base
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = outputs.last_hidden_state[:, 0, :]  # Take the [CLS] representation

        # Classification head
        logits = self.classifier(x)

        return torch.sigmoid(logits), x


def get_validation_loss(models, val_loader, device, loss_function, lambda_reg):
    val_loss = 0.0
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            fairness_loss, combined_loss, acc = loss_function(
                models, batch, device, lambda_reg
            )
            if fairness_loss is None:
                continue
            val_loss += combined_loss.item()

    avg_val_loss = val_loss / len(val_loader)
    return avg_val_loss


class Adversary(nn.Module):
    def __init__(self, classifier_hidden_dim, fairness, identity_labels=1):
        super(Adversary, self).__init__()
        self.fairness = fairness
        if self.fairness.startswith("eo"):
            classifier_hidden_dim += 1
        self.a1 = nn.Linear(classifier_hidden_dim, 32)
        self.a2 = nn.Linear(32, identity_labels)
        nn.init.xavier_normal_(self.a1.weight)

    def forward(self, input_ids, true_label):
        if self.fairness.startswith("eo"):
            input_ids = torch.cat([input_ids, true_label], axis=1)
        # Adversary
        adversary = F.relu(self.a1(input_ids))
        adversary_output = torch.sigmoid(self.a2(adversary))

        return adversary_output


class ClassifierWrapper(nn.Module):
    def __init__(self, classifier_with_multiple_outputs):
        super(ClassifierWrapper, self).__init__()
        self.classifier_with_multiple_outputs = classifier_with_multiple_outputs

    def forward(self, x):
        probs, _ = self.classifier_with_multiple_outputs(x)
        return probs


class ClassifierWithIntermediateOutput(nn.Module):
    def __init__(self, input_dim, hidden_layer_size=64):
        super(ClassifierWithIntermediateOutput, self).__init__()
        self.hidden_layer_size = hidden_layer_size

        # Define the first linear layer
        self.fc1 = nn.Linear(input_dim, hidden_layer_size)

        # Define the activation function
        self.relu = nn.ReLU()

        # Define the second linear layer
        self.fc2 = nn.Linear(hidden_layer_size, hidden_layer_size)

        # Define the output layer
        self.fc_out = nn.Linear(hidden_layer_size, 1)

        # Define the final activation function
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # First layer
        x1 = self.fc1(x)
        x1_activated = self.relu(x1)

        # Second layer
        x2 = self.fc2(x1_activated)
        x2_activated = self.relu(x2)

        # Output layer
        out = self.fc_out(x2_activated)
        out_activated = self.sigmoid(out)

        return out_activated, x2_activated


# Sample pretraining functions based on the logic provided earlier


def pretrain_classifier(classifier, optimizer_clf, train_loader, epochs, device):
    """
    Pretrain the classifier model
    """
    print("Pretraining the classifier...")
    loss_criterion = nn.BCELoss()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for i, data in enumerate(train_loader):
            inputs, labels, sensitive_attrs = data
            inputs, labels, sensitive_attrs = (
                inputs.to(device),
                labels.to(device),
                sensitive_attrs.to(device),
            )

            optimizer_clf.zero_grad()

            # classifier_output, _ = classifier(inputs)
            classifier_output = classifier(inputs)
            classifier_loss = loss_criterion(classifier_output, labels)  # compute loss
            classifier_loss.backward()  # backpropagation
            optimizer_clf.step()

            epoch_loss += classifier_loss.item()

        print(f"Epoch {epoch + 1}, Loss: {epoch_loss / len(train_loader)}")
    return classifier


def train_adversary(adversary, classifier, optimizer_adv, train_loader, epochs, device):
    """
    Train the adversary model
    """
    print("Training the adversary...")
    loss_criterion = nn.BCELoss()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for i, data in enumerate(train_loader):
            inputs, labels, sensitive_attrs = data
            inputs, labels, sensitive_attrs = (
                inputs.to(device),
                labels.to(device),
                sensitive_attrs.float().to(device),
            )

            optimizer_adv.zero_grad()

            _, classifier_prev_output = classifier(inputs)
            # classifier_output = classifier(inputs)
            adversary_output = adversary(classifier_prev_output, labels)
            adversary_loss = loss_criterion(
                adversary_output, sensitive_attrs
            )  # compute loss
            adversary_loss.backward()  # backpropagation
            optimizer_adv.step()

            epoch_loss += adversary_loss.item()
            print(f"i {i}, Loss: {epoch_loss / (i+1)}")
        print(f"Epoch {epoch + 1}, Loss: {epoch_loss / len(train_loader)}")
    return adversary


def adversarial_loss_function(models, batch, device, lambda_reg):
    """
    Compute the adversarial loss for training the classifier.

    Parameters:
    - models: Tuple containing the classifier and adversary models
    - batch: Data batch containing inputs and labels
    - device: Device to run the computations on
    - lambda_reg: Regularization parameter (lambda)

    Returns:
    - adversary_loss: Loss for the adversary
    - combined_loss: Combined loss for the classifier
    - acc: Classifier accuracy for the current batch (placeholder, can be implemented)
    """
    classifier, adversary = models  # Unpack the classifier and adversary models

    # Get the inputs and labels from the batch
    data, labels, sensitive_attr = batch
    data = data.to(device)
    labels = labels.to(device)
    sensitive_attr = sensitive_attr.float().to(device)

    # optimizer_clf.zero_grad()  # Zero out any existing gradients

    # Forward pass through the classifier
    classifier_output, classifier_prev_output = classifier(data)
    # classifier_output = classifier(data)

    # Compute the classifier loss
    loss_criterion = nn.BCELoss()
    classifier_loss = loss_criterion(classifier_output, labels)

    # Forward pass through the adversary using the classifier's output
    adversary_output = adversary(classifier_prev_output, labels)
    # adversary_output = adversary(classifier_output, labels)

    # Compute the adversary loss
    adversary_loss = loss_criterion(adversary_output, sensitive_attr)

    # Compute the combined loss for the classifier
    combined_loss = classifier_loss - lambda_reg * adversary_loss

    # Placeholder for accuracy (can be implemented)
    acc = (
        (
            classifier_output.reshape(-1) * (labels).reshape(-1)
            + (1 - classifier_output).reshape(-1) * (1 - labels).reshape(-1)
        )
        .sum()
        .item()
    )

    return adversary_loss, combined_loss, acc


def main(args):
    # Initialize a new wandb run
    run = wandb.init(project="adversarial-fairness")

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
    config.batch_size = args.batch_size
    config.fairness = args.fairness
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    tag = create_tag(
        args,
        ignore_args=[
            "output_dir",
            "model_dir",
            "load_default_config",
        ],
    )

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
    )

    if config.input_dim == 512:
        model, optimizer_clf = get_jigsaw_model_with_optimiser(device, yoto=False)
    else:
        if config.input_dim == 64 * 64 * 3:
            model = CelebAClassifier(hidden_layer_size=config.hidden_layer_size).to(
                device
            )
        else:
            # model = ClassifierWithIntermediateOutput(
            #     config.input_dim, hidden_layer_size=config.hidden_layer_size
            # ).to(device)
            model = Classifier(
                config.input_dim, hidden_layer_size=config.hidden_layer_size
            ).to(device)
        optimizer_clf = Adam(model.parameters(), lr=0.001, weight_decay=config.wd)

    # adversary = Adversary(
    #     classifier_hidden_dim=config.hidden_layer_size, fairness=config.fairness
    # ).to(device)
    adversary = Adversary(classifier_hidden_dim=768, fairness=config.fairness).to(
        device
    )
    optimizer_adv = Adam(adversary.parameters(), lr=0.001, weight_decay=0.0)

    if config.input_dim in [9, 102]:
        print("Pretraining the classifier...")
        for param in adversary.parameters():
            param.requires_grad = False
        model = pretrain_classifier(
            model, optimizer_clf, train_loader, epochs=3, device=device
        )
        for param in adversary.parameters():
            param.requires_grad = True
    elif config.input_dim == 512:
        model_found = False
        for filename in os.listdir("../pretrained_models/"):
            if (
                f"input_dim_{config.input_dim}" in filename
                and f"fairness_{config.fairness}" in filename
            ):
                model = BERT_Adversarial_Wrapper(f"../pretrained_models/{filename}").to(
                    device
                )
    else:
        model_found = False
        for filename in os.listdir("../pretrained_models/"):
            if (
                f"input_dim_{config.input_dim}" in filename
                and f"fairness_{config.fairness}" in filename
            ):
                model = torch.load(
                    f"../pretrained_models/{filename}", map_location=device
                ).to(device)
                model_found = True
        if not model_found:
            raise ("Value error: pretrained model not found")

    # Pretrain the adversary
    loss_function = adversarial_loss_function

    for param in model.parameters():
        param.requires_grad = False

    adversary = train_adversary(
        adversary, model, optimizer_adv, train_loader, epochs=3, device=device
    )
    for param in model.parameters():
        param.requires_grad = True

    # Lists for storing metrics
    train_losses = []
    test_losses = []
    fairness_losses = []
    validation_losses = []
    test_accuracies = []
    train_accuracies = []

    patience = 50  # set your patience level
    counter = 0
    best_val_loss = float("inf")

    # Create directory for saving models
    if not os.path.exists(config.model_dir):
        os.makedirs(config.model_dir)
    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)

    model.train()
    adversary.train()
    for epoch in range(config.max_epochs):
        for param in adversary.parameters():
            param.requires_grad = True
        for param in model.parameters():
            param.requires_grad = False
        adversary = train_adversary(
            adversary, model, optimizer_adv, train_loader, epochs=1, device=device
        )
        for param in model.parameters():
            param.requires_grad = True
        for param in adversary.parameters():
            param.requires_grad = False

        # Training
        train_loss = 0.0
        train_accuracy = 0.0
        fairness_loss_sum = 0.0
        for i, batch in enumerate(train_loader):
            optimizer_clf.zero_grad()
            fairness_loss, combined_loss, acc = loss_function(
                (model, adversary),
                batch,
                device,
                config.lambda_reg,
            )
            if fairness_loss is None:
                print("skipping iteration as both classes not in the minibatch")
                continue
            train_accuracy += acc
            combined_loss.backward()
            optimizer_clf.step()
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
                    (model, adversary),
                    batch,
                    device,
                    config.lambda_reg,
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
            f"Train loss: {avg_train_loss}, Test loss: {avg_test_loss}, Fairness loss: {avg_fairness_loss}, Test acc: {avg_test_accuracy}"
        )

        # Compute validation loss
        val_loss = get_validation_loss(
            (model, adversary),
            val_loader,
            device,
            loss_function,
            config.lambda_reg,
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
            # wrapped_classifier = ClassifierWrapper(model)
    wrapped_classifier = ClassifierWrapper(model)
    torch.save(wrapped_classifier, f"{config.model_dir}/model_{tag}.pth")
    torch.save(
        optimizer_clf.state_dict(),
        f"{config.model_dir}/optimizer_clf_{tag}.pth",
    )
    torch.save(adversary, f"{config.model_dir}/adv_{tag}.pth")
    torch.save(
        optimizer_adv.state_dict(),
        f"{config.model_dir}/optimizer_adv_{tag}.pth",
    )
    if config.fairness.startswith("eo"):
        accuracy, _, _, unfairness, _, _ = get_fairness_bounds_separate(
            model_filepath=f"{config.model_dir}/model_{tag}.pth",
            dataloader=test_loader,
            device=device,
            alpha=0.05,
            output_CIs=False,
            fairness="eo",
        )
        df1 = pd.DataFrame(
            {
                "input_dim": [config.input_dim],
                "Accuracy": [accuracy],
                "Unfairness": [unfairness],
                "lambda_reg": [config.lambda_reg],
                "seed": [config.seed],
            }
        )
        df1.to_csv(
            f"{config.output_dir}/results_{config.input_dim}_eo_lambda_reg_{config.lambda_reg}_seed_{config.seed}.csv",
            index=False,
        )
        accuracy, _, _, unfairness, _, _ = get_fairness_bounds_separate(
            model_filepath=f"{config.model_dir}/model_{tag}.pth",
            dataloader=test_loader,
            device=device,
            alpha=0.05,
            output_CIs=False,
            fairness="eop",
        )
        df2 = pd.DataFrame(
            {
                "input_dim": [config.input_dim],
                "Accuracy": [accuracy],
                "Unfairness": [unfairness],
                "lambda_reg": [config.lambda_reg],
                "seed": [config.seed],
            }
        )
        df2.to_csv(
            f"{config.output_dir}/results_{config.input_dim}_eop_lambda_reg_{config.lambda_reg}_seed_{config.seed}.csv",
            index=False,
        )
    elif config.fairness == "dp":
        accuracy, _, _, unfairness, _, _ = get_fairness_bounds_separate(
            model_filepath=f"{config.model_dir}/model_{tag}.pth",
            dataloader=test_loader,
            device=device,
            alpha=0.05,
            output_CIs=False,
            fairness="dp",
        )
        df = pd.DataFrame(
            {
                "input_dim": [config.input_dim],
                "Accuracy": [accuracy],
                "Unfairness": [unfairness],
                "lambda_reg": [config.lambda_reg],
                "seed": [config.seed],
            }
        )
        df.to_csv(
            f"{config.output_dir}/results_{config.input_dim}_dp_lambda_reg_{config.lambda_reg}_seed_{config.seed}.csv",
            index=False,
        )

    run.finish()


if __name__ == "__main__":
    args = set_args()
    main(args)
