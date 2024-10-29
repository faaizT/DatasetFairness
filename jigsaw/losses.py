import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class EOLossSep:
    def __init__(self, fairness, train_df):
        self.fairness = fairness
        self.Ps1 = train_df["gender"].mean()
        self.Ps0 = 1 - self.Ps1
        self.Ps1_y1 = (train_df["gender"] * train_df["toxic"]).mean()
        self.Ps1_y0 = (train_df["gender"] * (1 - train_df["toxic"])).mean()
        self.Ps0_y1 = ((1 - train_df["gender"]) * train_df["toxic"]).mean()
        self.Ps0_y0 = ((1 - train_df["gender"]) * (1 - train_df["toxic"])).mean()

    def loss_function(self, model, batch, device, lambda_reg, square_eo):
        if self.fairness == "eo":
            return self.compute_eo_regularized_loss_and_acc_per_batch(
                model, batch, device, lambda_reg, square_eo
            )
        if self.fairness == "eop":
            return self.compute_eop_regularized_loss_and_acc_per_batch(
                model, batch, device, lambda_reg, square_eo
            )

    def compute_eo_regularized_loss_and_acc_per_batch(
        self, model, batch, device, lambda_reg, square_eo
    ):
        data, labels, sensitive_attr = batch
        data, labels, sensitive_attr = (
            data.to(device),
            labels.to(device),
            sensitive_attr.to(device),
        )
        outputs = model(data)
        bce_loss = nn.BCELoss()
        classification_loss = bce_loss(outputs, labels)

        # True positive rates for each group
        tpr0 = (
            outputs[(~sensitive_attr & labels.long()).reshape(-1).bool()].sum()
            / self.Ps0_y1
        )
        tpr1 = (
            outputs[(sensitive_attr & labels.long()).reshape(-1).bool()].sum()
            / self.Ps1_y1
        )

        # False positive rates for each group
        fpr0 = (
            outputs[(~sensitive_attr & ~labels.long()).reshape(-1).bool()].sum()
            / self.Ps0_y0
        )
        fpr1 = (
            outputs[(sensitive_attr & ~labels.long()).reshape(-1).bool()].sum()
            / self.Ps1_y0
        )
        if square_eo:
            eo_loss = torch.square(tpr0 - tpr1) + torch.square(fpr0 - fpr1)
        else:
            eo_loss = torch.abs(tpr0 - tpr1) + torch.abs(fpr0 - fpr1)
        eo_loss /= outputs.shape[0]

        combined_loss = classification_loss + lambda_reg * eo_loss
        acc = (
            (
                outputs.reshape(-1) * labels.reshape(-1)
                + ((1 - outputs).reshape(-1) * (1 - labels).reshape(-1))
            )
            .sum()
            .item()
        )
        return (eo_loss, combined_loss, acc)

    def compute_eop_regularized_loss_and_acc_per_batch(
        self, model, batch, device, lambda_reg, square_eo
    ):
        data, labels, sensitive_attr = batch
        data, labels, sensitive_attr = (
            data.to(device),
            labels.to(device),
            sensitive_attr.to(device),
        )
        outputs = model(data)
        bce_loss = nn.BCELoss()
        classification_loss = bce_loss(outputs, labels)

        # True positive rates for each group
        tpr0 = (
            outputs[(~sensitive_attr & labels.long()).reshape(-1).bool()].sum()
            / self.Ps0_y1
        )
        tpr1 = (
            outputs[(sensitive_attr & labels.long()).reshape(-1).bool()].sum()
            / self.Ps1_y1
        )

        if square_eo:
            eo_loss = torch.square(tpr0 - tpr1)
        else:
            eo_loss = torch.abs(tpr0 - tpr1)
        eo_loss /= outputs.shape[0]

        combined_loss = classification_loss + lambda_reg * eo_loss
        acc = (
            (
                outputs.reshape(-1) * labels.reshape(-1)
                + ((1 - outputs).reshape(-1) * (1 - labels).reshape(-1))
            )
            .sum()
            .item()
        )
        return (eo_loss, combined_loss, acc)


class EOLossYoto:
    def __init__(self, fairness, train_df):
        self.fairness = fairness
        self.Ps1 = train_df["gender"].mean()
        self.Ps0 = 1 - self.Ps1
        self.Ps1_y1 = (train_df["gender"] * train_df["toxic"]).mean()
        self.Ps1_y0 = (train_df["gender"] * (1 - train_df["toxic"])).mean()
        self.Ps0_y1 = ((1 - train_df["gender"]) * train_df["toxic"]).mean()
        self.Ps0_y0 = ((1 - train_df["gender"]) * (1 - train_df["toxic"])).mean()

    def loss_function(self, model, batch, device, lambda_lb, lambda_ub, square_eo):
        if self.fairness == "eo":
            return self.compute_eo_regularized_loss_and_acc_per_batch_single_lambda(
                model, batch, device, lambda_lb, lambda_ub, square_eo
            )
        if self.fairness == "eop":
            return self.compute_eop_regularized_loss_and_acc_per_batch_single_lambda(
                model, batch, device, lambda_lb, lambda_ub, square_eo
            )

    def compute_eo_regularized_loss_and_acc_per_batch_single_lambda(
        self, model, batch, device, lambda_lb, lambda_ub, square_eo
    ):
        data, labels, sensitive_attr = batch
        data, labels, sensitive_attr = (
            data.to(device),
            labels.to(device),
            sensitive_attr.to(device),
        )

        lambda_reg = (
            torch.exp(
                torch.FloatTensor(1).uniform_(np.log(lambda_lb), np.log(lambda_ub))
            )
            .reshape(-1, 1)
            .to(device)
        )
        outputs = model(data, lambda_reg.repeat(data.shape[0], 1))
        bce_loss = nn.BCELoss()
        classification_loss = bce_loss(outputs, labels)

        # True positive rates for each group
        tpr0 = (
            outputs[(~sensitive_attr & labels.long()).reshape(-1).bool()].sum()
            / self.Ps0_y1
        )
        tpr1 = (
            outputs[(sensitive_attr & labels.long()).reshape(-1).bool()].sum()
            / self.Ps1_y1
        )

        # False positive rates for each group
        fpr0 = (
            outputs[(~sensitive_attr & ~labels.long()).reshape(-1).bool()].sum()
            / self.Ps0_y0
        )
        fpr1 = (
            outputs[(sensitive_attr & ~labels.long()).reshape(-1).bool()].sum()
            / self.Ps1_y0
        )
        if square_eo:
            eo_loss = torch.square(tpr0 - tpr1) + torch.square(fpr0 - fpr1)
        else:
            eo_loss = torch.abs(tpr0 - tpr1) + torch.abs(fpr0 - fpr1)
        eo_loss /= outputs.shape[0]

        combined_loss = classification_loss + lambda_reg.reshape(-1) * eo_loss
        acc = (
            (
                outputs.reshape(-1) * labels.reshape(-1)
                + ((1 - outputs).reshape(-1) * (1 - labels).reshape(-1))
            )
            .sum()
            .item()
        )
        return (eo_loss, combined_loss, acc)

    def compute_eop_regularized_loss_and_acc_per_batch_single_lambda(
        self, model, batch, device, lambda_lb, lambda_ub, square_eo
    ):
        data, labels, sensitive_attr = batch
        data, labels, sensitive_attr = (
            data.to(device),
            labels.to(device),
            sensitive_attr.to(device),
        )

        lambda_reg = (
            torch.exp(
                torch.FloatTensor(1).uniform_(np.log(lambda_lb), np.log(lambda_ub))
            )
            .reshape(-1, 1)
            .to(device)
        )
        outputs = model(data, lambda_reg.repeat(data.shape[0], 1))
        bce_loss = nn.BCELoss()
        classification_loss = bce_loss(outputs, labels)

        # True positive rates for each group
        tpr0 = (
            outputs[(~sensitive_attr & labels.long()).reshape(-1).bool()].sum()
            / self.Ps0_y1
        )
        tpr1 = (
            outputs[(sensitive_attr & labels.long()).reshape(-1).bool()].sum()
            / self.Ps1_y1
        )

        if square_eo:
            eo_loss = torch.square(tpr0 - tpr1)
        else:
            eo_loss = torch.abs(tpr0 - tpr1)
        eo_loss /= outputs.shape[0]

        combined_loss = classification_loss + lambda_reg.reshape(-1) * eo_loss
        acc = (
            (
                outputs.reshape(-1) * labels.reshape(-1)
                + ((1 - outputs).reshape(-1) * (1 - labels).reshape(-1))
            )
            .sum()
            .item()
        )
        return (eo_loss, combined_loss, acc)
