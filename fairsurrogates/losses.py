import torch
from torch import nn
import torch.nn.functional as F


class SurrogatesLoss:
    def __init__(self, form, fairness, train_df=None):
        self.form = form
        self.fairness = fairness
        if train_df is not None:
            self.Ps1 = train_df["gender"].mean()
            self.Ps0 = 1 - self.Ps1
            self.Ps1_y1 = (train_df["gender"] * train_df["toxic"]).mean()
            self.Ps1_y0 = (train_df["gender"] * (1 - train_df["toxic"])).mean()
            self.Ps0_y1 = ((1 - train_df["gender"]) * train_df["toxic"]).mean()
            self.Ps0_y0 = ((1 - train_df["gender"]) * (1 - train_df["toxic"])).mean()

    def loss_function(self, model, batch, device, lambda_reg, square_fl):
        if self.fairness == "dp":
            return self.compute_dp_regularized_loss_and_acc_per_batch(
                model, batch, device, lambda_reg
            )
        if self.fairness == "eo":
            return self.compute_eo_regularized_loss_and_acc_per_batch(
                model, batch, device, lambda_reg
            )
        if self.fairness == "eop":
            return self.compute_eop_regularized_loss_and_acc_per_batch(
                model, batch, device, lambda_reg
            )

    def compute_dp_regularized_loss_and_acc_per_batch(
        self, model, batch, device, lambda_reg
    ):
        data, labels, sensitive_attr = batch
        if (sensitive_attr == 0).sum() == 0 or (sensitive_attr == 1).sum() == 0:
            return None, None, None
        data, labels, sensitive_attr = (
            data.to(device),
            labels.to(device),
            sensitive_attr.to(device),
        )
        outputs = model(data)
        bce_loss = nn.BCELoss()
        classification_loss = bce_loss(outputs, labels)
        if self.form == "logsig":
            non_linearity = torch.log
        elif self.form == "linear":
            non_linearity = torch.logit
        dp_loss = -lambda_reg * (
            non_linearity(outputs[sensitive_attr == 0]).sum()/self.Ps0
            + non_linearity(1 - outputs[sensitive_attr == 1]).sum()/self.Ps0
        )
        dp_loss /= outputs.shape[0]
        combined_loss = classification_loss + torch.abs(dp_loss)
        acc = (
            (
                outputs.reshape(-1) * labels.reshape(-1)
                + ((1 - outputs).reshape(-1) * (1 - labels).reshape(-1))
            )
            .sum()
            .item()
        )
        return dp_loss, combined_loss, acc

    def compute_eo_regularized_loss_and_acc_per_batch(
        self, model, batch, device, lambda_reg
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
        if self.form == "logsig":
            non_linearity = torch.log
        elif self.form == "linear":
            non_linearity = torch.logit

        tpr0 = (
            non_linearity(
                outputs[(~sensitive_attr & labels.long()).reshape(-1).bool()]
            ).sum()
            / self.Ps0_y1
        )
        tpr1 = (
            non_linearity(
                1 - outputs[(sensitive_attr & labels.long()).reshape(-1).bool()]
            ).sum()
            / self.Ps1_y1
        )

        # False positive rates for each group
        fpr0 = (
            non_linearity(
                outputs[(~sensitive_attr & ~labels.long()).reshape(-1).bool()]
            ).sum()
            / self.Ps0_y0
        )
        fpr1 = (
            non_linearity(
                1 - outputs[(sensitive_attr & ~labels.long()).reshape(-1).bool()]
            ).sum()
            / self.Ps1_y0
        )
        eo_loss = -(tpr0 + tpr1 + fpr0 + fpr1)
        eo_loss /= outputs.shape[0]

        combined_loss = classification_loss + lambda_reg * torch.abs(eo_loss)
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
        self, model, batch, device, lambda_reg
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
        if self.form == "logsig":
            non_linearity = torch.log
        elif self.form == "linear":
            non_linearity = torch.logit
        tpr0 = (
            non_linearity(
                outputs[(~sensitive_attr & labels.long()).reshape(-1).bool()]
            ).sum()
            / self.Ps0_y1
        )
        tpr1 = (
            non_linearity(
                1 - outputs[(sensitive_attr & labels.long()).reshape(-1).bool()]
            ).sum()
            / self.Ps1_y1
        )

        eo_loss = -(tpr0 + tpr1)
        eo_loss /= outputs.shape[0]

        combined_loss = classification_loss + lambda_reg * torch.abs(eo_loss)
        acc = (
            (
                outputs.reshape(-1) * labels.reshape(-1)
                + ((1 - outputs).reshape(-1) * (1 - labels).reshape(-1))
            )
            .sum()
            .item()
        )
        return (eo_loss, combined_loss, acc)
