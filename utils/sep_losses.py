import numpy as np
import torch
from torch import nn
from fairsurrogates.losses import SurrogatesLoss

from jigsaw.losses import EOLossSep


def compute_dp_regularized_loss_and_acc_per_batch(
    model, batch, device, lambda_reg, square_dp
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
    if square_dp:
        dp_loss = torch.square(
            (outputs[sensitive_attr == 0].mean() - outputs[sensitive_attr == 1].mean())
        )
    else:
        dp_loss = torch.abs(
            (outputs[sensitive_attr == 0].mean() - outputs[sensitive_attr == 1].mean())
        )
    combined_loss = classification_loss + torch.abs(lambda_reg * dp_loss)
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
    model, batch, device, lambda_reg, square_eo
):
    data, labels, sensitive_attr, X_y0_s0, X_y0_s1, X_y1_s0, X_y1_s1 = batch
    data, labels, sensitive_attr, X_y0_s0, X_y0_s1, X_y1_s0, X_y1_s1 = (
        data.to(device),
        labels.to(device),
        sensitive_attr.to(device),
        X_y0_s0.to(device),
        X_y0_s1.to(device),
        X_y1_s0.to(device),
        X_y1_s1.to(device),
    )
    outputs = model(data)
    bce_loss = nn.BCELoss()
    classification_loss = bce_loss(outputs, labels)

    # True positive rates for each group
    tpr0 = model(X_y1_s0).mean()
    tpr1 = model(X_y1_s1).mean()

    # False positive rates for each group
    fpr0 = model(X_y0_s0).mean()
    fpr1 = model(X_y0_s1).mean()

    if square_eo:
        eo_loss = torch.square(tpr0 - tpr1) + torch.square(fpr0 - fpr1)
    else:
        eo_loss = torch.abs(tpr0 - tpr1) + torch.abs(fpr0 - fpr1)

    combined_loss = classification_loss + torch.abs(lambda_reg * eo_loss)
    acc = (
        (
            outputs.reshape(-1) * labels.reshape(-1)
            + ((1 - outputs).reshape(-1) * (1 - labels).reshape(-1))
        )
        .sum()
        .item()
    )
    return eo_loss, combined_loss, acc


def compute_eop_regularized_loss_and_acc_per_batch(
    model, batch, device, lambda_reg, square_eop
):
    data, labels, sensitive_attr, X_y0_s0, X_y0_s1, X_y1_s0, X_y1_s1 = batch
    data, labels, sensitive_attr, X_y0_s0, X_y0_s1, X_y1_s0, X_y1_s1 = (
        data.to(device),
        labels.to(device),
        sensitive_attr.to(device),
        X_y0_s0.to(device),
        X_y0_s1.to(device),
        X_y1_s0.to(device),
        X_y1_s1.to(device),
    )
    outputs = model(data)
    bce_loss = nn.BCELoss()
    classification_loss = bce_loss(outputs, labels)

    # True positive rates for each group
    tpr0 = model(X_y1_s0).mean()
    tpr1 = model(X_y1_s1).mean()

    if square_eop:
        eop_loss = torch.square(tpr0 - tpr1)
    else:
        eop_loss = torch.abs(tpr0 - tpr1)

    combined_loss = classification_loss + torch.abs(lambda_reg * eop_loss)
    acc = (
        (
            outputs.reshape(-1) * labels.reshape(-1)
            + ((1 - outputs).reshape(-1) * (1 - labels).reshape(-1))
        )
        .sum()
        .item()
    )
    return eop_loss, combined_loss, acc


def get_loss_function(input_dim, fairness, fairsurrogates, train_dataset, form="logsig"):
    if fairsurrogates:
        loss_per_batch = SurrogatesLoss(
            form, fairness=fairness, train_df=train_dataset
        ).loss_function
    elif input_dim == 512 and fairness.startswith("eo"):
        loss_per_batch = EOLossSep(
            fairness=fairness, train_df=train_dataset
        ).loss_function
    elif fairness == "eo":
        loss_per_batch = compute_eo_regularized_loss_and_acc_per_batch
    elif fairness == "eop":
        loss_per_batch = compute_eop_regularized_loss_and_acc_per_batch
    else:
        loss_per_batch = compute_dp_regularized_loss_and_acc_per_batch
    return loss_per_batch
