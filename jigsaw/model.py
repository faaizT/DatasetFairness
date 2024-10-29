from transformers import BertModel, AdamW, AutoTokenizer, BertConfig
from torch import nn
import torch


class BERT_Wrapper(nn.Module):
    def __init__(self):
        super(BERT_Wrapper, self).__init__()

        # Initialize BERT base model
        self.bert = BertModel.from_pretrained(
            "./jigsaw/bert_model/", output_attentions=False, output_hidden_states=False
        )

        # Classification head
        self.classifier = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, data):
        input_ids, attention_mask = data[:, :, 0], data[:, :, 1]
        # Forward pass through BERT base
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = outputs.last_hidden_state[:, 0, :]  # Take the [CLS] representation

        # Classification head
        logits = self.classifier(x)

        return torch.sigmoid(logits)


class YOTO_BERT(nn.Module):
    def __init__(self):
        super(YOTO_BERT, self).__init__()

        # Initialize BERT base model
        self.bert = BertModel.from_pretrained(
            "./jigsaw/bert_model/", output_attentions=False, output_hidden_states=False
        )

        # Classification head
        self.classifier = nn.Linear(self.bert.config.hidden_size, 1)

        # Initialize FiLM layers
        self.film_gamma = nn.Linear(1, self.bert.config.hidden_size)
        self.film_beta = nn.Linear(1, self.bert.config.hidden_size)

    def forward(self, data, lambda_reg):
        input_ids, attention_mask = data[:, :, 0], data[:, :, 1]
        # Forward pass through BERT base
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = outputs.last_hidden_state[:, 0, :]  # Take the [CLS] representation

        # Generate FiLM parameters from lambda
        gamma = self.film_gamma(lambda_reg)
        beta = self.film_beta(lambda_reg)

        # Apply FiLM layer
        x = gamma * x + beta

        # Classification head
        logits = self.classifier(x)

        return torch.sigmoid(logits)


def get_jigsaw_model_with_optimiser(device, yoto=True):
    if yoto:
        model = YOTO_BERT()
    else:
        model = BERT_Wrapper()
    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 1e-3,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    len([p for n, p in param_optimizer if not any(nd in n for nd in no_decay)]), len(
        [p for n, p in param_optimizer if any(nd in n for nd in no_decay)]
    )

    optimizer = AdamW(
        optimizer_parameters,
        lr=2e-5,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
        eps=1e-8,  # args.adam_epsilon  - default is 1e-8.
    )
    return model, optimizer
