import pandas as pd
from transformers import BertModel, AdamW, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.model_selection import train_test_split


class JigsawDataset(Dataset):
    def __init__(self, reviews, targets, genders, tokenizer, max_len):
        self.comment, self.targets, self.tokenizer, self.max_len = (
            reviews.to_numpy(),
            targets.to_numpy().reshape(-1, 1),
            tokenizer,
            max_len,
        )
        self.genders = genders.to_numpy().reshape(-1, 1)

    def __len__(self):
        return len(self.comment)

    def __getitem__(self, item):
        review = self.comment[item]
        tokens = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            return_token_type_ids=False,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
        )

        return (
            torch.stack(
                [tokens["input_ids"].flatten(), tokens["attention_mask"].flatten()],
                axis=1,
            ),
            torch.FloatTensor(self.targets[item, :]),
            torch.tensor(self.genders[item, :]).long(),
        )


def create_data_loader(df, tokenizer, max_length, batch_size):
    ds = JigsawDataset(
        df["comment_text"], df["toxic"], df["gender"], tokenizer, max_length
    )
    return DataLoader(ds, batch_size=batch_size)


def get_data_loaders_jigsaw(batch_size, n_samples, with_val=True, fairness=None):
    df = pd.read_csv("../jigsaw/data_with_genders.csv")

    df["toxic"] = (df["toxicity"] > 0) * 1
    df["gender"] = (df["gender"] == "male") * 1

    if fairness == "eop":
        unique_ids = df["id"].unique()
        train_ids, test_ids = train_test_split(
            unique_ids, test_size=0.1, random_state=42
        )
        val_ids, test_ids = train_test_split(test_ids, test_size=0.5, random_state=42)
        train_df = df[df["id"].isin(train_ids)].sample(frac=1, random_state=42)
        valid_df = df[df["id"].isin(val_ids)].sample(frac=1, random_state=42)
        test_df = df[df["id"].isin(test_ids)].sample(frac=1, random_state=42)
    else:
        train_df, test_df = train_test_split(
            df, test_size=0.1, random_state=42, stratify=df.toxic.values
        )
        valid_df, test_df = train_test_split(
            test_df, test_size=0.5, random_state=42, stratify=test_df.toxic.values
        )
        test_df = df[~df["id"].isin(train_df["id"])].sample(frac=1, random_state=42)

    if with_val:
        test_df = test_df[:1600]
    else:
        test_df = test_df[:n_samples]

    tokenizer = AutoTokenizer.from_pretrained("./jigsaw/tokenizer/")

    train_loader = create_data_loader(train_df, tokenizer, 512, batch_size)
    test_loader = create_data_loader(test_df, tokenizer, 512, batch_size)
    val_loader = create_data_loader(valid_df, tokenizer, 512, batch_size)
    if with_val:
        return train_loader, val_loader, test_loader, train_df, valid_df, test_df
    return test_loader, test_df
