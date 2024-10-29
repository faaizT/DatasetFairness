import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from fairlearn.metrics import demographic_parity_difference
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import torch
from torchvision import transforms
from PIL import Image
import os
from jigsaw.jigsaw_dataset_utils import get_data_loaders_jigsaw

import torchvision.transforms.functional as TF
from scripts.visualise_tradeoff_1d import plot_tradeoff


class CelebADataset(Dataset):
    def __init__(self, root_dir, attributes_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # Load attributes file
        attributes_df = pd.read_csv(attributes_file, delim_whitespace=True, header=1)
        # Convert gender column (-1 to 1) to binary (0 to 1)
        attributes_df["Male"] = (attributes_df["Male"] + 1) // 2
        self.attributes_df = attributes_df

    def __len__(self):
        return len(self.attributes_df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.attributes_df.index[idx])
        image = Image.open(img_name)

        if self.transform:
            image = self.transform(image)

        # Binary outcome (whether person is wearing glasses or not)
        glasses = self.attributes_df.iloc[idx]["Eyeglasses"]
        glasses = torch.tensor((glasses + 1) // 2, dtype=torch.float32)

        # Sensitive attribute (gender)
        gender = self.attributes_df.iloc[idx]["Male"]
        gender = torch.tensor(gender, dtype=torch.float32)

        return image, glasses, gender


class CustomDataset(Dataset):
    def __init__(self, X, y, sensitive_attr, surrogates=None):
        self.X = X
        self.y = y
        self.sensitive_attr = sensitive_attr
        self.surrogates = surrogates

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.sensitive_attr is None:
            return self.X[idx], self.y[idx]
        if self.surrogates is None:
            return self.X[idx], self.y[idx], self.sensitive_attr[idx]
        return (
            self.X[idx],
            self.y[idx],
            self.sensitive_attr[idx],
            self.surrogates[0][idx],
            self.surrogates[1][idx],
        )


class CustomDatasetForEO(Dataset):
    def __init__(self, X, y, sensitive_attr, surrogates=None):
        self.X = X
        self.y = y
        self.sensitive_attr = sensitive_attr

        # Define indices for each subgroup
        indices_y0_s0 = torch.where((self.y == 0) & (self.sensitive_attr == 0))[0]
        indices_y0_s1 = torch.where((self.y == 0) & (self.sensitive_attr == 1))[0]
        indices_y1_s0 = torch.where((self.y == 1) & (self.sensitive_attr == 0))[0]
        indices_y1_s1 = torch.where((self.y == 1) & (self.sensitive_attr == 1))[0]

        # Sample indices for each subgroup to match the length of X
        self.indices_y0_s0 = np.random.choice(indices_y0_s0, size=self.X.shape[0])
        self.indices_y0_s1 = np.random.choice(indices_y0_s1, size=self.X.shape[0])
        self.indices_y1_s0 = np.random.choice(indices_y1_s0, size=self.X.shape[0])
        self.indices_y1_s1 = np.random.choice(indices_y1_s1, size=self.X.shape[0])

        self.surrogates = surrogates

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X_y0_s0 = self.X[self.indices_y0_s0[idx]]
        X_y0_s1 = self.X[self.indices_y0_s1[idx]]
        X_y1_s0 = self.X[self.indices_y1_s0[idx]]
        X_y1_s1 = self.X[self.indices_y1_s1[idx]]
        if self.surrogates is None:
            return (
                self.X[idx],
                self.y[idx],
                self.sensitive_attr[idx],
                X_y0_s0,
                X_y0_s1,
                X_y1_s0,
                X_y1_s1,
            )
        return (
            self.X[idx],
            self.y[idx],
            self.sensitive_attr[idx],
            self.surrogates[0][idx],
            self.surrogates[1][idx],
            X_y0_s0,
            X_y0_s1,
            X_y1_s0,
            X_y1_s1,
        )


def create_dataset(n_samples, input_dim):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=input_dim,
        n_informative=2,
        n_redundant=input_dim // 2,
        random_state=42,
    )

    # Create a sensitive feature that is also highly correlated with y
    # Let's create two groups with different sizes and different rates of positive outcomes
    group1_samples = int(n_samples * 0.3)
    group2_samples = n_samples - group1_samples

    group1_outcomes = np.random.choice([0, 1], size=group1_samples, p=[0.8, 0.2])
    group2_outcomes = np.random.choice([0, 1], size=group2_samples, p=[0.2, 0.8])

    y_biased = np.concatenate([group1_outcomes, group2_outcomes])

    # Create a binary sensitive feature
    sensitive_feature = np.array([0] * group1_samples + [1] * group2_samples)

    # Check demographic parity difference
    dp_diff = demographic_parity_difference(
        y_biased, y_biased, sensitive_features=sensitive_feature
    )
    print(f"Demographic parity of the data: {dp_diff}")

    # relabel the true y's
    y = y_biased
    return X, y, sensitive_feature


def create_dataset_male_female_synth_1d(
    n_samples, plot_tradeoff_=False, fairness_metric="eo", ax=None
):
    male_mean = 1
    male_std_dev = 0.2
    female_mean = 0
    female_std_dev = 0.2
    threshold = 0.5

    # Generate male and female data
    n_male = n_samples // 2
    n_female = n_samples - n_male
    x_male = np.random.normal(male_mean, male_std_dev, n_male)
    x_female = np.random.normal(female_mean, female_std_dev, n_female)

    # Labels
    a_male = np.ones(n_male)
    a_female = np.zeros(n_female)
    y_male = (x_male > threshold).astype(int)
    y_female = (x_female > threshold).astype(int)

    # We add some noise when using "EO" as fairness metric
    if fairness_metric.startswith("eo"):
        # Determine the number of labels to flip for males
        num_to_flip_male = int(0.05 * len(y_male))

        # Generate random indices
        flip_indices = np.random.choice(
            len(y_male), size=num_to_flip_male, replace=False
        )

        # Flip labels at the selected indices
        y_male[flip_indices] = 1 - y_male[flip_indices]

        # Determine the number of labels to flip for females
        num_to_flip_female = int(0.05 * len(y_female))

        # Generate random indices
        flip_indices = np.random.choice(
            len(y_female), size=num_to_flip_female, replace=False
        )

        # Flip labels at the selected indices
        y_female[flip_indices] = 1 - y_female[flip_indices]

    # Concatenate
    x = np.concatenate([x_male, x_female])
    a = np.concatenate([a_male, a_female])
    y = np.concatenate([y_male, y_female])

    # Split the data
    if plot_tradeoff_:
        x_male = x[a == 1]
        x_female = x[a == 0]
        y_male = y[a == 1]
        y_female = y[a == 0]

        plot_tradeoff(
            x_male,
            x_female,
            y_male,
            y_female,
            male_mean,
            male_std_dev,
            female_mean,
            female_std_dev,
            fairness_metric=fairness_metric,
            ax=ax,
        )
    return x, y, a


def create_dataset_male_female_synth(n_samples, fairness_metric="dp"):
    # Set seed for reproducibility
    np.random.seed(42)

    # Define the total number of samples and samples per group
    n_samples_per_group = n_samples // 2  # Equal size groups for male and female

    # Sensitive feature is gender: Male=0, Female=1
    sensitive_attr = np.concatenate(
        [np.zeros(n_samples_per_group), np.ones(n_samples_per_group)]
    )

    # Non-sensitive features with a normal distribution
    # Assume males (group 0) are drawn from a 2D normal distribution with means (2, 3) and standard deviations (1, 1.5)
    # Assume females (group 1) are drawn from a 2D normal distribution with means (4, 5) and standard deviations (2, 2.5)
    non_sensitive_attr_1 = np.concatenate(
        [
            np.random.normal(2, 1, n_samples_per_group),
            np.random.normal(4, 2, n_samples_per_group),
        ]
    )
    non_sensitive_attr_2 = np.concatenate(
        [
            np.random.normal(3, 1.5, n_samples_per_group),
            np.random.normal(5, 2.5, n_samples_per_group),
        ]
    )

    # Binary target variable
    # Assume the positive class is more likely for higher values of the non-sensitive features
    # Here we are simply summing the two features and applying a threshold, but you could define this in many ways
    threshold = 4  # You can adjust this value
    y = ((non_sensitive_attr_1) > threshold).astype(int)

    if fairness_metric.startswith("eo"):
        # Determine the number of labels to flip
        num_to_flip = int(0.2 * len(y))

        # Generate random indices
        flip_indices = np.random.choice(len(y), size=num_to_flip, replace=False)

        # Flip labels at the selected indices
        y[flip_indices] = 1 - y[flip_indices]

    # Create a DataFrame for better visualization
    df = pd.DataFrame(
        {
            "gender": sensitive_attr,
            "feature_1": non_sensitive_attr_1,
            "feature_2": non_sensitive_attr_2,
            "target": y,
            # 'feature': np.concatenate([non_sensitive_attr_1.reshape(-1, 1), non_sensitive_attr_2], axis=1),
        }
    )

    # Shuffle the DataFrame
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    return (
        df[["feature_1", "feature_2"]].to_numpy(),
        df["target"].to_numpy(),
        df["gender"].to_numpy(),
    )


def get_adults_datset(n_samples, with_val):
    col_names = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "income",
    ]

    # Load data
    data = (
        pd.read_csv("./datasets/adult.data", names=col_names)
        .sample(frac=1.0, random_state=42)
        .reset_index(drop=True)
    )

    # Handle missing values
    data = data.replace(" ?", pd.NA)
    data = data.dropna()

    if with_val:
        data = data.iloc[:-5000].copy()

    # Define sensitive attribute
    sensitive_attr = data["sex"].apply(lambda x: "Female" in x).values

    # Define target variable
    y = LabelEncoder().fit_transform(data["income"])

    # Define features
    X = data.drop(["sex", "income"], axis=1)

    # Normalization for numeric features
    num_cols = X.select_dtypes(include=[np.number]).columns
    scaler = StandardScaler()
    X[num_cols] = scaler.fit_transform(X[num_cols])

    # One-hot encoding for categorical features
    X = pd.get_dummies(X).values
    if not with_val:
        X = X[-5000:, :]
        y = y[-5000:]
        sensitive_attr = sensitive_attr[-5000:]
    return X[:n_samples, :], y[:n_samples], sensitive_attr[:n_samples]


def get_compas_dataset(n_samples, with_val):
    col_names = [
        "sex",
        "age",
        "race",
        "juv_fel_count",
        "juv_misd_count",
        "juv_other_count",
        "priors_count",
        "c_charge_degree",
        "two_year_recid",
    ]

    # Load data
    data = (
        pd.read_csv("./datasets/compas-scores-two-years.csv")
        .sample(frac=1.0, random_state=42)
        .reset_index(drop=True)
    )

    data = data.loc[:, col_names]

    # Handle missing values
    data = data.replace(" ?", pd.NA)
    data = data.dropna()

    if with_val:
        data = data.iloc[:-2000].copy()
    else:
        data = data.iloc[-2000:].copy()

    # Define sensitive attribute
    sensitive_attr = data["race"].apply(lambda x: "African-American" in x).values

    # Define target variable
    y = LabelEncoder().fit_transform(data["two_year_recid"])

    # Define features
    X = data.drop(["race", "two_year_recid"], axis=1)

    # Normalization for numeric features
    num_cols = X.select_dtypes(include=[np.number]).columns
    scaler = StandardScaler()
    X[num_cols] = scaler.fit_transform(X[num_cols])

    # One-hot encoding for categorical features
    X = pd.get_dummies(X).values
    return X[:n_samples, :], y[:n_samples], sensitive_attr[:n_samples]


def get_celeba_dataset(n_samples, with_val=False):
    df = pd.read_csv("../celeba/list_attr_celeba.csv")
    if with_val:
        df = df.iloc[:-10000].copy()
    else:
        df = df.iloc[-10000:].copy()
    df.replace(to_replace=-1, value=0, inplace=True)
    df["image_id"] = "../celeba/img_align_celeba/img_align_celeba/" + df["image_id"]
    df = df[:n_samples]

    X = []
    transform = transforms.Compose(
        [transforms.Resize(64), transforms.CenterCrop(64), transforms.ToTensor()]
    )
    X = df["image_id"].apply(lambda x: transform(Image.open(x))).values

    y = df["Smiling"].values
    sensitive_attr = df["Male"].values

    (
        X_train,
        X_test,
        y_train,
        y_test,
        sensitive_attr_train,
        sensitive_attr_test,
    ) = train_test_split(X, y, sensitive_attr, test_size=0.4, random_state=42)

    # Split the temp data into test and validation sets (20-20 split of the total data)
    if with_val:
        (
            X_test,
            X_val,
            y_test,
            y_val,
            sensitive_attr_test,
            sensitive_attr_val,
        ) = train_test_split(
            X_test, y_test, sensitive_attr_test, test_size=0.5, random_state=42
        )
        X_val = torch.stack(list(X_val))
        y_val = torch.from_numpy(y_val.astype("float32")).reshape(-1, 1)
        sensitive_attr_val = torch.from_numpy(
            sensitive_attr_val.astype("float32")
        ).reshape(-1, 1)
    else:
        X_val, y_val, sensitive_attr_val = None, None, None

    X_train = torch.stack(list(X_train))
    y_train = torch.from_numpy(y_train.astype("float32")).reshape(-1, 1)
    sensitive_attr_train = torch.from_numpy(
        sensitive_attr_train.astype("float32")
    ).reshape(-1, 1)

    X_test = torch.stack(list(X_test))
    y_test = torch.from_numpy(y_test.astype("float32")).reshape(-1, 1)
    sensitive_attr_test = torch.from_numpy(
        sensitive_attr_test.astype("float32")
    ).reshape(-1, 1)

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        sensitive_attr_train,
        sensitive_attr_val,
        sensitive_attr_test,
    )


def get_data_loaders_celeba(
    input_dim, n_samples, split=True, for_eo=False, with_val=False, batch_size=32
):
    # TODO: implement for_eo functionality
    fairness_metric = "eo" if for_eo else "dp"

    if input_dim == 64 * 64 * 3:  # RGB images of size 64x64
        (
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test,
            sensitive_attr_train,
            sensitive_attr_val,
            sensitive_attr_test,
        ) = get_celeba_dataset(n_samples, with_val=with_val)
    else:
        raise ValueError("Input dimension >= 3 not implemented")

    # Create Datasets
    DatasetClass = CustomDatasetForEO if for_eo else CustomDataset
    train_dataset = DatasetClass(X_train, y_train, sensitive_attr_train)
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = DatasetClass(X_test, y_test, sensitive_attr_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    if with_val:
        val_dataset = DatasetClass(X_val, y_val, sensitive_attr_val)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
        return (
            train_loader,
            val_loader,
            test_loader,
            train_dataset,
            val_dataset,
            test_dataset,
        )
    if split:
        return train_loader, test_loader, train_dataset, test_dataset
    return train_loader, train_dataset


def get_data_loaders_without_validation(
    input_dim, n_samples, batch_size=32, split=True, for_eo=False, fairness=None
):
    fairness_metric = "eo" if for_eo else "dp"
    if input_dim == 1:
        X, y, sensitive_attr = create_dataset_male_female_synth_1d(
            n_samples, fairness_metric=fairness_metric
        )
    elif input_dim == 2:
        X, y, sensitive_attr = create_dataset_male_female_synth(
            n_samples, fairness_metric=fairness_metric
        )
    elif input_dim == 9:
        X, y, sensitive_attr = get_compas_dataset(n_samples, with_val=False)
    elif input_dim == 102:
        X, y, sensitive_attr = get_adults_datset(n_samples, with_val=False)
    elif input_dim == 64 * 64 * 3:
        return get_data_loaders_celeba(
            input_dim,
            n_samples,
            split=split,
            for_eo=False,
            with_val=False,
            batch_size=batch_size,
        )
    elif input_dim == 512:
        return get_data_loaders_jigsaw(
            batch_size, n_samples, with_val=False, fairness=fairness
        )
    else:
        raise ValueError("Input dimension >= 3 not implemented")
    # Assume X, y, sensitive_attr are your numpy arrays
    X = X.astype("float32")  # Convert to float32 for PyTorch
    y = y.astype("float32")
    sensitive_attr = sensitive_attr.astype("float32")

    if split:
        # Split the data into train and test sets (80-20 split)
        (
            X_train,
            X_test,
            y_train,
            y_test,
            sensitive_attr_train,
            sensitive_attr_test,
        ) = train_test_split(X, y, sensitive_attr, test_size=0.2, random_state=42)
    else:
        X_train, y_train, sensitive_attr_train = X, y, sensitive_attr

    # Convert to PyTorch tensors
    X_train_tensor = torch.from_numpy(X_train)
    y_train_tensor = torch.from_numpy(y_train).reshape(-1, 1)
    sensitive_attr_train_tensor = torch.from_numpy(sensitive_attr_train).reshape(-1, 1)

    if split:
        X_test_tensor = torch.from_numpy(X_test)
        y_test_tensor = torch.from_numpy(y_test).reshape(-1, 1)
        sensitive_attr_test_tensor = torch.from_numpy(sensitive_attr_test).reshape(
            -1, 1
        )

    if len(X_train.shape) == 1:
        X_train_tensor = X_train_tensor.reshape(-1, 1)
        if split:
            X_test_tensor = X_test_tensor.reshape(-1, 1)

    # Create Datasets
    train_dataset = CustomDataset(
        X_train_tensor, y_train_tensor, sensitive_attr_train_tensor
    )
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    if split:
        test_dataset = CustomDataset(
            X_test_tensor, y_test_tensor, sensitive_attr_test_tensor
        )
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, test_loader, train_dataset, test_dataset
    return train_loader, train_dataset


def get_data_loaders_with_validation(
    input_dim, n_samples, batch_size, for_eo=False, fairness=None
):
    fairness_metric = "eo" if for_eo else "dp"
    if input_dim == 1:
        X, y, sensitive_attr = create_dataset_male_female_synth_1d(
            n_samples, fairness_metric=fairness_metric
        )
    elif input_dim == 2:
        X, y, sensitive_attr = create_dataset_male_female_synth(
            n_samples, fairness_metric=fairness_metric
        )
    elif input_dim == 9:
        X, y, sensitive_attr = get_compas_dataset(n_samples, with_val=True)
    elif input_dim == 102:
        X, y, sensitive_attr = get_adults_datset(n_samples, with_val=True)
    elif input_dim == 64 * 64 * 3:
        return get_data_loaders_celeba(
            input_dim, n_samples, split=True, for_eo=for_eo, with_val=True
        )
    elif input_dim == 512:
        return get_data_loaders_jigsaw(
            batch_size, n_samples, with_val=True, fairness=fairness
        )
    else:
        raise ValueError("Input dimension >= 3 not implemented")
    X = X.astype("float32")
    y = y.astype("float32")
    sensitive_attr = sensitive_attr.astype("float32")

    # Split the data into train and temp sets (60-40 split)
    (
        X_train,
        X_temp,
        y_train,
        y_temp,
        sensitive_attr_train,
        sensitive_attr_temp,
    ) = train_test_split(X, y, sensitive_attr, test_size=0.4, random_state=42)

    # Split the temp data into validation and test sets (each 20% of the total data)
    (
        X_val,
        X_test,
        y_val,
        y_test,
        sensitive_attr_val,
        sensitive_attr_test,
    ) = train_test_split(
        X_temp, y_temp, sensitive_attr_temp, test_size=0.5, random_state=42
    )

    # Convert to PyTorch tensors and reshape y and sensitive_attr
    reshape_and_tensor = lambda arr: torch.from_numpy(arr).reshape(-1, 1)
    if len(X_train.shape) > 1:
        X_train, X_val, X_test = map(torch.from_numpy, (X_train, X_val, X_test))
    else:
        X_train, X_val, X_test = map(reshape_and_tensor, (X_train, X_val, X_test))
    y_train, y_val, y_test = map(reshape_and_tensor, (y_train, y_val, y_test))
    sensitive_attr_train, sensitive_attr_val, sensitive_attr_test = map(
        reshape_and_tensor,
        (sensitive_attr_train, sensitive_attr_val, sensitive_attr_test),
    )

    # Create Datasets
    dataset = CustomDatasetForEO if for_eo else CustomDataset
    train_dataset = dataset(X_train, y_train, sensitive_attr_train)
    val_dataset = dataset(X_val, y_val, sensitive_attr_val)
    test_dataset = dataset(X_test, y_test, sensitive_attr_test)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return (
        train_loader,
        val_loader,
        test_loader,
        train_dataset,
        val_dataset,
        test_dataset,
    )
