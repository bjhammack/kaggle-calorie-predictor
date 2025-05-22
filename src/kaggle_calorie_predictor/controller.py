import logging
import matplotlib.pyplot as plt
from model.criteria_optimizers import get_criterion, get_optimizer
from model.models import MODELS
from model.train import ModelEnvironment
import pandas as pd
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import TensorDataset, DataLoader
from utils import date_filename, init_logging, get_parameters


def prep_train_data(seed=42):
    df = pd.read_csv("data/train.csv")
    features = ["Age", "Duration", "Heart_Rate", "Body_Temp"]
    target = "Calories"

    X = df[features].values.astype("float32")
    y = df[[target]].values.astype("float32")

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.10, random_state=seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.1111, random_state=seed
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    X_train_tensor = torch.tensor(X_train)
    y_train_tensor = torch.tensor(y_train)
    X_val_tensor = torch.tensor(X_val)
    y_val_tensor = torch.tensor(y_val)
    X_test_tensor = torch.tensor(X_test)
    y_test_tensor = torch.tensor(y_test)

    train_ds = TensorDataset(X_train_tensor, y_train_tensor)
    val_ds = TensorDataset(X_val_tensor, y_val_tensor)
    test_ds = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)
    test_loader = DataLoader(test_ds, batch_size=32)
    return train_loader, test_loader, val_loader


def prep_inference_data(with_labels=False):
    df = pd.read_csv("data/test.csv")
    features = ["Age", "Duration", "Heart_Rate", "Body_Temp"]

    X = df[features].values.astype("float32")

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_tensor = torch.tensor(X)

    if with_labels:
        target = "Calories"
        y = df[[target]].values.astype("float32")
        y_tensor = torch.tensor(y)
        data_ds = TensorDataset(X_tensor, y_tensor)
        data_loader = DataLoader(data_ds, batch_size=32, shuffle=False)
    else:
        data_ds = TensorDataset(X_tensor)
        data_loader = DataLoader(data_ds, batch_size=32, shuffle=False)

    return data_loader


def prep_model(model_name, hp):
    model = MODELS[model_name]().to(hp["device"])
    hp["criterion_instance"] = get_criterion(hp["criterion"])
    hp["optimizer_instance"] = get_optimizer(
        hp["optimizer"], model.parameters(), hp["learning_rate"]
    )
    return model, hp


def train(model, train, test, val, hp):
    train_env = ModelEnvironment(model, train, test, val, hp)
    train_env.train()
    return train_env


def evaluate(env=None, dataset=None, model=None, hp=None):
    if env:
        env.evaluate()
        return env
    if dataset is not None and model is not None and hp is not None:
        test_env = ModelEnvironment(model, dataset, hp)
        test_env.create_tensors(use_split=False)
        test_env.create_dataloaders(use_split=False)
        test_env.test_dataloader = test_env.dataset_dataloader
        test_env.evaluate(test_env.dataset_dataloader)
        return test_env


def infer(model, data, hp):
    infer_env = ModelEnvironment(model, None, None, None, hp)
    preds = infer_env.infer(data)
    return preds


def create_submission_file(predictions, filepath="data/submissions/submission.csv"):
    df = pd.read_csv("data/test.csv")
    df["Calories"] = predictions
    submission_df = df[["id", "Calories"]]
    submission_df.to_csv(filepath, index=False)


def plot_losses(
    train_losses, val_losses=None, title="Training and Validation Loss", save_path=None
):
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)

    plt.plot(epochs, train_losses, label="Train Loss", linewidth=2)
    if val_losses is not None:
        plt.plot(epochs, val_losses, label="Val Loss", linewidth=2)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Loss plot saved to {save_path}")
    plt.show()


if __name__ == "__main__":
    param_path = sys.argv[1] if len(sys.argv) > 1 else "parameters.json"
    params, hp = get_parameters(param_path)
    hp["model_name"] = params["model_name"]

    hp_str = "".join(["\n\t{0}: {1}".format(key, value) for key, value in hp.items()])

    log_loc = init_logging(params["model_name"])
    logger = logging.getLogger(__name__)
    logger.info(f"Logging to {log_loc}")
    logger.info("\nHyperparameters:{0}".format(hp_str))

    model, hp = prep_model(params["model_name"], hp)

    if params["task"] == "train":
        data = prep_train_data(seed=params["seed"])
        logger.info("Training mode")
        train_env = train(model, *data, hp)
        filename = date_filename(hp["model_name"], ".pt")
        train_env = evaluate(env=train_env)
        plot_losses(
            train_env.train_losses,
            train_env.val_losses,
            title=f"Training and Validation Loss for {params['model_name']}",
            save_path=date_filename(
                f'model/visuals/loss_curves/{hp["model_name"]}',
                "_loss_curve.png",
                include_time=True,
            ),
        )
    elif params["task"] == "infer":
        data = prep_inference_data(False)
        logger.info("Inference mode")
        outputs, preds = infer(model, data, hp)
        print(outputs[0], preds[0])
    elif params["task"] == "test":
        data = prep_train_data(seed=params["seed"])
        logger.info("Testing mode")
        test_env = evaluate(model=model, dataset=data, hp=hp)
    elif params["task"] == "submission":
        data = prep_inference_data(False)
        logger.info("Submission mode")
        state = torch.load(params["model_path"])
        model, hp = prep_model(params["model_name"], hp)
        model.load_state_dict(state["model_state_dict"])
        preds = infer(model, data, hp)
        create_submission_file(preds)
        logger.info("Submission file created")
    else:
        raise ValueError("Invalid task. Please specify 'train', 'infer', or 'test'.")
