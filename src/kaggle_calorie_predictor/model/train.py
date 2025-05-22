import logging
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from utils import date_filename


logger = logging.getLogger(__name__)


class ModelEnvironment:
    def __init__(self, model, train, test, val, hyperparameters):
        self.model = model
        self.model_name = hyperparameters["model_name"]
        self.batch_size = hyperparameters["batch_size"]
        self.device = hyperparameters["device"]
        self.epochs = hyperparameters["epochs"]
        self.learning_rate = hyperparameters["learning_rate"]
        self.criterion = hyperparameters["criterion_instance"]
        self.optimizer = hyperparameters["optimizer_instance"]
        self.output_size = hyperparameters["output_size"]
        self.start_epoch = 0
        self.running_loss = 0.0
        self.val_loss = 0.0
        self.train_dataloader = train
        self.test_dataloader = test
        self.val_dataloader = val
        self.train_losses = []
        self.val_losses = []
        self.avg_test_loss = None
        self.test_accuracy = None
        self.test_predictions = []
        self.test_actuals = []
        self.test_losses = []

    def train_test_split(self, test_split, val_split=None, seed=42):
        train_data, test_data = train_test_split(
            self.dataset,
            test_size=test_split if not val_split else test_split + val_split,
            random_state=seed,
        )
        self.train_data = train_data
        self.test_data = test_data

        if val_split:
            val_data, test_data = train_test_split(
                test_data,
                test_size=test_split / (test_split + val_split),
                random_state=seed,
            )
            self.val_data = val_data
            self.test_data = test_data

    def train_epoch(self):
        self.model.train()
        for inputs, labels in self.train_dataloader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # self.scheduler.step()
            self.running_loss += loss.item()

    def train(self):
        logger.info(f"Training model for {self.epochs} epochs")
        best_val_loss = float("inf")
        for epoch in tqdm(range(self.start_epoch, self.epochs)):
            self.running_loss = 0.0
            self.val_loss = 0.0
            self.train_epoch()
            avg_loss = self.running_loss / len(self.train_dataloader)

            if self.val_dataloader:
                self.validate()
                avg_val_loss = self.val_loss / len(self.val_dataloader)
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    self.save_checkpoint(
                        date_filename(
                            f"model/saved_models/{self.model_name}_best_epoch", ".pt"
                        ),
                        epoch,
                    )
                    logger.info(f"Saved new best model at epoch {epoch+1}")
            else:
                avg_val_loss = "N/A"

            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(
                    f"Epoch [{epoch+1}/{self.epochs}], Loss/Val Loss: {avg_loss:.4f} / {avg_val_loss:.4f}"
                )

            self.train_losses.append(avg_loss)
            if self.val_dataloader:
                self.val_losses.append(avg_val_loss)
            else:
                self.val_losses.append(None)

        final_path = date_filename(
            f"model/saved_models/{self.model_name}", ".pt", include_time=True
        )
        self.save_checkpoint(final_path, self.epochs)
        logger.info(f"Training complete. Model saved to {final_path}")

    def validate(self):
        if not self.val_dataloader:
            return
        self.model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in self.val_dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                total_val_loss += loss.item()
        self.val_loss = total_val_loss

    def evaluate(self):
        self.model.eval()
        test_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in tqdm(self.test_dataloader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                test_loss += loss.item()
                all_preds.extend(outputs.squeeze().cpu().tolist())
                all_labels.extend(labels.squeeze().cpu().tolist())
                self.test_losses.append(loss.item())

        self.avg_test_loss = test_loss / len(self.test_dataloader)
        self.test_predictions = all_preds
        self.test_actuals = all_labels

        logger.info(f"Test Loss: {self.avg_test_loss:.4f}")

    def infer(self, dataloader):  # TODO: Fix to work with labels and without
        self.model.eval()
        all_outputs = []

        with torch.no_grad():
            for (inputs,) in tqdm(dataloader):  # assuming labels optional
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                all_outputs.append(outputs.cpu())

        predictions = torch.cat(all_outputs, dim=0).squeeze()
        return predictions

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.to(self.device)
        logger.info(f"Model loaded from {path}")

    def save_checkpoint(self, path, epoch):
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                # "scheduler_state_dict": self.scheduler.state_dict(),
                "epoch": epoch,
                "train_losses": self.train_losses,
                "val_losses": self.val_losses,
                "running_loss": self.running_loss,
                "val_loss": self.val_loss,
            },
            path,
        )
        logger.info(f"Checkpoint saved to {path}")
