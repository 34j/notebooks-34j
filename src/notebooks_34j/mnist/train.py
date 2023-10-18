import os
import warnings
from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.callbacks import RichProgressBar
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.tuner.tuning import Tuner
from lion_pytorch import Lion
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy
from torchvision import transforms
from torchvision.datasets import MNIST

__module__ = __name__.split(".")[-1]


class MNISTDataModule(LightningDataModule):
    def __init__(
        self, batch_size: int, data_dir: str = f"~/.cache/{__module__}"
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message=".*does not have many workers.*",
        )
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message=".*Lazy modules are a.*",
        )

    def prepare_data(self) -> None:
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: str) -> None:
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(
                mnist_full, [55000, 5000], generator=torch.Generator().manual_seed(42)
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.mnist_test = MNIST(
                self.data_dir, train=False, transform=self.transform
            )

        if stage == "predict":
            self.mnist_predict = MNIST(
                self.data_dir, train=False, transform=self.transform
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.mnist_train, batch_size=self.batch_size)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.mnist_val, batch_size=self.batch_size)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.mnist_test, batch_size=self.batch_size)

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(self.mnist_predict, batch_size=self.batch_size)


class MNISTLightning(LightningModule):
    def __init__(
        self, model: nn.Module, lr: float = 0.001, gamma: float = 0.98
    ) -> None:
        super().__init__()
        self.model = model
        self.lr = lr
        self.gamma = gamma
        self.val_accuracy = Accuracy(task="multiclass", num_classes=10)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return F.log_softmax(x, dim=1)

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.val_accuracy.update(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_accuracy, prog_bar=True)

    def test_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.test_accuracy.update(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", self.test_accuracy, prog_bar=True)

    def configure_optimizers(
        self,
    ) -> tuple[list[torch.optim.Optimizer], list[torch.optim.lr_scheduler.LRScheduler]]:
        # optim = torch.optim.AdamW(self.parameters(), lr=self.lr)
        optim = Lion(self.parameters(), lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=self.gamma)
        return [optim], [lr_scheduler]


def train(
    model: torch.nn.Module,
    max_epochs: int = 10,
    batch_size: int | None = None,
    lr: float | None = 0.001,
) -> list[Mapping[str, float]]:
    strategy = (
        (
            "ddp_find_unused_parameters_true"
            if os.name != "nt"
            else DDPStrategy(find_unused_parameters=True, process_group_backend="gloo")
        )
        if torch.cuda.device_count() > 1
        else "auto"
    )
    data_module = MNISTDataModule(batch_size=batch_size or 1)
    model = MNISTLightning(model, lr=lr or 0.001)
    trainer = Trainer(
        max_epochs=max_epochs,
        strategy=strategy,
        callbacks=[RichProgressBar()],
        benchmark=True,
        precision="bf16-mixed",
    )
    tuner = Tuner(trainer)
    if batch_size is None:
        tuner.scale_batch_size(model, data_module, steps_per_trial=1)
        data_module.batch_size //= 2
    if lr is None:
        lr_finder = tuner.lr_find(model, data_module)
        if lr_finder is not None:
            fig = lr_finder.plot(suggest=True)
            model.logger.experiment.add_figure("lr_finder", fig)
        else:
            warnings.warn("tuner.lr_find returned None", RuntimeWarning)
    trainer.fit(model, data_module)
    return trainer.test(model, data_module)
