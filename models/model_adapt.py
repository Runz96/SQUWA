import torch
from torch import nn, Tensor
from torch.optim.lr_scheduler import ExponentialLR
import torch.nn.functional as F
import torchmetrics
import pytorch_lightning as pl
from typing import Any, List, Tuple
from transformers import get_scheduler
import numpy as np
from omegaconf import DictConfig, OmegaConf
import io
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import ToTensor
from typing import List, Tuple, Union

from .compose_signal import CompositeSignalGeneratorModule
from .resnet_adapt import ResNet_Adapt
from .utils import ConditionalModel
from .dense import Dense
from .mapping import Mapping
from .cmc_loss import calculate_cmc_loss
from .weighting_adapt import Weighting_Adapt
from .loss_fn import get_loss_function
# from .cmc_loss import intra_cluster_loss, inter_cluster_loss


class CompleteModel(nn.Module):
    def __init__(self, hyperparams: DictConfig, config: DictConfig):
        super(CompleteModel, self).__init__()
        self.config = config
        self._initialize_modules(hyperparams, config)

    def _initialize_modules(self, hyperparams: DictConfig, config: DictConfig):
        self.composite_signal_generator = CompositeSignalGeneratorModule(
            **hyperparams.signal_generator, config=config.composite_signal_generator
        )

        feat_extractor = ResNet_Adapt(**hyperparams.feature_extractor)
        self.feature_extractor = ConditionalModel(
            feat_extractor, config.feature_extractor
        )

        if config.feature_extractor:
            rnn_in_dim = feat_extractor.feature_dim
            time_dim = feat_extractor.target_time_dim
        else:
            rnn_in_dim = 1
            time_dim = hyperparams.quality_assessor_dim
            self.downsampling_layer = nn.AdaptiveAvgPool1d(time_dim)

        if hyperparams.quality_assessor_dim != time_dim:
            raise ValueError("The length of quality assessor must equal to the length of extracted features!")

        if hyperparams.rnn_type == "LSTM":
            rnn = nn.LSTM(input_size = rnn_in_dim, **hyperparams.rnn)
        elif hyperparams.rnn_type == "GRU":
            rnn = nn.GRU(input_size = rnn_in_dim, **hyperparams.rnn)
        else:
            raise ValueError(f"Invalid rnn_type: {hyperparams.rnn_type}. Expected 'LSTM' or 'GRU'")

        self.rnn = ConditionalModel(rnn, config.rnn)

        if config.rnn:
            rnn_out_dim = hyperparams.rnn.hidden_size * (1 + hyperparams.rnn.bidirectional)
            clf_in_dim = rnn_out_dim
        else: 
            clf_in_dim = rnn_in_dim
            self.timeavg_layer = nn.AdaptiveAvgPool1d(1)

        self.dense = Dense(hyperparams.quality_assessor_dim, clf_in_dim)

        self.classifier = nn.Linear(clf_in_dim, hyperparams.classifier.num_classes)

        self.mapping = Mapping()
        
        self.dropout = nn.Dropout(0.5)

    def forward(self, x: Tensor, x_signal_quality_assessor: Tensor = None):
        if self.config.signal_compositor:
            composite_signal = self.composite_signal_generator(x)
            out = composite_signal
        else:
            out = x[:, 0]
        
        if self.config.feature_extractor:
            out = self.feature_extractor(out)
        else:
            out = self.downsampling_layer(out)
            out = out.permute(0, 2, 1)

        if self.config.rnn:
            out, _ = self.rnn(out)

        if self.config.quality_assessor:
            if x_signal_quality_assessor is None:
                raise ValueError("Required inputs for quality assessment are missing.")
            context_vector, attn_weights = self.dense(out, x_signal_quality_assessor)
        elif not self.config.quality_assessor and self.rnn:
            context_vector = out[:, -1, :]  # Use only the last hidden state if attention is not applied
        else:
            context_vector = self.timeavg_layer(out)
        
        context_vector = self.dropout(context_vector)
        # Pass the context vector through the classifier
        logits = self.classifier(context_vector)
        # cluster_vector = self.mapping(context_vector)

        return composite_signal, attn_weights, context_vector, logits

class ModelModule(pl.LightningModule):
    def __init__(
        self,
        task: str = "binary",
        num_classes: int = 2,
        lr: float = 0.0001,
        lr_warmup_ratio: float = 0.1,
        use_lr_scheduler: bool = True,
        loss_name: str = "cross_en",
        use_cmc_loss: bool = False,
        cluster_num: int = 2,
        weighting_stragegy: str = "EW",
        task_num: int = 2,
        device: str = "cuda",
        # switch_loss: int = 5,
        # cmc_alpha_intra: float = 0.1,
        # cmc_alpha_inter: float = 0.9,
        total_training_steps: int = 10000,
        config: DictConfig = None,
        **kwargs
    ):
        super(ModelModule, self).__init__()
        self.save_hyperparameters()
        self.task = task
        self.num_classes = num_classes
        self.lr = lr
        self.warmup_ratio = lr_warmup_ratio
        self.use_lr_scheduler = use_lr_scheduler
        self.loss_name = loss_name
        self.use_cmc_loss = use_cmc_loss
        self.cluster_num = cluster_num
        self.weighting_stragegy = weighting_stragegy
        self.task_num = task_num
        if device == "cuda" and torch.cuda.is_available():
            self.device_type = torch.device("cuda")
        else:
            self.device_type = torch.device("cpu")
        # self.switch_loss = switch_loss
        # self.cmc_alpha_ce = cmc_alpha_ce
        # self.cmc_alpha_intra = cmc_alpha_intra
        # self.cmc_alpha_inter = cmc_alpha_inter
        self.total_steps = total_training_steps
        # self.use_cmc_loss_this_epoch = False
        self.config = config

        # define P for forward and backward loss
        P = np.eye(num_classes)

        self.list_metrics = []
        for metric in self.config.metrics:
            if self.config.metrics[metric]:
                self.list_metrics.append(metric)
        
        print("Loss Function: ", self.loss_name, flush=True)

        print("Metrics: ", self.config.metrics, flush=True)

        # self.task = config.task   # or binary

        self.model = CompleteModel(hyperparams=config.hyperparams, config=config.model)
        # print("Device: ", self.device, flush=True)
        self.weighting_adaptor = Weighting_Adapt(self.weighting_stragegy, self.device_type, self.task_num)
        self.weighting_adaptor.init_param()

        if self.loss_name == "bce":
            self.loss_fn = get_loss_function(self.loss_name)
        elif self.loss_name == 'ce':
            self.loss_fn = get_loss_function(self.loss_name)
        elif self.loss_name == 'sce':
            alpha = 1
            beta = 5
            self.loss_fn = get_loss_function(self.loss_name)(alpha, beta)
        elif self.loss_name == 'lsr':
            self.loss_fn = get_loss_function(self.loss_name)
        elif self.loss_name == 'gce':
            self.loss_fn = get_loss_function(self.loss_name)
        elif self.loss_name == 'jol':
            self.loss_fn = get_loss_function(self.loss_name)
        elif self.loss_name == 'bs':
            self.loss_fn = get_loss_function(self.loss_name)(P)
        elif self.loss_name == 'bh':
            self.loss_fn = get_loss_function(self.loss_name)(P)
        else:
            raise ValueError(f"Invalid loss function: {self.loss_name}")

        self.metrics = nn.ModuleDict({
            "metrics_train": nn.ModuleDict({}),
            "metrics_valid": nn.ModuleDict({}),
            "metrics_test": nn.ModuleDict({})
        })

        for phase in ["train", "valid", "test"]:
            for metric in self.config.metrics:
                if metric == "accuracy":
                    self.metrics["metrics_" + phase][metric] = torchmetrics.Accuracy(
                        self.task, num_classes=self.num_classes, average="none"
                    )
                elif metric == "cf_matrix":
                    self.metrics["metrics_" + phase][metric] = torchmetrics.ConfusionMatrix(
                        self.task, num_classes=self.num_classes
                    )
                elif metric == "f1":
                    self.metrics["metrics_" + phase][metric] = torchmetrics.F1Score(
                        self.task, num_classes=self.num_classes
                    )

        self.step_outputs = {"train": [], "valid": [], "test": []}
        self.step_losses = {"loss_cl": [], "loss_cmc": []}

    def configure_optimizers(self):
        """Setup the optimizers and schedulers for training"""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        if self.use_lr_scheduler:
            scheduler = {
                "scheduler": get_scheduler(
                    "polynomial",
                    optimizer,
                    num_warmup_steps=round(self.warmup_ratio * self.total_steps),
                    num_training_steps=self.total_steps,
                ),
                "interval": "step",
                "frequency": 1,
            }
            return [optimizer], [scheduler]

        return optimizer

    def forward(self, x: Tensor, x_signal_quality_assessor: Tensor = None):
        """Forward pass of the model"""
        return self.model(x, x_signal_quality_assessor)
    
    def plot_confusion_matrix(self, matrix):
        """Convert confusion matrix into a plottable image tensor."""
        fig, ax = plt.subplots()
        cax = ax.matshow(matrix.cpu().numpy())
        fig.colorbar(cax)
    
        buf = io.BytesIO()
        plt.savefig(buf, format='jpeg')
        plt.close(fig)  # ensure to close the figure to free memory
        buf.seek(0)
    
        image = Image.open(buf)
        image_tensor = ToTensor()(image)
    
        return image_tensor

    def log_all(self, items: List[Tuple[str, Union[float, torch.Tensor]]], phase: str = "train", prog_bar: bool = True, sync_dist_group: bool = False):
        for key, value in items:
            if value is not None:
                # Check if value is a float
                if isinstance(value, float):
                    self.log(f"{phase}_{key}", value, prog_bar=prog_bar, sync_dist_group=sync_dist_group)
                # Check if value is a tensor
                elif isinstance(value, torch.Tensor):
                    if len(value.shape) == 0:  # Scalar tensor
                        self.log(f"{phase}_{key}", value, prog_bar=prog_bar, sync_dist_group=sync_dist_group)
                    elif len(value.shape) == 2:  # 2D tensor, assume confusion matrix and log as image
                        image_tensor = self.plot_confusion_matrix(value)
                        self.logger.experiment.add_image(f"{phase}_{key}", image_tensor, global_step=self.current_epoch)

    def update_metrics(self, outputs, targets, phase: str = "train"):
        # model_device =  next(self.model.parameters()).device
        for k in self.config.metrics:
            # metric_device = self.metrics["metrics_" + phase][k].device
            self.metrics["metrics_" + phase][k].update(outputs, targets)

    def reset_metrics(self, phase: str = "train"):
        for k in self.config.metrics:
            self.metrics["metrics_" + phase][k].reset()

    def training_step(self, batch, batch_idx):
        """Training step"""
        signal_image, quality_image, cluster, targets = batch
        _, _, cluster_vector, output_logits = self(signal_image, quality_image)
        preds = torch.argmax(output_logits, dim=1)
        if self.loss_name == "bce":
            loss_cl = self.loss_fn(targets, output_logits)
        elif self.loss_name == 'ce':
            loss_cl = self.loss_fn(targets, output_logits)
        elif self.loss_name == 'sce':
            loss_cl = self.loss_fn(targets, output_logits, self.num_classes)
        elif self.loss_name == 'lsr':
            loss_cl = self.loss_fn(targets, output_logits, self.num_classes)
        elif self.loss_name == 'gce':
            loss_cl = self.loss_fn(targets, output_logits, self.num_classes)
        elif self.loss_name == 'jol':
            loss_cl = self.loss_fn(targets, output_logits, self.num_classes)
        elif self.loss_name == 'bs':
            loss_cl = self.loss_fn(targets, output_logits, self.num_classes)
        elif self.loss_name == 'bh':
            loss_cl = self.loss_fn(targets, output_logits, self.num_classes)
        else:
            raise ValueError(f"Invalid loss function: {self.loss_name}")
        
        # print("Loss printing: ", loss_cl, flush=True)

        if self.use_cmc_loss:
            cluster = cluster.view(-1)
            unique_clusters = torch.unique(cluster)
            num_clusters = len(unique_clusters)

            cluster_vectors = []
            for i in unique_clusters:
                cluster_vectors.append(cluster_vector[cluster == i])
                loss_intra, loss_inter = calculate_cmc_loss(cluster_vectors)
                loss_cmc = loss_intra - loss_inter
                # loss_cmc, weights = self.weighting_adaptor.backward(torch.stack([loss_ce, loss_cmc]))
                loss = loss_cl + loss_cmc
                self.step_losses["loss_cl"].append(loss_cl.item())
                self.step_losses["loss_cmc"].append(loss_cmc.item())
        else:
            loss = loss_cl

            self.update_metrics(preds, targets, "train")
          
        self.step_outputs["train"].append(loss.item())
        return {"loss": loss}

    def on_train_epoch_end(self):
        """End of the training epoch"""
        avg_loss = sum(self.step_outputs["train"]) / len(self.step_outputs["train"])

        acc, matrix, f1 = None, None, None

        if "accuracy" in self.list_metrics:
            acc = self.metrics["metrics_" + "train"]["accuracy"].compute()

        if "cf_matrix" in self.list_metrics:
            matrix = self.metrics["metrics_" + "train"]["cf_matrix"].compute()

        if "f1" in self.list_metrics:
            f1 = self.metrics["metrics_" + "train"]["f1"].compute()
        
        if self.use_cmc_loss:
            avg_loss_cl = sum(self.step_losses["loss_cl"]) / len(self.step_losses["loss_cl"])
            avg_loss_cmc = sum(self.step_losses["loss_cmc"]) / len(self.step_losses["loss_cmc"])
            self.log_all(
                items=[
                    ("loss", avg_loss),
                    ("accuracy", acc),
                    ("confusion_matrix", matrix),
                    ("f1", f1),
                    ("loss_cl", avg_loss_cl),
                    ("loss_cmc", avg_loss_cmc),
                ],
                phase="train",
                prog_bar=True,
                sync_dist_group=False,
            )

            self.step_losses["loss_cl"].clear()
            self.step_losses["loss_cmc"].clear()
        else:
            self.log_all(
                items=[
                    ("loss", avg_loss),
                    ("accuracy", acc),
                    ("confusion_matrix", matrix),
                    ("f1", f1),
                ],
                phase="train",
                prog_bar=True,
                sync_dist_group=False,
            )                     
            
        self.reset_metrics("train")
        self.step_outputs["train"].clear()

    def validation_step(self, batch, batch_idx):
        """Validation step"""
        signal_image, quality_image, targets = batch
        _, _, _, output_logits = self(signal_image, quality_image)
        preds = torch.argmax(output_logits, dim=1)
        loss = F.cross_entropy(output_logits, targets)
        self.update_metrics(preds, targets, "valid")
        
        self.step_outputs["valid"].append(loss.item())
        return {"val_loss": loss}

    def on_validation_epoch_end(self):
        """End of the validation epoch"""
        # avg_loss = torch.stack(self.step_outputs["valid"]).mean()
        avg_loss = sum(self.step_outputs["valid"]) / len(self.step_outputs["valid"])
        acc, matrix, f1 = None, None, None

        if "accuracy" in self.list_metrics:
            acc = self.metrics["metrics_" + "valid"]["accuracy"].compute()

        if "cf_matrix" in self.list_metrics:
            matrix = self.metrics["metrics_" + "valid"]["cf_matrix"].compute()

        if "f1" in self.list_metrics:
            f1 = self.metrics["metrics_" + "valid"]["f1"].compute()

        self.log_all(
            items=[
                ("loss", avg_loss),
                ("accuracy", acc),
                ("confusion_matrix", matrix),
                ("f1", f1),
            ],
            phase="valid",
            prog_bar=True,
            sync_dist_group=False
        )

        self.reset_metrics("valid")
        self.step_outputs["valid"].clear()

    def test_step(self, batch, batch_idx):
        """test step"""
        signal_image, quality_image, cluster, targets = batch
        _, _, _, output_logits = self(signal_image, quality_image)
        preds = torch.argmax(output_logits, dim=1)
        loss = F.cross_entropy(output_logits, targets)
        self.update_metrics(preds, targets, "test")
        
        self.step_outputs["test"].append(loss.item())
        return {"test_loss": loss}

    def on_test_epoch_end(self):
        """End of the test epoch"""
        # avg_loss = torch.stack(self.step_outputs["test"]).mean()
        avg_loss = sum(self.step_outputs["test"]) / len(self.step_outputs["test"])
        acc, matrix, f1 = None, None, None

        if "acc" in self.list_metrics:
            acc = self.metrics["metrics_" + "test"]["accuracy"].compute()

        if "cf_matrix" in self.list_metrics:
            matrix = self.metrics["metrics_" + "test"]["cf_matrix"].compute()

        if "f1" in self.list_metrics:
            f1 = self.metrics["metrics_" + "test"]["f1"].compute()

        self.log_all(
            items=[
                ("loss", avg_loss),
                ("accuracy", acc),
                ("confusion_matrix", matrix),
                ("f1", f1),
            ],
            phase="test",
            prog_bar=True,
            sync_dist_group=False
        )

        self.reset_metrics("test")
        self.step_outputs["test"].clear()

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        """Predicting step"""
        signal_image, quality_image, targets = batch
        # quality_image = quality_image.float()
        outputs = self(signal_image, quality_image)

        return outputs
