########################################################################################
#
# Implementation of a prototype model for speaker recognition.
#
# It consists of a single CNN layer which convolutes the input spectrogram,
# a pooling layer which reduces the CNN output to a fixed-size speaker embedding,
# and a single FC classification layer.
#
# Author(s): Nik Vaessen
########################################################################################

from typing import Optional, Tuple, List

import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim

from pytorch_lightning import LightningModule
from torchmetrics import Accuracy

from skeleton.evaluation.evaluation import (
    EmbeddingSample,
    EvaluationPair,
    evaluate_speaker_trials,
)
from skeleton.layers.resnet import ResNet

from skeleton.layers.statistical_pooling import MeanStatPool1D

########################################################################################
# Implement the lightning module for training a prototype model
# for speaker recognition on tiny-voxceleb


class PrototypeSpeakerRecognitionModule(LightningModule):
    def __init__(
        self,
        num_inp_features: int,
        num_embedding: int,
        num_speakers: int,
        learning_rate: float,
        val_trials: Optional[List[EvaluationPair]] = None,
        test_trials: Optional[List[EvaluationPair]] = None,
    ):
        super().__init__()

        # hyperparameters
        self.num_inp_features = num_inp_features
        self.num_embedding = num_embedding
        self.num_speakers = num_speakers
        self.learning_rate = learning_rate
        self.original_lr = learning_rate

        # evaluation data
        self.val_trials = val_trials
        self.test_trials = test_trials

        # 1-dimensional convolution layer which transforms the spectrogram
        # of shape [BATCH_SIZE, NUM_MEL, NUM_FRAMES]
        # into embedding of shape [BATCH_SIZE, NUM_EMBEDDING, REDUCED_NUM_FRAMES]
        self.embedding_layer = nn.Sequential(
            nn.Conv1d(
                in_channels=num_inp_features,
                out_channels=128,
                kernel_size=3,
                stride=1,
            ),
            nn.ReLU(),
        )

        self.resnet = ResNet(((num_embedding, 2, num_embedding*2),(num_embedding*2, 2, num_embedding*4), (num_embedding*4, 2, num_embedding*8), (num_embedding*8, 2, num_embedding*16)))

        # Pooling layer
        # assuming input of shape [BATCH_SIZE, NUM_EMBEDDING, REDUCED_NUM_FRAMES]
        # reduced to shape [BATCH_SIZE, NUM_EMBEDDING]
        self.pooling_layer = MeanStatPool1D(dim_to_reduce=2)

        # Fully-connected layer which is responsible for transforming the
        # speaker embedding of shape [BATCH_SIZE, NUM_EMBEDDING] into a
        # speaker prediction of shape [BATCH_SIZE, NUM_SPEAKERS]
        self.prediction_layer = nn.Sequential(
            nn.Linear(in_features=num_embedding, out_features=num_speakers),
            nn.LogSoftmax(dim=1)
        )

        # The loss function. Be careful - some loss functions apply the (log)softmax
        # layer internally (e.g F.cross_entropy) while others do not
        # (e.g F.nll_loss)
        self.loss_fn = F.nll_loss

        # used to keep track of training/validation accuracy
        self.train_acc = Accuracy(task="multiclass", num_classes=self.num_speakers)
        self.val_acc = Accuracy(task="multiclass", num_classes=self.num_speakers)

        # save hyperparameters for easy reloading of model
        self.save_hyperparameters()

    def forward(self, spectrogram: t.Tensor) -> Tuple[t.Tensor, t.Tensor]:
        # we split the forward pass into 2 phases:

        # first compute the speaker embeddings based on the spectrogram:
        speaker_embedding = self.compute_embedding(spectrogram)

        # then compute the speaker prediction probabilities based on the
        # embedding
        speaker_prediction = self.compute_prediction(speaker_embedding)

        return speaker_embedding, speaker_prediction

    def compute_embedding(self, spectrogram: t.Tensor) -> t.Tensor:
        # modify to your liking!
        feature_representation = self.embedding_layer(spectrogram) # -> [128,128,239]
        resnet_output = self.resnet(feature_representation)


        resnet_output = resnet_output[:, :, None] # -> ([128, 128, 1])

        embedding = self.pooling_layer(resnet_output) # -> [128, 128]    
        return embedding

    def compute_prediction(self, embedding: t.Tensor) -> t.Tensor:
        # modify to your liking!
        # embedding = embedding[None, :, :]
        prediction = self.prediction_layer(embedding)
        # print(prediction.shape)
        return prediction


    # @property
    # def automatic_optimization(self) -> bool:
    #     return False

    def training_step(
        self, batch: Tuple[List[str], t.Tensor, t.Tensor], *args, **kwargs
    ) -> t.Tensor:
        # first unwrap the batch into the input tensor and ground truth labels
        # opt = self.optimizers()
        sample_id, network_input, speaker_labels = batch
        # opt = self.optimizers()
        # opt.zero_grad()
        assert network_input.shape[0] == speaker_labels.shape[0]
        assert network_input.shape[1] == self.num_inp_features
        assert len(network_input.shape) == 3

        # then compute the forward pass
        embedding, prediction = self.forward(network_input)

        # based on the output of the forward pass we compute the loss
        loss = self.loss_fn(prediction, speaker_labels)
        # self.manual_backward(loss)
        # opt.step()
        # based on the output of the forward pass we compute some metrics
        self.train_acc(prediction, speaker_labels)

        # log training loss
        self.log("loss", loss, prog_bar=False)

        # The value we return will be minimized
        return loss

    def training_epoch_end(self, outputs: List[t.Tensor]) -> None:
        # at the end of a training epoch we log our metrics
        self.log("train_acc", self.train_acc, prog_bar=True)

    def validation_step(
        self, batch: Tuple[List[str], t.Tensor, t.Tensor], *args, **kwargs
    ) -> Tuple[t.Tensor, t.Tensor, List[str]]:
        # first unwrap the batch into the input tensor and ground truth labels
        sample_id, network_input, speaker_labels = batch

        assert network_input.shape[0] == speaker_labels.shape[0]
        assert network_input.shape[1] == self.num_inp_features
        assert len(network_input.shape) == 3

        # then compute the forward pass
        embedding, prediction = self.forward(network_input)

        # based on the output of the forward pass we compute the loss
        loss = self.loss_fn(prediction, speaker_labels)

        # based on the output of the forward pass we compute some metrics
        self.val_acc(prediction, speaker_labels)

        # The value(s) we return will be saved until the end op the epoch
        # and passed to `validation_epoch_end`,
        # We move the embeddings to CPU to prevent taking up space on the GPU
        # for next batch(es)
        return embedding.to("cpu"), loss, sample_id

    def validation_epoch_end(
        self, outputs: List[Tuple[t.Tensor, t.Tensor, List[str]]]
    ) -> None:
        # at the end of a validation epoch we compute the validation EER
        # based on the embeddings and log all metrics

        # unwrap outputs
        embeddings = [embedding for embedding, _, _ in outputs]
        losses = [loss for _, loss, _ in outputs]
        sample_keys = [key for _, _, key in outputs]

        # log metrics
        self.log("val_acc", self.val_acc, prog_bar=True)
        self.log("val_loss", t.mean(t.stack(losses)), prog_bar=True)

        # compute an`d` log val EER
        if self.val_trials is not None:
            val_eer = self._evaluate(embeddings, sample_keys, self.val_trials)
            val_eer = t.tensor(val_eer, dtype=t.float32)

            self.log("val_eer", val_eer, prog_bar=True)

    def test_step(
        self, batch: Tuple[List[str], t.Tensor, t.Tensor], *args, **kwargs
    ) -> Tuple[t.Tensor, List[str]]:
        # first unwrap the batch into the input tensor and ground truth labels
        sample_id, network_input, speaker_labels = batch

        assert network_input.shape[0] == 1
        assert network_input.shape[1] == self.num_inp_features
        assert len(network_input.shape) == 3

        # then compute the speaker embedding
        embedding = self.compute_embedding(network_input)

        # The value(s) we return will be saved until the end op the epoch
        # and passed to `test_epoch_end`.
        # We move the embeddings to CPU to prevent taking up space on the GPU
        # for next batch(es)
        return embedding.to("cpu"), sample_id

    def test_epoch_end(self, outputs: List[t.Tensor]) -> None:
        # at the end of the test epoch we compute the test EER
        if self.test_trials is None:
            return

        # unwrap outputs
        embeddings = [embedding for embedding, _ in outputs]
        sample_keys = [key for _, key in outputs]

        # compute test EER
        test_eer = self._evaluate(embeddings, sample_keys, self.test_trials)

        # log EER
        self.log("test_eer", test_eer)

    def configure_optimizers(self):
        # setup the optimization algorithm
        optimizer = t.optim.Adam(self.parameters(), self.learning_rate)

        # setup the learning rate schedule.
        # Here StepLR acts as a constant lr.
        # Adapt schedule to your liking :).
        schedule = {
            # Required: the scheduler instance.
            "scheduler": t.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8),
            # The unit of the scheduler's step size, could also be 'step'.
            # 'epoch' updates the scheduler on epoch end whereas 'step'
            # updates it after an optimizer update.
            "interval": "epoch",
            # How many epochs/steps should pass between calls to
            # `scheduler.step()`. 1 corresponds to updating the learning
            # rate after every epoch/step.
            "frequency": 1,
            # Metric to monitor for schedulers like `ReduceLROnPlateau`
            "monitor": "val_loss",
            # If set to `True`, will enforce that the value specified 'monitor'
            # is available when the scheduler is updated, thus stopping
            # training if not found. If set to `False`, it will only produce a warning
            "strict": True,
            # If using the `LearningRateMonitor` callback to monitor the
            # learning rate progress, this keyword can be used to specify
            # a custom logged name
            "name": None,
        }

        return [optimizer], [schedule]

    def _evaluate(
        self,
        embeddings: List[t.Tensor],
        sample_keys: List[List[str]],
        pairs: List[EvaluationPair],
    ):
        # construct a list of embedding samples
        assert len(embeddings) == len(sample_keys)
        embedding_list: List[EmbeddingSample] = []

        for embedding_tensor, key_list in zip(embeddings, sample_keys):
            if len(key_list) != embedding_tensor.shape[0]:
                raise ValueError("batch dimension is missing or incorrect")

            assert len(embedding_tensor.shape) == 2
            assert embedding_tensor.shape[0] == len(key_list)

            # we have to loop over the batch dimension and access each embedding
            for idx, sample_id in enumerate(key_list):
                embedding_list.append(
                    # make sure embedding is 1-dimensional, and move to CPU to avoid OOM
                    EmbeddingSample(
                        sample_id, embedding_tensor[idx, :].squeeze().to("cpu")
                    )
                )

        # evaluate the embeddings based on the trial list of pairs
        result = evaluate_speaker_trials(
            trials=pairs, embeddings=embedding_list, skip_eer=False
        )

        return result["eer"]
