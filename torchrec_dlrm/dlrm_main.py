#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import itertools
import os
import sys
from typing import cast, Iterator, List, Optional, Dict, Any

import torch
#import torchmetrics as metrics
import torchmetrics
import sklearn.metrics
import numpy as np
from pyre_extensions import none_throws
from torch import nn, distributed as dist
from torch.autograd.profiler import record_function
from torch.utils.data import DataLoader
from torchrec import EmbeddingBagCollection
from torchrec.datasets.criteo import DEFAULT_CAT_NAMES, DEFAULT_INT_NAMES
from torchrec.datasets.utils import Batch
from torchrec.distributed import TrainPipelineSparseDist
from torchrec.distributed.embeddingbag import EmbeddingBagCollectionSharder
from torchrec.distributed.model_parallel import DistributedModelParallel
from torchrec.distributed.types import ModuleSharder
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.optim.keyed import CombinedOptimizer, KeyedOptimizerWrapper
from tqdm import tqdm
from fbgemm_gpu.split_embedding_configs import EmbOptimType as OptimType

from torch.utils.tensorboard import SummaryWriter

import pathlib
from os import fspath
p = pathlib.Path(__file__).absolute().parents[1].resolve()
sys.path.append(fspath(p))
import mlperf_logger

# OSS import
try:
    # pyre-ignore[21]
    # @manual=//torchrec/github/examples/dlrm/data:dlrm_dataloader
    from data.dlrm_dataloader import get_dataloader, STAGES

    # pyre-ignore[21]
    # @manual=//torchrec/github/examples/dlrm/modules:dlrm_train
    from modules.dlrm_train import DLRMTrain
    from multi_hot import multihot
    from multi_hot import multihot_uniform
except ImportError:
    pass

# internal import
try:
    from .data.dlrm_dataloader import (  # noqa F811
        get_dataloader,
        STAGES,
    )
    from .modules.dlrm_train import DLRMTrain  # noqa F811
except ImportError:
    pass

TRAIN_PIPELINE_STAGES = 3  # Number of stages in TrainPipelineSparseDist.

from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.embeddingbag import EmbeddingBagCollectionSharder
from torchrec.distributed.types import ModuleSharder, ShardingType
import torch.nn as nn

class TestEBCSharder(EmbeddingBagCollectionSharder):
    def __init__(
        self,
        #sharding_type: str,
        #kernel_type: str,
        fused_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        if fused_params is None:
            fused_params = {}
        # self._sharding_type = sharding_type
        # self._kernel_type = kernel_type
        self._fused_params = fused_params
        if True:
            self._sharding_type = [
                #ShardingType.DATA_PARALLEL.value,
                ShardingType.TABLE_WISE.value,
                #ShardingType.COLUMN_WISE.value,
                #ShardingType.ROW_WISE.value,
                #ShardingType.TABLE_ROW_WISE.value,
                #ShardingType.TABLE_COLUMN_WISE.value,
                ]
            self._kernel_type = [
                #EmbeddingComputeKernel.DENSE.value,
                #EmbeddingComputeKernel.SPARSE.value,
                #EmbeddingComputeKernel.BATCHED_DENSE.value,
                #EmbeddingComputeKernel.BATCHED_FUSED.value,
                #EmbeddingComputeKernel.BATCHED_FUSED_UVM.value,
                EmbeddingComputeKernel.BATCHED_FUSED_UVM_CACHING.value,
                #EmbeddingComputeKernel.BATCHED_QUANT.value,
            ]
    """
    Restricts sharding to single type only.
    """

    def sharding_types(self, compute_device_type: str) -> List[str]:
        return self._sharding_type

    """
    Restricts to single impl.
    """

    def compute_kernels(
        self, sharding_type: str, compute_device_type: str
    ) -> List[str]:
        return self._kernel_type

    @property
    def fused_params(self) -> Optional[Dict[str, Any]]:
        return self._fused_params

def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="torchrec dlrm example trainer")
    parser.add_argument(
        "--epochs", type=int, default=1, help="number of epochs to train"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="batch size to use for training"
    )
    parser.add_argument(
        "--limit_train_batches",
        type=int,
        default=None,
        help="number of train batches",
    )
    parser.add_argument(
        "--limit_val_batches",
        type=int,
        default=None,
        help="number of validation batches",
    )
    parser.add_argument(
        "--limit_test_batches",
        type=int,
        default=None,
        help="number of test batches",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="criteo_1t",
        help="dataset for experiment, current support criteo_1tb, criteo_kaggle",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="number of dataloader workers",
    )
    parser.add_argument(
        "--num_embeddings",
        type=int,
        default=100_000,
        help="max_ind_size. The number of embeddings in each embedding table. Defaults"
        " to 100_000 if num_embeddings_per_feature is not supplied.",
    )
    parser.add_argument(
        "--num_embeddings_per_feature",
        type=str,
        default=None,
        help="Comma separated max_ind_size per sparse feature. The number of embeddings"
        " in each embedding table. 26 values are expected for the Criteo dataset.",
    )
    parser.add_argument(
        "--dense_arch_layer_sizes",
        type=str,
        default="512,256,64",
        help="Comma separated layer sizes for dense arch.",
    )
    parser.add_argument(
        "--over_arch_layer_sizes",
        type=str,
        default="512,512,256,1",
        help="Comma separated layer sizes for over arch.",
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=64,
        help="Size of each embedding.",
    )
    parser.add_argument(
        "--undersampling_rate",
        type=float,
        help="Desired proportion of zero-labeled samples to retain (i.e. undersampling zero-labeled rows)."
        " Ex. 0.3 indicates only 30pct of the rows with label 0 will be kept."
        " All rows with label 1 will be kept. Value should be between 0 and 1."
        " When not supplied, no undersampling occurs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--pin_memory",
        dest="pin_memory",
        action="store_true",
        help="Use pinned memory when loading data.",
    )
    parser.add_argument(
        "--in_memory_binary_criteo_path",
        type=str,
        default=None,
        help="Path to a folder containing the binary (npy) files for the Criteo dataset."
        " When supplied, InMemoryBinaryCriteoIterDataPipe is used.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=15.0,
        help="Learning rate.",
    )
    parser.add_argument(
        "--shuffle_batches",
        type=bool,
        default=False,
        help="Shuffle each batch during training.",
    )
    parser.add_argument(
        "--validation_freq_within_epoch",
        type=int,
        default=None,
        help="Frequency at which validation will be run within an epoch.",
    )
    parser.add_argument(
        "--adagrad",
        dest="adagrad",
        action="store_true",
        help="Flag to determine if adagrad optimizer should be used.",
    )
    parser.add_argument(
        "--collect_freqs_stats",
        dest="collect_freqs_stats",
        action="store_true",
        help="Flag to determine whether to collect stats on freq of embedding access..",
    )
    parser.add_argument(
        "--tensor_board_filename",
        type=str,
        default="tensorboard_file",
        help="Tensorboard file that will store AUROC information to display in tensorboard.",
    )
    parser.add_argument(
       "--mlperf_logging",
        action="store_true",
        default=False
    )
    parser.add_argument(
        "--multi_hot_size",
        type=int,
        default=1,
        help="The number of Multi-hot indices to use. When 1, multi-hot is disabled.",
    )
    parser.add_argument(
        "--multi_hot_min_table_size",
        type=int,
        default=200,
        help="The minimum number of rows an embedding table must have to run multi-hot inputs.",
    )
    parser.add_argument(
        "--interaction_branch1_layer_sizes",
        type=str,
        default="",
        help="Comma separated layer sizes for interaction branch1.",
    )
    parser.add_argument(
        "--interaction_branch2_layer_sizes",
        type=str,
        default="",
        help="Comma separated layer sizes for interaction branch2.",
    )
    parser.set_defaults(pin_memory=None)
    return parser.parse_args(argv)


def _evaluate(
    args: argparse.Namespace,
    train_pipeline: TrainPipelineSparseDist,
    iterator: Iterator[Batch],
    next_iterator: Iterator[Batch],
    stage: str,
    writer: SummaryWriter,
    log_iter: int = None,

) -> None:
    """
    Evaluate model. Computes and prints metrics including AUROC and Accuracy. Helper
    function for train_val_test.

    Args:
        args (argparse.Namespace): parsed command line args.
        train_pipeline (TrainPipelineSparseDist): pipelined model.
        iterator (Iterator[Batch]): Iterator used for val/test batches.
        next_iterator (Iterator[Batch]): Iterator used for the next phase (either train
            if there are more epochs to train on or test if all epochs are complete).
            Used to queue up the next TRAIN_PIPELINE_STAGES - 1 batches before
            train_val_test switches to the next phase. This is done so that when the
            next phase starts, the first output train_pipeline generates an output for
            is the 1st batch for that phase.
        stage (str): "val" or "test".

    Returns:
        None.
    """
    if log_iter == None:
        return
    model = train_pipeline._model
    model.eval()
    device = train_pipeline._device
    limit_batches = (
        args.limit_val_batches if stage == "val" else args.limit_test_batches
    )
    if limit_batches is not None:
        limit_batches -= TRAIN_PIPELINE_STAGES - 1

    # Because TrainPipelineSparseDist buffer batches internally, we load in
    # TRAIN_PIPELINE_STAGES - 1 batches from the next_iterator into the buffers so that
    # when train_val_test switches to the next phase, train_pipeline will start
    # producing results for the TRAIN_PIPELINE_STAGES - 1 buffered batches (as opposed
    # to the last TRAIN_PIPELINE_STAGES - 1 batches from iterator).
    combined_iterator = itertools.chain(
        iterator
        if limit_batches is None
        else itertools.islice(iterator, limit_batches),
        itertools.islice(next_iterator, TRAIN_PIPELINE_STAGES - 1),
    )
    auroc = torchmetrics.AUROC(compute_on_step=False).to(device)
    accuracy = torchmetrics.Accuracy(compute_on_step=False).to(device)


    if args.mlperf_logging:
        scores = []
        targets = []


    # Infinite iterator instead of while-loop to leverage tqdm progress bar.
    for _ in tqdm(iter(int, 1), desc=f"Evaluating {stage} set"):
        try:
            _loss, logits, labels = train_pipeline.progress(combined_iterator)
            nn_output = torch.sigmoid(logits)
            labels = labels.int()
            auroc(nn_output, labels)
            accuracy(nn_output, labels)

            if args.mlperf_logging:
                Z_test = nn_output
                T_test = labels
                S_test = Z_test.detach().cpu().numpy()  # numpy array
                T_test = T_test.detach().cpu().numpy()  # numpy array
                scores.append(S_test)
                targets.append(T_test)

        except StopIteration:
            break
    auroc_result = auroc.compute().item()
    accuracy_result = accuracy.compute().item()
    if dist.get_rank() == 0:
        print(f"AUROC over {stage} set: {auroc_result}.")
        print(f"Accuracy over {stage} set: {accuracy_result}.")
        writer.add_scalar("Test/Acc", accuracy_result, log_iter)
        writer.add_scalar("AUROC", auroc_result, log_iter)

    if args.mlperf_logging:
        with record_function("DLRM mlperf sklearn metrics compute"):
            scores = np.concatenate(scores, axis=0)
            targets = np.concatenate(targets, axis=0)

            metrics = {
                "recall": lambda y_true, y_score: sklearn.metrics.recall_score(
                    y_true=y_true, y_pred=np.round(y_score)
                ),
                "precision": lambda y_true, y_score: sklearn.metrics.precision_score(
                    y_true=y_true, y_pred=np.round(y_score)
                ),
                "f1": lambda y_true, y_score: sklearn.metrics.f1_score(
                    y_true=y_true, y_pred=np.round(y_score)
                ),
                "ap": sklearn.metrics.average_precision_score,
                "roc_auc": sklearn.metrics.roc_auc_score,
                "accuracy": lambda y_true, y_score: sklearn.metrics.accuracy_score(
                    y_true=y_true, y_pred=np.round(y_score)
                ),
            }

        validation_results = {}
        for metric_name, metric_function in metrics.items():
            validation_results[metric_name] = metric_function(targets, scores)
            writer.add_scalar(
                "mlperf-metrics-test/" + metric_name,
                validation_results[metric_name],
                log_iter,
            )

    model.train(True)


def _train(
    args: argparse.Namespace,
    train_pipeline: TrainPipelineSparseDist,
    iterator: Iterator[Batch],
    next_iterator: Iterator[Batch],
    within_epoch_val_dataloader: DataLoader,
    epoch: int,
    writer: SummaryWriter,
    multihot_hash: multihot_uniform,
) -> None:
    """
    Train model for 1 epoch. Helper function for train_val_test.

    Args:
        args (argparse.Namespace): parsed command line args.
        train_pipeline (TrainPipelineSparseDist): pipelined model.
        iterator (Iterator[Batch]): Iterator used for training batches.
        next_iterator (Iterator[Batch]): Iterator used for validation batches. Used to
            queue up the next TRAIN_PIPELINE_STAGES - 1 batches before train_val_test
            switches to validation mode. This is done so that when validation starts,
            the first output train_pipeline generates an output for is the 1st
            validation batch (as opposed to a buffered train batch).
        epoch (int): Which epoch the model is being trained on.

    Returns:
        None.
    """
    train_pipeline._model.train()

    limit_batches = args.limit_train_batches
    # For the first epoch, train_pipeline has no buffered batches, but for all other
    # epochs, train_pipeline will have TRAIN_PIPELINE_STAGES - 1 from iterator already
    # present in its buffer.
    if limit_batches is not None and epoch > 0:
        limit_batches -= TRAIN_PIPELINE_STAGES - 1

    # Because TrainPipelineSparseDist buffer batches internally, we load in
    # TRAIN_PIPELINE_STAGES - 1 batches from the next_iterator into the buffers so that
    # when train_val_test switches to the next phase, train_pipeline will start
    # producing results for the TRAIN_PIPELINE_STAGES - 1 buffered batches (as opposed
    # to the last TRAIN_PIPELINE_STAGES - 1 batches from iterator).
    combined_iterator = itertools.chain(
        iterator
        if args.limit_train_batches is None
        else itertools.islice(iterator, limit_batches),
        itertools.islice(next_iterator, TRAIN_PIPELINE_STAGES - 1),
    )

    # Infinite iterator instead of while-loop to leverage tqdm progress bar.
    it = 0
    for _ in tqdm(iter(int, 1), desc=f"Epoch {epoch}"):
        try:
            _loss, logits, labels = train_pipeline.progress(combined_iterator)
            writer.add_scalar("Train/Loss", _loss, it)
            if (
                args.validation_freq_within_epoch
                and it > 0
                and it % args.validation_freq_within_epoch == 0
            ):
                if multihot_hash.collect_freqs_stats:
                    multihot_hash.collect_freqs_stats_temp_disable = True
                _evaluate(
                    args,
                    train_pipeline,
                    iter(within_epoch_val_dataloader),
                    iterator,
                    "val",
                    writer,
                    it
                )
                if multihot_hash.collect_freqs_stats:
                    multihot_hash.collect_freqs_stats_temp_disable = False
            it += 1
        except StopIteration:
            break
    if multihot_hash.collect_freqs_stats:
        multihot_hash.collect_freqs_stats_temp_disable = False
    return it

def train_val_test(
    args: argparse.Namespace,
    train_pipeline: TrainPipelineSparseDist,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    test_dataloader: DataLoader,
    multihot_hash: multihot_uniform
) -> None:
    """
    Train/validation/test loop. Contains customized logic to ensure each dataloader's
    batches are used for the correct designated purpose (train, val, test). This logic
    is necessary because TrainPipelineSparseDist buffers batches internally (so we
    avoid batches designated for one purpose like training getting buffered and used for
    another purpose like validation).

    Args:
        args (argparse.Namespace): parsed command line args.
        train_pipeline (TrainPipelineSparseDist): pipelined model.
        train_dataloader (DataLoader): DataLoader used for training.
        val_dataloader (DataLoader): DataLoader used for validation.
        test_dataloader (DataLoader): DataLoader used for testing.

    Returns:
        None.
    """
    global writer
    tb_file = "./" + args.tensor_board_filename
    #try:
    #    shutil.rmtree(tb_file)
    #except:
    #    pass
    writer = SummaryWriter(tb_file)
    if args.mlperf_logging:
        mlperf_logger.log_event(key=mlperf_logger.constants.CACHE_CLEAR, value=True)
        mlperf_logger.log_start(
            key=mlperf_logger.constants.INIT_START, log_all_ranks=True
        )
    if args.mlperf_logging:
        mlperf_logger.barrier()
        mlperf_logger.log_end(key=mlperf_logger.constants.INIT_STOP)
        mlperf_logger.barrier()
        mlperf_logger.log_start(key=mlperf_logger.constants.RUN_START)
        mlperf_logger.barrier()
    if args.mlperf_logging:
        mlperf_logger.mlperf_submission_log("dlrm")
        mlperf_logger.log_event(
            key=mlperf_logger.constants.SEED, value=0 #int(args.seed if args.seed is not None else 0)
        )
        mlperf_logger.log_event(
            key=mlperf_logger.constants.GLOBAL_BATCH_SIZE, value=args.batch_size
        )
    if args.mlperf_logging:
        # LR is logged twice for now because of a compliance checker bug
        mlperf_logger.log_event(
            key=mlperf_logger.constants.OPT_BASE_LR, value=args.learning_rate
        )
        mlperf_logger.log_event(
            key=mlperf_logger.constants.OPT_LR_WARMUP_STEPS,
            value=0,
        )

        # use logging keys from the official HP table and not from the logging library
        mlperf_logger.log_event(
            key="sgd_opt_base_learning_rate", value=args.learning_rate
        )
        mlperf_logger.log_event(
            key="lr_decay_start_steps", value=0
        )
        mlperf_logger.log_event(
            key="sgd_opt_learning_rate_decay_steps", value=0
        )
        mlperf_logger.log_event(key="sgd_opt_learning_rate_decay_poly_power", value=2)

    train_iterator = iter(train_dataloader)
    test_iterator = iter(test_dataloader)
    for epoch in range(args.epochs):

        if args.mlperf_logging:
            mlperf_logger.barrier()
            mlperf_logger.log_start(
                key=mlperf_logger.constants.BLOCK_START,
                metadata={
                    mlperf_logger.constants.FIRST_EPOCH_NUM: (epoch+1),
                    mlperf_logger.constants.EPOCH_COUNT: 1,
                },
            )
            mlperf_logger.barrier()
            mlperf_logger.log_start(
                key=mlperf_logger.constants.EPOCH_START,
                metadata={mlperf_logger.constants.EPOCH_NUM: (epoch+1)},
            )

        val_iterator = iter(val_dataloader)
        it = _train(
            args, train_pipeline, train_iterator, val_iterator, val_dataloader, epoch, writer, multihot_hash
        )
        train_iterator = iter(train_dataloader)
        val_next_iterator = (
            test_iterator if epoch == args.epochs - 1 else train_iterator
        )

        if args.mlperf_logging:
            mlperf_logger.barrier()
            mlperf_logger.log_start(
                key=mlperf_logger.constants.EVAL_START,
                metadata={
                    mlperf_logger.constants.EPOCH_NUM: epoch+1
                },
            )
        it = _evaluate(args, train_pipeline, val_iterator, val_next_iterator, "val", writer, it)

        if args.mlperf_logging:
            mlperf_logger.barrier()
            mlperf_logger.log_end(
                key=mlperf_logger.constants.EVAL_STOP,
                metadata={
                    mlperf_logger.constants.EPOCH_NUM: epoch+1
                },
            )

    it = _evaluate(args, train_pipeline, test_iterator, iter(test_dataloader), "test", writer, it)


def main(argv: List[str]) -> None:
    """
    Trains, validates, and tests a Deep Learning Recommendation Model (DLRM)
    (https://arxiv.org/abs/1906.00091). The DLRM model contains both data parallel
    components (e.g. multi-layer perceptrons & interaction arch) and model parallel
    components (e.g. embedding tables). The DLRM model is pipelined so that dataloading,
    data-parallel to model-parallel comms, and forward/backward are overlapped. Can be
    run with either a random dataloader or an in-memory Criteo 1 TB click logs dataset
    (https://ailab.criteo.com/download-criteo-1tb-click-logs-dataset/).

    Args:
        argv (List[str]): command line args.

    Returns:
        None.
    """
    args = parse_args(argv)

    rank = int(os.environ["LOCAL_RANK"])
    if rank == 0:
        print(argv)

    if torch.cuda.is_available():
        device: torch.device = torch.device(f"cuda:{rank}")
        backend = "nccl"
        torch.cuda.set_device(device)
    else:
        device: torch.device = torch.device("cpu")
        backend = "gloo"

    if not torch.distributed.is_initialized():
        dist.init_process_group(backend=backend)

    print(f"My local rank is: {rank}. \t My global rank is: {dist.get_rank()}")

    if args.num_embeddings_per_feature is not None:
        args.num_embeddings_per_feature = list(
            map(int, args.num_embeddings_per_feature.split(","))
        )
        args.num_embeddings = None


    # Sets default limits for random dataloader iterations when left unspecified.
    if args.in_memory_binary_criteo_path is None:
        # pyre-ignore[16]
        for stage in STAGES:
            attr = f"limit_{stage}_batches"
            if getattr(args, attr) is None:
                setattr(args, attr, 10)

    eb_configs = [
        EmbeddingBagConfig(
            name=f"t_{feature_name}",
            embedding_dim=args.embedding_dim,
            num_embeddings=none_throws(args.num_embeddings_per_feature)[feature_idx]
            if args.num_embeddings is None
            else args.num_embeddings,
            feature_names=[feature_name],
        )
        for feature_idx, feature_name in enumerate(DEFAULT_CAT_NAMES)
    ]
    sharded_module_kwargs = {}
    if args.over_arch_layer_sizes is not None:
        sharded_module_kwargs["over_arch_layer_sizes"] = list(
            map(int, args.over_arch_layer_sizes.split(","))
        )

    train_model = DLRMTrain(
        embedding_bag_collection=EmbeddingBagCollection(
            tables=eb_configs, device=torch.device("meta")
        ),
        dense_in_features=len(DEFAULT_INT_NAMES),
        dense_arch_layer_sizes=list(map(int, args.dense_arch_layer_sizes.split(","))),
        over_arch_layer_sizes=list(map(int, args.over_arch_layer_sizes.split(","))),
        interaction_branch1_layer_sizes=None
        if args.interaction_branch1_layer_sizes == ""
        else list(map(int, args.interaction_branch1_layer_sizes.split(","))),
        interaction_branch2_layer_sizes=None
        if args.interaction_branch2_layer_sizes == ""
        else list(map(int, args.interaction_branch2_layer_sizes.split(","))),
        dense_device=device,
    )
    fused_params = {
        "learning_rate": args.learning_rate,
        "optimizer": OptimType.EXACT_ROWWISE_ADAGRAD if args.adagrad else OptimType.EXACT_SGD,
    }

    if True:
        sharders = TestEBCSharder(
                        fused_params=fused_params,
                    )

        model = DistributedModelParallel(
            module=train_model,
            device=device,
            sharders=[
                cast(
                    ModuleSharder[torch.nn.Module],
                    sharders,
                )
            ],
        )
    else:
        sharders = [
            EmbeddingBagCollectionSharder(fused_params=fused_params),
        ]
        model = DistributedModelParallel(
            module=train_model,
            device=device,
            sharders=cast(List[ModuleSharder[nn.Module]], sharders),
        )

    def optimizer_with_params():
        if args.adagrad:
            return lambda params: torch.optim.Adagrad(params, lr=args.learning_rate)
        else:
            return lambda params: torch.optim.SGD(params, lr=args.learning_rate)

    dense_optimizer = KeyedOptimizerWrapper(
        dict(model.named_parameters()),
        optimizer_with_params(),
    )
    optimizer = CombinedOptimizer([model.fused_optimizer, dense_optimizer])

    if rank == 0:
        for collectionkey, plans in model._plan.plan.items():
            print(collectionkey)
            for key, plan in plans.items():
                print(key, "\n", plan, "\n")

    train_pipeline = TrainPipelineSparseDist(
        model,
        optimizer,
        device,
    )

    train_dataloader = get_dataloader(args, backend, "train")
    val_dataloader = get_dataloader(args, backend, "val")
    test_dataloader = get_dataloader(args, backend, "test")

    if 1 < args.multi_hot_size:
        #m = multihot(args.multi_hot_size, args.num_embeddings_per_feature, args.batch_size)
        m = multihot_uniform(
            args.multi_hot_size,
            args.multi_hot_min_table_size,
            args.num_embeddings_per_feature,
            args.batch_size,
            collect_freqs_stats = args.collect_freqs_stats,
        )
        train_dataloader = map(m.convert_to_multi_hot, train_dataloader)
        val_dataloader = map(m.convert_to_multi_hot, val_dataloader)
        test_dataloader = map(m.convert_to_multi_hot, test_dataloader)

    train_val_test(
        args, train_pipeline, train_dataloader, val_dataloader, test_dataloader, m
    )

    if 1 < args.multi_hot_size and m.collect_freqs_stats:
        m.save_freqs_stats()

if __name__ == "__main__":
    main(sys.argv[1:])
