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
from dataclasses import dataclass, field
from typing import cast, Iterator, List, Optional, Tuple

import numpy as np

import torch
import torchmetrics as metrics
from fbgemm_gpu.split_embedding_configs import EmbOptimType as OptimType
from pyre_extensions import none_throws
from torch import distributed as dist, nn
from torch.utils.data import DataLoader
from torchrec import EmbeddingBagCollection
from torchrec.datasets.criteo import (
    DEFAULT_CAT_NAMES,
    DEFAULT_INT_NAMES,
    TOTAL_TRAINING_SAMPLES,
)
from torchrec.datasets.utils import Batch
from torchrec.distributed import TrainPipelineSparseDist
from torchrec.distributed.embeddingbag import EmbeddingBagCollectionSharder
from torchrec.distributed.model_parallel import DistributedModelParallel
from torchrec.distributed.types import ModuleSharder
from torchrec.models.dlrm import DLRM, DLRMV2, DLRMTrain
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.optim.keyed import CombinedOptimizer, KeyedOptimizerWrapper
from tqdm import tqdm


# OSS import
try:
    # pyre-ignore[21]
    # @manual=//ai_codesign/benchmarks/dlrm/torchrec_dlrm/data:dlrm_dataloader
    from data.dlrm_dataloader import get_dataloader, STAGES

    # pyre-ignore[21]
    # @manual=//ai_codesign/benchmarks/dlrm/torchrec_dlrm:multi_hot
    from multi_hot import Multihot, RestartableMap
except ImportError:
    pass

# internal import
try:
    from .data.dlrm_dataloader import get_dataloader, STAGES  # noqa F811
    from .multi_hot import Multihot, RestartableMap # noqa F811
except ImportError:
    pass

def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="torchrec dlrm example trainer")
    parser.add_argument(
        "--batch_size", type=int, default=65536, help="batch size to use for training"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="criteo_1t_preprocessed_and_shuffled",
        help="dataset for experiment, current support criteo_1tb, criteo_kaggle",
    )
    parser.add_argument(
        "--num_embeddings_per_feature",
        type=str,
        default=None,
        help="Comma separated max_ind_size per sparse feature. The number of embeddings"
        " in each embedding table. 26 values are expected for the Criteo dataset.",
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
        "--mmap_mode",
        dest="mmap_mode",
        action="store_true",
        help="--mmap_mode mmaps the dataset."
        " That is, the dataset is kept on disk but is accessed as if it were in memory."
        " --mmap_mode is intended mostly for faster debugging. Use --mmap_mode to bypass"
        " preloading the dataset when preloading takes too long or when there is "
        " insufficient memory available to load the full dataset.",
    )
    parser.add_argument(
        "--multi_hot_save_path",
        type=str,
        default=None,
        help="Path to a folder in which to save the multi-hot synthetic dataset generated"
        " by this script.",
    )
    parser.add_argument(
        "--in_memory_binary_criteo_path",
        type=str,
        default=None,
        help="Path to a folder containing the binary (npy) files for the Criteo dataset."
        " When supplied, InMemoryBinaryCriteoIterDataPipe is used.",
    )
    parser.add_argument(
        "--shuffle_batches",
        dest="shuffle_batches",
        action="store_true",
        help="Shuffle each batch during training.",
    )
    parser.set_defaults(
        pin_memory=None,
        mmap_mode=None,
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
        "--multi_hot_distribution_type",
        type=str,
        choices=["uniform", "pareto"],
        default="uniform",
        help="Path to a folder containing the binary (npy) files for the Criteo dataset."
        " When supplied, InMemoryBinaryCriteoIterDataPipe is used.",
    )
    return parser.parse_args(argv)


#def save_multi_hot_dataset_to_disk():
def main(argv: List[str]) -> None:
    """
    Generates & saves multi-hot synthetic data to disk. This data is meant to then be read using
    ./dlrm_main.py

    Args:
        argv (List[str]): command line args.

    Returns:
        None.
    """
    args = parse_args(argv)

    rank = int(os.environ["LOCAL_RANK"])
    if torch.cuda.is_available():
        device: torch.device = torch.device(f"cuda:{rank}")
        backend = "nccl"
        torch.cuda.set_device(device)
    else:
        device: torch.device = torch.device("cpu")
        backend = "gloo"

    if not torch.distributed.is_initialized():
        dist.init_process_group(backend=backend)

    if args.num_embeddings_per_feature is not None:
        args.num_embeddings_per_feature = list(
            map(int, args.num_embeddings_per_feature.split(","))
        )
        args.num_embeddings = None

    # We use the same dataloader used for training, because it iterates over the
    # dataset 1 time. We save each sample's corresponding multi-hot encoding each iteration.
    # No training actuall happens. We just use the same dataloader
    # I think this is wrong? We also need to save the validation dataloader and test dataloader?
    train_dataloader = get_dataloader(args, backend, "train")

    if 1 >= args.multi_hot_size:
        raise ValueError(f"CLI option --multi_hot_size must be greater than 1.")

    multihot = Multihot(
        args.multi_hot_size,
        args.multi_hot_min_table_size,
        args.num_embeddings_per_feature,
        args.batch_size,
        collect_freqs_stats=False,
        dist_type=args.multi_hot_distribution_type,
    )
    train_dataloader = RestartableMap(multihot.convert_to_multi_hot, train_dataloader)



    """
    Format

    Save dataset_meta_data.npy
    String: "Dataset name"
    Int: batch_size
    List: ln_emb = [ table 1 row count, table 2 row count etc] ?
    String: "distribution_type"
    Int: pooling_factor_per_table_l = [ t1=1, t2=10, t3=10]
    Store offset record once.
    Store sparse indices, continuous values, and row number?
    """


    train_dataloader = get_dataloader(args, backend, "train")
    train_dataloader = RestartableMap(multihot.convert_to_multi_hot, train_dataloader)

    # Save info on dataset to dict in npy file.
    batch = next(iter(train_dataloader))
    #sparse_offsets = batch.sparse_features._offsets.reshape(-1, args.batch_size).tranpose(0,1) # <-- don't try to reduce size by reshaping. keep it simple for now until it's working.
    info_dict = {}
    info_dict["offset"] = batch.sparse_features._offsets
    info_dict["batch_size"] = args.batch_size
    info_dict["num_embeddings_per_feature"] = np.array(args.num_embeddings_per_feature)
    info_dict["dataset_name"] = args.dataset_name
    info_dict["multi_hot_distribution_type"] = args.multi_hot_distribution_type
    info_dict["random_seed_used"] = args.seed

    print(f"Saving START")
    np.save(args.multi_hot_save_path + "/multi_hot_encoded_dataset_info.npy", info_dict)
    print(f"Saving DONE")

    dataset_iter = iter(train_dataloader)
    for it in tqdm(itertools.count(), desc=f"Save multi-hot encoded categorical data to disk"):
        try:
            batch = next(dataset_iter)
            sparse_values = batch.sparse_features._values

            # NOTE: USE ORIGINAL CONTINUOUS NPY FILES AND LABELS FILES FOR MULTI-HOT DATASET?
            # sparse_offsets = batch.sparse_features._offsets
            # dense_values = batch.dense_features
            # labels = batch.labels
            start_sample = it * args.batch_size
            last_sample = start_sample + args.batch_size - 1
            #file_name = args.multi_hot_save_path + f"/multi_hot_encoded_samples_{start_sample}_to_{last_sample}"
            file_name = args.multi_hot_save_path + f"/multi_hot/rank_{rank}_batch_{it}.npy"
            print(f"Saving {file_name} START")
            np.save( file_name, sparse_values )
            print(f"Saving {file_name} DONE")
            #np.save(args.multi_hot_save_path + f"multi_hot_encoded_samples_{start_sample}_to_{last_sample}", sparse_values )
            #np.save(args.save_path + f"multi_hot_encoded_samples_{it * args.batch_size}_to_{(it+1) * args.batch_size - 1}", sparse_values )
        except StopIteration:
            break
        # return Batch(
        #     dense_features=batch.dense_features,
        #     sparse_features=new_sparse_features,
        #     labels=batch.labels,
        # )


        # save_each_batch_to_npy_file? Name the files by batch iteration number?

if __name__ == "__main__":

    sys.argv = ["synthetic_multi_hot.py",
        #"--batch_size", "65536",
        "--batch_size", "2048",
        "--dataset_name", "criteo_1t_preprocessed_and_shuffled",
        "--num_embeddings_per_feature", "45833188,36746,17245,7413,20243,3,7114,1441,62,29275261,1572176,345138,10,2209,11267,128,4,974,14,48937457,11316796,40094537,452104,12606,104,35",
        # "--seed",
        "--pin_memory",
        "--mmap_mode",
        "--multi_hot_save_path", "/home/ubuntu/mountpoint/shuffled",
        "--in_memory_binary_criteo_path", "/home/ubuntu/mountpoint/shuffled",
        "--multi_hot_size", "20",
        "--multi_hot_min_table_size", "200",
        "--multi_hot_distribution_type", "uniform",]

    main(sys.argv[1:])

# Run this script with:
# torchx run -s local_cwd dist.ddp -j 1x8 --script synthetic_multi_hot.py
#
# GET SINGLE THREADED VERSION WORKING FIRST.
# Single threaded version
# torchx run -s local_cwd dist.ddp -j 1x1 --script synthetic_multi_hot.py
#
# To delete:
# rm /home/ubuntu/mountpoint/shuffled/multi_hot_encoded*