import json
import os
import pickle
import random
import hashlib
from copy import deepcopy
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf

from src.utils.constants import DatasetConfig, DatasetName, SplitMethod  # Import the structured config

from data.utils.schemes.flower import flower_partition
from src.utils.tools import fix_random_seed
from data.utils.process import (
    exclude_domain,
    plot_distribution,
    prune_args,
    generate_synthetic_data,
    process_celeba,
    process_femnist,
)
from data.utils.schemes import (
    dirichlet,
    iid_partition,
    randomly_assign_classes,
    allocate_shards,
    semantic_partition,
)
from data.utils.datasets import DATASETS, BaseDataset

@hydra.main(config_path="config", config_name="dataset_default", version_base=None)
def main(cfg: DictConfig):
    # Adjust paths to be relative to the original script location
    CURRENT_DIR = Path(__file__).parent.absolute()
    dataset_root = CURRENT_DIR / "data" / cfg.dataset

    pruned_args = prune_args(cfg)
    split_argument_hash = hashlib.md5(json.dumps(OmegaConf.to_container(cfg, resolve=True)).encode()).hexdigest()

    if cfg.prepare:
        split_root = CURRENT_DIR / "data" / cfg.dataset / split_argument_hash
        split_root.mkdir(parents=True, exist_ok=True)
    else:
        split_root = CURRENT_DIR / "data" / cfg.dataset

    fix_random_seed(cfg.seed, cfg.use_cuda)

    if not dataset_root.is_dir():
        dataset_root.mkdir(parents=True, exist_ok=True)

    partition_md5_path = split_root / "partition_md5.txt"
    args_dict = OmegaConf.to_container(cfg, resolve=True)
    args_json = json.dumps(args_dict)
    args_md5 = hashlib.md5(args_json.encode()).hexdigest()

    if partition_md5_path.is_file():
        with open(partition_md5_path, "r") as f:
            existing_md5 = f.read().strip()
            if existing_md5 == args_md5:
                print("Partition file already exists. Skip partitioning.")
                return split_argument_hash

    client_num = cfg.client_num
    partition = {"separation": None, "data_indices": [[] for _ in range(client_num)]}
    # x: num of samples,
    # y: label distribution
    stats = {i: {"x": 0, "y": {}} for i in range(client_num)}
    dataset: BaseDataset = None

    if cfg.dataset == DatasetName.femnist:
        dataset = process_femnist(cfg, partition, stats)
        partition["val"] = []
    elif cfg.dataset == DatasetName.celeba:
        dataset = process_celeba(cfg, partition, stats)
        partition["val"] = []
    elif cfg.dataset == DatasetName.synthetic:
        dataset = generate_synthetic_data(cfg, partition, stats)
    else:  # MEDMNIST, COVID, MNIST, CIFAR10, ...
        dataset = DATASETS[cfg.dataset](dataset_root, cfg)
        targets = np.array(dataset.targets, dtype=np.int32)
        target_indices = np.arange(len(targets), dtype=np.int32)
        valid_label_set = set(range(len(dataset.classes)))
        if cfg.dataset == DatasetName.domain and cfg.ood_domains:
            metadata = json.load(open(dataset_root / "metadata.json", "r"))
            valid_label_set, targets, client_num = exclude_domain(
                client_num=client_num,
                domain_map=metadata["domain_map"],
                targets=targets,
                domain_indices_bound=metadata["domain_indices_bound"],
                ood_domains=set(cfg.ood_domains),
                partition=partition,
                stats=stats,
            )

        iid_data_partition = deepcopy(partition)
        iid_stats = deepcopy(stats)
        if 0 < cfg.iid <= 1.0:  # iid partition
            sampled_indices = np.array(
                random.sample(
                    target_indices.tolist(), int(len(target_indices) * cfg.iid)
                )
            )

            # if cfg.iid < 1.0, then residual indices will be processed by another partition method
            target_indices = np.array(
                list(set(target_indices) - set(sampled_indices)), dtype=np.int32
            )

            iid_partition(
                targets=targets[sampled_indices],
                target_indices=sampled_indices,
                label_set=valid_label_set,
                client_num=client_num,
                partition=iid_data_partition,
                stats=iid_stats,
            )

        if len(target_indices) > 0:
            if cfg.alpha > 0:  # Dirichlet(alpha)
                dirichlet(
                    targets=targets[target_indices],
                    target_indices=target_indices,
                    label_set=valid_label_set,
                    client_num=client_num,
                    alpha=cfg.alpha,
                    min_samples_per_client=cfg.min_samples_per_client,
                    partition=partition,
                    stats=stats,
                )
            elif cfg.classes != 0:  # randomly assign classes
                cfg.classes = max(1, min(cfg.classes, len(dataset.classes)))
                randomly_assign_classes(
                    targets=targets[target_indices],
                    target_indices=target_indices,
                    label_set=valid_label_set,
                    client_num=client_num,
                    class_num=cfg.classes,
                    partition=partition,
                    stats=stats,
                )
            elif cfg.shards > 0:  # allocate shards
                allocate_shards(
                    targets=targets[target_indices],
                    target_indices=target_indices,
                    label_set=valid_label_set,
                    client_num=client_num,
                    shard_num=cfg.shards,
                    partition=partition,
                    stats=stats,
                )
            elif cfg.semantic:
                semantic_partition(
                    dataset=dataset,
                    targets=targets[target_indices],
                    target_indices=target_indices,
                    label_set=valid_label_set,
                    efficient_net_type=cfg.efficient_net_type,
                    client_num=client_num,
                    pca_components=cfg.pca_components,
                    gmm_max_iter=cfg.gmm_max_iter,
                    gmm_init_params=cfg.gmm_init_params,
                    seed=cfg.seed,
                    use_cuda=cfg.use_cuda,
                    partition=partition,
                    stats=stats,
                )
            elif cfg.flower_partitioner_class != "":
                flower_partition(
                    targets=targets[target_indices],
                    target_indices=target_indices,
                    label_set=valid_label_set,
                    client_num=client_num,
                    flower_partitioner_class=cfg.flower_partitioner_class,
                    flower_partitioner_kwargs=OmegaConf.to_container(cfg.flower_partitioner_kwargs),
                    partition=partition,
                    stats=stats,
                )
            elif cfg.dataset == DatasetName.domain and cfg.ood_domains is None:
                with open(dataset_root / "original_partition.pkl", "rb") as f:
                    partition = {}
                    partition["data_indices"] = pickle.load(f)
                    partition["separation"] = None
                    cfg.client_num = len(partition["data_indices"])
            else:
                raise RuntimeError(
                    "Part of data indices are ignored. Please set arbitrary one arg from"
                    " [--alpha, --classes, --shards, --semantic] for partitioning."
                )

    # merge the iid and niid partition results
    if 0 < cfg.iid <= 1.0:
        num_samples = []
        for i in range(cfg.client_num):
            partition["data_indices"][i] = np.concatenate(
                [partition["data_indices"][i], iid_data_partition["data_indices"][i]]
            ).astype(np.int32)
            stats[i]["x"] += iid_stats[i]["x"]
            stats[i]["y"] = {
                cls: stats[i]["y"].get(cls, 0) + iid_stats[i]["y"].get(cls, 0)
                for cls in dataset.classes
            }
            num_samples.append(stats[i]["x"])
        num_samples = np.array(num_samples)
        stats["samples_per_client"] = {
            "mean": num_samples.mean().item(),
            "stddev": num_samples.std().item(),
        }

    if partition["separation"] is None:
        if cfg.split == SplitMethod.USER.value:
            test_clients_num = int(cfg.client_num * cfg.test_ratio)
            val_clients_num = int(cfg.client_num * cfg.val_ratio)
            train_clients_num = cfg.client_num - test_clients_num - val_clients_num
            clients_4_train = list(range(train_clients_num))
            clients_4_val = list(
                range(train_clients_num, train_clients_num + val_clients_num)
            )
            clients_4_test = list(
                range(train_clients_num + val_clients_num, cfg.client_num)
            )

        elif cfg.split == SplitMethod.SAMPLE.value:
            clients_4_train = list(range(cfg.client_num))
            clients_4_val = clients_4_train
            clients_4_test = clients_4_train

        partition["separation"] = {
            "train": clients_4_train,
            "val": clients_4_val,
            "test": clients_4_test,
            "total": cfg.client_num,
        }

    if cfg.dataset not in [DatasetName.femnist, DatasetName.celeba]:
        if cfg.split == SplitMethod.SAMPLE:
            for client_id in partition["separation"]["train"]:
                indices = partition["data_indices"][client_id]
                np.random.shuffle(indices)
                testset_size = int(len(indices) * cfg.test_ratio)
                valset_size = int(len(indices) * cfg.val_ratio)
                trainset, valset, testset = (
                    indices[testset_size + valset_size :],
                    indices[testset_size : testset_size + valset_size],
                    indices[:testset_size],
                )
                partition["data_indices"][client_id] = {
                    "train": trainset,
                    "val": valset,
                    "test": testset,
                }
        elif cfg.split == SplitMethod.USER:
            for client_id in partition["separation"]["train"]:
                indices = partition["data_indices"][client_id]
                partition["data_indices"][client_id] = {
                    "train": np.array([], dtype=np.int64),
                    "val": np.array([], dtype=np.int64),
                    "test": np.array([], dtype=np.int64),
                }

            for client_id in partition["separation"]["val"]:
                indices = partition["data_indices"][client_id]
                partition["data_indices"][client_id] = {
                    "train": np.array([], dtype=np.int64),
                    "val": indices,
                    "test": np.array([], dtype=np.int64),
                }

            for client_id in partition["separation"]["test"]:
                indices = partition["data_indices"][client_id]
                partition["data_indices"][client_id] = {
                    "train": np.array([], dtype=np.int64),
                    "val": np.array([], dtype=np.int64),
                    "test": indices,
                }

    if cfg.dataset == DatasetName.domain:
        class_targets = np.array(dataset.targets, dtype=np.int32)
        metadata = json.load(open(dataset_root / "metadata.json", "r"))

        def _idx_2_domain_label(index):
            for domain, bound in metadata["domain_indices_bound"].items():
                if bound["begin"] <= index < bound["end"]:
                    return metadata["domain_map"][domain]

        domain_targets = np.vectorize(_idx_2_domain_label)(
            np.arange(len(class_targets), dtype=np.int64)
        )
        for client_id in range(cfg.client_num):
            indices = np.concatenate(
                [
                    partition["data_indices"][client_id].get("train", []),
                    partition["data_indices"][client_id].get("val", []),
                    partition["data_indices"][client_id].get("test", []),
                ]
            ).astype(np.int64)
            stats[client_id] = {
                "x": len(indices),
                "class space": Counter(class_targets[indices].tolist()),
                "domain space": Counter(domain_targets[indices].tolist()),
            }
        stats["domain_map"] = metadata["domain_map"]

    # plot
    if cfg.plot_distribution:
        if cfg.dataset == DatasetName.domain:
            # class distribution
            counts = np.zeros((len(dataset.classes), cfg.client_num), dtype=np.int64)
            client_ids = range(cfg.client_num)
            for i, client_id in enumerate(client_ids):
                for j, cnt in stats[client_id]["class space"].items():
                    counts[j][i] = cnt
            plot_distribution(
                client_num=cfg.client_num,
                label_counts=counts,
                save_path=f"{split_root}/class_distribution.png",
            )
            # domain distribution
            counts = np.zeros(
                (len(metadata["domain_map"]), cfg.client_num), dtype=np.int64
            )
            client_ids = range(cfg.client_num)
            for i, client_id in enumerate(client_ids):
                for j, cnt in stats[client_id]["domain space"].items():
                    counts[j][i] = cnt
            plot_distribution(
                client_num=cfg.client_num,
                label_counts=counts,
                save_path=f"{split_root}/domain_distribution.png",
            )

        else:
            counts = np.zeros((len(dataset.classes), cfg.client_num), dtype=np.int64)
            client_ids = range(cfg.client_num)
            for i, client_id in enumerate(client_ids):
                for j, cnt in stats[client_id]["y"].items():
                    counts[j][i] = cnt
            plot_distribution(
                client_num=cfg.client_num,
                label_counts=counts,
                save_path=f"{split_root}/class_distribution.png",
            )

    with open(split_root / "partition.pkl", "wb") as f:
        pickle.dump(partition, f)

    with open(split_root / "all_stats.json", "w") as f:
        json.dump(stats, f, indent=4)

    with open(split_root / "args.json", "w") as f:
        json.dump(pruned_args, f, indent=4)

    with open(split_root / "partition_md5.txt", "w") as f:
        f.write(split_argument_hash)

    return split_argument_hash

if __name__ == "__main__":
    main()


