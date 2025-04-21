import torch
from torchmetrics import Accuracy
from torchmetrics.aggregation import MeanMetric
from torchmetrics.functional import pairwise_cosine_similarity
import time
import pandas as pd
import wandb
import re
import sys
import os
import copy
import random
from pathlib import Path
from rashomon_sets.protorset_factory import ProtoRSetFactory, DEFAULT_RSET_ARGS
from protopnet.datasets import training_dataloaders
from protopnet.weights_and_biases import run_report_on_sweeps
from tqdm import tqdm
from cvpr_experiments import (
    get_path_to_best_model_by_dataset_backbone,
    run_eval_epoch
)

def run_train_epoch(model, dataset, num_classes, l2_coef=0.0001, required_indices=[], epochs=10, early_stopping_thresh=1e-6):
    model.train()

    optim = torch.optim.SGD(model.prototype_prediction_head.class_connection_layer.parameters(), lr=0.001)
    
    acc = Accuracy(task="multiclass", num_classes=num_classes).to("cuda" if torch.cuda.is_available() else "cpu")
    prev_loss = None
    for epoch in range(epochs):
            
        X = torch.tensor(dataset.values[:, :-1], device="cuda" if torch.cuda.is_available() else "cpu")
        y = torch.tensor(dataset.values[:, -1], device="cuda" if torch.cuda.is_available() else "cpu").long()
        output = model.prototype_prediction_head(X)
            
        acc.update(target=y, preds=output["logits"])
        
        optim.zero_grad()

        cur_loss = torch.nn.functional.cross_entropy(output["logits"], y)
        if len(required_indices) > 0:
            required_entry_loss = torch.norm(
                torch.stack([model.prototype_prediction_head.class_connection_layer.weight[
                    torch.argmax(model.prototype_prediction_head.prototype_class_identity[v, :]).item(), 
                    v
                ] for v in required_indices]),
                p=1
            )
            cur_loss = cur_loss - required_entry_loss

        cur_loss.backward()
        optim.step()

        if prev_loss is None:
            prev_loss = cur_loss.item()
        elif abs(prev_loss - cur_loss.item()) <= early_stopping_thresh:
            print(f"Breaking on iter {epoch}")
            break
        else:
            prev_loss = cur_loss.item()
        

        del cur_loss

    return acc.compute().item(), \
        (prev_loss + l2_coef * torch.norm(model.prototype_prediction_head.class_connection_layer.weight.data)).item()

def sample_many_models(
    n_iters=10, 
    save_loc_root=Path("/usr/xtmp/jcd97/proto-rset/results/"), 
    target_config_index=None
):
    os.makedirs(save_loc_root, exist_ok=True)
    dataset_and_backbone_to_model = get_path_to_best_model_by_dataset_backbone()
    print(f"dataset_and_backbone_to_model: {dataset_and_backbone_to_model}")

    rset_args = DEFAULT_RSET_ARGS
    rset_args["max_iter"] = 5_000
    rset_args["directly_compute_hessian"] = True
    rset_args["rashomon_bound_multiplier"] = 1.1
    rset_args["lam"] = 0.0001
    rset_args["reg"] = "l2"
    rset_args["lr_for_opt"] = 1
    rset_args["num_models"] = 0
    rset_args["opt_tol"] = 1e-7
    additional_prototypes_to_sample = 0
    additional_prototypes_to_sample_ham = 0
    last_layer_retrain_epochs = rset_args["max_iter"]
    requirement_thresh = 1.0

    # coefs_list will store all sampled coefs, used to
    # generate a similarity heatmap
    coefs_list = []

    config_index = 0
    for dataset in dataset_and_backbone_to_model:
        if "." in dataset:
            print(f"Skipping {dataset}")
            continue
        if "HAM" in dataset:
            print(f"Skipping {dataset}")
            continue
            additional_prototypes_to_sample = additional_prototypes_to_sample_ham
        for backbone in dataset_and_backbone_to_model[dataset]:
            if target_config_index is not None and config_index != target_config_index:
                config_index += 1
                continue

            results = pd.DataFrame({
                "dataset": [],
                "backbone": [],
                "val_acc": [],
                "train_acc": [],
                "train_loss": [],
                "val_loss": [],
                "sample_time": [],
                "coef_similarity_to_orig": [],
                "is_original": [],
                "time_to_get_rset": [],
                "delta_train_accuracy": [],
                "delta_val_accuracy": [],
                "delta_train_loss": [],
                "delta_val_loss": [],
                "num_prototypes": [],
                "indices": [],
                "required_protos_avg_coef": [],
                "delta_test_loss": [],
                "delta_test_accuracy": [],
                "test_acc": [],
                "test_loss": [],
            })

            print(f"======== Running dataset {dataset} backbone {backbone} ==========")
            model = dataset_and_backbone_to_model[dataset][backbone][0]
            original_ppn = torch.load(model, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

            batch_sizes = {"train": 20, "project": 20, "val": 20}
            split_dataloaders = training_dataloaders(dataset, batch_sizes=batch_sizes, part_labels=False)
            split_dataloaders_test = training_dataloaders(dataset, batch_sizes=batch_sizes, val_dir="test", part_labels=False)

            train_loader = split_dataloaders.train_loader_no_aug
            val_loader = split_dataloaders.val_loader
            test_loader = split_dataloaders_test.val_loader

            start = time.time()
            factory = ProtoRSetFactory(
                split_dataloaders,
                Path(model),
                rashomon_set_args=rset_args,
                correct_class_connections_only=True,
                device=torch.device("cuda"),
                additional_prototypes_to_sample=additional_prototypes_to_sample
            )
            cur_time_to_get_rset = time.time() - start

            train_similarities = factory._build_similarities_dataset(train_loader)
            val_similarities = factory._build_similarities_dataset(val_loader)
            test_similarities = factory._build_similarities_dataset(test_loader)

            original_ppn = copy.deepcopy(factory.initial_protopnet)
            original_ppn.prune_duplicate_prototypes()
            original_coef = original_ppn.prototype_prediction_head.class_connection_layer.weight.data
            requirement_thresh = original_coef[original_coef != 0].mean().item()
            print(f"requirement_thresh: {requirement_thresh}")

            torch.save(original_coef, save_loc_root / f"original_coef.pth")

            og_train_accuracy, og_train_loss = run_eval_epoch(original_ppn, train_similarities, split_dataloaders.num_classes, l2_coef=rset_args["lam"])
            og_val_accuracy, og_val_loss = run_eval_epoch(original_ppn, val_similarities, split_dataloaders.num_classes, l2_coef=rset_args["lam"])
            og_test_accuracy, og_test_loss = run_eval_epoch(original_ppn, test_similarities, split_dataloaders.num_classes, l2_coef=rset_args["lam"])

            results = pd.concat([results, pd.DataFrame({
                "dataset": [dataset],
                "backbone": [backbone],
                "val_acc": [og_val_accuracy],
                "test_acc": [og_test_accuracy],
                "train_acc": [og_train_accuracy],
                "train_loss": [og_train_loss],
                "val_loss": [og_val_loss],
                "test_loss": [og_test_loss],
                "sample_time": [0],
                "coef_similarity_to_orig": [1],
                "model_type": ["original"],
                "time_to_get_rset": [cur_time_to_get_rset],
                "delta_train_accuracy": [0],
                "delta_val_accuracy": [0],
                "delta_test_accuracy": [0],
                "delta_train_loss": [0],
                "delta_val_loss": [0],
                "delta_test_loss": [0],
                "num_prototypes": [original_ppn.prototype_layer.num_prototypes],
                "indices": [-1],
                "required_protos_avg_coef": [-1]
            })], ignore_index=True)

            naive_ppn = copy.deepcopy(original_ppn)
            naive_adjusted_ppn = copy.deepcopy(original_ppn)

            required_indices = []
            required_indices_viable = []
            for i in range(n_iters):
                print(f"Iter {i}")

                # Find a prototype that can be required
                if i > 0:
                    can_require = False
                    proto_to_require = random.randint(0, original_ppn.prototype_layer.num_prototypes - 1)
                    viable_proto_to_require = proto_to_require
                    for _ in range(1_000):
                        start = time.time()
                        # try:
                        ppn = factory.produce_protopnet_object_with_requirements(
                            [(v, requirement_thresh) for v in required_indices_viable + [viable_proto_to_require]]
                        )
                        # except BaseException as e:
                            # print(f"ERROR: {e}")
                            # can_require = False
                        end = time.time()
                        if ppn is not None:
                            can_require = True
                            required_indices_viable.append(viable_proto_to_require)
                            break
                        else:
                            viable_proto_to_require = random.randint(0, original_ppn.prototype_layer.num_prototypes - 1)
                    
                    # If there are none left, stop early
                    if not can_require:
                        print("Failed to require prototype")
                        break
                else:
                    start = time.time()
                    ppn = factory.produce_protopnet_object_with_requirements(
                        []
                    )
                    end = time.time()

                # ====== Evaluate a model produced using the RSet
                sample_time = end - start

                all_coef = ppn.prototype_prediction_head.class_connection_layer.weight.data
                if i > 0:
                    required_protos_avg_coef = torch.mean(torch.tensor([
                        torch.abs(all_coef[
                            torch.argmax(ppn.prototype_prediction_head.prototype_class_identity[v, :]).item(), 
                            v
                        ]).item() for v in required_indices_viable#required_indices
                    ])).item()
                else:
                    required_protos_avg_coef = 0

                train_acc_cur, train_loss_cur = run_eval_epoch(ppn, train_similarities, split_dataloaders.num_classes, l2_coef=rset_args["lam"])
                val_acc_cur, val_loss_cur = run_eval_epoch(ppn, val_similarities, split_dataloaders.num_classes, l2_coef=rset_args["lam"])
                test_acc_cur, test_loss_cur = run_eval_epoch(ppn, test_similarities, split_dataloaders.num_classes, l2_coef=rset_args["lam"])

                results = pd.concat([results, pd.DataFrame({
                    "dataset": [dataset],
                    "backbone": [backbone],
                    "val_acc": [val_acc_cur],
                    "train_acc": [train_acc_cur],
                    "train_loss": [train_loss_cur],
                    "val_loss": [val_loss_cur],
                    "sample_time": [sample_time],
                    "coef_similarity_to_orig": [torch.nn.functional.cosine_similarity(original_coef.flatten(), all_coef.flatten(), dim=-1).item()],
                    "model_type": ["rset"],
                    "time_to_get_rset": [cur_time_to_get_rset],
                    "delta_train_accuracy": [train_acc_cur - og_train_accuracy],
                    "delta_val_accuracy": [val_acc_cur - og_val_accuracy],
                    "delta_train_loss": [train_loss_cur - og_train_loss],
                    "delta_val_loss": [val_loss_cur - og_val_loss],
                    "delta_test_loss": [test_loss_cur - og_test_loss],
                    "delta_test_accuracy": [test_acc_cur - og_test_accuracy],
                    "test_acc": [test_acc_cur],
                    "test_loss": [test_loss_cur],
                    "num_prototypes": [original_ppn.prototype_layer.num_prototypes],
                    "required_protos_avg_coef": [required_protos_avg_coef],
                    "indices": [i]
                })], ignore_index=True)
                os.makedirs(save_loc_root / f"iter_{i}/", exist_ok=True)
                torch.save(all_coef, save_loc_root / f"iter_{i}/rset_coef.pth")

                coefs_list.append(all_coef.flatten())

                # ====== Evaluate naive removal of this prototype
                start = time.time()
                if i > 0:
                    #required_indices.append(proto_to_require)
                    for req_v in required_indices_viable:
                        active_ind = torch.argmax(naive_ppn.prototype_prediction_head.class_connection_layer.weight.data[:, req_v])
                        naive_ppn.prototype_prediction_head.class_connection_layer.weight.data[active_ind, req_v] = torch.clamp(
                            naive_ppn.prototype_prediction_head.class_connection_layer.weight.data[active_ind, req_v],
                            min=requirement_thresh
                        )
                sample_time = time.time() - start

                all_coef = naive_ppn.prototype_prediction_head.class_connection_layer.weight.data
                if i > 0:
                    required_protos_avg_coef = torch.mean(torch.tensor([
                        torch.abs(all_coef[
                            torch.argmax(naive_ppn.prototype_prediction_head.prototype_class_identity[v, :]).item(), 
                            v
                        ]).item() for v in required_indices_viable
                    ])).item()
                else:
                    required_protos_avg_coef = 0

                train_acc_cur, train_loss_cur = run_eval_epoch(naive_ppn, train_similarities, split_dataloaders.num_classes)
                val_acc_cur, val_loss_cur = run_eval_epoch(naive_ppn, val_similarities, split_dataloaders.num_classes)
                test_acc_cur, test_loss_cur = run_eval_epoch(naive_ppn, test_similarities, split_dataloaders.num_classes)

                results = pd.concat([results, pd.DataFrame({
                    "dataset": [dataset],
                    "backbone": [backbone],
                    "val_acc": [val_acc_cur],
                    "train_acc": [train_acc_cur],
                    "train_loss": [train_loss_cur],
                    "val_loss": [val_loss_cur],
                    "sample_time": [sample_time],
                    "coef_similarity_to_orig": [torch.nn.functional.cosine_similarity(original_coef.flatten(), all_coef.flatten(), dim=-1).item()],
                    "model_type": ["naive_removal"],
                    "time_to_get_rset": [cur_time_to_get_rset],
                    "delta_train_accuracy": [train_acc_cur - og_train_accuracy],
                    "delta_val_accuracy": [val_acc_cur - og_val_accuracy],
                    "delta_train_loss": [train_loss_cur - og_train_loss],
                    "delta_val_loss": [val_loss_cur - og_val_loss],
                    "delta_test_loss": [test_loss_cur - og_test_loss],
                    "delta_test_accuracy": [test_acc_cur - og_test_accuracy],
                    "test_acc": [test_acc_cur],
                    "test_loss": [test_loss_cur],
                    "num_prototypes": [original_ppn.prototype_layer.num_prototypes],
                    "required_protos_avg_coef": [required_protos_avg_coef],
                    "indices": [i]
                })], ignore_index=True)
                torch.save(all_coef, save_loc_root / f"iter_{i}/naive_coef.pth")

                # ====== Evaluate naive removal of this prototype with reoptimization
                start = time.time()
                if i > 0:
                    for req_v in required_indices_viable:
                        active_ind = torch.argmax(naive_adjusted_ppn.prototype_prediction_head.class_connection_layer.weight.data[:, req_v])
                        naive_adjusted_ppn.prototype_prediction_head.class_connection_layer.weight.data[active_ind, req_v] = torch.clamp(
                            naive_adjusted_ppn.prototype_prediction_head.class_connection_layer.weight.data[active_ind, req_v],
                            min=requirement_thresh
                        )
                    run_train_epoch(
                        naive_adjusted_ppn, 
                        train_similarities, 
                        split_dataloaders.num_classes, 
                        epochs=last_layer_retrain_epochs, 
                        required_indices=required_indices_viable, 
                        early_stopping_thresh=rset_args["opt_tol"]
                    )
                sample_time = time.time() - start

                all_coef = naive_adjusted_ppn.prototype_prediction_head.class_connection_layer.weight.data
                if i > 0:
                    required_protos_avg_coef = torch.mean(torch.tensor([
                        torch.abs(all_coef[
                            torch.argmax(naive_adjusted_ppn.prototype_prediction_head.prototype_class_identity[v, :]).item(), 
                            v
                        ]).item() for v in required_indices_viable
                    ])).item()
                else:
                    required_protos_avg_coef = 0

                train_acc_cur, train_loss_cur = run_eval_epoch(naive_adjusted_ppn, train_similarities, split_dataloaders.num_classes)
                val_acc_cur, val_loss_cur = run_eval_epoch(naive_adjusted_ppn, val_similarities, split_dataloaders.num_classes)
                test_acc_cur, test_loss_cur = run_eval_epoch(naive_adjusted_ppn, test_similarities, split_dataloaders.num_classes)

                results = pd.concat([results, pd.DataFrame({
                    "dataset": [dataset],
                    "backbone": [backbone],
                    "val_acc": [val_acc_cur],
                    "train_acc": [train_acc_cur],
                    "train_loss": [train_loss_cur],
                    "val_loss": [val_loss_cur],
                    "sample_time": [sample_time],
                    "coef_similarity_to_orig": [torch.nn.functional.cosine_similarity(original_coef.flatten(), all_coef.flatten(), dim=-1).item()],
                    "model_type": ["naive_removal_retrained"],
                    "time_to_get_rset": [cur_time_to_get_rset],
                    "delta_train_accuracy": [train_acc_cur - og_train_accuracy],
                    "delta_val_accuracy": [val_acc_cur - og_val_accuracy],
                    "delta_train_loss": [train_loss_cur - og_train_loss],
                    "delta_val_loss": [val_loss_cur - og_val_loss],
                    "delta_test_loss": [test_loss_cur - og_test_loss],
                    "delta_test_accuracy": [test_acc_cur - og_test_accuracy],
                    "test_acc": [test_acc_cur],
                    "test_loss": [test_loss_cur],
                    "num_prototypes": [original_ppn.prototype_layer.num_prototypes],
                    "required_protos_avg_coef": [required_protos_avg_coef],
                    "indices": [i]
                })], ignore_index=True)
                torch.save(all_coef, save_loc_root / f"iter_{i}/naive_retrained_coef.pth")
                torch.save(torch.tensor(required_indices_viable), save_loc_root / f"iter_{i}/required_indices.pth")

                results.to_csv(save_loc_root / "summary.csv", index=False)
            config_index += 1

def main():
    slurm_id = int(sys.argv[1])
    torch.manual_seed(0)
    random.seed(0)
    sample_many_models(
        n_iters=100,
        save_loc_root=Path(f"/usr/xtmp/jcd97/proto-rset/results/require/configuration_{slurm_id}_with_test/"),
        target_config_index=slurm_id
    )

if __name__ == '__main__':
    main()