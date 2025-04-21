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
import shutil
import random
from pathlib import Path
from rashomon_sets.protorset_factory import ProtoRSetFactory, DEFAULT_RSET_ARGS
from protopnet.datasets import training_dataloaders
from protopnet.weights_and_biases import run_report_on_sweeps
from experiments.protodebug_train_wrapper import run_proto_debug
from tqdm import tqdm
import cv2
from matplotlib import pyplot as plt

def get_path_to_best_model_by_dataset_backbone():
    dataset_and_backbone_to_model = {}
    
    def extract_dataset(row):
        return row['name'].split('-')[-1]

    sweeps = run_report_on_sweeps(True)
    sweeps['dataset'] = sweeps.apply(extract_dataset, axis=1)
    api = wandb.Api()

    for run_id in sweeps['best_run_id']:
        run = api.run(f"duke-interp/proto-rset/{run_id}")

        cur_sweep = sweeps[sweeps['best_run_id'] == run.id]

        try:
            logs = run.file("output.log").download(replace=True).readlines()
        except:
            continue
        save_entries = [log for log in logs if "Saving model with" in log]

        if len(save_entries) > 0:
            save_entries = save_entries[-1]  # Get the last (best) save of the run
        else:
            save_entries = ""

        pattern = (
            r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d{3} INFO.*?to (/.+\.pth)"
        )
        match = re.search(pattern, save_entries)
        if match:
            timestamp, save_path = match.groups()
        else:
            timestamp, save_path = None, None

        if cur_sweep['dataset'].values[0] in dataset_and_backbone_to_model:
            dataset_and_backbone_to_model[cur_sweep['dataset'].values[0]][cur_sweep['backbone'].values[0]] = (save_path, run)
        else:
            dataset_and_backbone_to_model[cur_sweep['dataset'].values[0]] = {cur_sweep['backbone'].values[0]: (save_path, run)}
    
    return dataset_and_backbone_to_model

def run_train_epoch(model, dataset, num_classes, l2_coef=0.0001, removed_indices=[], epochs=10, early_stopping_thresh=1e-6):
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
        if len(removed_indices) > 0:
            removed_entry_loss = torch.norm(model.prototype_prediction_head.class_connection_layer.weight[:, removed_indices], p=1)
            cur_loss = cur_loss + removed_entry_loss

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

def run_eval_epoch(model, dataset, num_classes, l2_coef=0.0001):
    model.eval()
    
    acc = Accuracy(task="multiclass", num_classes=num_classes).to("cuda" if torch.cuda.is_available() else "cpu")
    loss = MeanMetric().to("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        X = torch.tensor(dataset.values[:, :-1], device="cuda" if torch.cuda.is_available() else "cpu")
        y = torch.tensor(dataset.values[:, -1], device="cuda" if torch.cuda.is_available() else "cpu").long()
        output = model.prototype_prediction_head(X)
        
        acc.update(target=y, preds=output["logits"])
        loss.update(
            torch.nn.functional.cross_entropy(output["logits"], y)
        )

    return acc.compute().item(), \
        (loss.compute() + l2_coef * torch.norm(model.prototype_prediction_head.class_connection_layer.weight.data, p=2)).item()

def sample_many_models(
    n_iters=10, 
    protodebug_eval_points=[24, 49, 74, 99],
    save_loc_root=Path("/usr/xtmp/jcd97/proto-rset/results/"), 
    target_config_index=None
):
    os.makedirs(save_loc_root, exist_ok=True)
    dataset_and_backbone_to_model = get_path_to_best_model_by_dataset_backbone()
    print(f"dataset_and_backbone_to_model: {dataset_and_backbone_to_model}")

    rset_args = DEFAULT_RSET_ARGS
    rset_args["max_iter"] = 20_000
    rset_args["directly_compute_hessian"] = True
    rset_args["rashomon_bound_multiplier"] = 1.1
    rset_args["lam"] = 0.0001
    rset_args["reg"] = "l2"
    rset_args["lr_for_opt"] = 1
    rset_args["num_models"] = 0
    rset_args["opt_tol"] = 1e-7
    additional_prototypes_to_sample = 100
    additional_prototypes_to_sample_ham = 100
    last_layer_retrain_epochs = rset_args["max_iter"]

    analysis_save_dir = Path("/usr/xtmp/jcd97/proto-rset/visualizations/")

    # coefs_list will store all sampled coefs, used to
    # generate a similarity heatmap
    coefs_list = []

    config_index = 0
    for dataset in dataset_and_backbone_to_model:
        if "." in dataset:
            print(f"Skipping {dataset}")
            continue
        if "HAM" in dataset:
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
                "removed_protos_avg_coef": [],
                "delta_test_loss": [],
                "delta_test_accuracy": [],
                "test_acc": [],
                "test_loss": [],
            })

            print(f"======== Running dataset {dataset} backbone {backbone} ==========")
            model = dataset_and_backbone_to_model[dataset][backbone][0]
            wandb_run = dataset_and_backbone_to_model[dataset][backbone][1]
            wandb_config = wandb_run.config
            wandb_config['post_project_phases'] = 5
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
                "removed_protos_avg_coef": [-1]
            })], ignore_index=True)

            naive_ppn = copy.deepcopy(original_ppn)
            naive_adjusted_ppn = copy.deepcopy(original_ppn)

            removed_indices = []
            removed_indices_viable = []
            for i in range(n_iters):
                print(f"Iter {i}")

                # Find a prototype that can be removed
                if i > 0:
                    can_remove = False
                    proto_to_remove = random.randint(0, original_ppn.prototype_layer.num_prototypes - 1)
                    saved_image = os.path.exists(proto_save_loc / "prototypes" / f"proto_{proto_to_remove}_cropped_region.png")
                    while (proto_to_remove in removed_indices) or not saved_image:
                        proto_to_remove = random.randint(0, original_ppn.prototype_layer.num_prototypes - 1)
                        saved_image = os.path.exists(proto_save_loc / "prototypes" / f"proto_{proto_to_remove}_cropped_region.png")

                    viable_proto_to_remove = proto_to_remove
                    for _ in range(1_000):
                        start = time.time()
                        try:
                            can_remove = factory.require_to_avoid_prototype(viable_proto_to_remove)
                        except BaseException as e:
                            print(f"ERROR: {e}")
                            can_remove = False
                        if can_remove:
                            removed_indices_viable.append(viable_proto_to_remove)
                            break
                        else:
                            viable_proto_to_remove = random.randint(0, original_ppn.prototype_layer.num_prototypes - 1)
                    
                    # If there are none left, stop early
                    if not can_remove:
                        break
                else:
                    start = time.time()

                # ====== Evaluate a model produced using the RSet
                ppn = factory.produce_protopnet_object().to("cuda" if torch.cuda.is_available() else "cpu")
                sample_time = time.time() - start

                all_coef = ppn.prototype_prediction_head.class_connection_layer.weight.data
                print(f"Cur RSet coef: {all_coef}")

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
                    "removed_protos_avg_coef": [torch.mean(torch.abs(all_coef[:, removed_indices_viable])).item() if i > 0 else 0],
                    "indices": [i]
                })], ignore_index=True)

                coefs_list.append(all_coef.flatten())

                # ====== Evaluate naive removal of this prototype
                start = time.time()
                if i > 0:
                    removed_indices.append(proto_to_remove)
                    naive_ppn.prototype_prediction_head.class_connection_layer.weight.data[:, proto_to_remove] = 0.0
                sample_time = time.time() - start
                assert len(removed_indices) == len(set(removed_indices)), "Error: duplicate removal"

                all_coef = naive_ppn.prototype_prediction_head.class_connection_layer.weight.data

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
                    "removed_protos_avg_coef": [torch.mean(torch.abs(all_coef[:, proto_to_remove])).item() if i > 0 else 0],
                    "indices": [i]
                })], ignore_index=True)

                # ====== Evaluate naive removal of this prototype with reoptimization
                start = time.time()
                if i > 0:
                    naive_adjusted_ppn.prototype_prediction_head.class_connection_layer.weight.data[:, proto_to_remove] = 0.0
                    run_train_epoch(
                        naive_adjusted_ppn, 
                        train_similarities, 
                        split_dataloaders.num_classes, 
                        epochs=last_layer_retrain_epochs, 
                        removed_indices=removed_indices, 
                        early_stopping_thresh=rset_args["opt_tol"]
                    )
                sample_time = time.time() - start

                all_coef = naive_adjusted_ppn.prototype_prediction_head.class_connection_layer.weight.data

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
                    "removed_protos_avg_coef": [torch.mean(torch.abs(all_coef[:, proto_to_remove])).item() if i > 0 else 0],
                    "indices": [i]
                })], ignore_index=True)


                # ====== Evaluate direct removal of this prototype with reoptimization
                hard_pruned_ppn = copy.deepcopy(original_ppn)
                start = time.time()
                kept_columns = [i for i in range(train_similarities.shape[1]) if i not in removed_indices]
                print(f"removed_indices: {removed_indices}")
                print(f"kept_columns: {kept_columns}")
                if i > 0:
                    for p in sorted(removed_indices, reverse=True):
                        hard_pruned_ppn.prune_prototype(p)
                    print(f"hard_pruned_ppn.num_prototypes: {hard_pruned_ppn.prototype_layer.num_prototypes}")
                    run_train_epoch(
                        hard_pruned_ppn, 
                        train_similarities.iloc[:, kept_columns], 
                        split_dataloaders.num_classes, 
                        epochs=last_layer_retrain_epochs, 
                        removed_indices=[], #removed_indices,  set to emptty list so we don't penalize random entries
                        early_stopping_thresh=rset_args["opt_tol"]
                    )
                sample_time = time.time() - start

                train_acc_cur, train_loss_cur = run_eval_epoch(hard_pruned_ppn, train_similarities.iloc[:, kept_columns], split_dataloaders.num_classes)
                val_acc_cur, val_loss_cur = run_eval_epoch(hard_pruned_ppn, val_similarities.iloc[:, kept_columns], split_dataloaders.num_classes)
                test_acc_cur, test_loss_cur = run_eval_epoch(hard_pruned_ppn, test_similarities.iloc[:, kept_columns], split_dataloaders.num_classes)
                del hard_pruned_ppn

                results = pd.concat([results, pd.DataFrame({
                    "dataset": [dataset],
                    "backbone": [backbone],
                    "val_acc": [val_acc_cur],
                    "train_acc": [train_acc_cur],
                    "train_loss": [train_loss_cur],
                    "val_loss": [val_loss_cur],
                    "sample_time": [sample_time],
                    "coef_similarity_to_orig": [-2],
                    "model_type": ["hard_removal_retrained"],
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
                    "removed_protos_avg_coef": [0],
                    "indices": [i]
                })], ignore_index=True)

                proto_save_loc = (
                    analysis_save_dir / f"vis_{Path(model).parent.name + '_' + Path(model).name}"
                )

                if i in protodebug_eval_points:
                    # ===== Generate the forbid dataset
                    file_source_root = proto_save_loc / "prototypes"
                    file_save_root = Path(f"/usr/xtmp/jcd97/proto-rset/protodebug_datasets/{dataset}/{backbone}")
                    # Make sure we don't have any junk floating around in the forbit directory
                    if os.path.exists(file_save_root):
                        shutil.rmtree(file_save_root)

                    for removal in removed_indices:
                        cropped_region = cv2.imread(str(file_source_root / f"proto_{removal}_cropped_region.png"))
                        img_class = torch.argmax(naive_ppn.prototype_layer.prototype_class_identity[removal]).item()
                        save_dir = file_save_root / f"{img_class}/"

                        os.makedirs(str(save_dir), exist_ok=True)
                        try:
                            vert = (224 - cropped_region.shape[0]) // 2
                            horiz = (224 - cropped_region.shape[1]) // 2
                            padded_image = cv2.copyMakeBorder(cropped_region, vert, vert, horiz, horiz, cv2.BORDER_CONSTANT, value=(255, 255, 255))
                            plt.imsave(
                                str(save_dir / f"prototype_{removal}.png"),
                                padded_image
                            )
                        except:
                            print(f"WARNING: Not removing {removal} because that prototype image didn't save")

                    # ====== Train a model with the same params as our best, with debugging added
                    protodebug_run = run_proto_debug(
                        dataset=dataset,
                        backbone=backbone,
                        debug_round=True,
                        debug_forbid_dir=str(file_save_root),
                        use_test_dataset=True,
                        debug_remember_dir='/usr/xtmp/zg78/protodbug/debug_folder/remember/',
                        resume_weight_path=model,
                        **wandb_config
                    )
                    # As it turns out, the iternal and API wandb run objects have different structures.
                    # Converting between the two
                    api = wandb.Api()
                    protodebug_run = api.run(protodebug_run.path)
                    history = protodebug_run.history()
                    best_step_stats = history[history['_step'] == protodebug_run.summary['best_accuracy_step']]

                    results = pd.concat([results, pd.DataFrame({
                        "dataset": [dataset],
                        "backbone": [backbone],
                        "train_acc": [best_step_stats['train.accu'].values[0] / 100],
                        "train_loss": [best_step_stats['train.total_loss'].values[0]],
                        "sample_time": [best_step_stats['_runtime'].values[0]],
                        "model_type": ["protopdebug"],
                        "time_to_get_rset": [cur_time_to_get_rset],
                        "delta_train_accuracy": [best_step_stats['train.accu'].values[0] / 100 - og_train_accuracy],
                        "delta_test_accuracy": [best_step_stats['eval.accu'].values[0] / 100 - og_test_accuracy],
                        "test_acc": [best_step_stats['eval.accu'].values[0] / 100],
                        "test_loss": [best_step_stats['eval.total_loss'].values[0]],
                        "num_prototypes": [original_ppn.prototype_layer.num_prototypes],
                        "indices": [i],
                        "wandb_runid": [protodebug_run.id]
                    })], ignore_index=True)

                pairwise_similarities = pairwise_cosine_similarity(
                    torch.stack(coefs_list, dim=0), zero_diagonal=False
                )

                torch.save(
                    pairwise_similarities,
                    save_loc_root / "pairwise_similarities.pth"
                )

                results.to_csv(save_loc_root / "summary.csv", index=False)
                del ppn, all_coef
            config_index += 1

def main():
    slurm_id = int(sys.argv[1])
    torch.manual_seed(0)
    random.seed(0)
    sample_many_models(
        n_iters=100,
        save_loc_root=Path(f"/usr/xtmp/jcd97/proto-rset/results/configuration_{slurm_id}_with_test_more_iter_TEST/"),
        target_config_index=slurm_id
    )

if __name__ == '__main__':
    main()