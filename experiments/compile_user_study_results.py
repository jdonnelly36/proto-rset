import pandas as pd
import os
import copy
import time
from app import prep_rset
from cvpr_experiments import run_eval_epoch, run_train_epoch
from protodebug_train_wrapper import run_proto_debug
from pathlib import Path
from app import GOLD_STD_REMOVALS
import cv2
import torch
from matplotlib import pyplot as plt
import sys
import wandb

def get_valid_result_dfs():
    user_to_df = {}
    assert False, \
        "To run this compilation script, remove this line and provide a path to a dir containing user study results"
    results_root = None ###

    return [pd.read_csv(results_root + file) for file in os.listdir(results_root)]


if __name__ == "__main__":
    slurm_id = int(sys.argv[1])
    num_jobs = int(sys.argv[2])
    api = wandb.Api()

    factory = prep_rset()
    initial_protopnet_path = factory.initial_protopnet_path
    initial_rset_acc, initial_rset_loss = run_eval_epoch(factory.produce_protopnet_object(), factory.val_similarities_dataset, 200)
    print(initial_rset_acc)
    
    train_similarities = factory.train_similarities_dataset

    df = pd.DataFrame({
        "delta_val_acc": [],
        "prolific_pid": [],
        "num_removed": [],
        "total_removal_time": [],
        "total_time_spent": [],
        "method": [],
    })
    user_dfs = get_valid_result_dfs()
    print(f"len(user_dfs): {len(user_dfs)}")

    assert False, \
        "To run this compilation script, remove this line and provide a path to a complete user study csv below"
    # completed_results = pd.read_csv( ### )
    for df_ind, user_df in enumerate(user_dfs):
        factory = prep_rset()
        if df_ind % num_jobs != slurm_id:
            continue
        print(user_df['prolific_pid'].values[0] in completed_results['prolific_pid'].values, user_df['prolific_pid'].values[0], completed_results['prolific_pid'].values)
        # if user_df['prolific_pid'].values[0] in completed_results['prolific_pid'].values:
        #     print(f"DF so far for this pid: {completed_results[completed_results['prolific_pid'] == user_df['prolific_pid'].values[0]]}")
        #     if 'ProtoPDebug' in completed_results[completed_results['prolific_pid'] == user_df['prolific_pid'].values[0]]['method'].values:
        #         print(f"Skipping {user_df['prolific_pid'].values[0]} because it's done")
        #         continue
        print(f"Running {df_ind}: {user_df}")
        naive_adjusted_ppn = copy.deepcopy(factory.initial_protopnet)

        # ======== Get ProtoRSet result
        for t in user_df['target'].unique():
            factory.require_to_avoid_prototype(int(t))

        updated_rset_acc, updated_rset_loss = run_eval_epoch(factory.produce_protopnet_object(), factory.val_similarities_dataset, 200)
        df = pd.concat(
            [df, pd.DataFrame({
                "delta_val_acc": [updated_rset_acc - initial_rset_acc],
                "prolific_pid": [user_df['prolific_pid'].values[0]],
                "num_removed": [user_df['target'].nunique()],
                "total_removal_time": [user_df['runtime'].sum()],
                "total_time_spent": [user_df['time_to_action'].max()],
                "method": ["ProtoRSet"]
            })], axis=0
        )

        # ====== Evaluate naive removal of this prototype without reoptimization
        start = time.time()
        for proto_to_remove in user_df['target'].unique():
            naive_adjusted_ppn.prototype_prediction_head.class_connection_layer.weight.data[:, int(proto_to_remove)] = 0.0
        sample_time = time.time() - start

        all_coef = naive_adjusted_ppn.prototype_prediction_head.class_connection_layer.weight.data

        train_acc_cur, train_loss_cur = run_eval_epoch(naive_adjusted_ppn, train_similarities, 200)
        val_acc_cur, val_loss_cur = run_eval_epoch(naive_adjusted_ppn, factory.val_similarities_dataset, 200)

        df = pd.concat(
            [df, pd.DataFrame({
                "delta_val_acc": [val_acc_cur - initial_rset_acc],
                "prolific_pid": [user_df['prolific_pid'].values[0]],
                "num_removed": [user_df['target'].nunique()],
                "total_removal_time": [sample_time],
                "total_time_spent": [user_df['time_to_action'].max() - user_df['runtime'].sum() + sample_time],
                "method": ["Naive Removal (No Retrain)"]
            })], axis=0
        )

        # ===== Generate the forbid dataset
        proto_save_loc = (
            factory.analysis_save_dir / f"vis_{factory.initial_protopnet_path.parent.name + '_' + factory.initial_protopnet_path.name}"
        )
        file_source_root = proto_save_loc / "prototypes"
        file_save_root = Path(f"/usr/xtmp/jcd97/proto-rset/protodebug_datasets/{user_df['prolific_pid'].values[0]}")
        val_similarities_dataset = factory.val_similarities_dataset

        del factory

        for removal in user_df['target'].unique():
            cropped_region = cv2.imread(str(file_source_root / f"proto_{int(removal)}_cropped_region.png"))
            img_class = torch.argmax(naive_adjusted_ppn.prototype_layer.prototype_class_identity[int(removal)]).item()
            save_dir = file_save_root / f"{img_class}/"

            os.makedirs(str(save_dir), exist_ok=True)

            vert = (224 - cropped_region.shape[0]) // 2
            horiz = (224 - cropped_region.shape[1]) // 2
            padded_image = cv2.copyMakeBorder(cropped_region, vert, vert, horiz, horiz, cv2.BORDER_CONSTANT, value=(255, 255, 255))
            plt.imsave(
                str(save_dir / f"prototype_{int(removal)}.png"),
                padded_image
            )
        df.to_csv(f"/usr/xtmp/jcd97/proto-rset/user_study_agg_{slurm_id}_with_bias.csv")

        # ====== Evaluate naive removal of this prototype with reoptimization
        start = time.time()
        for proto_to_remove in user_df['target'].unique():
            naive_adjusted_ppn.prototype_prediction_head.class_connection_layer.weight.data[:, int(proto_to_remove)] = 0.0
        
        run_train_epoch(
            naive_adjusted_ppn, 
            train_similarities, 
            200, 
            epochs=5_000, 
            removed_indices=user_df['target'].unique(), 
            early_stopping_thresh=1e-7
        )
        sample_time = time.time() - start

        all_coef = naive_adjusted_ppn.prototype_prediction_head.class_connection_layer.weight.data

        train_acc_cur, train_loss_cur = run_eval_epoch(naive_adjusted_ppn, train_similarities, 200)
        val_acc_cur, val_loss_cur = run_eval_epoch(naive_adjusted_ppn, val_similarities_dataset, 200)

        df = pd.concat(
            [df, pd.DataFrame({
                "delta_val_acc": [val_acc_cur - initial_rset_acc],
                "prolific_pid": [user_df['prolific_pid'].values[0]],
                "num_removed": [user_df['target'].nunique()],
                "total_removal_time": [sample_time],
                "total_time_spent": [user_df['time_to_action'].max() - user_df['runtime'].sum() + sample_time],
                "method": ["Naive Removal (Retrain)"]
            })], axis=0
        )
        df.to_csv(f"/usr/xtmp/jcd97/proto-rset/user_study_agg_{slurm_id}_with_bias.csv")
        del naive_adjusted_ppn

        # ====== Train a model with the same params as our best, with debugging added
        protodebug_run = run_proto_debug(
            backbone="vgg19",
            pre_project_phase_len=11,
            phase_multiplier=1,  # for online augmentation
            latent_dim_multiplier_exp=-4,
            joint_lr_step_size=8,
            post_project_phases=5,
            joint_epochs_per_phase=10,
            last_only_epochs_per_phase=10,
            cluster_coef=-1.2,
            separation_coef=0.03,
            l1_coef=0.00001,
            orthogonality_loss=0.0004,
            num_addon_layers=1,
            fa_type=None,
            fa_coef=0.0,
            num_prototypes_per_class=14,
            joint_add_on_lr_multiplier=1,
            warm_lr_multiplier=1,
            lr_multiplier=0.89,
            lr_step_gamma=0.1,
            class_specific=False,
            dry_run=False,
            verify=False,
            interpretable_metrics=False,
            dataset="cub200",
            resume=True,
            resume_weight_path=initial_protopnet_path,
            bias_rate=1.0,
            debug_round=True, # turn this on if running the second round of debug, first run should be false
            debug_forbid_coef=100.0, # this is the default from protodebug config
            debuf_remeber_coef=0, # this should be 0 given protodebug conif, but could be a number > 0
            debug_forbid_dir=str(file_save_root),
            debug_remember_dir='/usr/xtmp/zg78/protodbug/debug_folder/remember/'
        )
        
        protodebug_run = api.run(protodebug_run.path)
        history = protodebug_run.history()
        best_step_stats = history[history['_step'] == protodebug_run.summary['best_accuracy_step']]

        df = pd.concat(
            [df, pd.DataFrame({
                "delta_val_acc": [best_step_stats['eval.accu'].values[0] / 100 - initial_rset_acc],
                "prolific_pid": [user_df['prolific_pid'].values[0]],
                "num_removed": [user_df['target'].nunique()],
                "total_removal_time": [best_step_stats['_runtime'].values[0]],
                "total_time_spent": [user_df['time_to_action'].max() - user_df['runtime'].sum() + best_step_stats['_runtime'].values[0]],
                "method": ["ProtoPDebug"]
            })], axis=0
        )

        df.to_csv(f"/usr/xtmp/jcd97/proto-rset/user_study_agg_{slurm_id}_with_bias.csv")

        del train_acc_cur, train_loss_cur, val_acc_cur, val_loss_cur, protodebug_run, updated_rset_acc