import asyncio
import functools
import os
import pathlib
import time
import copy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from math import ceil
import multiprocessing

from protopnet.datasets import torch_extensions
from protopnet.visualization import (
    global_analysis,
    local_analysis,
    reproject_prototypes,
    run_analyses,
    save_prototype_images_to_file,
    KeyReturningDict
)
from rashomon_sets.linear_rashomon_set import (
    CorrectClassOnlyMultiClassLogisticRSet,
    MultiClassLogisticRSet,
)

DEFAULT_RSET_ARGS = {
    "rashomon_bound_multiplier": 1.05,
    "num_models": 0,  # Right now, I'm randomly sampling models from the ellipsoid during fit
    "reg": "l2",
    "lam": 0.0001,
    "compute_hessian_batched": False,  # Not batching for memory's sake
    "max_iter": 10_000,  # The max number of iterations allowed when fitting the LR,
    "directly_compute_hessian": True,
    "device": torch.device("cuda"),
}


class ListDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


class ProtoRSetFactory:
    def __init__(
        self,
        split_dataloaders: torch_extensions.FilesystemSplitDataloaders,
        initial_protopnet_path: pathlib.Path,
        use_bias: bool = False,
        rashomon_set_args: dict = DEFAULT_RSET_ARGS,
        additional_prototypes_to_sample: int = 0,
        prototype_sampling_method: str = "uniform_random",
        device: str = "cuda",
        analysis_save_dir: pathlib.Path = Path(
            "/usr/xtmp/zg78/proto_rset/rashomon_sets/rset_analysis"
        ),
        correct_class_connections_only: bool = True,
        run_complete_vislization_in_init: bool = False,
        reproject: bool = False,
        verbose: bool = False
    ):
        self.train_dataloader, self.viz_dataloader, self.val_dataloader = (
            split_dataloaders.train_loader_no_aug, # split_dataloaders.train_loader_no_aug,
            split_dataloaders.project_loader,
            split_dataloaders.val_loader,
        )
        try:
            self.class_name_ref_dict = split_dataloaders.class_name_ref_dict
        except:
            self.class_name_ref_dict = KeyReturningDict()
        self.correct_class_connections_only = correct_class_connections_only

        # ===== Load and store class assignments for the original model
        self.initial_protopnet_path = initial_protopnet_path
        self.initial_protopnet = torch.load(self.initial_protopnet_path, weights_only=False)

        self.device = device
        rashomon_set_args["device"] = device
        self.use_bias = use_bias
        self.verbose = verbose


        # prototype info dict is a must before running any visulization, running reproject to check beforehand
        self.analysis_save_dir = analysis_save_dir
        self.vis_save_loc = (
            self.analysis_save_dir
            / f"vis_{initial_protopnet_path.parent.name + '_' + initial_protopnet_path.name}"
        )
        self.initial_protopnet = reproject_prototypes(
            self.initial_protopnet, self.vis_save_loc, self.train_dataloader#split_dataloaders.project_loader
        )

        # TODO: Initialize this using the class assignment tensor from the
        # underlying protopnet
        self.model_index_to_class = {0: 0}

        # ===== Sample in the init
        self.prototype_sampling_method = prototype_sampling_method
        if additional_prototypes_to_sample > 0:
            self.sample_additional_prototypes(
                target_number_of_samples=additional_prototypes_to_sample,
                prototype_sampling=self.prototype_sampling_method,
                dataloader=self.viz_dataloader,
                immediately_refit=False
            )

        # ===== Remove duplicate prototypes
        self.initial_protopnet.prune_duplicate_prototypes()

        # ===== Build our similarities dataset
        self._prepare_datasets(self.train_dataloader, self.val_dataloader)

        if self.correct_class_connections_only:
            self.rset = CorrectClassOnlyMultiClassLogisticRSet(
                prototype_class_identity=self.initial_protopnet.prototype_layer.prototype_class_identity,
                **rashomon_set_args,
                verbose=self.verbose
            )
        else:
            self.rset = MultiClassLogisticRSet(**rashomon_set_args, verbose=self.verbose)

        self.rset.fit(self.X_train, self.y_train)
        # Update our internal ProtoPNet to the best thing we could find in the RSet
        new_weights = self.rset.optimal_model.linear_weights.data
        self.initial_protopnet = self._update_protopnet_last_layer(new_weights)

        # ===== Initialize objects to store user interaction data
        self.rating_int_to_string = {
            0: "Have not viewed",
            1: "Viewed and no feedback",
            2: "Required",
            3: "Required to avoid",
        }
        self.prototype_ratings_so_far = [
            0 for _ in range(self.train_similarities_dataset.shape[1])
        ]

        # ===== History storing all user interactions
        self.user_interaction_history = []
        pass

        self.device = device
        # ===== Cuncurrent run analysis
        self.normalize_mean, self.normalize_std = (
            split_dataloaders.normalize_mean,
            split_dataloaders.normalize_std,
        )
        # prototype info dict is a must before running any visulization, running reproject to check beforehand
        if reproject:
            self.initial_protopnet = reproject_prototypes(
                self.initial_protopnet, self.vis_save_loc, split_dataloaders.project_loader
            )

        self.img_size = split_dataloaders.image_size
        if run_complete_vislization_in_init:
            loop = asyncio.get_event_loop()
            loop.create_task(
                self.run_analysis_async(
                    model=self.initial_protopnet,
                    initial_protopnet_path=initial_protopnet_path,
                    project_dataloader=split_dataloaders.project_loader,
                    val_dataloader=split_dataloaders.val_loader,
                    img_size=self.img_size,
                    device=device,
                )
            )

    async def run_analysis_async(
        self,
        model,
        initial_protopnet_path,
        project_dataloader,
        val_dataloader,
        img_size,
        device,
    ):
        print(f"[{time.strftime('%X')}] run_analysis_async started", flush=True)
        loop = asyncio.get_running_loop()
        render_proto_call = functools.partial(
            run_analyses,
            model=model,
            model_path=initial_protopnet_path,
            project_dataloader=project_dataloader,
            class_name_ref_dict=self.class_name_ref_dict,
            val_dataloader=val_dataloader,
            vis_save_loc=None,
            analysis_type="render-prototypes",
            output_dir=self.analysis_save_dir,
            img_size=img_size,
            device=device,
            normlization_params=(self.normalize_mean, self.normalize_std),
            last_layer_coef=self.rset.optimal_model.get_params(),
        )

        await loop.run_in_executor(None, render_proto_call)
        print(f"[{time.strftime('%X')}] render_proto finished", flush=True)

        global_analysis_call = functools.partial(
            run_analyses,
            model=model,
            model_path=initial_protopnet_path,
            project_dataloader=project_dataloader,
            val_dataloader=val_dataloader,
            class_name_ref_dict=self.class_name_ref_dict,
            vis_save_loc=None,
            analysis_type="global",
            output_dir=self.analysis_save_dir,
            img_size=img_size,
            device=device,
            normlization_params=(self.normalize_mean, self.normalize_std),
            last_layer_coef=self.rset.optimal_model.get_params(),
        )

        await loop.run_in_executor(None, global_analysis_call)
        print(f"[{time.strftime('%X')}] global_analysis finished", flush=True)

        local_analysis_call = functools.partial(
            run_analyses,
            model=model,
            model_path=initial_protopnet_path,
            project_dataloader=project_dataloader,
            val_dataloader=val_dataloader,
            class_name_ref_dict=self.class_name_ref_dict,
            vis_save_loc=None,
            analysis_type="local",
            output_dir=self.analysis_save_dir,
            img_size=img_size,
            device=device,
            normlization_params=(self.normalize_mean, self.normalize_std),
            last_layer_coef=self.rset.optimal_model.get_params(),
        )

        await loop.run_in_executor(None, local_analysis_call)
        print(f"[{time.strftime('%X')}] local_analysis finished", flush=True)

    def _best_val_acc(self):
        m = self.produce_protopnet_object(return_optimal=True)
        all_preds = torch.zeros(0).to(self.device)
        all_lbls = torch.zeros(0).to(self.device)
        for item in self.val_dataloader:
            img = item["img"].to(self.device)
            targets = item["target"].to(self.device)

            preds = m(img)['logits']
            all_lbls = torch.concat((all_lbls, targets))
            all_preds = torch.concat((all_preds, preds.argmax(dim=-1)))
        return (1.0*(all_lbls == all_preds)).mean()


    def _build_similarities_dataset(self, dataloader, split=None):
        """
        Build a dataset in which each row is a sample from dataloader, and
        each column (except the last) is the maximum similarity between a prototype
        and that image. The last column is the image label.
        Args:
            dataloader -- a torch dataloader that returns a dictionary with keys
                'img' and 'target'
        Returns:
            combined_dataset: pd.DataFrame -- The described dataset
        """
        with torch.no_grad():
            all_similarities = torch.empty(0).to(self.device)
            all_targets = torch.empty(0)
            for item in tqdm(dataloader, desc=f"Computing prototype activation table{ f'for {split} split' if split is not None else ''}"):
                img = item["img"].to(self.device)
                targets = item["target"]

                prototype_activations = self.initial_protopnet(
                    img, return_prototype_layer_output_dict=True
                )["prototype_activations"]
                max_similarities, _ = torch.max(
                    prototype_activations.view(
                        prototype_activations.shape[0],
                        prototype_activations.shape[1],
                        -1,
                    ),
                    dim=-1,
                )

                all_similarities = torch.cat(
                    [all_similarities, max_similarities], axis=0
                )
                all_targets = torch.cat([all_targets, targets], axis=0)

            all_similarities = all_similarities.detach().cpu()

            combined_dataset = pd.DataFrame(
                all_similarities.numpy(),
                columns=[
                    f"prototype_{i}_similarity"
                    for i in range(all_similarities.shape[1])
                ],
            )
            combined_dataset["y"] = all_targets.numpy()

            return combined_dataset

    def _prepare_datasets(self, train_loader, val_loader):
        self.train_similarities_dataset = self._build_similarities_dataset(
            train_loader, split="train"
        )
        self.val_similarities_dataset = self._build_similarities_dataset(
            val_loader, split="val"
        )

        # ===== Fit our Rashomon set over this data
        if self.use_bias:
            # Add a dummy column of all ones to function as bias
            self.X_train = torch.concatenate(
                [
                    torch.ones((self.train_similarities_dataset.shape[0], 1)),
                    torch.tensor(self.train_similarities_dataset.values[:, :-1]),
                ],
                axis=1,
            )
            self.X_val = torch.concatenate(
                [
                    torch.ones((self.val_similarities_dataset.shape[0], 1)),
                    torch.tensor(self.val_similarities_dataset.values[:, :-1]),
                ],
                axis=1,
            )
        else:
            self.X_train = torch.tensor(self.train_similarities_dataset.values[:, :-1])
            self.X_val = torch.tensor(self.val_similarities_dataset.values[:, :-1])

        self.y_train = torch.LongTensor(self.train_similarities_dataset.values[:, -1])
        self.y_val = torch.LongTensor(self.val_similarities_dataset.values[:, -1])

    def _push_to_history(self, interaction: str):
        self.user_interaction_history.append(interaction)

    def _update_protopnet_last_layer(self, coef: torch.Tensor):
        """
        Updates the last layer weights in our internal ProtoPNet
        to match those of our current Rashomon set center
        Args:
            coef : torch.Tensor (num_prototypes) -- The coefficients
                to update our ProtoPNet to use
        Returns:
            ProtoPNet : a ProtoPNet object reflecting the given coefficients
        """
        cur_protopnet = copy.deepcopy(self.initial_protopnet)
        new_weights = (
            coef.unsqueeze(1) * self.rset.optimal_model.prototype_class_identity
        )
        cur_protopnet.prototype_prediction_head.class_connection_layer.weight.data = (
            new_weights.T
        )
        return cur_protopnet

    def display_local_analysis(
        self, 
        target_img_indices, 
        include_coef=True, 
        random_model=False, 
        prespecified_model=None, 
        run_proto_vis=False,
        reasoning_for_top_c=None,
        sort_using_logit=False
    ):
        """
        Build and display a local analysis for the given
        image index
        Args:
            target_img_index: int or list -- the index of the image to analyze
        """
        if type(target_img_indices) is int:
            target_img_indices = [target_img_indices]

        for target_img_index in target_img_indices:
            self._push_to_history(f"Local analysis for image {target_img_index}")
        # Every prototype that appears in local analysis should be set to 1

        if prespecified_model is not None:
            target_model = prespecified_model
        elif not random_model:
            target_model = self.initial_protopnet
        else:
            target_model = self.produce_protopnet_object(False)

        target_model.eval()

        def weight_matrix_to_vector(mat):
            '''
            mat should be num_classes x num_protos
            '''
            am = torch.argmax(mat, dim=0)
            return torch.tensor(
                [mat[am[i], i].item() for i in range(mat.shape[1])],
                device=mat.device
            )

        path_rets = [local_analysis(
            target_model,
            save_loc=self.vis_save_loc,
            push_dataloader=self.viz_dataloader,
            eval_dataloader=self.val_dataloader,
            class_name_ref_dict=self.class_name_ref_dict,
            img_size=self.img_size,
            device=self.device,
            run_proto_vis=run_proto_vis,
            specify_img_idx=target_img_index,
            normlization_params=(self.normalize_mean, self.normalize_std),
            reasoning_for_top_c=reasoning_for_top_c,
            last_layer_coef=weight_matrix_to_vector(target_model.prototype_prediction_head.class_connection_layer.weight.data) if include_coef else None,
            sort_using_logit=sort_using_logit
        ) for target_img_index in target_img_indices]
        return path_rets

    def display_global_analysis_for_proto(self, target_proto_index):
        """
        Visualize the nearest neighbors for some prototype
        """

        # NOTE: it seems the most accessible way (current standard) to reference prototype is by prototype_index
        #       but would this be fragile if we are doing frequent back and forth sampling and removing protos?
        #       where previouse index could be lose its meaning (even though we are current only doing adding,
        #       which makes index increasing only, but we are not encforcing this), could be a non issue.
        self._push_to_history(f"Global analysis for prototype {target_proto_index}")
        self.prototype_ratings_so_far[target_proto_index] = 1

        path_ret = global_analysis(
            self.initial_protopnet,
            save_loc=self.vis_save_loc,
            proto_save_dir=self.vis_save_loc / "prototypes",
            proto_index=target_proto_index,
            project_dataloader=self.train_dataloader,
            img_size=self.img_size,
            normlization_params=(self.normalize_mean, self.normalize_std),
            device=self.device,
        )

        return path_ret

    def display_proto_based_on_coef(self, mode, k=None, threshold=None):
        """
        Visualizes prototypes based on the coefficients from the model's parameters using different modes.

        Args:
            mode (str): The mode to use for selecting prototypes. Must be one of:
                - "highest": Selects the prototype with the highest coefficient.
                - "topk": Selects the top-k prototypes with the highest coefficients. Requires `k` to be specified.
                - ">_threshold": Selects all prototypes whose coefficients are greater than the specified `threshold`. Requires `threshold` to be specified.
            k (int, optional): The number of top-k prototypes to visualize when `mode` is "topk". Must be a positive integer.
            threshold (float, optional): The threshold value for selecting prototypes when `mode` is ">_threshold". Prototypes with coefficients greater than this value will be selected.

        Raises:
            AssertionError: If `mode` is not one of ["highest", "topk", ">_threshold"].
            AssertionError: If `k` is not provided or not an integer when `mode` is "topk".
            Warning: If no prototypes are selected when using `mode` = ">_threshold" and the `threshold` is too high.

        Returns:
            path_ret (list): A list of paths to the saved prototype images.
        """

        assert mode in [
            "highest",
            "topk",
            ">_threshold",
        ], 'Invalid mode option, must one of ["highest", "topk", ">_threshold"].'

        if mode == "highest":
            highest_coef_index = torch.argmax(
                self.rset.optimal_model.get_params()
            ).item()
            self._push_to_history(
                f"Visualize prototype with highest coefficient {highest_coef_index}"
            )
            path_ret = save_prototype_images_to_file(
                model=self.initial_protopnet,
                push_dataloader=self.train_dataloader,
                save_loc=self.vis_save_loc,
                img_size=self.img_size,
                normalize_for_fwd=None,
                box_color=(0, 255, 255),
                device=self.device,
                specify_proto_idx=[highest_coef_index],
            )

        elif mode == "topk":
            assert k is not None and isinstance(
                k, int
            ), "When mode is topk, must specify k."

            _, topk_idxs = torch.topk(self.rset.optimal_model.get_params(), k=k, dim=-1)
            topk_idxs = topk_idxs.squeeze().cpu().numpy().astype(np.int32)

            self._push_to_history(
                f"Visualize prototype with highest topk k={k} coef {topk_idxs}"
            )
            path_ret = save_prototype_images_to_file(
                model=self.initial_protopnet,
                push_dataloader=self.train_dataloader,
                save_loc=self.vis_save_loc,
                img_size=self.img_size,
                normalize_for_fwd=None,
                box_color=(0, 255, 255),
                device=self.device,
                specify_proto_idx=topk_idxs,
            )

        elif mode == ">_threshold":
            proto_indices = (
                torch.nonzero(self.rset.optimal_model.get_params() > threshold)
                .squeeze()
                .cpu()
                .numpy()
                .astype(np.int32)
            )

            if len(proto_indices) == 0:
                raise Warning(
                    f"Threshold of {threshold} is too high, causing empty selection"
                )

            self._push_to_history(
                f"Visualize prototype with coefficient > {threshold} {len(proto_indices)} selected."
            )

            path_ret = save_prototype_images_to_file(
                model=self.initial_protopnet,
                push_dataloader=self.train_dataloader,
                save_loc=self.vis_save_loc,
                img_size=self.img_size,
                normalize_for_fwd=None,
                box_color=(0, 255, 255),
                device=self.device,
                specify_proto_idx=proto_indices,
            )

        return path_ret

    def display_proto_in_class(self, target_class):
        """
        Visualize the highest coef prototype in the best model
        """

        self._push_to_history(f"Visualize prototype in class {target_class}")
        path_ret = save_prototype_images_to_file(
            model=self.initial_protopnet,
            push_dataloader=self.train_dataloader,
            save_loc=self.vis_save_loc,
            img_size=self.img_size,
            normalize_for_fwd=None,
            box_color=(0, 255, 255),
            device=self.device,
            specify_proto_class=target_class,
        )

        return path_ret

    def display_proto_collage(self, start_idx, end_idx, num_cols=10, use_bbox=True, specified_protos=[]):
        """
        Creates and saves a collage of prototype images arranged in a grid.

        This function loads a specified set of prototype images from a directory, arranges them in
        a grid of specified rows and columns, and saves the collage to a designated location.
        The function returns the path to the saved collage image.

        Parameters
        ----------
        num_rows : int
            The number of rows in the collage grid.
        num_cols : int
            The number of columns in the collage grid.
        collage_idx : int
            The index of the collage to display, used to calculate which images to load.
        specified_protos: list (optional)
            A list of specific prototypes to visualize. If empty, ignored; otherwise, overrides
            start and end idx

        Returns
        -------
        Path
            The file path of the saved collage image.

        Notes
        -----
        - Images are loaded from `self.vis_save_loc / "prototypes"`, filtered to include only files
        ending in "_proto_bbox.png".
        - Collages are saved to `self.vis_save_loc / "prototypes" / "{collage_idx}th_collage.png"`.
        - The function assumes that `num_rows * num_cols` images are available for each collage index.
        - Any extra subplot spaces in the grid are hidden if fewer than `num_rows * num_cols` images are loaded.
        - The num_rows and num_cols should be fixed to avoid missing any protos.
        - This function should be called after the async processing is done at the begining, it depends on the saved proto images.

        """
        img_save_path = self.vis_save_loc / "prototypes"
        if use_bbox:
            all_proto_files = sorted(
                [f for f in os.listdir(img_save_path) if f.endswith("_proto_bbox.png")],
                key=lambda x: int(x.split("_")[1])
            )
        else:
            all_proto_files = sorted(
                [f for f in os.listdir(img_save_path) if f.endswith("_overlayheatmap.png")],
                key=lambda x: int(x.split("_")[1])
            )

        # start_idx = collage_idx * num_rows * num_cols
        # end_idx = start_idx + num_rows * num_cols
        if len(specified_protos) > 0:
            num_rows = ceil(len(specified_protos)/ num_cols)
        else:
            num_rows = ceil((end_idx - start_idx) / num_cols)

        removed_protos = []
        for interaction in self.user_interaction_history:
            if "Required avoiding prototype" in interaction:
                removed_protos.append(int(interaction.split(" ")[-1]))
        
        selected_proto_files = []
        if len(specified_protos) > 0:
            for i in specified_protos:
                if i not in removed_protos:
                    selected_proto_files.append(all_proto_files[i])
        else:
            for i in range(start_idx, end_idx):
                if i not in removed_protos:
                    selected_proto_files.append(all_proto_files[i])

        selected_images = [Image.open(img_save_path / f) for f in selected_proto_files]
        prototype_class_identity = (
            self.initial_protopnet.prototype_layer.prototype_class_identity
        )

        fig, axes = plt.subplots(
            num_rows, num_cols, figsize=(num_cols * 2, num_rows * 2)
        )
        axes = axes.flatten() if num_rows * num_cols > 1 else [axes]
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)

        for ax, (img, file_name) in zip(
            axes, zip(selected_images, selected_proto_files)
        ):
            ax.imshow(img)
            ax.axis("off")

            # getting label through indeitity matrix
            proto_idx = int(file_name.split("_")[1])
            proto_label = torch.argmax(prototype_class_identity[proto_idx]).item()
            ax.text(
                80,
                10,
                f"Idx:{proto_idx}\nClass:{proto_label}",
                color="white",
                fontsize=10,
                ha="right",
                va="top",
                bbox=dict(
                    facecolor="black", edgecolor="black", boxstyle="round,pad=0.2"
                ),
            )

        for ax, img in zip(axes, selected_images):
            ax.imshow(img)
            ax.axis("off")

        for ax in axes[len(selected_images) :]:
            ax.axis("off")

        collage_save_path = img_save_path / f"{start_idx}_{end_idx}_collage.png"
        plt.tight_layout()
        plt.savefig(collage_save_path, bbox_inches="tight")
        plt.close(fig)
        os.chmod(collage_save_path, 0o777)

        return collage_save_path

    def constrain_rashomon_set(self):
        """
        Run through and apply any new constraints that have been added
        since we last recomputed our rashomon set
        """
        # self.rset.constrain_to_meet_requirements(
        #     self.prototype_ratings_so_far
        # )
        pass

    def require_prototype(self, target_proto):
        self._push_to_history(f"Required prototype {target_proto}")
        self.prototype_ratings_so_far[target_proto] = 2
        pass

    def require_to_avoid_prototype(self, target_proto):
        """
        See if a prototype can be removed from models while remaining
        in the Rashomon set
        Args:
            target_proto : int -- the index of the prototype to remove
        Modifies:
            self.rset : MultiClassLogisticRSet -- If successful, updates
                this factory's running RSet to exclude the specified
                prototype. If not, does not modify this Rset
        Returns:
            successfully_removed : bool -- wheter or not we successfully
                removed the specified variable
        """
        self.prototype_ratings_so_far[target_proto] = 3
        updated_rset, successfully_removed = self.rset._drop_prototype(
            target_proto, resample=True
        )
        self.rset = updated_rset
        if successfully_removed:
            # Correct for floating point error on 0 value coefficients
            new_weights = self.rset.optimal_model.linear_weights.data
            new_weights[abs(new_weights) < 1e-5] = 0.0
            self.initial_protopnet = self._update_protopnet_last_layer(new_weights)
            self._push_to_history(f"Required avoiding prototype {target_proto}")
        return successfully_removed

    def sample_additional_prototypes(
        self,
        target_number_of_samples: int,
        prototype_sampling,
        dataloader: torch.utils.data.DataLoader,
        immediately_refit: bool = True
    ):
        """
        Samples additional candidate prototypes and augments the current similarity table with these new samples.

        This method generates a specified number of additional prototype candidates, samples them from the provided
        DataLoader based on the sampling strategy (`random` or list of indices), and computes the corresponding
        prototype vectors and class confidence (CC) vectors. It finally augments the initial prototype network with
        the newly sampled prototypes.

        Args:
            target_number_of_samples (int): The number of prototype samples to be selected from the dataset.
            prototype_sampling (Union[str, List[int]]): The sampling strategy to use. If 'random', a specified number
                of samples are selected randomly. If a list of indices is provided, those specific indices are sampled.
            dataloader (torch.utils.data.DataLoader): The PyTorch DataLoader that provides batches of data from which
                the prototypes will be sampled.

        Raises:
            Exception: If `prototype_sampling` is not a valid sampling strategy or contains non-integer values when
                provided as a list.

        Returns:
            None: The method directly modifies the `initial_protopnet` object by adding the newly sampled prototypes
            to its prototype layer.

        """

        # TODO: attempt to gurantees that it is selecting new protos, will not overlap with existing ones
        total_samples = len(dataloader.dataset)

        if (
            isinstance(prototype_sampling, str)
            and prototype_sampling == "uniform_random"
        ):
            existing_proto_mask = []
            for _, items in enumerate(dataloader):
                sample_ids = items["sample_id"]
                for sample_id in sample_ids:
                    # existing_proto_mask.append(int(sample_id in existing_proto_sample_ids))
                    existing_proto_mask.append(0)

            existing_proto_mask = torch.tensor(existing_proto_mask)
            non_proto_indices = torch.where(existing_proto_mask == 0)[0]
            target_number_of_samples = min(
                target_number_of_samples, len(non_proto_indices)
            )

            if target_number_of_samples <= 0:
                raise Warning("Cannot sample more prototypes.")

            binary_mask = torch.zeros(len(existing_proto_mask))
            selected_indices = non_proto_indices[
                torch.randperm(len(non_proto_indices))[:target_number_of_samples]
            ]
            binary_mask[selected_indices] = 1

        elif isinstance(prototype_sampling, list):
            binary_mask = torch.zeros((total_samples,))
            try:
                binary_mask[prototype_sampling] = 1
            except Exception as e:
                raise Exception(
                    f"Cannot have prototype_sampling list of non-ints => {e}"
                )
        else:
            raise Exception("Invalid sampling")

        sampled_items = []
        mask_index = 0
        set_batch_size = dataloader.batch_size

        for _, items in enumerate(dataloader):
            batch_size = items["img"].shape[0]  # Get the actual batch size
            for i in range(batch_size):  # Iterate over each sample in the batch
                if (
                    binary_mask[mask_index] == 1
                ):  # Check the binary mask for the current sample
                    # Append the entire sample (i.e., for the i-th element of the batch)
                    sampled_items.append({key: items[key][i] for key in items})

                mask_index += 1

                # Stop once the target number of samples is reached
                if len(sampled_items) >= target_number_of_samples:
                    break

            if len(sampled_items) >= target_number_of_samples:
                break

        sampled_dataset = ListDataset(sampled_items)
        sampled_dataloader = DataLoader(
            sampled_dataset, batch_size=set_batch_size, shuffle=False
        )

        sampled_proto_cc_vectors = []
        sampled_proto_vectors = []

        self.initial_protopnet.eval()
        for item in tqdm(sampled_dataloader):
            x, targets = (
                item["img"],
                item["target"],
            )  # should be on cuda at this point
            x = x.to(self.device)
            latent_vectors = self.initial_protopnet.backbone(x)
            latent_vectors = self.initial_protopnet.add_on_layers(latent_vectors)
            # NOTE: we may want to make this process fancier in the future
            random_h = torch.randint(0, latent_vectors.shape[-2], (1,))[0]
            random_w = torch.randint(0, latent_vectors.shape[-1], (1,))[0]
            latent_vectors = (
                latent_vectors[:, :, random_h, random_w].unsqueeze(-1).unsqueeze(-1)
            )
            cc_vectors = torch.zeros(
                (x.shape[0], self.initial_protopnet.prototype_layer.num_classes),
                device=self.device,
            )  # bsz x num_classes
            # NOTE: under the assumption that we set CC to 1 for all positive class positions and rest are 0
            cc_vectors[torch.arange(cc_vectors.shape[0]), targets] = 1

            sampled_proto_vectors.append(latent_vectors)
            sampled_proto_cc_vectors.append(cc_vectors)

        sampled_proto_vectors = torch.cat(
            sampled_proto_vectors, dim=0
        )  # target_number_of_samples x ...
        sampled_proto_cc_vectors = torch.cat(
            sampled_proto_cc_vectors, dim=0
        )  # target_number_of_samples x ...

        self.initial_protopnet.add_additional_prototype(
            sampled_proto_vectors, sampled_proto_cc_vectors
        )

        # we need to call project again, so we have the correct prototype_info_dict
        with torch.no_grad():
            self.initial_protopnet.project(self.viz_dataloader)

        if immediately_refit:
            self._prepare_datasets(self.train_dataloader, self.val_dataloader)

            # we need to call fit again, so the shapes are correct
            self.rset.prototype_class_identity = self.initial_protopnet.prototype_layer.prototype_class_identity
            self.rset.fit(self.X_train, self.y_train)

    def sample_prototypes_by_index(
        self,
        dataloader: torch.utils.data.DataLoader,
        idx,
        target_h=None,
        target_w=None,
        immediately_refit=False,
        immediately_reproject: bool = False
    ):
        item = dataloader.dataset.__getitem__(idx)

        x, targets = (
            item["img"],
            item["target"],
        )  # should be on cuda at this point
        x = x.to(self.device).unsqueeze(0)

        latent_vectors = self.initial_protopnet.backbone(x)
        latent_vectors = self.initial_protopnet.add_on_layers(latent_vectors)
        # NOTE: we may want to make this process fancier in the future
        if target_h is None:
            target_h = torch.randint(0, latent_vectors.shape[-2], (1,))[0]
        if target_w is None:
            target_w = torch.randint(0, latent_vectors.shape[-1], (1,))[0]
        latent_vectors = (
            latent_vectors[:, :, target_h, target_w].unsqueeze(-1).unsqueeze(-1)
        )
        cc_vectors = torch.zeros(
            (x.shape[0], self.initial_protopnet.prototype_layer.num_classes),
            device=self.device,
        )  # bsz x num_classes
        # NOTE: under the assumption that we set CC to 1 for all positive class positions and rest are 0
        cc_vectors[torch.arange(cc_vectors.shape[0]), targets] = 1

        self.initial_protopnet.add_additional_prototype(
            latent_vectors, cc_vectors
        )

        # we need to call project again, so we have the correct prototype_info_dict
        if immediately_reproject:
            with torch.no_grad():
                self.initial_protopnet.project(self.viz_dataloader)

        if immediately_refit:
            self._prepare_datasets(self.train_dataloader, self.val_dataloader)

            # we need to call fit again, so the shapes are correct
            self.rset.prototype_class_identity = self.initial_protopnet.prototype_layer.prototype_class_identity
            self.rset.fit(self.X_train, self.y_train)

    def produce_protopnet_object(self, return_optimal=True):
        """
        Pick a protopnet from our constrained RSet, randomly at first but ideally
        using some better selection criterion
        Args:
            return_optimal : bool -- if True, return the ProtoPNet at the center of
                our running Rashomon set. Else, sample a random model from our running
                Rashomon set and use that
        Returns:
            protopnet : ProtoPNet -- a protopnet fitting all of our selection criteria
        """
        if return_optimal:
            self.initial_protopnet.eval()
            return self.initial_protopnet
        else:
            new_weights = self.rset._sample_model(target_dist_to_edge=0.1).linear_weights.data
            new_weights[abs(new_weights) < 1e-5] = 0.0
            model = self._update_protopnet_last_layer(new_weights)
            model.eval()
            return model


    def produce_protopnet_object_with_requirements(self, required_protos):
        """
        Pick a protopnet from our constrained RSet, randomly at first but ideally
        using some better selection criterion
        Args:
            return_optimal : bool -- if True, return the ProtoPNet at the center of
                our running Rashomon set. Else, sample a random model from our running
                Rashomon set and use that
        Returns:
            protopnet : ProtoPNet -- a protopnet fitting all of our selection criteria
        """
        new_weights = self.rset._sample_model_with_constraints(required_protos).linear_weights.data
        new_weights[abs(new_weights) < 1e-5] = 0.0
        model = self._update_protopnet_last_layer(new_weights)
        model.eval()
        return model