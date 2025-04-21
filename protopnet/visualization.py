import argparse
import logging
import pathlib
import random
import warnings
from typing import List, Tuple

import cv2
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from protopnet.datasets import training_dataloaders
from protopnet.prototypical_part_model import ProtoPNet

from .utilities.general_utilities import find_high_activation_crop
from .utilities.project_utilities import custom_unravel_index, hash_func
from .utilities.visualization_utilities import indices_to_upsampled_boxes

log = logging.getLogger(__name__)


class KeyReturningDict(dict):
    """
    Dictionary that gives us default values for missing keys.
    Used to make some of the bootstrapping for the visualization less onerous.
    """

    def __missing__(self, key):
        return key


def denormalize_tensor(tensor, mean, std):
    """
    Reverts the normalization applied to the tensor image.
    Args:
        tensor: the normalized image as a torch.Tensor.
        mean: list or tuple of means for each channel.
        std: list or tuple of standard deviations for each channel.

    Returns:
        Reverted image in numpy form.
    """
    tensor = tensor.clone()

    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)

    reverted_image = tensor.permute(1, 2, 0).numpy()
    reverted_image = np.clip(reverted_image, 0, 1)

    return reverted_image


def denormalize_numpy(ori_img, mean, std):
    """
    Reverts the normalization applied to the image in numpy format.
    Args:
        ori_img: The normalized image as a numpy array (shape: (H, W, C)).
        mean: list or tuple of means for each channel.
        std: list or tuple of standard deviations for each channel.

    Returns:
        Reverted image in numpy form with shape (H, W, C).
    """
    ori_img = ori_img.copy()

    # Reverse normalization per channel
    for c in range(ori_img.shape[2]):  # Loop over the channels (C dimension)
        ori_img[:, :, c] = ori_img[:, :, c] * std[c] + mean[c]

    # Clip the values to [0, 1]
    reverted_image = np.clip(ori_img, 0, 1)

    return reverted_image


def cv2_overlay_heatmap(activation_0_to_1, ori_image):
    heatmap = cv2.applyColorMap(
        np.uint8(activation_0_to_1 * 255),
        cv2.COLORMAP_JET,
    )
    heatmap = np.float32(heatmap) / 255
    heatmap = heatmap[..., ::-1]
    overlayed_img = 0.5 * ori_image + 0.3 * heatmap
    return heatmap, overlayed_img


def save_prototype_images_to_file(
    model,
    push_dataloader,
    save_loc,
    img_size,
    normalize_for_fwd=None,
    denormalize_mean=None,
    denormalize_std=None,
    box_color=(0, 255, 255),
    device="cpu",
    specify_proto_idx=None,
    specify_proto_class=None,
):
    """
    Save prototype images, and images overlaid with heatmaps, to save_loc.
    """

    prototype_info_dict = model.prototype_layer.prototype_info_dict

    assert not (
        specify_proto_idx is not None and specify_proto_class is not None
    ), "specify_proto_idx and specify_proto_class cannot be both optioned."

    if specify_proto_idx is not None:
        assert isinstance(
            specify_proto_idx, (List, np.ndarray)
        ), "specify_proto_idx needs to be a list or numpy array."
        assert all(
            isinstance(i, (int, np.integer)) for i in specify_proto_idx
        ), "All elements in specify_proto_idx need to be integers."

    specified_returns = []

    proto_dict_inv = dict()
    for proto_index, proto_info in prototype_info_dict.items():
        proto_dict_inv[proto_info.sample_id] = proto_info.sample_hash

    protos_vizualized = [0] * len(prototype_info_dict)
    batches_not_run = 0
    batches_run = 0
    non_matching_hashes = 0

    all_sample_ids = []
    proto_idx_found = []
    for batch_data_dict in tqdm(push_dataloader):
        search_batch_images = batch_data_dict["img"].to(device)
        labels = batch_data_dict["target"]
        sample_ids = batch_data_dict["sample_id"]

        for x in sample_ids:
            all_sample_ids.append(x.numpy().tobytes())

        run_batch = False
        for sample_id in sample_ids:
            sample_id_bytes = sample_id.numpy().tobytes()
            if sample_id_bytes in proto_dict_inv.keys():
                run_batch = True
                break

        if not run_batch:
            batches_not_run += 1
            continue

        batches_run += 1
        model_outputs = model(
            (
                normalize_for_fwd(search_batch_images)
                if normalize_for_fwd
                else search_batch_images
            ),
            return_prototype_layer_output_dict=True,
        )

        latent_space_size_activation_maps = model_outputs["prototype_activations"]

        # (batch, proto_h, proto_w, H, W, 2)
        # prototype_sample_location_map
        if "prototype_sample_location_map" in model_outputs:
            prototype_sample_location_map = model_outputs[
                "prototype_sample_location_map"
            ]
        else:
            prototype_sample_location_map = None

        # TODO: get img_h, img_w values in a better way than direct passing (grab from dataloader? model?)
        img_height, img_width = img_size
        pixel_space_size_activation_maps = torch.nn.Upsample(
            size=(img_height, img_width), mode="bilinear", align_corners=False
        )(latent_space_size_activation_maps)

        for index_in_batch, sample_id in enumerate(sample_ids):

            label = labels[index_in_batch]

            if specify_proto_class is not None and label != specify_proto_class:
                continue

            sample_id_bytes = sample_id.numpy().tobytes()

            for proto_index, proto_info in prototype_info_dict.items():

                if specify_proto_idx is not None:
                    if proto_index not in specify_proto_idx:
                        continue

                img_save_path = save_loc / "prototypes"
                img_save_path.mkdir(parents=True, exist_ok=True)

                (
                    overlay_heatmap_path,
                    original_path,
                    cropped_part_path,
                    heatmap_path,
                    proto_bbox_path,
                    part_locs_path,
                ) = (
                    img_save_path / f"proto_{proto_index}_overlayheatmap.png",
                    img_save_path / f"proto_{proto_index}_original.png",
                    img_save_path / f"proto_{proto_index}_cropped_region.png",
                    img_save_path / f"proto_{proto_index}_heatmap.png",
                    img_save_path / f"proto_{proto_index}_proto_bbox.png",
                    img_save_path / f"proto_{proto_index}_part_locs.png",
                )

                if sample_id_bytes != proto_info.sample_id:
                    continue

                proto_idx_found.append(sample_id_bytes)
                # if (
                #     not proto_dict_inv[sample_id_bytes]
                #     == hash_func(search_batch_images[index_in_batch])
                #     and non_matching_hashes < 2
                # ):
                #     non_matching_hashes += 1
                #     warnings.warn(
                #         f"Hash of prototype sample does not match hash of visualization sample for {sample_id}, prototype {proto_index}."
                #     )
                # elif non_matching_hashes == 2:
                #     warnings.warn("Will not report subsequent unmatching hashes.")

                original_img = (
                    search_batch_images[index_in_batch].clone().cpu().detach().numpy()
                )
                # original_img is height width channel at this point
                original_img = np.transpose(original_img, [1, 2, 0])
                if denormalize_mean and denormalize_std:
                    original_img = denormalize_numpy(original_img, denormalize_mean, denormalize_std)

                # make heat map
                pixel_space_size_activation_map = pixel_space_size_activation_maps[
                    index_in_batch, proto_index, :, :
                ]
                pixel_space_size_activation_map = (
                    pixel_space_size_activation_map
                    - torch.min(pixel_space_size_activation_map)
                )
                pixel_space_size_activation_map = (
                    pixel_space_size_activation_map
                    / torch.max(pixel_space_size_activation_map)
                )

                (
                    proto_part_lower_y,
                    proto_part_upper_y,
                    proto_part_right_x,
                    proto_part_left_x,
                ) = find_high_activation_crop(
                    pixel_space_size_activation_map.clone().detach().cpu().numpy(),
                    percentile=95,
                )

                cropped_part_img = original_img.copy()[
                    proto_part_lower_y:proto_part_upper_y-1,
                    proto_part_right_x:proto_part_left_x-1
                ]

                ori_img_bgr = cv2.cvtColor(
                    np.uint8(255 * original_img), cv2.COLOR_RGB2BGR
                )
                ori_img_bgr_w_box = ori_img_bgr.copy()
                cv2.rectangle(
                    ori_img_bgr_w_box,
                    (proto_part_right_x, proto_part_lower_y),
                    (proto_part_left_x - 1, proto_part_upper_y - 1),
                    box_color,
                    thickness=2,
                )
                ori_img_rgb_w_box = ori_img_bgr_w_box[..., ::-1]
                ori_img_rgb_w_box = np.float32(ori_img_rgb_w_box) / 255

                # TODO: change from np to only tensor use if possible here
                heatmap, overlayed_img = cv2_overlay_heatmap(
                    pixel_space_size_activation_map.clone().cpu().detach().numpy(),
                    original_img,
                )

                # saving images
                plt.imsave(
                    overlay_heatmap_path,
                    overlayed_img,
                )
                plt.imsave(
                    original_path,
                    original_img,
                )
                plt.imsave(
                    heatmap_path,
                    heatmap,
                )
                plt.imsave(
                    cropped_part_path,
                    cropped_part_img
                )
                plt.imsave(
                    proto_bbox_path,
                    ori_img_rgb_w_box,
                )

                protos_vizualized[proto_index] = 1

                if prototype_sample_location_map is None:
                    # TODO - why would this happen?
                    # when this gets triggered, it's the end
                    specified_returns.append(
                        (
                            overlay_heatmap_path,
                            original_path,
                            heatmap_path,
                            proto_bbox_path,
                        )
                    )
                    continue

                cur_proto_act = latent_space_size_activation_maps[
                    index_in_batch, proto_index
                ]
                batch_argmax_cur_proto_act = list(
                    custom_unravel_index(
                        torch.argmax(cur_proto_act, axis=None),
                        cur_proto_act.shape,
                    )
                )
                log.info(f"batch_argmax_cur_proto_act: {batch_argmax_cur_proto_act}")
                log.info(f"cur_proto_act: {cur_proto_act}")
                best_locs_for_proto = prototype_sample_location_map[
                    index_in_batch,
                    :,
                    :,
                    batch_argmax_cur_proto_act[0],
                    batch_argmax_cur_proto_act[1],
                ]
                img_with_boxes = original_img.copy()
                for part_h in range(best_locs_for_proto.shape[0]):
                    for part_w in range(best_locs_for_proto.shape[1]):
                        box_coords = indices_to_upsampled_boxes(
                            best_locs_for_proto[part_h, part_w],
                            cur_proto_act.shape[-2:],
                            original_img.shape[:-1],
                        )
                        cv2.rectangle(
                            img_with_boxes,
                            box_coords[0],
                            box_coords[1],
                            (1.0, 0.0, 0.0),
                            2,
                        )

                plt.imsave(
                    part_locs_path,
                    img_with_boxes,
                )
                del heatmap, pixel_space_size_activation_map
                # TODO: save activation map itself after picking a format for that

                specified_returns.append(
                    (
                        overlay_heatmap_path,
                        original_path,
                        heatmap_path,
                        proto_bbox_path,
                        part_locs_path,
                    )
                )

        del pixel_space_size_activation_maps
        del latent_space_size_activation_maps

    log.info(
        f"Batches not run because they contained no prototype samples: {batches_not_run}. Batches run: {batches_run}."
    )
    protos_vizualized = np.asarray(protos_vizualized)

    if specify_proto_idx is None and specify_proto_class is None:
        assert np.sum(protos_vizualized) == len(
            prototype_info_dict
        ), f"protos_vizualized is {np.sum(protos_vizualized)}; len(prototype_info_dict) = {len(prototype_info_dict)}. These should match."
    elif specify_proto_idx is not None:
        assert np.sum(protos_vizualized) == len(
            specify_proto_idx
        ), "protos_vizualized should be equal to len(specify_proto_idx) when specify_proto_idx is set."

    return specified_returns if len(specified_returns) > 1 else specified_returns[0]


def local_analysis_plotter(
    prototype_class_identity,
    plot_save_path,
    ori_image,
    proto_actmaps_on_image,
    proto_indices,
    proto_save_dir,
    sim_scores,
    class_connections,
    num_top_protos_viewed,
    pred,
    truth,
    class_name_ref_dict,
    close_fig=True,
):
    anno_opts_cen = dict(
        xy=(0.4, 0.5), xycoords="axes fraction", va="center", ha="center"
    )
    anno_opts_symb = dict(
        xy=(1, 0.5), xycoords="axes fraction", va="center", ha="center"
    )
    anno_opts_sum = dict(xy=(0, -0.1), xycoords="axes fraction", va="center", ha="left")

    fig = plt.figure(constrained_layout=False)
    fig.set_size_inches(28, 4 * num_top_protos_viewed)

    ncols, nrows = 7, num_top_protos_viewed
    spec = gridspec.GridSpec(ncols=ncols, nrows=nrows, figure=fig)

    f_axes = []
    for row in range(nrows):
        f_axes.append([])
        for col in range(ncols):
            f_axes[-1].append(fig.add_subplot(spec[row, col]))

    # FIXME: shouldn't be setting this for all of matplotlib
    plt.rcParams.update({"font.size": 14})

    for ax_num, ax in enumerate(f_axes[0]):
        if ax_num == 0:
            ax.set_title("Test image", fontdict=None, loc="left", color="k")
        elif ax_num == 1:
            ax.set_title(
                "Test image activation\nby prototype",
                fontdict=None,
                loc="left",
                color="k",
            )
        elif ax_num == 2:
            ax.set_title("Prototype", fontdict=None, loc="left", color="k")
        elif ax_num == 3:
            ax.set_title(
                "Self-activation of\nprototype", fontdict=None, loc="left", color="k"
            )
        elif ax_num == 4:
            ax.set_title("Similarity score", fontdict=None, loc="left", color="k")
        elif ax_num == 5:
            ax.set_title("Class connection", fontdict=None, loc="left", color="k")
        elif ax_num == 6:
            ax.set_title("Contribution", fontdict=None, loc="left", color="k")
        else:
            pass

    plt.rcParams.update({"font.size": 22})

    for ax in [f_axes[r][4] for r in range(nrows)]:
        ax.annotate("x", **anno_opts_symb)

    for ax in [f_axes[r][5] for r in range(nrows)]:
        ax.annotate("=", **anno_opts_symb)

    for ax in [f_axes[r][0] for r in range(nrows)]:
        ax.imshow(ori_image)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])

    for top_p in range(num_top_protos_viewed):
        # put info in place
        proto_idx = proto_indices[top_p]
        proto_label = torch.argmax(prototype_class_identity[proto_idx]).item()
        sim_score = sim_scores[top_p]
        proto_class_str = str(
            class_name_ref_dict[proto_label]
            if class_name_ref_dict is not None
            else proto_label
        )
        top_cc = class_connections[0, proto_idx].item()

        for ax in [f_axes[top_p][4]]:
            ax.annotate(round(sim_score, 3), **anno_opts_cen)
            ax.set_axis_off()
        for ax in [f_axes[top_p][5]]:
            ax.annotate(str(round(top_cc, 3)) + "\n" + proto_class_str, **anno_opts_cen)
            ax.set_axis_off()
        for ax in [f_axes[top_p][6]]:
            tc = round(top_cc * sim_score, 3)
            ax.annotate("{0:.3f}".format(tc) + "\n" + proto_class_str, **anno_opts_cen)
            ax.set_axis_off()
        # put images in place
        p_img = Image.open(proto_save_dir / f"proto_{proto_idx}_original.png")
        for ax in [f_axes[top_p][2]]:
            ax.imshow(p_img)
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])

            ax.text(
                30,
                10,
                f"{proto_idx}",
                color="white",
                fontsize=17,
                ha="right",
                va="top",
                bbox=dict(
                    facecolor="black", edgecolor="black", boxstyle="round,pad=0.2"
                ),
            )

        p_act_img = Image.open(proto_save_dir / f"proto_{proto_idx}_overlayheatmap.png")
        for ax in [f_axes[top_p][3]]:
            ax.imshow(p_act_img)
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
        _, act_img = cv2_overlay_heatmap(proto_actmaps_on_image[top_p], ori_image)
        for ax in [f_axes[top_p][1]]:
            ax.imshow(act_img)
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])

    # summary
    f_axes[2][4].annotate(
        f"This {class_name_ref_dict[int(truth)]} is classified as {class_name_ref_dict[int(pred)]}.",
        **anno_opts_sum,
    )

    plt.savefig(plot_save_path, bbox_inches="tight", pad_inches=0)
    if close_fig:
        plt.close(fig)


def local_analysis(
    model,
    save_loc,
    push_dataloader,
    eval_dataloader,
    img_size,
    device,
    sample=None,
    num_top_protos_viewed=3,
    class_name_ref_dict=KeyReturningDict(),
    run_proto_vis=True,
    specify_img_idx=None,
    normlization_params=((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    last_layer_coef=None,  # turn on sort by contrib points by pass in the last_layer_coef
    reasoning_for_top_c=None, # If int, show reasoning or c_th highest class logit
    sort_using_logit=False # Set true to sort using logit instead of activation
):

    all_save_paths = []

    if run_proto_vis:
        save_prototype_images_to_file(
            model, push_dataloader, save_loc, img_size, device=device
        )

    local_ana_save_dir = save_loc / "local_analysis_saves"
    local_ana_save_dir.mkdir(parents=True, exist_ok=True)

    num_saved_analyses = 0
    total_img_count = 0

    if specify_img_idx is not None:
        for i, packet in enumerate(eval_dataloader):
            inds_for_batch = [i*eval_dataloader.batch_size + j for j in range(eval_dataloader.batch_size)]
            if specify_img_idx in inds_for_batch:
                dict_final = {}
                for k in packet:
                    dict_final[k] = packet[k][specify_img_idx % eval_dataloader.batch_size]
                break
        dataset = [dict_final]
        eval_dataloader = DataLoader(dataset, batch_size=20)  # any batchsize will do

    for batch_data_dict in tqdm(eval_dataloader):

        labels = batch_data_dict["target"]

        batch_images = batch_data_dict["img"].to(device)

        model_outputs = model(
            batch_images,
            return_prototype_layer_output_dict=True,
            return_similarity_score_to_each_prototype=True,
        )
        # batch size, num_protos, latent_height, latent_width
        latent_space_size_activation_maps = model_outputs["prototype_activations"]

        # batch size, num_protos
        similarity_score_to_each_prototype = model_outputs[
            "similarity_score_to_each_prototype"
        ]

        # batch size, k
        if reasoning_for_top_c is not None:
            _, target_class = torch.kthvalue(-model_outputs["logits"].squeeze(0), k=reasoning_for_top_c)
            last_layer_coef = last_layer_coef * model.prototype_layer.prototype_class_identity[:, target_class]
        if sort_using_logit: #last_layer_coef is not None:
            last_layer_coef = last_layer_coef.unsqueeze(0)
            _, top_k_proto_indicies = torch.topk(
                similarity_score_to_each_prototype * last_layer_coef,
                k=num_top_protos_viewed,
                dim=1,
            )
            top_k_proto_sim_scores = torch.gather(
                similarity_score_to_each_prototype, 1, top_k_proto_indicies
            )
        elif last_layer_coef is not None:
            last_layer_coef = last_layer_coef.unsqueeze(0)
            _, top_k_proto_indicies = torch.topk(
                similarity_score_to_each_prototype * 1.0 * (last_layer_coef > 0),
                k=num_top_protos_viewed,
                dim=1,
            )
            top_k_proto_sim_scores = torch.gather(
                similarity_score_to_each_prototype, 1, top_k_proto_indicies
            )
        else:
            top_k_proto_sim_scores, top_k_proto_indicies = torch.topk(
                similarity_score_to_each_prototype, k=num_top_protos_viewed, dim=1
            )

        predictions = torch.argmax(model_outputs["logits"], dim=1)

        for index_in_batch in range(batch_images.shape[0]):

            ori_img = batch_images[index_in_batch].clone().cpu().detach().numpy()
            sample_id = hash_func(ori_img)
            # ori_img is height width channel at this point
            ori_img = np.transpose(ori_img, [1, 2, 0])
            mean, std = normlization_params
            ori_img = denormalize_numpy(ori_img, mean, std)

            # k, latent_height, latent_width
            proto_actmaps_on_image = latent_space_size_activation_maps[
                index_in_batch, top_k_proto_indicies[index_in_batch], :, :
            ].unsqueeze(1)
            proto_actmaps_on_image_normed = (
                proto_actmaps_on_image - proto_actmaps_on_image.min()
            ) / (proto_actmaps_on_image.max() - proto_actmaps_on_image.min())
            img_height, img_width = img_size
            proto_actmaps_on_image_normed = (
                torch.nn.Upsample(
                    size=(img_height, img_width), mode="bilinear", align_corners=False
                )(proto_actmaps_on_image_normed)
                .squeeze(1)
                .clone()
                .detach()
                .cpu()
                .numpy()
            )

            local_prototype_dir = save_loc / "prototypes"
            local_prototype_dir.mkdir(parents=True, exist_ok=True)
            save_path = local_ana_save_dir / f"{sample_id}.png"

            local_analysis_plotter(
                prototype_class_identity=model.prototype_layer.prototype_class_identity,
                plot_save_path=save_path,
                ori_image=ori_img,
                proto_actmaps_on_image=proto_actmaps_on_image_normed,
                proto_indices=top_k_proto_indicies[index_in_batch]
                .clone()
                .detach()
                .cpu()
                .numpy(),
                proto_save_dir=local_prototype_dir,
                sim_scores=top_k_proto_sim_scores[index_in_batch]
                .clone()
                .detach()
                .cpu()
                .numpy(),
                class_connections=(
                    model.prototype_prediction_head.class_connection_layer.weight
                    if last_layer_coef is None
                    else last_layer_coef
                ),  # num_classes, num_protos or # num_potatos
                class_name_ref_dict=class_name_ref_dict,
                num_top_protos_viewed=num_top_protos_viewed,
                pred=predictions[index_in_batch].item(),
                truth=labels[index_in_batch].item(),
            )

            all_save_paths.append(save_path)

            del ori_img, proto_actmaps_on_image, proto_actmaps_on_image_normed

            num_saved_analyses += 1

            if sample is not None and num_saved_analyses >= sample:
                return None

        total_img_count += len(labels)

        del (
            batch_images,
            labels,
            model_outputs,
            latent_space_size_activation_maps,
            similarity_score_to_each_prototype,
            top_k_proto_sim_scores,
            top_k_proto_indicies,
            predictions,
        )

        return all_save_paths[0] if len(all_save_paths) == 1 else all_save_paths


def global_analysis(
    model,
    save_loc,
    proto_save_dir,
    proto_index,
    project_dataloader,
    img_size,
    device,
    num_top_samples_viewed=3,
    box_color=(0, 255, 255),
    normlization_params=(0, 1)
):

    log.info("Running global analysis for prototype at index %s", proto_index)
    global_ana_save_dir = save_loc / "global_analysis_saves"
    global_ana_save_dir.mkdir(parents=True, exist_ok=True)

    heatmap_path = (
        global_ana_save_dir / f"{proto_index}_view_{num_top_samples_viewed}_heatmap.png"
    )
    views_path = (
        global_ana_save_dir / f"{proto_index}_view_{num_top_samples_viewed}.png"
    )

    if heatmap_path.exists() and views_path.exists():
        return views_path, heatmap_path

    top_k_sim_scores = []
    top_k_sample_infos = []
    all_sample_to_proto_similarity_scores = {}

    model = model.eval()
    for batch_data_dict in tqdm(project_dataloader):
        batch_images = batch_data_dict["img"].to(device)
        labels = batch_data_dict["target"]
        sample_ids = batch_data_dict["sample_id"]

        with torch.no_grad():
            model_outputs = model(
                # eval_normalize(batch_images),
                batch_images,
                return_prototype_layer_output_dict=True,
                return_similarity_score_to_each_prototype=True,
            )
        # batch size, num_protos, latent_height, latent_width
        latent_space_size_activation_maps = model_outputs["prototype_activations"]

        img_height, img_width = img_size
        pixel_space_size_activation_maps = torch.nn.Upsample(
            size=(img_height, img_width), mode="bilinear", align_corners=False
        )(latent_space_size_activation_maps)

        # batch size, num_protos
        similarity_score_to_each_prototype = model_outputs[
            "similarity_score_to_each_prototype"
        ]
        log.debug("all similarity scores %s", similarity_score_to_each_prototype.shape)
        sample_to_proto_similarity_scores = similarity_score_to_each_prototype[
            :, proto_index
        ]
        log.debug("prototype similarity scores %s", sample_to_proto_similarity_scores)

        for idx, sample_id in enumerate(sample_ids):
            sample_id_bytes = sample_id.numpy().tobytes()
            all_sample_to_proto_similarity_scores[sample_id_bytes] = (
                sample_to_proto_similarity_scores.detach().cpu().numpy()[idx]
            )

        for in_batch_idx in range(batch_images.shape[0]):

            pixel_space_size_activation_map = pixel_space_size_activation_maps[
                in_batch_idx, proto_index, :, :
            ]
            pixel_space_size_activation_map = (
                pixel_space_size_activation_map
                - torch.min(pixel_space_size_activation_map)
            )
            pixel_space_size_activation_map = (
                pixel_space_size_activation_map
                / torch.max(pixel_space_size_activation_map)
            )

            original_img = np.transpose(
                batch_images[in_batch_idx].clone().cpu().detach().numpy(), [1, 2, 0]
            )
            mean, std = normlization_params
            original_img = denormalize_numpy(original_img, mean, std)
            heatmap, overlayed_img = cv2_overlay_heatmap(
                pixel_space_size_activation_map.clone().cpu().detach().numpy(),
                original_img,
            )

            (
                proto_part_lower_y,
                proto_part_upper_y,
                proto_part_right_x,
                proto_part_left_x,
            ) = find_high_activation_crop(
                pixel_space_size_activation_map.clone().detach().cpu().numpy(),
                percentile=95,
            )

            packet = [
                (
                    proto_part_lower_y,
                    proto_part_upper_y,
                    proto_part_right_x,
                    proto_part_left_x,
                ),
                denormalize_numpy(np.transpose(
                    batch_images[in_batch_idx].clone().cpu().detach().numpy(), [1, 2, 0]
                ), mean, std),
                overlayed_img,
                labels[in_batch_idx].item(),
                sample_ids[in_batch_idx],
                sample_to_proto_similarity_scores[in_batch_idx].item(),
            ]

            if len(top_k_sim_scores) < num_top_samples_viewed:
                top_k_sim_scores.append(
                    sample_to_proto_similarity_scores[in_batch_idx].item()
                )
                top_k_sample_infos.append(packet)
            else:
                min_value = min(top_k_sim_scores)
                min_index = top_k_sim_scores.index(min_value)

                if sample_to_proto_similarity_scores[in_batch_idx] > min_value:
                    top_k_sim_scores[min_index] = sample_to_proto_similarity_scores[
                        in_batch_idx
                    ]
                    top_k_sample_infos[min_index] = packet

            combined = list(zip(top_k_sim_scores, top_k_sample_infos))
            combined_sorted = sorted(combined, key=lambda x: x[0])
            top_k_sim_scores = [element[0] for element in combined_sorted]
            top_k_sample_infos = [element[1] for element in combined_sorted]

        del (
            model_outputs,
            pixel_space_size_activation_map,
            latent_space_size_activation_maps,
        )

    sorted_items = sorted(
        all_sample_to_proto_similarity_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )
    _ = sorted_items[:num_top_samples_viewed]  # was topk

    actual_proto_sample_id = model.prototype_layer.prototype_info_dict[
        proto_index
    ].sample_id
    log.debug(
        "%s %s, %s",
        proto_index,
        actual_proto_sample_id,
        all_sample_to_proto_similarity_scores[actual_proto_sample_id],
    )
    log.debug(top_k_sample_infos)
    p_img = Image.open(proto_save_dir / f"proto_{proto_index}_proto_bbox.png")

    selected_samples_to_target_proto = [p_img]
    selected_samples_to_target_proto_heatmap = [p_img]
    # titles = [f"Prototype {proto_index}"]
    titles = ["Prototype"]

    for idx, packet in reversed(list(enumerate(top_k_sample_infos))):

        bbox, original_img, overlayed_img, label, sample_id, sim_score = packet
        (
            proto_part_lower_y,
            proto_part_upper_y,
            proto_part_right_x,
            proto_part_left_x,
        ) = bbox

        ori_img_bgr = cv2.cvtColor(np.uint8(255 * original_img), cv2.COLOR_RGB2BGR)
        ori_img_bgr_w_box = ori_img_bgr.copy()
        cv2.rectangle(
            ori_img_bgr_w_box,
            (proto_part_right_x, proto_part_lower_y),
            (proto_part_left_x - 1, proto_part_upper_y - 1),
            box_color,
            thickness=2,
        )
        ori_img_rgb_w_box = ori_img_bgr_w_box[..., ::-1]
        ori_img_rgb_w_box = np.float32(ori_img_rgb_w_box) / 255
        selected_samples_to_target_proto.append(ori_img_rgb_w_box)
        # titles.append(f"Sample {num_top_samples_viewed - idx} \nSimilarity score {round(sim_score, 3)}")
        titles.append(f"Sample {num_top_samples_viewed - idx}")

        selected_samples_to_target_proto_heatmap.append(overlayed_img)

    N = len(selected_samples_to_target_proto)

    _ = plt.figure(figsize=(5 * N + 2, 5))
    gs = gridspec.GridSpec(1, N + 1, width_ratios=[1, 0.3] + [1] * (N - 1))

    axes = []
    for i in range(N):
        if i == 1:
            axes.append(plt.subplot(gs[i + 1]))
        else:
            axes.append(plt.subplot(gs[i if i == 0 else i + 1]))

    for i, ax in enumerate(axes):
        ax.imshow(selected_samples_to_target_proto[i])
        # ax.set_title(titles[i], fontsize=20)
        ax.axis("off")

    plt.savefig(
        views_path,
        bbox_inches="tight",
    )
    plt.clf()

    _ = plt.figure(figsize=(5 * N + 2, 5))
    gs = gridspec.GridSpec(1, N + 1, width_ratios=[1, 0.3] + [1] * (N - 1))

    axes = []
    for i in range(N):
        if i == 1:
            axes.append(plt.subplot(gs[i + 1]))
        else:
            axes.append(plt.subplot(gs[i if i == 0 else i + 1]))

    for i, ax in enumerate(axes):
        ax.imshow(selected_samples_to_target_proto_heatmap[i])
        # ax.set_title(titles[i], fontsize=20)
        ax.axis("off")

    plt.savefig(
        heatmap_path,
        bbox_inches="tight",
    )
    plt.clf()

    return views_path, heatmap_path


def recover_proto_info_dict(model, push_dataloader, model_save_loc, device):
    prototype_info_dict = model.prototype_layer.prototype_info_dict

    if len(prototype_info_dict) == 0:

        prototype_info_dict = {}
        prototype_tensors = model.prototype_layer.prototype_tensors
        prototype_matches = [0] * model.prototype_layer.num_prototypes
        for batch_data_dict in tqdm(push_dataloader):
            search_batch_images = batch_data_dict["img"].to(device)
            sample_ids = batch_data_dict["sample_id"].numpy().tobytes()

            search_batch_latent_vectors = model.add_on_layers(
                model.backbone(search_batch_images)
            )
            log.info(search_batch_latent_vectors.shape)
            log.info(prototype_tensors.shape)

            for in_batch_idx in range(search_batch_latent_vectors.shape[0]):
                difference = (
                    search_batch_latent_vectors[in_batch_idx] - prototype_tensors
                )
                zero_vector = torch.zeros(256)
                is_zero_vector = torch.isclose(
                    difference, zero_vector.view(1, -1, 1, 1)
                )
                is_zero_vector_reduced = is_zero_vector.all(dim=1, keepdim=True)

                for match_idx in torch.nonzero(is_zero_vector_reduced):
                    prototype_matches[match_idx] += 1
                    prototype_info_dict[sample_ids[in_batch_idx]] = hash_func(
                        search_batch_images[in_batch_idx]
                    )

        log.info(sum(prototype_matches) / len(prototype_matches))
        log.info(len(prototype_info_dict))

        model.prototype_layer.prototype_info_dict = prototype_info_dict

        torch.save(
            obj=model,
            f=model_save_loc,
        )


def reproject_prototypes(model, output_path, project_dataloader):
    """
    Project the prototypes and save the reprojected model to the output path.
    """

    # Combine parent directory with the new file name
    reprojected_model_save_loc = output_path / "reprojected_model.pth"
    reprojected_model_infodict_save_loc = output_path / "reprojected_proto_info_dict.pt"
    reprojected_model_save_loc.parent.mkdir(parents=True, exist_ok=True)

    log.info("Reprojecting model and saving to %s", reprojected_model_save_loc)

    if reprojected_model_infodict_save_loc.exists():
        # Allow for caching of the prototype info dict between runs
        proto_info_dict = torch.load(reprojected_model_infodict_save_loc, weights_only=False)

        if (
            len(proto_info_dict.keys())
            == model.prototype_layer.prototype_tensors.shape[0]
        ):
            model.prototype_layer.prototype_info_dict = proto_info_dict
            log.info(
                f"Saved exist, reloading info dict len={len(model.prototype_layer.prototype_info_dict.keys())}"
            )
            return model

    # push old model to get proto hashes (kind of hacky)
    # FIXME: check that this didn't change results somehow
    with torch.no_grad():
        model.project(project_dataloader)
    # log.info("description of protos without vis: ", model.describe_prototypes())
    log.info("Project complete, saving model to %s", reprojected_model_save_loc)

    torch.save(
        obj=model,
        f=reprojected_model_save_loc,
    )
    torch.save(
        obj=model.prototype_layer.prototype_info_dict,
        f=reprojected_model_infodict_save_loc,
    )
    log.debug(
        f"After project info dict len={len(model.prototype_layer.prototype_info_dict.keys())}"
    )
    log.info(
        "Reloading saved model after project from %s", str(reprojected_model_save_loc)
    )

    model = torch.load(reprojected_model_save_loc, weights_only=False)
    log.info(
        f"After reloading project info dict len={len(model.prototype_layer.prototype_info_dict.keys())}"
    )

    return model


def sample_activations(folder_path, n, title):
    """
    Example usage
    sample_activations(
        'protopnext/all_analysis_2/vis_c3tskyq5_90_project_0.7196_reprojected.pth/global_analysis_saves/',
        15,
        ' Prototype' + ' '*79+'Nearest 10 Samples'
    )
    """

    # List all files in the directory
    files = pathlib.Path(folder_path)

    # Filter out files that do not end with ".png" or end with "_heatmap.png"
    images = [
        file
        for file in files
        if file.endswith(".png") and not file.endswith("_heatmap.png")
    ]

    # Randomly select N images
    selected_images = random.sample(images, n)

    # Create figure and axes
    _, axs = plt.subplots(n, 1, figsize=(20, n * 2))

    # Check if we have multiple axes or just one
    if n == 1:
        axs = [axs]  # Make it iterable if there is only one subplot

    # Plot each image
    for idx, (ax, image_name) in tqdm(enumerate(zip(axs, selected_images)), total=n):
        img_path = folder_path / image_name
        img = Image.open(img_path)
        ax.imshow(img)
        if idx == 0:
            ax.set_title(title, loc="left", fontsize=21)
        ax.axis("off")  # Turn off axis

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.01)  # Adjust the horizontal space between images
    plt.savefig(f"plops/global_choose_{n}.png", dpi=200, bbox_inches="tight")


def run_analyses(
    *,
    model: ProtoPNet,
    model_path: pathlib.Path,
    project_dataloader,
    val_dataloader,
    vis_save_loc: pathlib.Path,
    analysis_type: str,
    output_dir: pathlib.Path = pathlib.Path("analyses/"),
    img_size: Tuple[int, int] = (224, 224),
    batch_size: int = 4,
    global_analysis_top_samples: int = 10,
    sample: int = None,
    local_analysis_top_comparisons: int = 3,
    device: torch.device = (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    ),
    normlization_params=((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    last_layer_coef=None,
    hard_reproject=False,
    prune_duplicates=True,
    denormalize_viz=False,
    class_name_ref_dict=KeyReturningDict()
):
    """
    Run Global and Local Analyses for a trained model.

    This should be broken into individual functions for each type of analysis in the future.
    """
    if prune_duplicates:
        model.prune_duplicate_prototypes()

    # TODO - why is this necessary?
    model.prototype_layer.latent_spatial_size = None

    if model_path:
        vis_save_loc = (
            output_dir / f"vis_{model_path.parent.name + '_' + model_path.name}"
        )

    vis_save_loc.mkdir(parents=True, exist_ok=True)

    # check to see if all prototypes are matched into the info_dict
    if (
        len(model.prototype_layer.prototype_info_dict.keys())
        != model.prototype_layer.prototype_tensors.shape[0]
    ) or hard_reproject:
        print("About to reproject")
        model = reproject_prototypes(model, vis_save_loc, project_dataloader)

    if analysis_type == "render-prototypes":

        log.info(f"rendering prototypes to {vis_save_loc}")

        save_prototype_images_to_file(
            model,
            project_dataloader,
            vis_save_loc,
            img_size,
            device=device,
            denormalize_mean=normlization_params[0] if denormalize_viz else None,
            denormalize_std=normlization_params[1] if denormalize_viz else None,
        )

        log.info("Completed rendering of prototypes.")

    elif analysis_type == "global":
        for proto_index in range(model.prototype_layer.num_prototypes):
            if sample is not None and proto_index >= sample:
                break

            # proto_index = 11
            global_analysis(
                model,
                save_loc=vis_save_loc,
                proto_save_dir=vis_save_loc / "prototypes",
                proto_index=proto_index,
                project_dataloader=project_dataloader,
                img_size=img_size,
                device=device,
                num_top_samples_viewed=global_analysis_top_samples,
                normlization_params=normlization_params,
            )

        log.info("Completed global analysis.")

    elif analysis_type == "local":

        local_analysis(
            model,
            save_loc=vis_save_loc,
            class_name_ref_dict=class_name_ref_dict,
            push_dataloader=project_dataloader,
            eval_dataloader=val_dataloader,
            img_size=img_size,
            device=device,
            sample=sample,
            num_top_protos_viewed=local_analysis_top_comparisons,
            run_proto_vis=False,
            normlization_params=normlization_params,
            last_layer_coef=last_layer_coef,
        )

        log.info("Completed local analysis.")


def main():

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--debug", default=False, action="store_true")
    argparser.add_argument(
        "analysis_type",
        choices=["global", "local", "render-prototypes"],
        help="Type of analysis",
    )
    argparser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=argparse.SUPPRESS,
    )
    argparser.add_argument(
        "--model-path",
        type=pathlib.Path,
        required=True,
    )
    argparser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["cub200", "cub200-cropped", "cars", "dogs", "cifar10"],
    )
    argparser.add_argument("--sample", type=int, default=None)

    args = vars(argparser.parse_args())
    debug = args.pop("debug")

    if debug:
        log.setLevel(logging.DEBUG)

    log.debug(f"args: {args}")

    dataloaders = training_dataloaders(args["dataset"])
    debug = args.pop("dataset")
    model_path = args["model_path"]
    debug = args.pop("model_path")

    run_analyses(
        **args,
        model=torch.load(
            model_path,
            map_location=(
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            ),
            weights_only=False
        ),
        model_path=model_path,
        vis_save_loc=args["output_dir"],
        project_dataloader=dataloaders.project_loader,
        val_dataloader=dataloaders.val_loader,
        device=(
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        ),
    )
