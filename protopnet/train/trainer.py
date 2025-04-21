import logging
import os
import time
from pathlib import Path

import torch

from ..datasets.util import calculate_class_weights
from ..losses import (
    BalancedCrossEntropyCost,
    ClusterCost,
    FineAnnotationCost,
    L1CostClassConnectionLayer,
    LossTerm,
    OrthogonalityLoss,
    SeparationCost,
    ProtoForbidLoss,
    ProtoRememberLoss
)
from ..model_losses import ProtoPNetLoss
from ..utilities.trainer_utilities import get_learning_rates, init_or_update
from .logging.tensorboard_logger import TensorBoardLogger
from .metrics import TrainingMetrics
from .scheduling import (
    ProtoPNetBackpropEpoch,
    ProtoPNetProjectEpoch,
    prototype_embedded_epoch,
)

log = logging.getLogger(__name__)


class ProtoPNetTrainer:
    def __init__(
        self,
        model,
        split_loaders,
        activation_function,
        optimizers_with_schedulers,
        device,
        coefs=None,
        with_fa=False,
        fa_type="l2",
        use_ortho_loss=False,
        class_specific=True,
        deformable=False,
        logger=None,
        early_stopping_patience=None,
        save_dir=Path(os.environ.get("PPNXT_ARTIFACT_DIR", "models")),
        min_save_threshold=0.0,
        min_post_project_target_metric=0.0,
        num_accumulation_batches=1,
        training_metrics: TrainingMetrics = None,
        target_metric_name="accu",
        compute_metrics_for_embed_only=True,
        balance_ce=False,
        debug_round=False
    ):
        # Change to use a real logger (create_logger)

        # model, dataloader, activation_function, optimizer=None, device="cuda"
        self.model = model
        self.dataloader = split_loaders.train_loader

        # TODO: Add assert statement to ensure there is an optimizer for each phase
        self.optimizers_with_schedulers = optimizers_with_schedulers

        self.compute_metrics_for_embed_only = compute_metrics_for_embed_only
        self.device = device
        self.with_fa = with_fa
        self.fa_type = fa_type
        self.use_ortho_loss = use_ortho_loss
        self.class_specific = class_specific
        self.coefs = coefs
        self.num_accumulation_batches = num_accumulation_batches

        self.model = self.model.to(self.device)

        # Number of projects without improvement before stopping (0 means stop on first project without improvement over previous best)
        self.early_stopping_patience = early_stopping_patience
        self.min_post_project_target_metric = min_post_project_target_metric

        if logger is None:
            # TODO consolidate our logger initializations
            self.logger = TensorBoardLogger(
                use_ortho_loss=use_ortho_loss,
                class_specific=class_specific,
                device=self.device,
            )
        else:
            self.logger = logger
            self.logger.device = self.device

        self.debug_round = debug_round
        if self.debug_round:
            log.info("Loaded debug loaders")
            self.debug_forbid_loader = split_loaders.debug_forbid_loader
            self.debug_remember_loader = split_loaders.debug_remember_loader

        # TODO: Determine if this is where it should be set
        # But also want to allow for a Callable coefficient that would be function of the epoch

        if balance_ce:
            class_weights = calculate_class_weights(self.dataloader).to(self.device)
        else:
            class_weights = None

        batch_losses = [
            LossTerm(
                loss=BalancedCrossEntropyCost(class_weights=class_weights),
                coefficient=self.coefs["cross_entropy"],
            ),
            LossTerm(
                loss=ClusterCost(class_specific=self.class_specific),
                coefficient=self.coefs["cluster"],
            ),
            LossTerm(loss=SeparationCost(), coefficient=self.coefs["separation"]),
        ]

        if self.with_fa:
            batch_losses.append(
                LossTerm(
                    loss=FineAnnotationCost(fa_loss=self.fa_type),
                    coefficient=self.coefs["fa"],
                )
            )

        model_losses = [
            LossTerm(loss=L1CostClassConnectionLayer(), coefficient=self.coefs["l1"]),
        ]

        if "orthogonality_loss" in self.coefs:
            model_losses.append(
                LossTerm(
                    loss=OrthogonalityLoss(),
                    coefficient=self.coefs["orthogonality_loss"],
                )
            )


        if debug_round:
            debug_losses = [
                LossTerm(
                    loss=ProtoRememberLoss(debug_remember_loader=self.debug_remember_loader, device=device),
                    coefficient=self.coefs["debug_remember"],
                ),
                LossTerm(
                    loss=ProtoForbidLoss(debug_forbid_loader=self.debug_forbid_loader, device=device),
                    coefficient=self.coefs["debug_forbid"],
                )
            ]

        # TODO: Determine a better way to update the device without passing device into the loss
        self.loss = ProtoPNetLoss(
            batch_losses=batch_losses, 
            model_losses=model_losses, 
            debug_round=debug_round, 
            debug_loss=debug_losses if debug_round else None,
            device=self.device
        )

        self.forward_calc_flags = {
            f"return_{req}": True
            for req in self.loss.batch_loss.required_forward_results()
        }

        # TODO: Consolidate this with above
        self.forward_calc_flags["return_prototype_layer_output_dict"] = True


        # self.metric_logger = MetricLogger(device=self.device)

        self.project_dataloader = split_loaders.project_dataloader
        # TODO: check self.project_dataloader return dictionary has a string return for sample_id (only needs to be checked once)

        # TODO: Determine if this needs to be passed in here or if it can be passed in at a different time (seems inflexible rn)
        self.val_dataloader = split_loaders.val_dataloader

        self.save_dir = save_dir
        self.min_save_threshold = min_save_threshold
        self.target_metric_name = target_metric_name

        self.training_metrics = training_metrics

    def update_training_phase(self, epoch_settings):
        # Map model components to training settings dynamically
        for name, param in self.model.named_parameters():
            # Extract the component name from the parameter name
            # Assuming the naming convention follows the pattern "<component_name>_..."
            component_name = name.split(".")[0]  # Get the first part of the name

            # Construct the setting attribute name dynamically
            setting_attr = f"train_{component_name}"

            # Check if the corresponding setting attribute exists in epoch_settings
            # TODO: Pre-validate
            assert hasattr(
                epoch_settings, setting_attr
            ), f"Attribute '{setting_attr}' not found in epoch_settings"

            # Since the attribute exists, use getattr to fetch its value
            # The third argument in getattr is not needed anymore since we're asserting the attribute's existence
            should_train = getattr(epoch_settings, setting_attr)

            # Update the requires_grad based on the setting
            param.requires_grad = should_train

    def train(self, training_schedule, val_each_epoch=True, save_model=True):
        # TODO: Should this check be here?
        if val_each_epoch:
            assert (
                self.val_dataloader
            ), "val_dataloader must be provided if val_each_epoch is True"

        log.info("Training with the following schedule:")
        log.info("%s", repr(training_schedule))

        if save_model:
            assert val_each_epoch, "Must run evaluation epochs to save model"

            os.makedirs(self.save_dir, exist_ok=True)

        # parsimonious early stopping
        last_eval_target_metric = float("-inf")
        best_preproject_target_metric = float("-inf")
        best_project_target_metric = float("-inf")
        early_stopping_project_count = 0

        for epoch_index, epoch_settings in enumerate(training_schedule.get_epochs()):
            # If epoch_settings of type ProtoPNetBackpropEpoch
            if isinstance(epoch_settings, ProtoPNetBackpropEpoch):
                log.info(
                    f"Starting Epoch {epoch_index} of Phase {epoch_settings.phase} with settings: {epoch_settings.training_layers()}"
                )
                self.update_training_phase(epoch_settings)
                self.train_epoch(
                    phase=epoch_settings.phase,
                    epoch_index=epoch_index,
                    epoch_settings=epoch_settings,
                )
            elif isinstance(epoch_settings, ProtoPNetProjectEpoch):
                # TODO: Make a fail fast that ensures that self.project_dataloader is not None (while only checking once)
                log.info(f"Starting Epoch {epoch_index} as Project Epoch")
                self.project_epoch(epoch_index, epoch_settings)
                # TODO: Run self.eval_epoch() but with dataloader=self.dataloader
                # Want to evaluate on train

            # TODO: Make a fail-fast using assert statements
            else:
                raise ValueError(
                    f"Unsupported type of epoch_settings: {type(epoch_settings)}"
                )

            if val_each_epoch:
                log.info(f"Starting Validation Epoch {epoch_index}")
                target_metric = self.eval_epoch(epoch_index, epoch_settings)
            else:
                # TODO: smarter about determining accuracy without eval
                target_metric = float("-inf")

            # TODO: Undo hard-coding
            if save_model and prototype_embedded_epoch(epoch_settings):
                previous_best = self.logger.bests[self.target_metric_name][
                    "prototypes_embedded"
                ]
                # TODO: weird order - technically we just updated best
                if (
                    target_metric >= previous_best
                    and target_metric > self.min_save_threshold
                ):
                    model_name = f"{str(epoch_index)}_{epoch_settings.phase}"
                    metric_path = "_{0:.4f}.pth".format(float(target_metric))
                    model_path = os.path.join(self.save_dir, model_name + metric_path)
                    log.info(
                        "Saving model with %s %s to %s",
                        self.target_metric_name,
                        target_metric,
                        model_path,
                    )
                    torch.save(
                        obj=self.model,
                        f=model_path,
                    )
                else:
                    log.debug(
                        "skipping saving model state with %s %s",
                        self.target_metric_name,
                        target_metric,
                    )
            # parsimonious early stopping
            if isinstance(epoch_settings, ProtoPNetProjectEpoch):
                if self.early_stopping_patience is not None:
                    if (
                        last_eval_target_metric <= best_preproject_target_metric
                        and target_metric <= best_project_target_metric
                    ):
                        early_stopping_project_count += 1
                    else:
                        early_stopping_project_count = 0

                    if early_stopping_project_count > self.early_stopping_patience:
                        log.info("Early stopping after %s epochs", epoch_index + 1)
                        log.info(
                            "Best accuracy before project: %s",
                            best_preproject_target_metric,
                        )
                        log.info(
                            "Best accuracy after project: %s",
                            best_project_target_metric,
                        )
                        log.info(
                            "%s projects without improvement",
                            early_stopping_project_count,
                        )
                        break
                    else:
                        best_project_target_metric = max(
                            best_project_target_metric, target_metric
                        )
                        best_preproject_target_metric = max(
                            best_preproject_target_metric, last_eval_target_metric
                        )

                if (
                    self.min_post_project_target_metric is not None
                    and target_metric <= self.min_post_project_target_metric
                ):
                    log.info(
                        "Early stopping after %s epochs because post project threshold of %s not exceeded by %s",
                        epoch_index + 1,
                        self.min_post_project_target_metric,
                        target_metric,
                    )
                    break

            last_eval_target_metric = target_metric

        log.info("Training complete after %s epochs", epoch_index + 1)
        return epoch_index

    def project_epoch(self, epoch_index, epoch_settings):
        # TODO: Combine logic with train_epoch?

        start = time.time()
        self.model.eval()
        with torch.no_grad():
            self.model.project(
                self.project_dataloader, class_specific=self.class_specific
            )
        end = time.time()
        log.info(f"Completed project epoch in {end - start} seconds")

    def train_epoch(self, phase, epoch_index, epoch_settings):
        """
        Conducts a single epoch of training.
        """
        self.model.train()
        with torch.enable_grad():
            self.run_epoch(
                dataloader=self.dataloader,
                optimizer_scheduler=self.optimizers_with_schedulers[phase],
                epoch_index=epoch_index,
                epoch_settings=epoch_settings,
                compute_metrics_this_epoch=False,
            )

    def eval_epoch(self, epoch_index, epoch_settings):
        """
        Conducts a single epoch of testing/validation.
        """
        self.model.eval()
        with torch.no_grad():
            target_metric = self.run_epoch(
                dataloader=self.val_dataloader,
                epoch_index=epoch_index,
                epoch_settings=epoch_settings,
                compute_metrics_this_epoch=(
                    prototype_embedded_epoch(epoch_settings)
                    or not self.compute_metrics_for_embed_only
                ),
            )

        return target_metric

    def run_epoch(
        self,
        dataloader,
        epoch_index,
        epoch_settings,
        optimizer_scheduler=None,
        compute_metrics_this_epoch=False,
    ):
        # model,dataloader,activation_function,optimizer=None,class_specific=True,use_l1_mask=True,coefs=None,log=print,subtractive_margin=True,use_ortho_loss=False,finer_loader=None,fa_loss="serial",device="cuda",

        # Should be able to use functions rather than all these if statements
        # Create TrainAdapter, TestAdapter, and ValidationAdapter classes
        # Also create Adapters for Deformable, Fine Annotation, etc

        # TODO: Add a way to track variables as None/NA if they aren't used
        # Or make this more flexible to be more flexible in the terms it tracks
        epoch_metrics_dict = {
            "time": None,
            "n_examples": None,
            "n_correct": None,
            "n_batches": None,
            "cross_entropy": None,
            "cluster": None,
            "separation": None,
            "fine_annotation": None,
            "accu": None,
            "l1": None,
            "total_loss": None,
            "n_unique_proto_parts": None,
            "n_unique_protos": None,
            "prototype_non_sparsity": None,
        }

        # FIXME: there should always be metrics
        if self.training_metrics is not None and compute_metrics_this_epoch:
            self.training_metrics.start_epoch(phase=epoch_settings.phase)

        optimizer, scheduler = (
            optimizer_scheduler if optimizer_scheduler is not None else (None, None)
        )

        if optimizer is not None:
            lr_log = get_learning_rates(
                optimizer=optimizer, model=self.model, detailed=False
            )
            self.logger.log_backdrops(lr_log, step=epoch_index)

        start = time.time()

        # Use a helper function to handle None values

        if optimizer:
            optimizer.zero_grad()
        agg_loss = 0
        for i, batch_data_dict in enumerate(dataloader):
            # TODO: Make this formatting better

            # Intended to include sample IDs for metadata logging while also allowing for the case where the dataloader does not return sample IDs
            # Perhaps could make dataloader create a list of sample IDs if it does not already have one
            image = batch_data_dict["img"]
            label = batch_data_dict["target"]
            if self.with_fa:
                fine_anno = batch_data_dict["fine_anno"]
                fine_annotation = fine_anno.to(self.device)
            else:
                fine_anno = None
                fine_annotation = None

            # TODO: Remove this ugly formatting
            input = image.to(self.device)
            target = label.to(self.device)

            # TODO: Subtractive margin
            output = self.model(input, **self.forward_calc_flags)

            # conv_features = self.model.backbone(input)

            # TODO: Move to forward of Deformable Proto Layer
            with torch.no_grad():
                logits = output["logits"]

                # evaluation statistics
                _, predicted = torch.max(logits.data, 1)

                init_or_update(epoch_metrics_dict, "n_examples", target.size(0))
                init_or_update(
                    epoch_metrics_dict, "n_correct", (predicted == target).sum().item()
                )
                init_or_update(epoch_metrics_dict, "n_batches", 1)

            loss = (
                self.loss(
                    target=target,
                    model=self.model,
                    metrics_dict=epoch_metrics_dict,
                    fine_annotation=fine_annotation,
                    **output,
                )
                / self.num_accumulation_batches
            )

            agg_loss += loss.item()
            if optimizer:
                loss.backward(retain_graph=True)

            # Check if we have reached our accumulation threshold
            if ((i + 1) % self.num_accumulation_batches == 0) or (
                i + 1 == len(dataloader)
            ):
                init_or_update(epoch_metrics_dict, "total_loss", agg_loss)

                agg_loss = 0
                if optimizer:
                    # self.optimizer.step())
                    optimizer.step()
                    optimizer.zero_grad()

            # FIXME: There should always be metrics
            if self.training_metrics is not None and compute_metrics_this_epoch:
                log.debug("updating extra metrics")

                # FIXME: somewhere these tensors are being moved off of gpu
                for key, maybe_tensor in batch_data_dict.items():
                    if (
                        isinstance(maybe_tensor, torch.Tensor)
                        and maybe_tensor.device != self.device
                    ):
                        batch_data_dict[key] = maybe_tensor.to(self.device)
                for key, maybe_tensor in output.items():
                    if (
                        isinstance(maybe_tensor, torch.Tensor)
                        and maybe_tensor.device != self.device
                    ):
                        output[key] = maybe_tensor.to(self.device)

                with torch.no_grad():
                    self.training_metrics.update_all(
                        batch_data_dict, output, phase=epoch_settings.phase
                    )
                log.debug("Extra metrics updated")

            # Removed: offsets, batch_max, additional_returns
            del (
                input,
                target,
                fine_annotation,
                output,
                predicted,
            )  # similarity_score_to_each_prototype, batch_max

            del image, label, fine_anno

            del loss

        end = time.time()

        if scheduler:
            # Step scheduler if possible
            scheduler.step()

        epoch_metrics_dict["time"] = end - start

        # TODO: Determine where to make these calculations
        epoch_metrics_dict["accu"] = (
            100 * epoch_metrics_dict["n_correct"] / epoch_metrics_dict["n_examples"]
        )
        if "offset_bias_l2" in self.coefs and "avg_l2" in epoch_metrics_dict:
            epoch_metrics_dict["avg_l2_with_weight"] = (
                self.coefs["offset_bias_l2"] * epoch_metrics_dict["avg_l2"]
            )

        if (
            "orthogonality_loss" in self.coefs
            and "orthogonality_loss" in epoch_metrics_dict
        ):
            epoch_metrics_dict["orthogonality_loss_with_weight"] = (
                self.coefs["orthogonality_loss"]
                * epoch_metrics_dict["orthogonality_loss"]
            )

        if self.debug_round:
            log.info(f'Logging debug loss forbid={epoch_metrics_dict["debug_forbid"]}, remember={epoch_metrics_dict["debug_remember"]}')
            epoch_metrics_dict["debug_forbid_loss_with_weight"] = (
                self.coefs["debug_forbid"]
                * epoch_metrics_dict["debug_forbid"]
            )
            epoch_metrics_dict["debug_remember_loss_with_weight"] = (
                self.coefs["debug_remember"]
                * epoch_metrics_dict["debug_remember"]
            )

        if self.training_metrics is not None and compute_metrics_this_epoch:
            log.debug("Computing extra metrics")
            start = time.time()
            with torch.no_grad():
                extra_training_metrics = self.training_metrics.compute_dict()
            log.info("Extra metrics calculated in %s", time.time() - start)
        else:
            extra_training_metrics = None

        # TODO: Assess if flags should be passed in here or into the init of the Logger class

        self.logger.end_epoch(
            epoch_metrics_dict,
            is_train=True if optimizer else False,
            epoch_index=epoch_index,
            prototype_embedded_epoch=prototype_embedded_epoch(epoch_settings),
            precalculated_metrics=extra_training_metrics,
        )

        # FIXME: There should always be metrics
        if self.training_metrics is not None:
            self.training_metrics.end_epoch(phase=epoch_settings.phase)

        # FIXME: Consolidate metrics
        if self.target_metric_name in epoch_metrics_dict:
            return epoch_metrics_dict[self.target_metric_name]
        elif (
            extra_training_metrics and self.target_metric_name in extra_training_metrics
        ):
            return extra_training_metrics[self.target_metric_name]
        else:
            # FIXME: this is hack for eval-only target metrics, but it shouldn't break anything
            return 0.0
