import logging

import wandb

from .types import TrainLogger

log = logging.getLogger(__name__)


class WeightsAndBiasesTrainLogger(TrainLogger):
    def __init__(
        self,
        device="cpu",
        calculate_best_for=["accu"],
    ):
        super().__init__(device=device, calculate_best_for=calculate_best_for)

    def log_metrics(
        self,
        is_train,
        prototypes_embedded_state=False,
        precalculated_metrics=None,
        step=None,
    ):
        metric_group, metrics, commit = (
            ("train", self.train_metrics, False)
            if is_train
            else ("eval", self.val_metrics, True)
        )

        metrics_for_log = {
            metric_group: {name: metric.compute() for name, metric in metrics.items()},
        }

        if precalculated_metrics:
            metrics_for_log[metric_group].update(precalculated_metrics)

        wandb.log(metrics_for_log, step=step, commit=commit)

        for metric in metrics.values():
            # TODO - it's very bad that we're resetting metrics in a logging function
            metric.reset()

    def process_new_best(self, metric_name, metric_value, step):
        """
        This method is called whenever a new "best" value of a metric is found with the value of the metric, the current, step,
        and whether the prototype layer is embedded or not.

        This updates the weights and biases run summary with the new best value of the metric and the step at which it was found.
        """
        wandb.run.summary[metric_name] = metric_value
        wandb.run.summary[f"{metric_name}_step"] = step

    def end_epoch(
        self,
        epoch_metrics_dict,
        is_train,
        epoch_index,
        prototype_embedded_epoch,
        precalculated_metrics=None,
    ):
        for key in epoch_metrics_dict:
            # DO NOTHING FOR THESE KEYS
            if (
                key
                not in [
                    "time",
                    "n_batches",
                    "l1",
                    "max_offset",
                    "n_correct",
                    "n_examples",
                    "accu",
                    "is_train",
                ]
                and epoch_metrics_dict[key]
            ):
                epoch_metrics_dict[key] /= epoch_metrics_dict["n_batches"]

        self.update_metrics(epoch_metrics_dict, is_train)

        complete_metrics = epoch_metrics_dict.copy()
        if precalculated_metrics is not None:
            complete_metrics.update(precalculated_metrics)

        self.update_bests(
            complete_metrics,
            step=epoch_index,
            prototype_embedded_epoch=prototype_embedded_epoch,
        )

        self.log_metrics(
            is_train,
            step=epoch_index,
            prototypes_embedded_state=prototype_embedded_epoch,
            precalculated_metrics=precalculated_metrics,
        )

    @staticmethod
    def log_backdrops(backdrop_dict, step=None):
        # log dict to wandb
        wandb.log(backdrop_dict, step=step)
