import torchmetrics


class TrainLogger:
    def __init__(
        self,
        use_ortho_loss=False,
        class_specific=True,
        # FIXME: this should consistently be called accuracy
        calculate_best_for=["accu"],
        device="cpu",
    ):
        self.use_ortho_loss = use_ortho_loss
        self.class_specific = class_specific
        # self.coefs = coefs
        # FIXME: this should support min and max
        self.bests = self.__setup_bests(calculate_best_for)

        # Create separate metrics dictionaries for train and validation
        self.train_metrics = self.create_metrics(device)
        self.val_metrics = self.create_metrics(device)

    # FIXME: this should be part of the metrics class, not the logger
    def __setup_bests(self, calculate_best_for):
        bests = {}
        for metric_name in calculate_best_for:
            # FIXME: this should support min and max
            bests[metric_name] = {
                "any": float("-inf"),
                "prototypes_embedded": float("-inf"),
            }

        return bests

    # FIXME: this should be part of the metrics class, not the logger
    def update_bests(self, metrics_dict, step, prototype_embedded_epoch=False):
        for metric_name, metric_value in metrics_dict.items():
            if metric_name in self.bests and metric_value is not None:
                if metric_value > self.bests[metric_name]["any"]:
                    self.bests[metric_name]["any"] = metric_value
                    self.process_new_best(
                        self.__metric_best_name(metric_name, False), metric_value, step
                    )

                if prototype_embedded_epoch:
                    if metric_value > self.bests[metric_name]["prototypes_embedded"]:
                        self.bests[metric_name]["prototypes_embedded"] = metric_value
                        self.process_new_best(
                            self.__metric_best_name(metric_name, True),
                            metric_value,
                            step,
                        )

    def __metric_best_name(self, metric_name, prototype_embedded_state):
        # FIXME: we should consistently call this accuracy throughout
        maybe_prototypes_embedded = (
            "prototypes_embedded_" if prototype_embedded_state else ""
        )
        return f"best_{maybe_prototypes_embedded}{metric_name}"

    def serialize_bests(self):
        bests_flat = {}
        for metric_name, metric_values in self.bests.items():
            bests_flat[self.__metric_best_name(metric_name, False)] = metric_values[
                "any"
            ]
            bests_flat[self.__metric_best_name(metric_name, True)] = metric_values[
                "prototypes_embedded"
            ]
        return bests_flat

    def process_new_best(
        self, metric_name, metric_value, step, prototype_embedded_state=False
    ):
        """
        This method is called whenever a new "best" value of a metric is found with the value of the metric, the current, step,
        and whether the prototype layer is embedded or not. It provides a hook to capture the new value and take any necessary actions.

        The default is a no-op. Subclasses can override this method to implement custom behavior.
        """
        pass

    def create_metrics(self, device):
        # Helper method to initialize metrics

        metric_dict = {
            "n_examples": torchmetrics.SumMetric().to(device),
            "n_correct": torchmetrics.SumMetric().to(device),
            "n_batches": torchmetrics.SumMetric().to(device),
            "cross_entropy": torchmetrics.MeanMetric().to(device),
            "cluster": torchmetrics.MeanMetric().to(device),
            "separation": torchmetrics.MeanMetric().to(device),
            "fine_annotation": torchmetrics.MeanMetric().to(device),
            "accu": torchmetrics.MeanMetric().to(
                device
            ),  # Using torchmetrics.Accuracy directly for accuracy
            "l1": torchmetrics.MeanMetric().to(device),
            "total_loss": torchmetrics.MeanMetric().to(device),
            "debug_forbid_loss_with_weight": torchmetrics.MeanMetric().to(device),
            "debug_remember_loss_with_weight": torchmetrics.MeanMetric().to(device),

        }
        return metric_dict

    def update_metrics(self, metrics_dict, is_train):
        metrics = self.train_metrics if is_train else self.val_metrics

        # Update each metric from metrics_dict
        for key, value in metrics_dict.items():
            # TODO: Is this desired- not tracking Nones (torchmetrics does not like them)
            if key in metrics and value is not None:
                metrics[key].update(value)
