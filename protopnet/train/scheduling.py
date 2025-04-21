import collections
from dataclasses import dataclass
from typing import Union


class TrainingSchedule:
    def __init__(
        self,
        max_epochs=3000,
        num_warm_epochs=0,
        num_last_only_epochs=0,
        num_warm_pre_offset_epochs=0,
        num_joint_epochs=0,
        last_layer_fixed=False,
        project_epochs=[],
        num_last_only_epochs_after_each_project=20,
    ):
        # Check that num_last_only_epochs_after_each_project and num_last_only_epochs = 0
        if last_layer_fixed:
            assert (
                num_last_only_epochs == 0
                and num_last_only_epochs_after_each_project == 0
            ), "Cannot have last only epochs if last layer is fixed"

        self.max_epochs = max_epochs
        self.train_prototype_prediction_head = not last_layer_fixed
        self.num_last_only_epochs_after_each_project = (
            num_last_only_epochs_after_each_project
        )
        self.project_epochs = (
            self._convert_project_epochs_to_include_project_and_last_only_epochs(
                project_epochs
            )
        )

        self.epochs = self.build_vanilla_protopnet_training_schedule(
            num_warm_epochs,
            num_last_only_epochs,
            num_warm_pre_offset_epochs,
            num_joint_epochs,
            last_layer_fixed,
            self.project_epochs,
        )

    def _convert_project_epochs_to_include_project_and_last_only_epochs(
        self, project_epochs
    ):
        # Initialize a list to store the updated project epochs
        updated_project_epochs = []
        epochs_added_by_project_and_post_project_lastlayer_optimization = 0

        for project_epoch in project_epochs:
            updated_project_epochs.append(
                project_epoch
                + epochs_added_by_project_and_post_project_lastlayer_optimization
            )
            epochs_added_by_project_and_post_project_lastlayer_optimization += (
                1 + self.num_last_only_epochs_after_each_project
            )

        return updated_project_epochs

    def check_project_epoch_validity(self, project_epochs, base_training_epochs):
        if not project_epochs:
            return

        # Initialize a flag to indicate if the project epochs are valid
        are_project_epochs_valid = True

        for i, project_epoch in enumerate(project_epochs):
            # Calculate total epochs required up to this project epoch, considering last-only epochs after each project
            total_required_epochs = base_training_epochs + (i - 1) * (
                self.num_last_only_epochs_after_each_project + 1
            )

            # Check if the current project epoch is valid within the sequential timeline
            if project_epoch > total_required_epochs:
                are_project_epochs_valid = False
                break

        assert (
            are_project_epochs_valid
        ), "Project epochs must fit within the structured epoch timeline without creating a gap."

        # Ensure that project epochs are non-negative
        assert min(project_epochs) >= 0, "Project epochs must be non-negative. "

        # If project epoch is in project epochs at epoch i, then there should not be another project epoch for another num_last_only_epochs_after_each_project epochs
        sorted_project_epochs = sorted(project_epochs)
        assert all(
            [
                sorted_project_epochs[i] + self.num_last_only_epochs_after_each_project
                < sorted_project_epochs[i + 1]
                for i in range(len(sorted_project_epochs) - 1)
            ]
        ), "Project epochs must be separated by num_last_only_epochs_after_each_project epochs. "

    def build_vanilla_protopnet_training_schedule(
        self,
        num_warm_epochs,
        num_last_only_epochs,
        num_warm_pre_offset_epochs,
        num_joint_epochs,
        last_layer_fixed,
        project_epochs=[],
    ):
        assert any(
            [
                num_warm_epochs,
                num_last_only_epochs,
                num_warm_pre_offset_epochs,
                num_joint_epochs,
                project_epochs,
            ]
        ), "At least one of the epochs must be greater than 0 to train"

        assert (
            self.train_prototype_prediction_head or num_last_only_epochs == 0
        ), "Cannot have last only epochs if last layer is fixed"

        # num_last_only_epochs_after_each_project = 10
        # num_last_only_epochs = len(project_epochs) * num_last_only_epochs_after_each_project

        base_training_epochs = (
            num_warm_epochs
            + num_last_only_epochs
            + num_warm_pre_offset_epochs
            + num_joint_epochs
        )

        total_training_epochs = base_training_epochs + len(project_epochs)

        if not last_layer_fixed:
            total_training_epochs = (
                total_training_epochs
                + len(project_epochs) * self.num_last_only_epochs_after_each_project
            )

        self.check_project_epoch_validity(project_epochs, total_training_epochs)

        current_epoch = 0
        # TODO: Seems as though this logic could be removed if we just didn't count project epochs as epochs here
        # Can still count them for visualization/logging purposes too
        epochs_added_by_project_and_post_project_lastlayer_optimization = 0
        schedule = []

        while current_epoch < total_training_epochs:
            if current_epoch in project_epochs:
                schedule.append(ProtoPNetProjectEpoch())

                for i in range(self.num_last_only_epochs_after_each_project):
                    schedule.append(self._create_epoch("last_only"))
                    current_epoch += 1
                    epochs_added_by_project_and_post_project_lastlayer_optimization += 1

                # No increment to current_epoch because project epochs are not counted
                current_epoch += 1
                epochs_added_by_project_and_post_project_lastlayer_optimization += 1
                continue

            # Determine the phase based on the current_epoch
            if (
                current_epoch
                - epochs_added_by_project_and_post_project_lastlayer_optimization
                < num_warm_epochs
            ):
                schedule.append(self._create_epoch("warm"))
            # TODO: Is this necessary/right? Don't have a precedent for where to put this...
            elif (
                current_epoch
                - epochs_added_by_project_and_post_project_lastlayer_optimization
                < num_warm_epochs + num_last_only_epochs
            ):
                schedule.append(self._create_epoch("last_only"))
            elif (
                current_epoch
                - epochs_added_by_project_and_post_project_lastlayer_optimization
                < num_warm_epochs + num_last_only_epochs + num_warm_pre_offset_epochs
            ):
                schedule.append(self._create_epoch("warm_pre_offset"))
            else:
                schedule.append(self._create_epoch("joint"))

            current_epoch += 1  # Move to the next epoch considering all types

        # TODO: What to do if after num epochs
        # for epoch in project_epochs:
        #     if epoch >= total_training_epochs:
        #         schedule.append(ProtoPNetProjectEpoch())

        return schedule[: self.max_epochs]

    def _create_epoch(self, phase):
        # This helper function returns the appropriate epoch configuration based on the phase
        if phase == "warm":
            return ProtoPNetBackpropEpoch(
                phase=phase,
                train_backbone=False,
                train_add_on_layers=True,
                train_prototype_layer=True,
                train_conv_offset=False,
                train_prototype_prediction_head=self.train_prototype_prediction_head,
            )
        elif phase == "last_only":
            return ProtoPNetBackpropEpoch(
                phase=phase,
                train_backbone=False,
                train_add_on_layers=False,
                train_prototype_layer=False,
                train_conv_offset=False,
                train_prototype_prediction_head=self.train_prototype_prediction_head,
            )
        elif phase == "warm_pre_offset":
            return ProtoPNetBackpropEpoch(
                phase=phase,
                train_backbone=True,
                train_add_on_layers=True,
                train_prototype_layer=True,
                train_conv_offset=False,
                train_prototype_prediction_head=self.train_prototype_prediction_head,
            )
        elif phase == "joint":
            return ProtoPNetBackpropEpoch(
                phase=phase,
                train_backbone=True,
                train_add_on_layers=True,
                train_prototype_layer=True,
                train_conv_offset=True,
                train_prototype_prediction_head=False,
            )
        else:
            raise ValueError(f"Unsupported phase: {phase}")

    def get_epochs(self):
        return self.epochs

    # Naive __repr__ implementation that lists every single epoch
    def __repr_long__(self):
        schedule_repr = ",\n    ".join(repr(epoch) for epoch in self.epochs)
        return (
            f"{self.__class__.__name__}(max_epochs={self.max_epochs}, "
            f"train_prototype_prediction_head={self.train_prototype_prediction_head}, "
            f"epochs=[\n    {schedule_repr}\n])"
        )

    # __repr implementation that lists ranges of epochs
    def __repr__(self):
        """
        Returns a string representation of the TrainingSchedule object, summarizing the training epochs and their phases.

        The method groups consecutive epochs with the same phase together and displays them as ranges, providing a concise overview of the training plan. Phases for backprop epochs include 'warm', 'warm_pre_offset', 'last_only', and 'joint'. All project epochs have the phase 'project'. If the schedule is empty, it returns a placeholder string indicating an empty training schedule.

        Example Outputs:
            - If the schedule consists of 20 'warm' epochs followed by 10 'last_only' epochs, and then 5 'project' epochs, the output will be:
                TrainingSchedule(max_epochs=35, train_prototype_prediction_head=False, phases=[
                    1-20: ProtoPNetBackpropEpoch(phase=warm),
                    21-30: ProtoPNetBackpropEpoch(phase=last_only),
                    31-35: ProtoPNetProjectEpoch(phase=project)
                ])

            - If the schedule is empty, the output will be:
                <Empty TrainingSchedule>

        Returns:
            str: A string representation of the TrainingSchedule object.

        """

        phase_ranges = []
        if not self.epochs:
            return "<Empty TrainingSchedule>"

        # Initialize with the first epoch's phase
        current_phase = repr(self.epochs[0])
        start_epoch = 1
        handled_last_epoch = False

        for i, epoch in enumerate(self.epochs[1:], start=2):
            if repr(epoch) != current_phase or i == len(self.epochs):
                # Determine the end epoch for the current phase range
                end_epoch = (
                    i
                    if i == len(self.epochs) and repr(epoch) == current_phase
                    else i - 1
                )

                # Append the current phase range
                if start_epoch == end_epoch:
                    phase_ranges.append(f"{start_epoch}: {current_phase}")
                else:
                    phase_ranges.append(f"{start_epoch}-{end_epoch}: {current_phase}")

                if i == len(self.epochs) and repr(epoch) == current_phase:
                    handled_last_epoch = True  # The last epoch has been included

                current_phase = repr(epoch)
                start_epoch = i

        # Append the last phase range if it hasn't been handled
        if not handled_last_epoch:
            end_epoch = len(self.epochs)
            if start_epoch == end_epoch:
                phase_ranges.append(f"{start_epoch}: {current_phase}")
            else:
                phase_ranges.append(f"{start_epoch}-{end_epoch}: {current_phase}")

        return (
            f"{self.__class__.__name__}(max_epochs={self.max_epochs}, "
            f"train_prototype_prediction_head={self.train_prototype_prediction_head}, "
            f"phases=[\n    " + ",\n    ".join(phase_ranges) + "\n])"
        )


class ProtoPNetBackpropEpoch:
    def __init__(
        self,
        phase,
        train_backbone,
        train_add_on_layers,
        train_prototype_layer,
        train_conv_offset,
        train_prototype_prediction_head,
    ):
        self.phase = phase
        self.train_backbone = train_backbone
        self.train_add_on_layers = train_add_on_layers
        self.train_prototype_layer = train_prototype_layer
        self.train_prototype_prediction_head = train_prototype_prediction_head

        # TODO: Should we have a Deformable ProtoPNetEpoch?
        # To handle the case where we want to train the offset layer
        self.train_conv_offset = train_conv_offset

    def training_layers(self):
        # Features -> backbone
        # Prototype -> prototype layer

        # Add on layers
        # Conv offset
        return {
            "backbone": self.train_backbone,
            "add_on_layers": self.train_add_on_layers,
            "prototype_layer": self.train_prototype_layer,
            "conv_offset": self.train_conv_offset,
            "head": self.train_prototype_prediction_head,
        }

    def __repr__(self):
        return f"{self.__class__.__name__}(phase={self.phase})"


class ProtoPNetProjectEpoch:
    def __init__(self):
        self.phase = "project"

    def training_layers(self):
        return {
            "backbone": False,
            "add_on_layers": False,
            "prototype": True,
            "conv_offset": False,
            "head": False,
        }

    def __repr__(self):
        return f"{self.__class__.__name__}(phase={self.phase})"


def prototype_embedded_epoch(
    epoch: Union[ProtoPNetBackpropEpoch, ProtoPNetProjectEpoch]
):
    """
    Determines whether prototypes will match their embedding images after an epoch with the given settings.
    This is only True if the epoch is Project epoch, or the training does not affect the embedding.

    Returns:
        bool: True if prototypes will match their embedding images, False otherwise.
    """

    if isinstance(epoch, ProtoPNetProjectEpoch):
        return True

    layers = epoch.training_layers().copy()

    del layers["head"]

    return not any(layers.values())


@dataclass
class EarlyStopping:
    patience: int
    min_delta: float
    mode: str
    metric_source: collections.abc.Callable
    after_project: bool = True
    best: float = None
    counter: int = 0
    stop: bool = False

    def __post_init__(self):
        self.mode = self.mode.lower()
        assert self.mode in ["min", "max"], "mode must be 'min' or 'max'"
        self.best = float("inf") if self.mode == "min" else float("-inf")

    def check(self):
        value = self.metric_source()
        if self.mode == "min":
            if value < self.best - self.min_delta:
                self.best = value
                self.counter = 0
            else:
                self.counter += 1
        else:
            if value > self.best + self.min_delta:
                self.best = value
                self.counter = 0
            else:
                self.counter += 1

        if self.counter >= self.patience:
            self.stop = True

        return self.stop

    def reset(self):
        self.best = float("inf") if self.mode == "min" else float("-inf")
        self.counter = 0
        self.stop = False

    def __repr__(self):
        return f"{self.__class__.__name__}(patience={self.patience}, min_delta={self.min_delta}, mode={self.mode}, monitor={self.monitor}, best={self.best}, counter={self.counter}, stop={self.stop})"

    def __str__(self):
        return f"EarlyStopping(patience={self.patience}, min_delta={self.min_delta}, mode={self.mode}, monitor={self.monitor}, best={self.best}, counter={self.counter}, stop={self.stop})"
