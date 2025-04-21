# To debug these tests, the easiest thing is to run them from the command line.
# Just take the string from the call to python() below and run it from the command line as 'python ' + cmd
# If you need to debug, you can use your IDE debugger but stil running from the base command.

import os
import subprocess
import sys
from contextlib import contextmanager

import pytest

pytestmark = [
    pytest.mark.e2e,
]


@contextmanager
def disable_exception_traceback():
    """
    All traceback information is suppressed and only the exception type and value are printed
    """
    default_value = getattr(
        sys, "tracebacklimit", 1000
    )  # `1000` is a Python's default value
    sys.tracebacklimit = 0
    yield
    sys.tracebacklimit = default_value


def print_output(stdout, stderr):
    output = ""
    output += "STDOUT:\n"
    for line in stdout.splitlines():
        output += "> " + line + "\n"
    output += "\n"
    output += "STDERR:" + "\n"
    for line in stderr.splitlines():
        output += "> " + line + "\n"
    return output


def python(cmd, extra_env={}):
    env = os.environ.copy()
    env.update(extra_env)
    """Executes a command, captures stdout and stderr, and raises an error if it returns a non-0 exit code."""
    try:
        # If you want the command output as a string, use subprocess.PIPE as the stdout and stderr value
        result = subprocess.run(
            sys.executable + " " + cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=True,
            env=env,
        )
        print(print_output(result.stdout, result.stderr))

        return result.stdout, result.stderr

    except subprocess.CalledProcessError as e:
        # Suppress the runtime exception since we care about the subprocess exception and it's already been printed
        pytest.fail(
            reason=f"Command '{cmd}' returned non-zero exit status {e.returncode}:\n\n{print_output(e.output, e.stderr)}",
            pytrace=False,
        )


def assert_4_epoch_training(stdout: str, *, phase_2="joint", completed_epochs: int):
    # Define the pattern for the training schedule
    pattern_lines = [
        "TrainingSchedule(max_epochs=4, train_prototype_prediction_head=True, phases=[",
        "1: ProtoPNetBackpropEpoch(phase=warm),",
        f"2: ProtoPNetBackpropEpoch(phase={phase_2}),",
        "3: ProtoPNetProjectEpoch(phase=project),",
        "4: ProtoPNetBackpropEpoch(phase=last_only)",
        "])",
    ]
    remaining_lines = stdout
    for line in pattern_lines:
        assert line in remaining_lines, f"Expected line not found in output: {line}"
        remaining_lines = remaining_lines.split(line, 1)[1]
    assert (
        f"Training complete after {completed_epochs} epochs" in stdout
    ), "Training did not complete successfully"


def test_train_vanilla_cos():
    stdout, _ = python(
        "-u -m protopnet train-vanilla-cos --verify --dataset=cifar10 --backbone=squeezenet1_0",
        {"WANDB_MODE": "dryrun"},
    )
    assert_4_epoch_training(stdout, completed_epochs=4)


def test_train_vanilla():
    stdout, _ = python(
        "-u -m protopnet train-vanilla --verify --dataset=cifar10 --backbone=squeezenet1_0",
        {"WANDB_MODE": "dryrun"},
    )
    # 3 epochs because it will not pass the random threshold
    assert_4_epoch_training(stdout, completed_epochs=4)


def test_train_deformable():
    stdout, _ = python(
        "-u -m protopnet train-deformable --verify --dataset=cifar10 --backbone=squeezenet1_0",
        {"WANDB_MODE": "dryrun"},
    )
    assert_4_epoch_training(stdout, phase_2="warm_pre_offset", completed_epochs=4)


def test_render_prototypes():
    stdout, _ = python(
        "-u -m protopnet viz render-prototypes --model-path=test/dummy_test_files/ppn_squeezenet_cifar10.pth --dataset=cifar10 --output-dir=test/tmp/analysis",
        {"WANDB_MODE": "dryrun"},
    )
    assert "Completed rendering of prototypes" in stdout


def test_local_analysis():
    stdout, _ = python(
        "-u -m protopnet viz --debug local --model-path=test/dummy_test_files/ppn_squeezenet_cifar10.pth --dataset=cifar10 --sample=6 --output-dir=test/tmp/analysis",
        {"WANDB_MODE": "dryrun"},
    )
    assert "Completed local analysis." in stdout


@pytest.mark.skip(
    reason="Runtime is currently too long for CI. However, if you are changing global analysis, you should check this runs."
)
def test_global_analysis():
    stdout, _ = python(
        "-u -m protopnet viz global --model-path=test/dummy_test_files/ppn_squeezenet_cifar10.pth --dataset=cifar10 --sample=1 --output-dir=test/tmp/analysis",
        {"WANDB_MODE": "dryrun"},
    )
    assert "Completed global analysis." in stdout
