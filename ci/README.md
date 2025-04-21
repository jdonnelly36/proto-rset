# CI Build Containers

These are a simple pre-built docker files that have most the dependencies of the system preinstalled.
This significantly accelerates the ci job time, which was previously over 80% installing dependencies.

The CI jobs will still install all the relevant dependencies when they run.
This will ensure that small changes to the depedencies don't require full rebuilds of the containers.
However, with these files, those dependencies should mostly be pre-installed.
**Note:** The containers must be rebuilt if the pytorch (or torchvision) version is changed.

## Refreshing the Build Containers

### Prequisites

You must have docker installed and running in the environment where you will refresh from and the environment connected to OIT GitLab.

### Setting up OIT GitLab Docker Registry

You must create an OIT GitLab Access Token to access to the Docker Registry.
The token is associated with your netid, but you cannot use your passwork.
Go to `User Icon > Preferences > Access Tokens` and create a new token with registry `read` and `write` permissions for the container registry:

![Git Access Token Screenshow](gitlab-registry-access.png)

Then, login to GitLab with `docker login`:

```{sh}
> docker login gitlab-registry.oit.duke.edu
Username: <netid>
Password: <access token>
Login Succeeded
```

### Rebuilding the Containers

1. Update the project version in `pyproject.toml`
1. Run `ci/build-docker.sh push` from the project root. This script builds the images and pushes them to the [project registry](https://gitlab.oit.duke.edu/jcd97/proto-rset/container_registry) in OIT GitLab. It will take several minutes (>10).
1. Update the `image` field in `.gitlab-ci.yaml` to point to the containers

### Deleting Old Containers

OIT's registry space is not unlimited.
As old build images are no longer needed, please remove them in the [gitlab registry web interface](https://gitlab.oit.duke.edu/jcd97/proto-rset/container_registry).

## Design Notes

The `lint` container was created specifically because linting happens first, can generally be done very quickly, and only requires a small percentage of the system dependencies.

These images are also not rebuilt each time a dependency is updated.
`pip install` will keep the dependencies up-to-date within specific steps.
This avoids the overhead of rebuilding each time a small dependency is changed.
