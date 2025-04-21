#!/bin/bash
set -eu
# Run this build from the parent directory with ci/build-docker.sh

version=$(cat pyproject.toml | grep '^version\s*=\s*"' | cut -d'"' -f 2)
docker_path=gitlab-registry.oit.duke.edu/jcd97/proto-rset

# The tag names come from https://gitlab.oit.duke.edu/help/user/packages/container_registry/index#naming-convention-for-your-container-images
docker build --platform linux/x86_64 --file ci/lint.Dockerfile \
    --tag lint:$version --tag $docker_path/lint:$version .

# build requirements image for testing runs
docker build --platform linux/x86_64 --file ci/cpu-test.Dockerfile \
    --tag cpu-test:$version --tag $docker_path/cpu-test:$version .

# test that all the tests in the container
# this takes a few minutes in an undersized container
# docker run -v $(pwd):/proto-rset -w /proto-rset \
#     gitlab-registry.oit.duke.edu/jcd97/proto-rset/cpu-test:0.2.1 pytest --log-cli-level=INFO

# if you want to push, call as `ci/build-docker.sh push`
if [ ${1-unset} == 'push' ]; then
    docker push $docker_path/lint:$version
    docker push $docker_path/cpu-test:$version
fi
