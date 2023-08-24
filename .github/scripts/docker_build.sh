#!/bin/bash

if   [ $1 = "base" ]; then
    export DOCKER_IMAGE=openpilot-base
    export DOCKER_FILE=Dockerfile.openpilot_base
elif [ $1 = "docs" ]; then
    export DOCKER_IMAGE=openpilot-docs
    export DOCKER_FILE=docs/docker/Dockerfile
elif [ $1 = "sim" ]; then
    export DOCKER_IMAGE=openpilot-sim
    export DOCKER_FILE=tools/sim/Dockerfile.sim
elif [ $1 = "prebuilt" ]; then
    export DOCKER_IMAGE=openpilot-prebuilt
    export DOCKER_FILE=Dockerfile.openpilot
elif [ $1 = "cl" ]; then
    export DOCKER_IMAGE=openpilot-base-cl
    export DOCKER_FILE=Dockerfile.openpilot_base_cl
else
    echo "Invalid docker build image $1"
    exit 1
fi

export DOCKER_REGISTRY=ghcr.io/commaai
export COMMIT_SHA=$(git rev-parse HEAD);

LOCAL_TAG=$DOCKER_IMAGE
REMOTE_TAG=$DOCKER_REGISTRY/$LOCAL_TAG
REMOTE_SHA_TAG=$REMOTE_TAG:$COMMIT_SHA

REMOTE_TAG_CACHE=$REMOTE_TAG-cache

if [[ ! -z "$PUSH_IMAGE" ]];
then
    CACHE_TO="--cache-to type=registry,ref=$REMOTE_TAG_CACHE,mode=max"
fi

DOCKER_BUILDKIT=1 docker buildx build $CACHE_TO --cache-from type=registry,ref=$REMOTE_TAG_CACHE -t $REMOTE_TAG -t $LOCAL_TAG -f $DOCKER_FILE .

# if [[ ! -z "$PUSH_IMAGE" ]];
# then
#     docker push $REMOTE_TAG
#     docker tag $REMOTE_TAG $REMOTE_SHA_TAG
#     docker push $REMOTE_SHA_TAG
# fi