#!/bin/bash
set -e
# setup x auth environment for visual support
XAUTH=$(mktemp /tmp/.docker.xauth.XXXXXXXXX)
xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge -

###################################################################
########### UPDATE PATHS BELOW BEFORE RUNNING #####################
###################################################################

# Provide full path to pedgraph folder
CODE_FOLDER=/path/to/data/   # edit this line!!

IMAGE_NAME=pytorch_18  # edit this according to the pytorch version in the docker image
TAG=pedgraph           # edit this according to the tag in docker image
CONTAINER_NAME=pytorch # edit the name of the docker container

WORKING_DIR=$HOME/ped/

# gpu and memory limit
GPU_DEVICE=0

# options
INTERACTIVE=1
LOG_OUTPUT=1

while [[ $# -gt 0 ]]
do key="$1"

case $key in
	-im|--image_name)
	IMAGE_NAME="$2"
	shift # past argument
	shift # past value
	;;
	-t|--tag)
	TAG="$2"
	shift # past argument
	shift # past value
	;;
	-i|--interactive)
	INTERACTIVE="$2"
	shift # past argument
	shift # past value
	;;
	-gd|--gpu_device)
	GPU_DEVICE="$2"
	shift # past argument
	shift # past value
	;;
	-m|--memory_limit)
	MEMORY_LIMIT="$2"
	shift # past argument
	shift # past value
	;;
	-cn|--container_name)
	CONTAINER_NAME="$2"
	shift # past argument
	shift # past value
	;;
	-h|--help)
	shift # past argument
	echo "Options:"
	echo "	-im, --image_name 	name of the docker image (default \"base_images/pytorchlightning/pytorch_lightning\")"
	echo "	-t, --tag 		image tag name (default \"tf2-gpu\")"
	echo "	-gd, --gpu_device 	gpu to be used inside docker (default 1)"
	echo "	-cn, --container_name	name of container (default \"base-cuda-py3.8-torch1.7\" )"
	echo "	-m, --memory_limit 	RAM limit (default 32g)"
	exit
	;;
	*)
	echo " Wrong option(s) is selected. Use -h, --help for more information "
	exit
	;;
esac
done

echo "GPU_DEVICE 	= ${GPU_DEVICE}"
echo "CONTAINER_NAME 	= ${CONTAINER_NAME}"
echo "folder host       = ${WORKING_DIR}"


echo "Running docker in interactive mode"

docker run --rm -it --gpus "device=${GPU_DEVICE}"  \
	--mount type=bind,source=${CODE_FOLDER},target=$WORKING_DIR \
	-w ${WORKING_DIR} \
	-e log=/home/log.txt \
	-e HOST_UID=$(id -u) \
	-e HOST_GID=$(id -g) \
	-u $(id -u):$(id -g) \
	-e DISPLAY=$DISPLAY \
	-e XAUTHORITY=$XAUTH \
	-v $XAUTH:$XAUTH \
	-p 9009:7007 \
	-v /tmp/.X11-unix:/tmp/.X11-unix \
	--ipc=host \
	--name ${CONTAINER_NAME} \
	--net=host \
	-env="DISPLAY" \
	--volume="$HOME/.Xauthority:/root/.Xauthority:rw" \
	${IMAGE_NAME}:${TAG}
