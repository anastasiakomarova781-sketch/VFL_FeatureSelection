#!/bin/bash

# examples to run docker servers

WORKDIR_ACTIVE=./example/workdir/active
WORKDIR_PASSIVE=./example/workdir/passive

# Training Servers (50000/50001)
docker run -d \
  --name ucbvfl-server-training_passive \
  --network host \
  -v "$WORKDIR_PASSIVE":/mnt/ext \
  ucbvfl-server training_passive \
    --work_dir=/mnt/ext \
    --listening_address=0.0.0.0:50001

docker run -d \
  --name ucbvfl-server-training_active \
  --network host \
  -v "$WORKDIR_ACTIVE":/mnt/ext \
  ucbvfl-server training_active \
    --work_dir=/mnt/ext \
    --listening_address=0.0.0.0:50000 \
    --passive_server_address=localhost:50001

# Prediction Servers (50050/50051)
docker run -d \
  --name ucbvfl-server-prediction_passive \
  --network host \
  -v "$WORKDIR_PASSIVE":/mnt/ext \
  ucbvfl-server prediction_passive \
    --work_dir=/mnt/ext \
    --listening_address=0.0.0.0:50051

docker run -d \
  --name ucbvfl-server-prediction_active \
  --network host \
  -v "$WORKDIR_ACTIVE":/mnt/ext \
  ucbvfl-server prediction_active \
    --work_dir=/mnt/ext \
    --listening_address=0.0.0.0:50050 \
    --passive_server_address=localhost:50051
