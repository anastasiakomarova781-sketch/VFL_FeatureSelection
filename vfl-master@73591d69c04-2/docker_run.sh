#!/bin/bash

# examples to run docker servers

WORKDIR_ACTIVE=./example/workdir/active
WORKDIR_PASSIVE=./example/workdir/passive

docker container stop  ucbvfl-server-training_active
docker container remove ucbvfl-server-training_active
docker container stop  ucbvfl-server-training_passive
docker container remove ucbvfl-server-training_passive
docker container stop  rabbitmq
docker container remove rabbitmq

docker network rm vflnet
docker network create vflnet

docker run -d \
 --name rabbitmq \
 --hostname rabbitmq \
 --network vflnet \
 -p 5672:5672 \
 -p 15672:15672 rabbitmq:3-management

sleep 10
docker exec rabbitmq rabbitmqctl add_user rabbitmqadmin 12345
docker exec rabbitmq rabbitmqctl set_user_tags rabbitmqadmin administrator


sleep 10
docker exec rabbitmq rabbitmqctl add_vhost arbiter
docker exec rabbitmq rabbitmqctl add_vhost guest
docker exec rabbitmq rabbitmqctl add_vhost host
docker exec rabbitmq rabbitmqctl set_permissions -p arbiter rabbitmqadmin ".*" ".*" ".*"
docker exec rabbitmq rabbitmqctl set_permissions -p guest rabbitmqadmin ".*" ".*" ".*"
docker exec rabbitmq rabbitmqctl set_permissions -p host rabbitmqadmin ".*" ".*" ".*"


# Training Servers (50000/50001)
docker run -d \
  --name ucbvfl-server-training_passive \
  --network vflnet \
  -p 50001:50001 \
  -v "$WORKDIR_PASSIVE":/mnt/ext \
  ucbvfl-server training_passive \
    --work_dir=/mnt/ext \
    --listening_address=0.0.0.0:50001

docker run -d \
  --name ucbvfl-server-training_active \
  --network vflnet \
  -p 50000:50000 \
  -v "$WORKDIR_ACTIVE":/mnt/ext \
  ucbvfl-server training_active \
    --work_dir=/mnt/ext \
    --listening_address=0.0.0.0:50000 \
    --passive_server_address=ucbvfl-server-training_passive:50001

# Prediction Servers (50050/50051)
  