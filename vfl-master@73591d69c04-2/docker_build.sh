#!/bin/bash

docker build --progress=plain --no-cache -f docker/runtime.docker -t ucbvfl-server .
