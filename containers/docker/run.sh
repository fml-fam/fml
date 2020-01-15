#!/bin/sh

cd ../../
sudo docker build -t fml -f containers/docker/Dockerfile . \
  && sudo docker run -i -t --rm fml
