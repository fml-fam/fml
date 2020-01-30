#!/bin/sh

sudo docker build -t fml-dev-gpu . \
  && sudo docker run -i -t --rm fml-dev-gpu
