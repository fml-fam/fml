#!/bin/sh

sudo docker build -t fml-dev . \
  && sudo docker run -i -t --rm fml-dev
