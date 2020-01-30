FROM nvidia/cuda:10.1-devel-ubuntu18.04
ENV MAKE="make -j 4"

RUN apt-get update && \
  DEBIAN_FRONTEND=noninteractive apt-get install -y \
    git locales gcc-8 \
    libopenmpi-dev openmpi-common openmpi-bin ssh \
    libopenblas-dev libscalapack-openmpi-dev

RUN mkdir -p /opt/fml
RUN cd /opt && git clone --recurse-submodules https://github.com/wrathematics/fml.git

RUN ln -s /opt/fml/src /usr/include/fml

RUN ["/bin/bash"]
