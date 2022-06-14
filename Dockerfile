# sodetlib dockerfile.
FROM tidair/pysmurf-client:v6.0.0

#################################################################
# SPT3G Install
#################################################################
WORKDIR /usr/local/src/

ENV TZ=Etc/UTC
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update

RUN apt-get install -y \
    libboost-all-dev \
    libflac-dev \
    libnetcdf-dev \
    libfftw3-dev \
    libgsl0-dev \
    tcl \
    environment-modules \
    gdb \
    rsync \
    cmake \
    libblas-dev \
    # so3g reqs
    automake \
    gfortran \
    build-essential \
    libbz2-dev \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/CMB-S4/spt3g_software.git
RUN cd spt3g_software \
    && mkdir -p build \
    && cd build \
    && cmake .. -DPYTHON_EXECUTABLE=`which python3` \
    && make core version

ENV SPT3G_SOFTWARE_PATH /usr/local/src/spt3g_software
ENV SPT3G_SOFTWARE_BUILD_PATH /usr/local/src/spt3g_software/build
ENV PYTHONPATH /usr/local/src/spt3g_software/build:${PYTHONPATH}

#################################################################
# SO3G Install
#################################################################
WORKDIR /usr/local/src
ENV LANG C.UTF-8

RUN git clone https://github.com/simonsobs/so3g.git
WORKDIR /usr/local/src/so3g
RUN pip3 install -r requirements.txt

ENV Spt3g_DIR /usr/local/src/spt3g_software
# Install qpoint
RUN /bin/bash /usr/local/src/so3g/docker/qpoint-setup.sh

# Build so3g
RUN mkdir build \
    && cd build \
    && cmake .. -DCMAKE_PREFIX_PATH=$SPT3G_SOFTWARE_BUILD_PATH \
    && make \
    && make install

#################################################################
# SOTODLIB Install
#################################################################
WORKDIR /usr/local/src
RUN git clone https://github.com/simonsobs/sotodlib.git
RUN pip3 install quaternionarray sqlalchemy
RUN pip3 install ./sotodlib

#################################################################
# OCS Install
#################################################################
RUN git clone --branch v0.9.1 https://github.com/simonsobs/ocs.git

RUN pip3 install cryptography==3.3.2
# RUN pip3 install -r ocs/requirements.txt
RUN pip3 install ./ocs

# Sets ocs configuration environment
ENV OCS_CONFIG_DIR=/config

#################################################################
# sodetlib Install
#################################################################
COPY . /sodetlib
WORKDIR /sodetlib
RUN pip3 install -e .
RUN pip3 install -r requirements.txt

# This is to get the leap-second download out of the way
RUN python3 -c "import so3g"
