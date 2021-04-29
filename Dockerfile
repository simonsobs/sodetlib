# sodetlib dockerfile.
FROM tidair/pysmurf-client:v5.0.0

# Installs spt3g to /usr/local/src/
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

####  Installs OCS
WORKDIR /usr/local/src
RUN git clone https://github.com/simonsobs/ocs.git

COPY ./requirements.txt ocs/requirements.txt
RUN pip3 install -r ocs/requirements.txt
RUN pip3 install ./ocs

# Sets ocs configuration environment
ENV OCS_CONFIG_DIR=/config

#Copies and installs sodetlib
COPY . /sodetlib

WORKDIR /sodetlib

RUN pip3 install -e .
RUN pip3 install -U jupyterlab

