# sodetlib dockerfile.
FROM tidair/pysmurf-client:v5.0.0

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
RUN git clone https://github.com/simonsobs/so3g.git
WORKDIR /usr/local/src/so3g
ENV LANG C.UTF-8
RUN apt-get update
RUN apt-get install -y build-essential \
    automake \
    gfortran \
    libopenblas-dev

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
RUN git clone --branch load_g3_file https://github.com/simonsobs/sotodlib.git
RUN pip3 install quaternionarray sqlalchemy
RUN pip3 install ./sotodlib

#################################################################
# OCS Install
#################################################################
WORKDIR /usr/local/src
RUN git clone https://github.com/simonsobs/ocs.git

COPY ./requirements.txt ocs/requirements.txt
RUN pip3 install -r ocs/requirements.txt
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
RUN pip3 install -U jupyterlab
RUN pip3 install jedi==0.17.1

