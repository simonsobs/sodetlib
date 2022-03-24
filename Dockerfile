# sodetlib dockerfile.
FROM tidair/pysmurf-client:v7.1.0

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

RUN git clone https://github.com/CMB-S4/spt3g_software.git && cd spt3g_software && git checkout  5f30121395129de9c9a6af2976de8ba8e876b5a8
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
RUN mkdir /usr/local/src/IPMC && mkdir /usr/local/src/FirmwareLoader
RUN git clone --depth 1 https://github.com/slaclab/smurf-base-docker.git 
RUN tar -xf /usr/local/src/smurf-base-docker/packages/IPMC.tar.gz \
         -C /usr/local/src/IPMC
RUN tar -xf /usr/local/src/smurf-base-docker/packages/FirmwareLoader.tar.gz \
         -C /usr/local/src/FirmwareLoader

ENV LD_LIBRARY_PATH /usr/local/src/IPMC/lib64:${LD_LIBRARY_PATH}
ENV PATH /usr/local/src/IPMC/bin/x86_64-linux-dbg:${PATH}
ENV PATH /usr/local/src/FirmwareLoader:${PATH}

# Install EPICS
RUN mkdir -p /usr/local/src/epics/base-3.15.5
WORKDIR /usr/local/src/epics/base-3.15.5
RUN wget -c base-3.15.5.tar.gz https://github.com/epics-base/epics-base/archive/R3.15.5.tar.gz -O - | tar zx --strip 1 && \
    make clean && make && make install && \
    find . -maxdepth 1 \
    ! -name bin -a ! -name lib -a ! -name include \
    -exec rm -rf {} + \
    || true
ENV EPICS_BASE /usr/local/src/epics/base-3.15.5
ENV EPICS_HOST_ARCH linux-x86_64
ENV PATH /usr/local/src/epics/base-3.15.5/bin/linux-x86_64:${PATH}
ENV LD_LIBRARY_PATH /usr/local/src/epics/base-3.15.5/lib/linux-x86_64:${LD_LIBRARY_PATH}
ENV PYEPICS_LIBCA /usr/local/src/epics/base-3.15.5/lib/linux-x86_64/libca.so

RUN pip3 install ipython numpy pyepics

#################################################################
# SOTODLIB Install
#################################################################
WORKDIR /usr/local/src

# Freeze sotodlib before so3g was added as requirement
RUN git clone https://github.com/simonsobs/sotodlib.git && cd sotodlib && git checkout e37d2c0b342f609cf640ee79541c98f7a2d7485a
RUN pip3 install quaternionarray sqlalchemy
RUN pip3 install ./sotodlib

#################################################################
# OCS Install
#################################################################
RUN git clone --branch py36 https://github.com/simonsobs/ocs.git

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

RUN rm -rf /var/lib/apt/lists/*
