# sodetlib dockerfile.
FROM tidair/pysmurf-client:v4.0.0

# Sets ocs configuration environment
ENV OCS_CONFIG_DIR=/config

#Copies and installs sodetlib
COPY . /sodetlib

WORKDIR /sodetlib

RUN pip3 install -e .