# sodetlib dockerfile.
FROM tidair/pysmurf-client:v4.0.0

WORKDIR /usr/local/src
RUN git clone https://github.com/simonsobs/ocs.git
RUN pip3 install -r ocs/requirements.txt
RUN pip3 install ./ocs

# Sets ocs configuration environment
ENV OCS_CONFIG_DIR=/config

#Copies and installs sodetlib
COPY . /sodetlib

WORKDIR /sodetlib

RUN pip3 install -e .

