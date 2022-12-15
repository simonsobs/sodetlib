FROM simonsobs/so_smurf_base:v0.0.2-1-g4a75f5b

#################################################################
# sodetlib Install
#################################################################
COPY . /sodetlib
WORKDIR /sodetlib
RUN pip3 install -e .
RUN pip3 install -r requirements.txt