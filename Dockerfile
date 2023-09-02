FROM simonsobs/so_smurf_base:v0.0.4-1-g74a18de

#################################################################
# sodetlib Install
#################################################################
COPY . /sodetlib
WORKDIR /sodetlib
RUN pip3 install -e .
RUN pip3 install -r requirements.txt
