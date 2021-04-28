#!/usr/bin/env bash

port=$1

user="cryo"
docker run -it --rm  \
  --log-opt tag=guis \
  -u $(id -u ${user}):$(id -g ${user}) \
  --security-opt apparmor=docker-smurf \
  --net host \
  -e EPICS_CA_AUTO_ADDR_LIST=NO \
  -e EPICS_CA_ADDR_LIST=127.255.255.255 \
  -e EPICS_CA_MAX_ARRAY_BYTES=80000000 \
  -e DISPLAY \
  -v /etc/group:/etc/group:ro \
  -v /etc/passwrd:/etc/passerd:ro \
  -v /home/${user}/.bash_history:/home/${user}/.bash_history \
  -v /home/${user}/.Xauthority:/home/${user}/.Xauthority \
  tidair/smurf-rogue:R2.8.2 \
  python3 -m pyrogue gui --server=localhost:$port

