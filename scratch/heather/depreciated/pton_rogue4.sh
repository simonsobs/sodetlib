# 11/19/19 : smurf-srv15
# Drop into rogue 3 configuration 
wd=$PWD
PYSMURF=R4-rc1
#PYROGUE=dev_fw/v4.0.0-rc15
PYROGUE=dev_fw/slotN/v4.0.0-rc21
STARTUP_CFG=pton/pton_smurf_startup.cfg
PYSMURF_CFG=pton/experiment_pd009_cc02-01_lbOnlyBay0_r4.cfg

rm -v /data/smurf_startup_cfg/smurf_startup.cfg
ln -s /home/cryo/docker/pysmurf/${PYSMURF}/pysmurf/cfg_files/${STARTUP_CFG} /data/smurf_startup_cfg/smurf_startup.cfg

rm -v /data/pysmurf_cfg/${PYSMURF_CFG##*/}
ln -s /home/cryo/docker/pysmurf/${PYSMURF}/pysmurf/cfg_files/${PYSMURF_CFG} /data/pysmurf_cfg/${PYSMURF_CFG##*/}

cd /home/cryo/docker/pysmurf/${PYSMURF}/pysmurf/scratch/shawn/scripts/
sudo ./install.sh

rm /home/cryo/docker/smurf/current
ln -s /home/cryo/docker/smurf/${PYROGUE} /home/cryo/docker/smurf/current

cd $wd
