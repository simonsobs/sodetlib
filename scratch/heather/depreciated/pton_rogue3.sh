# 11/19/19 : smurf-srv15
# Drop into rogue 3 configuration 
wd=$PWD
PYSMURF=pton-dspv3
PYROGUE=dev_fw/R3.1.1-3854241-c03lb-bay0
STARTUP_CFG=pton/pton_smurf_startup.cfg
PYSMURF_CFG=pton/smurf-srv08/experiment_pc004_smurfsrv08_noExtRef_dspv3.cfg

rm -v /data/smurf_startup_cfg/smurf_startup.cfg
ln -s /home/cryo/docker/pysmurf/${PYSMURF}/pysmurf/cfg_files/${STARTUP_CFG} /data/smurf_startup_cfg/smurf_startup.cfg

rm -v /data/pysmurf_cfg/${PYSMURF_CFG##*/}
ln -s /home/cryo/docker/pysmurf/${PYSMURF}/pysmurf/cfg_files/${PYSMURF_CFG} /data/pysmurf_cfg/${PYSMURF_CFG##*/}

cd /home/cryo/docker/pysmurf/${PYSMURF}/pysmurf/scratch/shawn/scripts/
sudo ./install.sh

rm /home/cryo/docker/smurf/current
ln -s /home/cryo/docker/smurf/${PYROGUE} /home/cryo/docker/smurf/current

cd $wd
