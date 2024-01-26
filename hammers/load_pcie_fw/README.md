Loading PCIe FW
-----------------
It's possible the PCIe is not loaded with the correct firmware.
Download from here: https://github.com/slaclab/smurf-pcie/releases/tag/v3.0.1

We want to use the files:
 - SmurfKcu1500RssiOffload10GbE-0x03000000-20230830025732-ruckman-d89c87b_primary.mcs.gz 
 - SmurfKcu1500RssiOffload10GbE-0x03000000-20230830025732-ruckman-d89c87b_secondary.mcs.gz

steps to load fw:
- From this directory, run `sh download_fw.sh`
- Run `docker-compose run smurf-pcie`
- select the number corresponding to the correct files as stated above

This will load fw and then reboot computer
Afterwards check `cat /proc/datadev_0`
The result Axi Version should ideally look something like this:
```
cryo@smurf-srv-SO6:~$ cat /proc/datadev_0
-------------- Axi Version ----------------
     Firmware Version : 0x3000000
           ScratchPad : 0x0
        Up Time Count : 132
            Device ID : 0x0
             Git Hash : d89c87b6b29ddc5dab8a3be374e749725d598399
            DNA Value : 0x00000000400200010117f1801c8102c5
         Build String : SmurfKcu1500RssiOffload10GbE: Vivado v2023.1, rdsrv407 (Ubuntu 20.04.6 LTS), Built Wed 30 Aug 2023 02:57:32 AM PDT by ruckman
...
```
