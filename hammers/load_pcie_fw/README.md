SO SMuRF Utils
==================
This repo contains a bunch of utility scripts for managing and debugging
smurf.

Loading PCIe FW
-----------------
It's possible the PCIe is not loaded with the correct firmware.
Download from here: https://github.com/slaclab/smurf-pcie/releases/tag/v2.3.0

We want to use the files:
 - SmurfKcu1500RssiOffload10GbE-0x02030000-20200512134423-ruckman-7aa0ea1_primary.mcs
 - SmurfKcu1500RssiOffload10GbE-0x02030000-20200512134423-ruckman-7aa0ea1_secondary.mcs

steps to load fw:
- From this directory, run `sh download_fw.sh`
- Run `docker-compose run smurf-pcie`
- select 0 when prompted

This will load fw and then reboot computer
Afterwards check `cat /proc/datadev_0`
The result should ideally look something like this:
```
cryo@smurf-srv24:~$ cat /proc/datadev_0 
-------------- Axi Version ----------------
   Firmware Version : 0x2030000
      ScratchPad : 0x0
    Up Time Count : 3164802
      Device ID : 0x0
       Git Hash : 7aa0ea1a0ae6890e9e594a3d8a37476183075ed5
      DNA Value : 0x000000004002000101181bc404306145
     Build String : SmurfKcu1500RssiOffload10GbE: Vivado v2019.2, rdsrv302 (x86_64), Built Tue 12 May 2020 01:44:23 PM PDT by ruckman

-------------- General HW -----------------
     Int Req Count : 0
    Hw Dma Wr Index : 779
    Sw Dma Wr Index : 779
    Hw Dma Rd Index : 3458
    Sw Dma Rd Index : 3458
   Missed Wr Requests : 0
    Missed IRQ Count : 1382663943
     Continue Count : 274338
     Address Count : 4096
  Hw Write Buff Count : 1024
   Hw Read Buff Count : 0
      Cache Config : 0x0
      Desc 128 En : 1
      Envable Var : 0x2010101
   Driver Load Count : 1

-------------- General --------------------
     Dma Version : 0x5
     Git Version : v5.7.0

-------------- Read Buffers ---------------
     Buffer Count : 1024
     Buffer Size : 131072
     Buffer Mode : 1
   Buffers In User : 0
    Buffers In Hw : 1024
 Buffers In Pre-Hw Q : 0
 Buffers In Rx Queue : 0
   Missing Buffers : 0
    Min Buffer Use : 770541
    Max Buffer Use : 770549
    Avg Buffer Use : 770544
    Tot Buffer Use : 789037835

-------------- Write Buffers ---------------
     Buffer Count : 1024
     Buffer Size : 131072
     Buffer Mode : 1
   Buffers In User : 0
    Buffers In Hw : 0
 Buffers In Pre-Hw Q : 0
 Buffers In Sw Queue : 1024
   Missing Buffers : 0
    Min Buffer Use : 772450
    Max Buffer Use : 772452
    Avg Buffer Use : 772451
    Tot Buffer Use : 790990210
```


