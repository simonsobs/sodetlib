global POSTINGDIR
global ANALYSISDIR

POSTINGDIR = fullfile(pwd());
[r,ANALYSISDIR] = system('ls -d $HOME/keck');
ANALYSISDIR = strtrim(ANALYSISDIR);
clear r

cd(ANALYSISDIR)
run('./startup.m')
addpath(genpath([POSTINGDIR '/src']))

