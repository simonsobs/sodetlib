#Author: Kaiwen Zheng
#Description: Functions to map smurf channels to detector UIDs
import numpy as np
import pandas as pd
import math
import re


def smurf_chan_to_realized_freq(filename, band=None):

    smurf2Rfreq=pd.DataFrame({"smurf_band":{},"smurf_chan":{},"smurf_freq":{}})

    smurf_file=pd.read_csv(filename,header=None)
    freq=np.array(smurf_file[0])

    #Make sure band input, filename and frequency matches
    if band!=None:
        assert type(band) is int or float
        assert int(band)==math.floor((freq[0]-4000)/500)
    else:
        band=math.floor((freq[0]-3998)/500)

    searcher=re.search('_channel_assignment_b(.+).txt',filename)
    if searcher==None:
        pass        
    else:
        assert band==int(filename[-5])
            
    #Correct for AMC frequency
    if max(freq)>8e3 or min(freq)<4e3:
        raise ValueError("Smurf frequency out of range")
    if min(freq)>6e3:
        freq=freq-2000

    smurf2Rfreq["smurf_band"]=np.zeros(len(freq))+band
    smurf2Rfreq["smurf_chan"]=np.array(smurf_file[2])
    smurf2Rfreq["smurf_freq"]=freq

    valid_smurf2Rfreq=smurf2Rfreq.loc[smurf2Rfreq["smurf_chan"]!=-1]
    valid_smurf2Rfreq.index=np.arange(len(valid_smurf2Rfreq))

    return valid_smurf2Rfreq

def all_smurf_chan_to_realized_freq(filenames,band=None):
    """
    Returns a dataframe of smurf channel assignments
    Parameters
    ----------
    filenames : filename or an array of filenames
        Names for smurf channel assignments

    Returns
    -------
    smurf2Rfreq: DataFrame
        a table of smurf channel number and measured frequencies
        All frequency are corrected to be 4-6 GHz
    """
    smurf2Rfreq=pd.DataFrame({"smurf_band":{},"smurf_chan":{},"smurf_freq":{}})
    if band!=None:
        assert np.array([filenames]).size==np.array([band]).size

    if np.array([filenames]).size==1:
        smurf2Rfreq=smurf_chan_to_realized_freq(filename, band)
    else:
        for i,onefile in enumerate(filenames):
            
            if band!=None:
                oneband= band[i]
            else:
                oneband=None

            df_oneband=smurf_chan_to_realized_freq(onefile, oneband)
            smurf2Rfreq=smurf2Rfreq.append(df_oneband, ignore_index=True)
    return smurf2Rfreq

def vna_freq_to_muxpad(vna2designfile,design_file="umux_32_map.pkl"):
    """
    Reads a map of mux chip realized-frequency-to-frequency-index, and output the 
    bondpad location of these frequencies
    ----------
    vna2designfile: string
        A csv file with header 
        | Band | Index | UFM Frequency |
        Band is the muxband which has integers between 0-13, 
        Index is the frequency order within a mux band which is integer between 0-65
        UFM Frequency is resonator frequency in MHz or Hz as measured in VNA
        This file needs to be input into the script as of April 2021

    design_file:
        filepath to the mux chip design map, for v3.2 this is can be found in 
        http://simonsobservatory.wikidot.com/local--files/muxdesigns/umux_32_map.pkl

    Returns:
    -------
    vna2pad: DataFrame
        a table of VNA frequency to mux band and mux bondpad
    """
    df_design=pd.read_pickle(design_file)
    df_design['Frequency(MHz)']=df_design['Frequency(MHz)']/1e6 #A mistake in header,freq is in Hz

    df_vna=pd.read_csv(vna2designfile)

    freq=np.array(df_vna['UFM Frequency'])
    if ((2e9 < freq) & (freq < 8e9)).all():
        freq=freq/1e6
        df_vna['UFM Frequency']=freq

    assert ((2e3 < freq) & (freq < 8e3)).all()


    vna2pad=pd.DataFrame({"mux_band":{},"pad":{},"index":{},"design_freq":{},"vna_freq":{}})

    for i in np.arange(len(df_vna)):
        for j in np.arange(len(df_design)):
            if df_vna['Band'][i]==df_design['Band'][j]:
                if df_vna['Index'][i]==df_design['Freq-index'][j]:
                    vna2pad=vna2pad.append({"mux_band":df_vna['Band'][i],"pad":df_design['Pad'][j],
                        "index":df_vna['Index'][i],"design_freq":df_design['Frequency(MHz)'][j],
                        "vna_freq":df_vna['UFM Frequency'][i]},ignore_index=True)
    return vna2pad



def smurf_to_mux(smurf2Rfreq,vna2pad,threshold=0.01):
    """
    Reads SmuRF information and VNA-2-bondpad information and produce a map from
    smurf channel to mux band and bondpads
    ----------
    smurf2Rfreq:
        DataFrame that includes SMuRF tuning frequency to smurf bands and channels
    vna2pad:
        DataFrane that includes mux chip frequency and bondpad information
    threshold:
        The expected difference between VNA and SmuRF found resonance frequency, in MHz.

    Returns:
    -------
    smurf2mux: DataFrame
        A table of smurf band and channel to mux band and pad
    """

    smurf2mux=pd.DataFrame({"smurf_band":{},"smurf_chan":{},"smurf_freq":{},"mux_band":{},"pad":{},
                                  "index":{},"design_freq":{},"vna_freq":{}})

    for i,smurf_freq in enumerate(smurf2Rfreq["smurf_freq"]):
        found=False
        for j,vna_freq in enumerate(vna2pad["vna_freq"]):
            if found==False and abs(smurf_freq-vna_freq)<threshold:
                row=smurf2Rfreq[i:i+1]
                row2=vna2pad[j:j+1]
                row2.index =row.index
                smurf2mux=smurf2mux.append(pd.concat([row,row2],axis=1))
                found=True
    return  smurf2mux



def mux_band_to_mux_posn(smurf2mux,mux_band2_mux_posn,highband='S'):
    """
    Find the wafer location index of each mux chip.
    ----------
    smurf2mux:
        A dataframe that includes information from each smurf band to mux band
    mux_band2_mux_posn:
        A file of mux_band to 'mux_posn' correspondence    
    highband:
        'S' or 'N', indicates which side of the wafer is connected to SMuRF band 4-7

    Returns:
    -------
    smurf2padloc: DataFrame
        A table of smurf band and channel to mux location
    """
    smurf2mux=smurf2mux.reset_index(drop=True)
    smurf2padloc=pd.concat([smurf2mux,pd.DataFrame({"mux_posn":[]})],axis=1)

    band2posn=pd.read_csv(mux_band2_mux_posn)
    band = np.array(band2posn['mux_band'])
    posn = np.array(band2posn['mux_posn'])
    all_posn=[]

    assert ((-1 < band) & (band< 14)).all()
    assert ((-1 < posn) & (posn < 28)).all()

    for i, smurf_band in enumerate(smurf2mux["smurf_band"]):
        in_south=(smurf_band>3 and highband=='S') or (smurf_band<=3 and highband=='N')
        in_north=(smurf_band>3 and highband=='N') or (smurf_band<=3 and highband=='S')
        assert (in_south and in_north)==False

        twosides=posn[np.where(np.array(smurf2mux["mux_band"])[i]==band)]
        assert len(twosides)<=2

        if in_south:
            mux_posn=[x for x in twosides if x>13]
        elif in_north:
            mux_posn=[x for x in twosides if x<=13]
        else:
            raise ValueError("mux band not on wafer layout file")
        
        try:
            smurf2padloc["mux_posn"][i]=mux_posn[0]
        except:
            pass

    return smurf2padloc

def get_pad_to_wafer(filename, dark_bias_lines=[]):
    """
    Extracts routing wafer to detector wafer map
    Mostly from Zach Atkin's script 
    ----------
    filename:
        Path to the detector-routing wafer map created by NIST and Princeton
    dark_bias_lines:
        Bias lines that are dark in a particular test

    Returns:
    -------
    wafer_info
        A table from mux chip position and bondpad to detector information
        In particular, freq column indicates 90ghz, 150ghz, D for dark 
        detectors which is 90ghz but has different property as optical ones,
        and NC for no-coupled resonators
    """


    wafer_file = pd.read_csv(filename)
    wafer_info=pd.DataFrame({"mux_posn":{},"pad":{},"biasline":{},"pol":{},"freq":{},"det_row":{},
        "det_col":{},"rhomb":{},"opt":{},"det_x":{},"det_y":{}})

    for index, row in wafer_file.iterrows():
            
        pad_re = 'SQ_(.+)_Ch_(.+)_\+'
        pad_str = row['SQUID_PIN']
        searcher = re.search(pad_re, pad_str)
        if searcher==None:
            pass
        else:
            _, pad_str = searcher.groups() 
            pad = int(pad_str)
            posn = row['Mux chip position']

            pol_str = row['DTPadlabel']
            if pol_str[0] in ['T', 'R']: 
                pol = 'A'
            elif pol_str[0] in ['B', 'L']:
                pol = 'B'
            elif pol_str[0] in ['X']:
                pol = 'D'
            else:
                assert False

            rhomb = row['DTPixelsection']


            bias_line = int(row['Bias line'])


            r = int(row['DTPixelrow'])
            c = int(row['DTPixelcolumn'])

            det_x = float(row['x']) / 1e3
            det_y = float(row['y']) / 1e3

            if (bias_line in dark_bias_lines) or (pol == 'D') or row['DTSignaldescription']=='NC':
                opt = False
            else:
                opt = True


            freq_re = '(.+)ghz'
            freq_str = row['DTSignaldescription']
            searcher = re.search(freq_re, freq_str)
            if searcher is None:
                if row['DTSignaldescription']=='NC':
                    freq='NC'
                if pol=='D':
                    freq ='D'
            else:
                freq_str = searcher.groups()
                freq = int(*freq_str)

            wafer_info=wafer_info.append({"mux_posn":posn,"pad":pad,"biasline":bias_line,"pol":pol,"freq":freq,"det_row":r,
                "det_col":c,"rhomb":rhomb,"opt":opt,"det_x":det_x,"det_y":det_y},ignore_index=True)
    return wafer_info

def smurf_to_detector(smurf2padloc,wafer_info):
    """
    Produces a map from smurf channel to detector information
    """

    smurf2det=pd.concat([smurf2padloc[:0],wafer_info[:0]],axis=1)
    for i,smurf_posn in enumerate(smurf2padloc["mux_posn"]):
        for j, wafer_posn in enumerate(wafer_info["mux_posn"]):
            if smurf_posn==wafer_posn and smurf2padloc["pad"][i]==wafer_info["pad"][j]:
                row=smurf2padloc[i:i+1]
                row2=wafer_info[j:j+1]
                row2.index=row.index
                smurf2det=smurf2det.append(pd.concat([row,row2],axis=1))
    smurf2det=smurf2det.loc[:,~smurf2det.columns.duplicated()]
    smurf2det=smurf2det.reindex(columns=["smurf_band","smurf_chan","smurf_freq","vna_freq","design_freq","index",
                               "mux_band","pad","mux_posn","biasline","pol","freq","det_row","det_col",
                               "rhomb","opt","det_x","det_y"]) 
    return smurf2det





