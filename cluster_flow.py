#import all the processing functions from processing_routines
from processing_routines import *
#import all important packages to avoid errors
from obspy import UTCDateTime
import obspy
from obspy.clients.fdsn import Client
import random
from obspy.clients.fdsn import RoutingClient
from obspy import Stream
from obspy.geodetics import gps2dist_azimuth
from obspy import signal
from obspy import read
import matplotlib.pyplot as plt
import folium
import numpy as np
import json
import pandas as pd
from obspy.clients.fdsn.header import FDSNNoDataException
from scipy.signal import savgol_filter
import os
import folium
import numpy as np
import branca.colormap as cm
from scipy.signal import resample
from obspy import Stream
from matplotlib.colors import LogNorm
import pandas as pd
#set important parameters
Dtmin_Noise=-25
Dtmax_Noise=-5
Dtmin_Pn=-5.
Dtmax_Pn=10.
Dtmin_Sn=-5.
Dtmax_Sn=10.
vLg_max=3.5
vLg_min=3.1
vLg=0.5*(vLg_max+vLg_min)
vPg_max=6.2
vPg_min=5.2
vPg=0.5*(vPg_max+vPg_min)
tminCoda=300.
tmaxCoda=320.
















from urllib.error import URLError
from http.client import HTTPException
import os
import warnings
warnings.filterwarnings("ignore")


def processing_data_routine(datacenters=['RESIF', 'ODC', 'ETH', 'INGV', 'GEOFON', 'IRIS', 'ICGC','LMU','BGR',"http://fdsnws.sismologia.ign.es"],
                            directory='/bettik/PROJECTS/pr-terracorr/schreinl/Data/',distmin=0.5,distmax=10.0,catalogue='/bettik/PROJECTS/pr-terracorr/schreinl/Data/trial.csv',
                            factor=1.1,fmin=[0.5,2,4,6],fmax=[1.5,3,6,8],codawindow='cutoff',snr_threshold=5):
    '''
    this function does all the heavy processing steps. It downloads first the data (all of it when the checking rhythm is not active),
    then subsequently the data is filtered, and the envelopes are calculated. then the SNR routine is started, removing data
    with a weak body wave SNR, and calculating the cutoff distance, on which the coda window is based. 
    
    '''
    
    
    eq_list = pd.read_csv(catalogue)
    #in order to run without checking the existence of the file, we have to write here 
    existent = False
    #and comment the checking rhythm
    for event in range(len(eq_list)):
        print("event", event, "out of", len(eq_list))
        start = UTCDateTime(eq_list["time"][event]) -25
        end = start + 700
        eq_lon = float(eq_list["longitude"][event])
        eq_lat = float(eq_list["latitude"][event])
        
        time_string = UTCDateTime.strftime(start, format="%Y_%m_%dT%H_%M_%S")
        #existent = True
        for i in range(len(fmin)):
            print(i)
            print(fmin[i])
            amplitude_test = f"/bettik/PROJECTS/pr-terracorr/schreinl/Data/{time_string}/{time_string}_new_{codawindow}_fac_{factor}_{fmin[i]}_{fmax[i]}Hz_dict.txt"
            #if os.path.exists(amplitude_test):
            #    continue
            #else:
                #existent = False
        retry_attempts = 3
        if existent == True:
            print(f'event {event} is already handled entirely')
            continue
        if existent == False:
            for attempt in range(retry_attempts):
                try:
                    st_all, stations_all, plot = big_downloader2(datacenters, start, end, eq_lon, eq_lat, distmin, distmax, directory, plot=False)
                    break
                except (URLError, FDSNNoDataException, HTTPException) as e:
                    print(f"Failed to download data for event {event} on attempt {attempt + 1}: {str(e)}")
                    if attempt < retry_attempts - 1:
                        print("Retrying with a smaller time window...")
                        end -= 100  # Reduce the time window by 100 seconds and retry
                    else:
                        print("Max retry attempts reached. Skipping event...")
                        continue




        for i in range(len(fmin)):
            print(f'Frequency range {fmin[i]}-{fmax[i]}')
            envelope_file = f"/bettik/PROJECTS/pr-terracorr/schreinl/Data/{time_string}/{time_string}_new_{codawindow}_fac_{factor}_{fmin[i]}_{fmax[i]}Hz_stream1.mseed"
            amplitude_file = f"/bettik/PROJECTS/pr-terracorr/schreinl/Data/{time_string}/{time_string}_new_{codawindow}_fac_{factor}_{fmin[i]}_{fmax[i]}Hz_dict1.txt"
            
            if os.path.exists(envelope_file) and os.path.exists(amplitude_file):
                print(f"Files for event {event} already exist. Skipping...")
                continue
            
            

            st_plot_filt_all = st_all.copy()
            st_plot_filt_all.filter("bandpass", freqmin=fmin[i], freqmax=fmax[i])

            st_envelope = obspy.Stream()
            smallest = 7000
            for tr in st_plot_filt_all:
                if tr.data is None or len(tr.data) == 0:
                    print(f"Skipping trace {tr.id} due to empty data.")
                    continue
                data_envelope = envelope_calculator(tr.data)
                npts = tr.stats.npts
                if npts >= smallest:
                    samprate = tr.stats.sampling_rate
                    t = np.arange(0, npts / samprate, 1 / samprate)
                    tr_envelope = obspy.Trace(data=data_envelope, header=tr.stats)
                    st_envelope.append(tr_envelope)

            snr_threshold = snr_threshold
            eq_start = start

            filtered_stations_with_SNR, stations_with_SNR, distance_dict, tcoda_test, filtered_st, stations_with_amps, amp_plot = SNR_all(
                stations_all, st_plot_filt_all, Dtmin_Pn, Dtmax_Pn, Dtmin_Sn, Dtmax_Sn, vLg_min, vLg_max, vPg_min, vPg_max, tminCoda, tmaxCoda,
                Dtmin_Noise, Dtmax_Noise, eq_start, eq_lat, eq_lon, snr_threshold=snr_threshold, plot_SNR=False, plot_amps=False, wavecode="Lg_Coda", dB=True, codawindow=codawindow, factor=factor)

            if filtered_stations_with_SNR is None or len(filtered_stations_with_SNR) == 0:
                continue

            with open(f"/bettik/PROJECTS/pr-terracorr/schreinl/Data/{time_string}_{snr_threshold}_thresh_dict.txt", "w") as file:
                json.dump(distance_dict, file, indent=4)
            
            # Save stations_with_amps to a file
            with open(f"/bettik/PROJECTS/pr-terracorr/schreinl/Data/{time_string}/{time_string}_{fmin[i]}_{fmax[i]}Hz_{snr_threshold}_thresh_stations_with_amps1.txt", "w") as ampls:
                json.dump(stations_with_amps.tolist(), ampls, indent=4)

                # Save filtered stations with their corresponding SNR
            with open(f"/bettik/PROJECTS/pr-terracorr/schreinl/Data/{time_string}/{time_string}_{snr_threshold}_thresh_filtered_stations_SNR.txt", "w") as snrfile:
                json.dump(filtered_stations_with_SNR.tolist(), snrfile, indent=4)

                # Save the stations with SNR, unfiltered
            with open(f"/bettik/PROJECTS/pr-terracorr/schreinl/Data/{time_string}/{time_string}_unfiltered_stations_SNR.txt", "w") as unsnrfile:
                json.dump(stations_with_SNR.tolist(), unsnrfile, indent=4)
            amplitudes_full = calc_amps(stations_all, st_plot_filt_all, Dtmin_Pn, Dtmax_Pn, Dtmin_Sn, Dtmax_Sn, vLg_min,vLg_max,vPg_min,vPg_max, tcoda_test, tcoda_test+100, Dtmin_Noise, Dtmax_Noise, eq_start)
            amplitudes_small = calc_amps(stations_all, st_plot_filt_all, Dtmin_Pn, Dtmax_Pn, Dtmin_Sn, Dtmax_Sn, vLg_min,vLg_max,vPg_min,vPg_max, tcoda_test+80, tcoda_test+100, Dtmin_Noise, Dtmax_Noise, eq_start)
            SNR_dict = select_ratio_dict("Coda_Noise", amplitudes_full)
            SNR_dict_small = select_ratio_dict("Coda_Noise", amplitudes_small)
            envelopes_amps, st_smooth = envelopes_routine1(time_string, st_envelope, coda_dist_start=tcoda_test, coda_dist_end=tcoda_test + 100, plotting=False, method='cutoff', snr=SNR_dict, snr_window=SNR_dict_small)
            st_smooth.write(envelope_file, format="MSEED")

            with open(amplitude_file, "w") as ampls:
                json.dump(envelopes_amps, ampls, indent=4)

            filtered_station_names = set(row[1] for row in filtered_stations_with_SNR)
            filtered_smooth_stream = obspy.Stream()
            filtered_stream = obspy.Stream()
            for trace in st_smooth:
                if trace.stats.station in filtered_station_names:
                    filtered_smooth_stream.append(trace)
            for trace in st_envelope:
                if trace.stats.station in filtered_station_names:
                    filtered_stream.append(trace)
test = processing_data_routine(directory='/bettik/PROJECTS/pr-terracorr/schreinl/Data/',catalogue='/bettik/PROJECTS/pr-terracorr/schreinl/Data/trial.csv')