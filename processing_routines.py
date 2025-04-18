from obspy import UTCDateTime, read
from obspy.clients.syngine import Client
from obspy.taup import TauPyModel
from obspy.geodetics import gps2dist_azimuth
from obspy.core import Stream
import os
from tqdm import tqdm
import folium
from obspy.clients.fdsn import Client
from obspy import read_events
import numpy as np
import pandas as pd
import json

def get_data2(client, inventory, start, end, eq_lon, eq_lat, distmin, distmax, directory='/home/schreinl/Stage/Data/',datacenter='datacenter'):
    """
    -function that downloads data from given client and inventory for a given time window
    -filters the stations based on their location
    -checks if data is already downloaded and reads it from disk if it is
    -gives back stream with all the data {st_final_target}, as well as a list of stations and their metadata,
      including expected arrival times of Pn and Sn {stations_target}
    """
    #initialize variables
    stations_target = []
    time_string = UTCDateTime.strftime(start, format="%Y_%m_%dT%H_%M_%S")

    priorities = ["BHZ", "HHZ", "SHZ"]
    stat_count = 0
    st_final_target = None
    
    #create target directory
    target_directory = f'{directory}{time_string}' #format /../
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)
    logfile_path = os.path.join(target_directory, f"{time_string}.txt")
    
    #Read existing log file if it exists
    log_data = {}
    if os.path.exists(logfile_path):
        with open(logfile_path, "r") as logfile:
            for line in logfile:
                station_channel, status = line.strip().split(',')
                log_data[station_channel] = status

    #open log file for appending in a+ mode
    logfile = open(logfile_path, "a+")

    # Initialize tqdm progress bar
    total_stations = sum(len(network.stations) for network in inventory)
    pbar = tqdm(total=total_stations, desc=f"Processing stations of {datacenter}")

    for network in inventory:
        for station in network.stations:
            # Update progress bar
            pbar.update(1)
            
            #calculate distance between station and earthquake
            epi_dist, az, baz = gps2dist_azimuth(eq_lat, eq_lon, station.latitude, station.longitude)
            epi_dist_deg = epi_dist / 1000 / 111. 
            
            #check if station is in the desired distance range
            if distmin < epi_dist_deg < distmax:
                prio = [0, 0, 0]
                for location in station:
                    if location.code == priorities[0]: prio[0] = 1
                    if location.code == priorities[1]: prio[1] = 1
                    if location.code == priorities[2]: prio[2] = 1
                
                #create filepath for each channel
                chan_to_get = priorities[prio.index(1)]
                station_channel = f"{network.code}_{station.code}_{chan_to_get}"
                filename = f'{station_channel}_{time_string}.mseed'
                file_path = os.path.join(target_directory, filename)
                
                # Check log data for existing status
                if station_channel in log_data:
                    status = log_data[station_channel]
                    if status == "downloaded":
                        st = read(file_path)
                        conv_step = True
                    elif status == "unknown":
                        #print(f"Retrying download for {station_channel}")
                        try:
                            st = client.get_waveforms(
                                network.code,
                                station.code,
                                "*",
                                chan_to_get,
                                starttime=start,
                                endtime=end,
                                attach_response=False
                            )
                            #deconcolve with instrument response and store
                            st_prec, conv_step = preproc_single(st, inventory)
                            st_prec.write(file_path, format="MSEED")
                            st = st_prec
                            logfile.write(f"{station_channel},downloaded\n")
                        except Exception as e:
                            logfile.write(f"{station_channel},unknown\n")
                            continue
                    else:
                        continue
                else:
                    #if it does not exist, download it from the client
                    try:
                        st = client.get_waveforms(
                            network.code,
                            station.code,
                            "*",
                            chan_to_get,
                            starttime=start,
                            endtime=end,
                            attach_response=False
                        )
                        #deconcolve with instrument response and store
                        st_prec, conv_step = preproc_single(st, inventory)
                        st_prec.write(file_path, format="MSEED")
                        st = st_prec
                        logfile.write(f"{station_channel},downloaded\n")
                    except Exception as e:
                        if "204" or "404" in str(e):
                            logfile.write(f"{station_channel},204/404\n")
                        elif "403" in str(e):
                            logfile.write(f"{station_channel},403\n")
                        elif "URLError" in str(e):
                            logfile.write(f"{station_channel},URLError\n")
                        else:
                            logfile.write(f"{station_channel},unknown\n")
                        continue
                
                #if data is available, calculate expected arrival times of Pn and Sn, and store 
                if conv_step:
                    t_Pn = get_Pn_time(epi_dist_deg)
                    t_Sn = get_Sn_time(epi_dist_deg)
                    t_Pg = get_Pg_time(epi_dist_deg)
                    stations_target.append([network.code, station.code, station.latitude, station.longitude, 
                                            station.elevation, epi_dist, az, t_Pn, t_Sn,t_Pg])
                    stat_count += 1

                    #handle cases where st_final_target is None
                    if st_final_target is None:
                        st_final_target = st.copy()
                    else:
                        st_final_target += st
    #if no data is available at all, return None Stream
    if st_final_target is None:
        from obspy import Stream
        st_final_target = Stream()

    logfile.close()
    pbar.close()
    return st_final_target, stations_target

def big_downloader2(datacenters, start, end, eq_lon, eq_lat, distmin, distmax, directory='/home/schreinl/Stage/Data/', plot=False):
    """
    -input: list of datacenters type str, start and end time of the time window, minimum and maximum distance in degrees
    -function that downloads data from multiple datacenters
    -filters the stations based on their location
    -checks if data is already downloaded and reads it from disk if it is
    -possibility to plot the filtered stations
    -gives back stream with all the data {st_final_target}, as well as a list of stations and their metadata,
      including expected arrival times of Pn and Sn {stations_target}
    """

    minlat_st = 40.
    maxlat_st = 52
    minlon_st = -5            
    maxlon_st = 16

    # Write the earthquake info in a file
    events_list = Client("USGS").get_events(
        minlatitude=37,
        maxlatitude=50,
        minlongitude=-5,
        maxlongitude=20,
        minmagnitude=3,
        starttime=start,
        endtime=end
    )

    eqo = events_list[0].origins[0]
    eq_start = eqo.time
    eq_mag = events_list[0].magnitudes[0].mag
    t_duration = 500.
    eq_end = eq_start + t_duration
    eq_lat = eqo.latitude
    eq_lon = eqo.longitude

    # Define output file
    time_string = UTCDateTime.strftime(start, format="%Y_%m_%dT%H_%M_%S")
    output_file = f"/home/schreinl/Stage/Data/Metadata/{time_string}.txt"

    with open(output_file, "w") as f:
        f.write(f"Start Time: {eq_start}\n")
        f.write(f"End Time: {eq_end}\n")
        f.write(f"Latitude: {eq_lat}\n")
        f.write(f"Longitude: {eq_lon}\n")
        f.write(f"Magnitude: {eq_mag}\n")

    print(f"Earthquake at {eq_start} with magnitude {eq_mag}")

    st_final_target = None
    st_final_stations = []

    for i, datacenter in enumerate(datacenters):
        client = Client(datacenter)

        try:
            inventory = client.get_stations(
                network="*", station="*", channel="HHZ,BHZ,SHZ",
                location="*", starttime=start, endtime=end,
                minlatitude=minlat_st, maxlatitude=maxlat_st,
                minlongitude=minlon_st, maxlongitude=maxlon_st,
                level="response"
            )
        except Exception as e:
            if "no data" in str(e).lower():
                print(f"Warning: No data available for datacenter {datacenter}. Skipping...")
                continue
            else:
                raise  # Re-raise unexpected errors

        # Get data for the datacenter
        st_final_center, stations_center = get_data2(client, inventory, start, end, eq_lon, eq_lat, distmin, distmax, directory, datacenter)

        if st_final_target is None:
            st_final_target = st_final_center
        else:
            st_final_target += st_final_center

        st_final_stations += stations_center

    # Plotting option
    if plot:
        filtered_stations = folium.Map(location=[eq_lat, eq_lon], zoom_start=5)

        for station in st_final_stations:
            folium.RegularPolygonMarker(
                location=[station[2], station[3]],
                tooltip=f"{station[0]}.{station[1]}",
                color="red",
                fill=True,
                number_of_sides=3,
                radius=3,
                rotation=30,
            ).add_to(filtered_stations)
        return st_final_target, st_final_stations, filtered_stations

    return st_final_target, st_final_stations, None




def preproc_single(st,inv) :
    st_work=st.copy()
    st_work=st_work.detrend("linear")
    st_work.taper(max_percentage=0.02)
    conv_step=True
    try :
        st_work.remove_response(output="VEL", water_level=10, inventory=inv)   
    except Exception as e:
        conv_step=False
        print(st,conv_step)
    return st_work, conv_step



def get_Pn_time(dist_deg) :


    from obspy.taup import TauPyModel

    model = TauPyModel(model='ak135') #crust at 35 (?) , but only 3s difference with 11km crust 
    t_Pn=111.*dist_deg/8.  #default value
    
    arrivals = model.get_travel_times(source_depth_in_km=0,
                                  distance_in_degree=dist_deg,phase_list=["Pn"])


    try:
        t_Pn=arrivals [0].time +25
    except Exception as e:
        print('no Pn ', dist_deg, e)
        

    return t_Pn


def get_Sn_time(dist_deg) :

    from obspy.taup import TauPyModel

    model = TauPyModel(model='ak135') #crust at 35 (?) , but only 3s difference with 11km crust 
    t_Sn=111.*dist_deg/8.  #default value
    
    arrivals = model.get_travel_times(source_depth_in_km=0,
                                  distance_in_degree=dist_deg,phase_list=["Sn"])


    try:
        t_Sn=arrivals [0].time + 25
    except Exception as e:
        print('no Sn ', dist_deg, e)

    return t_Sn



def get_Pg_time(dist_deg) :
    t_Pg = 111.*dist_deg/5.7 +25
    #from obspy.taup import TauPyModel

    #model = TauPyModel(model='ak135') 
    #t_Pg=111.*dist_deg/6. + 25
   # 
    #arrivals = model.get_travel_times(source_depth_in_km=0,
    #                              distance_in_degree=dist_deg,phase_list=["Pg"])


    #try:
    #    t_Pg=arrivals [0].time +25
    #except Exception as e:
    ##    t_Pg = 111.*dist_deg/5.7 +25
        #print('no Pg ', dist_deg, e)

    return t_Pg





def select_ratio(wavecode, stations_with_amps):
    '''
    this only works when the stations_with amps file is in this format:
    net (0), sta (1), lat (2), lon (3), elev (4) , dist(5), az(6), 
    t_Pn (7), t_Sn (8), t_Pg(9), A_Pn (10), A_Sn (11), A_Lg(12), A_Coda(13), A_Noise(14), A_pg(15)

    '''
    distDraw=stations_with_amps[:,5].astype(float)/1000.
    azDraw=stations_with_amps[:,6].astype(float) 

    if wavecode == 'Pn' :
        Amp_Draw=stations_with_amps[:,10].astype(float)
    elif wavecode == 'Sn' :
        Amp_Draw=stations_with_amps[:,11].astype(float)
    elif wavecode == 'Lg' :
        Amp_Draw=stations_with_amps[:,12].astype(float)  
    elif wavecode == 'Coda' :
        Amp_Draw=stations_with_amps[:,13].astype(float)    
    elif wavecode == 'Noise' :
        Amp_Draw=stations_with_amps[:,14].astype(float) 
    elif wavecode == 'Pg' :
        Amp_Draw=stations_with_amps[:,15].astype(float)
    elif wavecode == 'Lg_Coda' :
        Amp_Draw=np.divide(stations_with_amps[:,12].astype(float),stations_with_amps[:,13].astype(float))
    elif wavecode == 'Lg_Pn' :
        Amp_Draw=np.divide(stations_with_amps[:,12].astype(float),stations_with_amps[:,10].astype(float))
    elif wavecode == 'Lg_Pg' :
        Amp_Draw=np.divide(stations_with_amps[:,12].astype(float),stations_with_amps[:,15].astype(float))
    elif wavecode == 'Lg_Sn' :
        Amp_Draw=np.divide(stations_with_amps[:,12].astype(float),stations_with_amps[:,11].astype(float))
    elif wavecode == 'Pn_Coda' :
        Amp_Draw=np.divide(stations_with_amps[:,10].astype(float),stations_with_amps[:,13].astype(float))
    elif wavecode == 'Sn_Coda' :
        Amp_Draw=np.divide(stations_with_amps[:,11].astype(float),stations_with_amps[:,13].astype(float))
    elif wavecode == 'Lg_Noise' :
        Amp_Draw=np.divide(stations_with_amps[:,12].astype(float),stations_with_amps[:,14].astype(float))
    elif wavecode == 'Coda_Noise' :
        Amp_Draw=np.divide(stations_with_amps[:,13].astype(float),stations_with_amps[:,14].astype(float))
    else:
        Amp_Draw = np.zeros(stations_with_amps.shape[0])
        print('wavecode not recognized')
    Amp_Draw[np.isnan(Amp_Draw)] = 0    
    Amp_Draw[np.isinf(Amp_Draw)] = 0
    return Amp_Draw






def select_ratio_dict(wavecode, stations_with_amps):
    '''
    this only works when the stations_with amps file is in this format:
    net (0), sta (1), lat (2), lon (3), elev (4) , dist(5), az(6), 
    t_Pn (7), t_Sn (8), t_Pg(9), A_Pn (10), A_Sn (11), A_Lg(12), A_Coda(13), A_Noise(14), A_pg(15)

    '''
    distDraw=stations_with_amps[:,5].astype(float)/1000.
    azDraw=stations_with_amps[:,6].astype(float) 
    stationname = stations_with_amps[:,1]

    if wavecode == 'Pn' :
        Amp_Draw=stations_with_amps[:,10].astype(float)
    elif wavecode == 'Sn' :
        Amp_Draw=stations_with_amps[:,11].astype(float)
    elif wavecode == 'Lg' :
        Amp_Draw=stations_with_amps[:,12].astype(float)  
    elif wavecode == 'Coda' :
        Amp_Draw=stations_with_amps[:,13].astype(float)    
    elif wavecode == 'Noise' :
        Amp_Draw=stations_with_amps[:,14].astype(float) 
    elif wavecode == 'Pg' :
        Amp_Draw=stations_with_amps[:,15].astype(float)
    elif wavecode== 'Pg_Coda':
        Amp_Draw=np.divide(stations_with_amps[:,15].astype(float),stations_with_amps[:,13].astype(float))
    elif wavecode == 'Lg_Coda' :
        Amp_Draw=np.divide(stations_with_amps[:,12].astype(float),stations_with_amps[:,13].astype(float))
    elif wavecode == 'Lg_Pn' :
        Amp_Draw=np.divide(stations_with_amps[:,12].astype(float),stations_with_amps[:,10].astype(float))
    elif wavecode == 'Lg_Pg' :
        Amp_Draw=np.divide(stations_with_amps[:,12].astype(float),stations_with_amps[:,15].astype(float))
    elif wavecode == 'Lg_Sn' :
        Amp_Draw=np.divide(stations_with_amps[:,12].astype(float),stations_with_amps[:,11].astype(float))
    elif wavecode == 'Pn_Coda' :
        Amp_Draw=np.divide(stations_with_amps[:,10].astype(float),stations_with_amps[:,13].astype(float))
    elif wavecode == 'Sn_Coda' :
        Amp_Draw=np.divide(stations_with_amps[:,11].astype(float),stations_with_amps[:,13].astype(float))
    elif wavecode == 'Lg_Noise' :
        Amp_Draw=np.divide(stations_with_amps[:,12].astype(float),stations_with_amps[:,14].astype(float))
    elif wavecode == 'Coda_Noise' :
        Amp_Draw=np.divide(stations_with_amps[:,13].astype(float),stations_with_amps[:,14].astype(float))

    else:
        Amp_Draw = np.zeros(stations_with_amps.shape[0])
        print('wavecode not recognized')
    Amp_Draw[np.isnan(Amp_Draw)] = 0    
    Amp_Draw[np.isinf(Amp_Draw)] = 0
    station_amp_dict =dict(zip(stationname, Amp_Draw))
    return station_amp_dict


def SNR_all(stations, st, Dtmin_Pn, Dtmax_Pn, Dtmin_Sn, Dtmax_Sn, vLg_min,vLg_max,vPg_min, vPg_max, tmin_Coda, tmax_Coda,
         Dtmin_Noise, Dtmax_Noise,eq_start,eq_lat,eq_lon,snr_threshold=2,plot_SNR=False,plot_amps=False,wavecode="Lg_Pg",dB=False,codawindow="cutoff", factor=1.3):
    """
stations: list with stations as processed with bid_downloader
st: data stream
other variables are float
"""
    signal_windows = ['Pn', 'Pg', 'Sn', 'Lg']
    #initialize table, which has dims (MxN), M amount of stations N amount of phases for which the SNR is computed
    snrs = np.zeros((len(stations),len(signal_windows)))
    for j, window in enumerate(signal_windows):
        print(f'calculating SNR for {window}  phase')
        
    #calculating the SNR for a specific phase
        tsmax = 0
        distmax = 0
            
        for k, station in enumerate(stations):
            net, sta, lat, lon, elev, dist, az, t_Pn, t_Sn, t_Pg = station
            A_Noise=0.
            A_Pn=0.
            A_Sn=0.
            A_Lg=0.
            A_Coda=0.
            A_LgAP=0.
            A_LgACoda=0.
            #tmin_Noise=t_Pn+Dtmin_Noise
            #tmax_Noise=t_Pn+Dtmax_Noise
            
            tmin_Pn=t_Pn+Dtmin_Pn
            tmax_Pn=t_Pn+Dtmax_Pn
            tmin_Sn=t_Sn+Dtmin_Sn
            tmax_Sn=t_Sn+Dtmax_Sn
            

            for tr in st :
                if tr.stats.network == net and tr.stats.station == sta:
                    tminLg=dist/1000/vLg_max
                    tmaxLg=dist/1000/vLg_min
                    tmin_Pg=dist/1000/vPg_max
                    tmax_Pg=dist/1000/vPg_min
                    trace_start=tr.stats.starttime - eq_start
                    #the noise window is not related to any phase, but relative to the start if of the trace
                    tmin_Noise = trace_start +5
                    tmax_Noise = tmin_Noise + 30
                    dt=tr.stats.delta
                    nt=tr.stats.npts
                    trace_end=trace_start+dt*(nt-1)
                    tvector=np.arange(trace_start,trace_end+dt,dt)
                    datavector=tr.data
                

                    if window == 'Lg':
                        iminLg=int((tminLg-trace_start)/dt)
                        imaxLg=int((tmaxLg-trace_start)/dt)
                        iminNoise=int((tmin_Noise-trace_start)/dt)
                        imaxNoise=int((tmax_Noise-trace_start)/dt)
                        dataselectLg=(datavector[iminLg:imaxLg])
                        dataselectNoise=(datavector[iminNoise:imaxNoise])                        
                        signal_power = np.sqrt(np.dot(dataselectLg,np.transpose(dataselectLg)))/len(dataselectLg) 
                        noise_power = np.sqrt(np.dot(dataselectNoise,np.transpose(dataselectNoise)))/len(dataselectNoise)
                        if noise_power == 0 or (signal_power/noise_power) == 0:
                            snr = 0
                        if dB:
                            snr = 10 * np.log10(signal_power / noise_power)
                        else:
                            snr = signal_power / noise_power

                    if window == 'Pn':
                        iminPn=int((tmin_Pn-trace_start)/dt)
                        imaxPn=int((tmax_Pn-trace_start)/dt)
                        iminNoise=int((tmin_Noise-trace_start)/dt)
                        imaxNoise=int((tmax_Noise-trace_start)/dt)
                        dataselectPn=(datavector[iminPn:imaxPn])
                        dataselectNoise=(datavector[iminNoise:imaxNoise])                        
                        signal_power = np.sqrt(np.dot(dataselectPn,np.transpose(dataselectPn)))/len(dataselectPn) 
                        noise_power = np.sqrt(np.dot(dataselectNoise,np.transpose(dataselectNoise)))/len(dataselectNoise)
                        if noise_power == 0 or (signal_power/noise_power) == 0:
                            snr = 0
                        if dB:
                            snr = 10 * np.log10(signal_power / noise_power)
                        else:
                            snr = signal_power / noise_power
                    
                    if window == 'Sn':
                        iminSn=int((tmin_Sn-trace_start)/dt)
                        imaxSn=int((tmax_Sn-trace_start)/dt)
                        iminNoise=int((tmin_Noise-trace_start)/dt)
                        imaxNoise=int((tmax_Noise-trace_start)/dt)
                        dataselectSn=(datavector[iminSn:imaxSn])
                        dataselectNoise=(datavector[iminNoise:imaxNoise])
                        signal_power = np.sqrt(np.dot(dataselectSn,np.transpose(dataselectSn)))/len(dataselectSn) 
                        noise_power = np.sqrt(np.dot(dataselectNoise,np.transpose(dataselectNoise)))/len(dataselectNoise)
                        if noise_power == 0 or (signal_power/noise_power) == 0:
                            snr = 0
                        if dB:
                            snr = 10 * np.log10(signal_power / noise_power)
                        else:
                            snr = signal_power / noise_power

                    if window == 'Pg':
                        iminPg=int((tmin_Pg-trace_start)/dt)
                        imaxPg=int((tmax_Pg-trace_start)/dt)
                        iminNoise=int((tmin_Noise-trace_start)/dt)
                        imaxNoise=int((tmax_Noise-trace_start)/dt)
                        dataselectPg=(datavector[iminPg:imaxPg])
                        dataselectNoise=(datavector[iminNoise:imaxNoise])
                        signal_power = np.sqrt(np.dot(dataselectPg,np.transpose(dataselectPg)))/len(dataselectPg) 
                        noise_power = np.sqrt(np.dot(dataselectNoise,np.transpose(dataselectNoise)))/len(dataselectNoise)
                        if noise_power == 0 or (signal_power/noise_power) == 0:
                            snr = 0
                        if dB:
                            snr = 10 * np.log10(signal_power / noise_power)
                        else:
                            snr = signal_power / noise_power
            #write value of snr with phase at index j and station at index k in the initialized table            
            snrs[k,j] = snr
    #now station with SNR has all the information of the stations (net, sta, lat, lon, elev, dist, az, t_Pn, t_Sn, t_Pg), as well 
    #as the SNR ratios of the given phases (signal_windows) as subsequent columns 'Pn', 'Pg', 'Sn', 'Lg'
    stations_with_SNR=np.append(np.array(stations),np.array(snrs),axis=1)



    
    #plotting possibility, here all phases are plotted
    if plot_SNR==True:
        for l, window in enumerate(signal_windows):
            Amp_Draw[np.isnan(Amp_Draw)] = 0    
            Amp_Draw[np.isinf(Amp_Draw)] = 0
            plotit =plot_stations_amps(stations_with_SNR, 1, 0.7, Amp_Draw, origin=[eq_lat,eq_lon], zoom=5, forcescale=False)
            plotit

    #now find the cutoff distances for all the phases
    phase_distance = {}
    for w, window in enumerate(signal_windows):
        SNR_vals = stations_with_SNR[:, 10+w].astype(float)
        dist_vals = stations_with_SNR[:, 5].astype(float) / 1000. 
        #SNR_vals = SNR_vals[np.isfinite(SNR_vals)]
        #dist_vals = dist_vals[np.isfinite(SNR_vals)]
        snr_threshold = snr_threshold
        filtered_distances = dist_vals[SNR_vals > snr_threshold]

        if len(filtered_distances) > 0:
            percentile_distance = np.percentile(filtered_distances, 90)
            phase_distance[window] = percentile_distance
            #print(f"Distance where 90% of SNR values are above 2: {percentile_distance}")
        else:
            print("No valid SNR values above 2.")
            return None,None,None,None,None,None,None
        #calculate the slope for for the regression snr = a*dist + b, when dist < percentile_distance, so sufficient SNR
        coef = np.polyfit(dist_vals[dist_vals < percentile_distance],np.nan_to_num(SNR_vals[dist_vals < percentile_distance], nan=0.0, posinf=0.0, neginf=0.0),1)
        #calculate the slope for the regression snr = a*dist + b, when dist > percentile_distance, so insufficient SNR
        coef1 = np.polyfit(dist_vals[dist_vals > percentile_distance],np.nan_to_num(SNR_vals[dist_vals > percentile_distance], nan=0.0, posinf=0.0, neginf=0.0),1)
        phase_distance[window] = {
            'percentile_distance': percentile_distance,
            'coef_quad': coef[0],
            'coef1': coef1[0]
        }
    #now find the average cutoff distance
    dist_mean = (phase_distance['Pg']['percentile_distance'] + phase_distance['Pn']['percentile_distance'] + phase_distance['Sn']['percentile_distance'])/3    
    dist_Lg = phase_distance['Lg']['percentile_distance']
    #filter the stations_with_SNR, based upon their distance, if it is larger than dist_mean, the row is deleted
    #and collect the station names of the dropped rows

    rows_to_drop_dist = stations_with_SNR[stations_with_SNR[:, 5].astype(float) / 1000. > dist_Lg]
    dropped_values_dist = rows_to_drop_dist[:, 1].tolist()
    filtered_arr = stations_with_SNR[stations_with_SNR[:, 5].astype(float) / 1000. <= dist_Lg]
    #we also filter out the rows, where the mean of the SNR of the Pn, Sn and the Pg is below 2
    mask2 = np.mean(filtered_arr[:, 10:13].astype(float), axis=1) >= 2
    rows_to_drop = filtered_arr[~mask2]    
    dropped_values = rows_to_drop[:, 1].tolist()
    dropped_list = dropped_values_dist + dropped_values
    filtered_arr = filtered_arr[mask2]
    print("Reduced from  ", len(stations_with_SNR), " stations to  ", len(filtered_arr), " stations due to insufficient SNR or distance > " ,  dist_Lg)
    
    #with the earthquake specific cutoff distance we can now set tmin_coda:
    #so here we base the cutoff distance only on the Lg wave
    if codawindow == "cutoff":
        tmin_Coda = factor * (dist_Lg/3)
        tmax_Coda = tmin_Coda + 100
        print(f"coda window set from {tmin_Coda}-{tmax_Coda}s based on Lg cutoff distance")
        phase_distance['tmin_Coda'] = tmin_Coda
    elif codawindow == "S_phase":
        tsmax = 0
        distmax = 0
        for station,net, lat, lon, elev, dist, az, t_Pn, t_Sn, t_Pg, Pn_snr, Pg_snr, Sn_snr, Lg_snr in filtered_arr:
            if float(dist) >= distmax:
                distmax = float(dist)
                tsmax = float(t_Sn)
        tmin_Coda = factor * tsmax
        tmax_Coda = tmin_Coda + 100
        print(f"coda window set from {tmin_Coda}-{tmax_Coda}s based on S wave arrival at dist {distmax} and S time {tsmax}")

    #using this information we can calculate now all the amplitudes:
    #keep only station names that satisfy the two conditions, and delete all the traces of the unsufficient stations
    filtered_stations = filtered_arr[:,:10]
    st1 = st.copy()
    for tr in st1:
        if tr.stats.station in dropped_list:
            st1.remove(tr)

    #station_with_amps now also has the following structure in columns:
    #(net, sta, lat, lon, elev, dist, az, t_Pn, t_Sn, t_Pg, A_Pn, A_Sn, A_Lg, A_Coda, A_Noise, A_Pg         
    stations_with_amps = calc_amps(filtered_stations,st1, Dtmin_Pn, Dtmax_Pn, Dtmin_Sn, Dtmax_Sn, vLg_min,vLg_max,vPg_min,
                                   vPg_max, tmin_Coda, tmax_Coda, Dtmin_Noise, Dtmax_Noise,eq_start)

    
    
    # Add the mean cutoff distance as a new column to stations_with_amps
    #dist_mean_column = np.full((stations_with_amps.shape[0], 1), dist_mean)
    #stations_with_amps = np.hstack((stations_with_amps, dist_mean_column))

    if plot_amps:
        Amp_Draw = select_ratio(wavecode, stations_with_amps)
        print(f"plotting {wavecode} amplitudes")
        amp_plot = plot_stations_amps(stations_with_amps, 1, 0.7, Amp_Draw, origin=[eq_lat, eq_lon], zoom=5, forcescale=False)
    amp_plot   




    return filtered_arr,stations_with_SNR, phase_distance, tmin_Coda, st, stations_with_amps,  amp_plot
    












def calc_amps(stations, st, Dtmin_Pn, Dtmax_Pn, Dtmin_Sn, Dtmax_Sn, vLg_min,vLg_max,vPg_min,vPg_max, tmin_Coda, tmax_Coda, Dtmin_Noise, Dtmax_Noise, eq_start):


    stations_amplitudes=[]
    stations = np.array(stations) 
    #  [:,:10]
    for net, sta, lat, lon, elev , dist, az, t_Pn, t_Sn, t_Pg  in stations[:,:10]:
        A_Noise=0.
        A_Pn=0.
        A_Sn=0.
        A_Pg=0
        A_Lg=0.
        A_Coda=0.
        A_LgAP=0.
        A_LgACoda=0.
        #tmin_Noise=float(t_Pn)+Dtmin_Noise
        #tmax_Noise=float(t_Pn)+Dtmax_Noise
        tmin_Pn=float(t_Pn)+Dtmin_Pn
        tmax_Pn=float(t_Pn)+Dtmax_Pn
        tmin_Sn=float(t_Sn)+Dtmin_Sn
        tmax_Sn=float(t_Sn)+Dtmax_Sn

        for tr in st :
            if tr.stats.network == net and tr.stats.station == sta:
                tminLg=float(dist)/1000/vLg_max
                tmaxLg=float(dist)/1000/vLg_min 
                tminPg = float(dist)/1000/vPg_max
                tmaxPg = float(dist)/1000/vPg_min
                trace_start=tr.stats.starttime - eq_start
                tmin_Noise = trace_start + 5
                tmax_Noise = tmin_Noise + 30
                
                #print(tmax_Noise)
                dt=tr.stats.delta
                nt=tr.stats.npts
                trace_end=trace_start+dt*(nt-1)
                tvector=np.arange(trace_start,trace_end+dt,dt)
                datavector=tr.data
            
                if (trace_start<tmin_Pn) and (trace_end>tmax_Pn) :
                    iminPn=int((tmin_Pn-trace_start)/dt)
                    imaxPn=int((tmax_Pn-trace_start)/dt)
                    dataselectPn=(datavector[iminPn:imaxPn])
                    A_Pn=np.sqrt(np.dot(dataselectPn,np.transpose(dataselectPn)))/len(dataselectPn)
                if (trace_start<tmin_Sn) and (trace_end>tmax_Sn) :
                    iminSn=int((tmin_Sn-trace_start)/dt)
                    imaxSn=int((tmax_Sn-trace_start)/dt)
                    dataselectSn=(datavector[iminSn:imaxSn])
                    A_Sn=np.sqrt(np.dot(dataselectSn,np.transpose(dataselectSn)))/len(dataselectSn)
                if (trace_start<tminLg) and (trace_end>tmaxLg) :
                    iminLg=int((tminLg-trace_start)/dt)
                    imaxLg=int((tmaxLg-trace_start)/dt)
                    dataselectLg=(datavector[iminLg:imaxLg])
                    A_Lg=np.sqrt(np.dot(dataselectLg,np.transpose(dataselectLg)))/len(dataselectLg)
                if (trace_start<tmin_Coda) and (trace_end>tmax_Coda) :
                    iminCoda=int((tmin_Coda-trace_start)/dt)
                    imaxCoda=int((tmax_Coda-trace_start)/dt)
                    dataselectCoda=(datavector[iminCoda:imaxCoda])
                    A_Coda=np.sqrt(np.dot(dataselectCoda,np.transpose(dataselectCoda)))/len(dataselectCoda)
                if (trace_start<tmin_Noise) and (trace_end>tmax_Noise) :
                    iminNoise=int((tmin_Noise-trace_start)/dt)
                    imaxNoise=int((tmax_Noise-trace_start)/dt)
                    dataselectNoise=(datavector[iminNoise:imaxNoise])
                    A_Noise=np.sqrt(np.dot(dataselectNoise,np.transpose(dataselectNoise)))/len(dataselectNoise)
                    #print(A_Noise)
                if (trace_start<tminPg) and (trace_end>tmaxPg) :
                    iminPg=int((tminPg-trace_start)/dt)
                    imaxPg=int((tmaxPg-trace_start)/dt)
                    dataselectPg=(datavector[iminPg:imaxPg])
                    A_Pg=np.sqrt(np.dot(dataselectPg,np.transpose(dataselectPg)))/len(dataselectPg)

    
        stations_amplitudes.append([A_Pn, A_Sn, A_Lg, A_Coda, A_Noise, A_Pg])


    stations_with_amps=np.append(np.array(stations[:,:10]),np.array(stations_amplitudes),axis=1)

    return stations_with_amps







def envelopes_routine1(event, st_envelope, codastart=350, codaend=470, method='cutoff',  coda_dist_start=300, coda_dist_end=350, plotting=False, n_traces=50, snr=None, snr_window=None):
    '''
    Takes as input a stream with the envelopes. It subsequently calculates the smoothed envelopes with a moving averaging
    window of 50s. These smooth envelopes are put in a new stream.
    '''
    st_envelope_smooth = obspy.Stream()
    
    for trace in st_envelope:
        npts = len(trace.data)
        samprate = trace.stats.sampling_rate
        t = np.arange(0, npts / samprate, 1 / samprate)

        window_length = min(50 * samprate, npts) 
        if window_length % 2 == 0:
            window_length -= 1
        
        yhat = savgol_filter(trace.data, int(window_length), 3) 
        t = t[:len(yhat)]

        tr_envelope_smooth = obspy.Trace(data=yhat, header=trace.stats)
        st_envelope_smooth.append(tr_envelope_smooth)

    if plotting:
        smooth_plot_envelope(event, n_traces, st_envelope_smooth)

    station_data = {}
    
    #Calculate the amplitude of the coda and the slope of the coda and the distance coda
    #but also, calculate the noise level, and the snr of the coda window in its entirety as well as the snr of the last 20 seconds of the coda window

    for trace in st_envelope_smooth:
        dt = trace.stats.delta
        station = trace.stats.station
        startcoda = int(codastart / dt) 
        #we have the noise in the stations_with_amplitudes for each event, so read in the noise amplitude and calculate the SNRs
        
        
        endcoda = int(codaend / dt) 
        startcoda_dist = int(coda_dist_start / dt)
        endcoda_dist = int(coda_dist_end / dt)



        
        if station in snr:
            snr_coda = snr[station]
            snr_coda_end = snr_window[station]
            

        if startcoda >= len(trace.data) or endcoda > len(trace.data) or startcoda >= endcoda:
            continue

        coda = trace.data[startcoda:endcoda]
        coda_dist = trace.data[startcoda_dist:endcoda_dist]

        if len(coda) < 2 or len(coda_dist) < 2:
            continue  

        x_coda = np.linspace(0, (len(coda) - 1) * dt, len(coda))
        x_coda_dist = np.linspace(0, (len(coda_dist) - 1) * dt, len(coda_dist))

        coef_coda = np.polyfit(x_coda, coda, 1) if not np.all(coda == coda[0]) else [0, 0]
        coef_coda_dist = np.polyfit(x_coda_dist, coda_dist, 1) if not np.all(coda_dist == coda_dist[0]) else [0, 0]

        station_name = trace.stats.station 
        if method == 'cutoff':
            amplitude = (np.sqrt(np.dot(coda_dist, coda_dist.T)) / len(coda_dist))
        elif method == 'S_phase':
            amplitude = (np.sqrt(np.dot(coda, coda.T)) / len(coda))
            
        if snr:
            station_data[station_name] = {
            "amplitude": amplitude,
            "coda_slope": coef_coda[0],
            "coda_dist_slope": coef_coda_dist[0],
            "snr_coda": snr_coda,
            "snr_coda_end": snr_coda_end
        }
        else:
            station_data[station_name] = {
            "amplitude": amplitude,
            "coda_slope": coef_coda[0],
            "coda_dist_slope": coef_coda_dist[0]
        }
            

    return station_data, st_envelope_smooth




def create_coda_amplitude_dict(event_file='/home/schreinl/Stage/Data/eq_4_france.csv', codawindow="cutoff", factor=1.1, fmin=3, fmax=4, data_dir='Data1',envelope_name='envelope_amps'):
    eq_list = pd.read_csv(event_file)
    amplitudes_dict = {}

    for i in range(len(eq_list)):
        start = UTCDateTime(eq_list["time"][i]) - 25
        time_string = UTCDateTime.strftime(start, format="%Y_%m_%dT%H_%M_%S")
        magnitude = eq_list["mag"][i]
        latitude = eq_list["latitude"][i]
        longitude = eq_list["longitude"][i]
        
        file_path = f"/home/schreinl/Stage/{data_dir}/{time_string}/{time_string}_{envelope_name}_{codawindow}_fac_{factor}_{fmin}_{fmax}Hz_dict.txt"
        #print(f"Checking file: {file_path}")
        
        if not os.path.exists(file_path):
            #print(f"File does not exist: {file_path}")
            continue
        
        try:
            with open(file_path, "r") as file:
                amp_dict = json.load(file)
            for station, data in amp_dict.items():
                if station not in amplitudes_dict:
                    amplitudes_dict[station] = [None] * i 
                amplitudes_dict[station].append({
                    "amplitude": data["amplitude"], 
                    "time": time_string, 
                    "magnitude": magnitude, 
                    "latitude": latitude,
                    "longitude": longitude,
                    "snr_coda": data.get("snr_coda", None), 
                    "snr_last_window": data.get("snr_coda_end", None)
                })
            
            for station in amplitudes_dict:
                if station not in amp_dict:
                    amplitudes_dict[station].append(None)
        except FileNotFoundError:
            #print(f"File not found: {file_path}")
            for station in amplitudes_dict:
                amplitudes_dict[station].append(None)

    return amplitudes_dict




def reference_stations_median(reference_stations,eventfile, fmin, fmax,envelope_name='envelope_amps'):
    '''
    calculates the ratios of the amplitudes within the coda window for each reference station and the main
    reference station ECH, and takes the median of it

    takes:
    - reference stations as a list of str
    - eventfile as a str of the path of the event catalog
    - fmin and fmax which are the corners of the applied filter as floats
    returns:
    - dictionary, the median of each ratio reference station/main reference station
    '''

    ratios_dict = {}    
    amplitudes_dict = create_coda_amplitude_dict(event_file=eventfile, codawindow="cutoff",fmin=fmin,fmax=fmax, factor=1.1,data_dir='Data1',envelope_name=envelope_name)



    for station in reference_stations:
        if station == "ECH":
            continue
        
        ech_amplitudes = [amp for amp in amplitudes_dict["ECH"] if amp is not None and amp['snr_coda'] > 2 and amp['snr_last_window'] > 2]
        ref_amplitudes = [amp for amp in amplitudes_dict[station] if amp is not None and amp['snr_coda'] > 2 and amp['snr_last_window'] > 2]
        
        ech_times = set(amp['time'] for amp in ech_amplitudes)
        ref_times = set(amp['time'] for amp in ref_amplitudes)
        
        common_times = ech_times.intersection(ref_times)
        
        ratios = []
        for time in common_times:
            ech_amp = next(amp['amplitude'] for amp in ech_amplitudes if amp['time'] == time)
            ref_amp = next(amp['amplitude'] for amp in ref_amplitudes if amp['time'] == time)
            ratio = ech_amp / ref_amp
            ratios.append(ratio)
        
        ratios_dict[f"{station}"] = ratios

    ref_median_dict = {key: np.median(value) for key, value in ratios_dict.items() if value}

    return ref_median_dict





def site_effect_dict(eventfile, fmin,fmax,reference_stations,envelope_name='envelope_amps'):
    '''
    Function that finally calculates the site effects, for each station and event.
    If for an event ECH was recording, this is used for reference. If not, the closest of the other reference stations
    in case it was recording should be used. This is done for each possible station-event combination. For each ratio that
    is built, we write in the dict the ratio, as well as info on the event and the used reference station.

    Input:
    - eventfile, str of path of event catalog
    - fmin and fmax, corners of the applied filter
    - reference_stations, list of strings with the names of the reference stations

    Returns:
    - dictionary of site effects with its belonging info
    '''


    site_effect = {}
    amplitudes_dict = create_coda_amplitude_dict(event_file=eventfile, codawindow="cutoff",fmin=fmin,fmax=fmax, factor=1.1,data_dir='Data1',envelope_name=envelope_name)
    ref_median_dict = reference_stations_median(reference_stations,eventfile,fmin,fmax)
    num_events = len(next(iter(amplitudes_dict.values())))

    for event in range(num_events):
        ech_amp = None
        reference_station = None

        if "ECH" in amplitudes_dict and event < len(amplitudes_dict["ECH"]):
            ech_entry = amplitudes_dict["ECH"][event]
            if ech_entry and ech_entry.get("snr_coda", 0) > 2:
                ech_amp = ech_entry["amplitude"]
                reference_station = "ECH"

        if ech_amp is None:
            for ref in reference_stations:
                if ref in amplitudes_dict and event < len(amplitudes_dict[ref]):
                    ref_entry = amplitudes_dict[ref][event]
                    if ref_entry and ref_entry.get("snr_coda", 0) > 2:
                        ref_amp = ref_entry["amplitude"]
                        ref_median_ratio = ref_median_dict.get(ref, None)
                        
                        if ref_median_ratio is not None:
                            ech_amp = ref_amp * ref_median_ratio 
                            reference_station = ref
                            #print(f'used reference station {reference_station}')
                        break 
        if ech_amp is None:
            continue

        for station in amplitudes_dict:
            if event < len(amplitudes_dict[station]): 
                station_entry = amplitudes_dict[station][event]
                if station_entry and station_entry.get("snr_coda", 0) > 2:
                    station_amp = station_entry["amplitude"]
                    ratio = station_amp / ech_amp 
                    magnitude = station_entry["magnitude"]
                    latitude = station_entry['latitude']
                    longitude = station_entry['longitude']

                    if station not in site_effect:
                        site_effect[station] = []
                    if ratio <= 100:
                        site_effect[station].append({
                            "ratio": ratio,
                            "time": station_entry["time"],
                            "reference_station": reference_station,
                            "magnitude": magnitude,
                            "latitude": latitude,
                            "longitude": longitude
                        })

    return site_effect





def site_effect_overall(fmin,fmax, reference_stations, frequenciesmin=None, frequenciesmax=None, method='single', eventfile='/home/schreinl/Stage/Data/big_box_4.5.csv',map_plot=False,envelope_name='envelope_amps'):
    '''
    Before the calling of this function all the data has to be downloaded, the SNR has to be calculated, 
    stations filtered out by low SNR and too large distance, and the coda window is set. Then after that, the envelopes 
    are calculated, and the slopes and amplitudes within the coda window are calculated and saved to disk in a dict form.
    After this point this function can be called, which handles all the big data and post-processig steps which
    do not require any computational power. 

    Required arguments:
    - fmin, fmax, which are the borders of the frequency windows for which we filtered
    - reference_stations, which is a list of strings, containing the names of the reference stations
    - eventfile, which contains the catalog of the events
    - method: this decides if we handle only a single frequency or multiple ones

    The workflow in this function is the following:

    - reading in from all the amplitudes in the coda time windows of all the events and stations
    - writing the amplitude along with the event info and the snr in the window in one dict for all the stations
    - then finally w
    '''
    #first handling the case when we only use a single frequency

    if method == 'single':
     # reading in a dict which contains all the amplitudes and snr for all the stations and events

        site_effect = site_effect_dict(eventfile, fmin, fmax, reference_stations, envelope_name=envelope_name)
        
        site_effect_median = {}
        for station, events in site_effect.items():
            ratios = np.array([event['ratio'] for event in events])
            
            Z = (ratios - np.mean(ratios)) / np.std(ratios)
            outlier_indices = np.where(Z > 1.3)[0]
            
            ratios_filtered = np.delete(ratios, outlier_indices)
            
            site_effect_median[station] = {
                "median_before": np.median(ratios),
                "std_before": np.std(ratios),
                "num_points_before": len(ratios),
                "median_after": np.median(ratios_filtered),
                "std_after": np.std(ratios_filtered),
                "num_points_after": len(ratios_filtered)
            }
        
        if map_plot:
            map_result = map_site_effect(fmin, fmax, site_effect_median)
            return site_effect_median, map_result
        
        return site_effect_median

    #second handling the case of multiple frequencies
    #in this case we take lists for the minimum and maximum frequency, and we get rid of outliers in this case
    # conflict: 

    elif method == 'multiple':
        site_effect_medians = {}

        for i in range(len(frequenciesmin)):    
            site_effect = site_effect_dict(eventfile,frequenciesmin[i],frequenciesmax[i], reference_stations,envelope_name=envelope_name)

        


            site_effect_median_std = {}
            for station, events in site_effect.items():
                #print(f'station: {station}, ratios: {ratios}')
                ratios = np.array([event['ratio'] for event in events])
                Z = (ratios - np.mean(ratios)) / np.std(ratios)
                outlier_indices = np.where(Z > 1.3)[0]

                ratios_filtered = np.delete(ratios, outlier_indices)

                median_before = np.median(ratios)
                std_before = np.std(ratios)
                median_after = np.median(ratios_filtered)
                std_after = np.std(ratios_filtered)

                site_effect_median_std[station] = {
                    "median_before": median_before,
                    "std_before": std_before,
                    "num_points_before": len(ratios),
                    "median_after": median_after,
                    "std_after": std_after,
                    "num_points_after": len(ratios_filtered)
                }

            site_effect_medians[f"{frequenciesmin[i]}-{frequenciesmax[i]}Hz"] = site_effect_median_std
            
            if map_plot:
                map_result = map_site_effect(fmin, fmax, site_effect_medians)
                return site_effect_medians, map_result
        






    return site_effect_medians


def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0  
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1 
    dlon = lon2 - lon1 
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    return R * 2 * np.arcsin(np.sqrt(a))





def Lg_amplitude_dict(base_directory, fmin, fmax):
    '''
    Input:
    - base_directory: str, the directory where the files are stored, mostly "/home/schreinl/Stage/Data1"
    - fmin: list of float, the minimum frequency
    - fmax: list of float, the maximum frequency

    This function reads in the Lg amplitude from all available events, and stores in a dict, where for all events the amplitude
    of all station is stored, if existent, for all the frequency ranges.
    
    Can be combined with: output_path = "..."
    with open(output_path, "w") as f:
        json.dump(Lg_dict, f, indent=4), in order to store the output file.


    '''
    events_data = {}
    
    for timestamp_dir in sorted(os.listdir(base_directory)):
        dir_path = os.path.join(base_directory, timestamp_dir)
        
        if os.path.isdir(dir_path):
            event_stations = {}
            
            for f1, f2 in zip(fmin, fmax):
                frequency_range = f"{f1}_{f2}Hz" 
                filename = f"{timestamp_dir}_{frequency_range}_5_thresh_stations_with_amps.txt"
                file_path = os.path.join(dir_path, filename)
                
                if os.path.exists(file_path):
                    try:
                        with open(file_path, "r") as f:
                            data = json.load(f)
                        
                        if isinstance(data, list):
                            for entry in data:
                                if isinstance(entry, list) and len(entry) > 12:
                                    network = entry[0]
                                    station_name = entry[1]
                                    amplitude = float(entry[12])
                                    station_key = f"{network}.{station_name}"
                                    
                                    if station_key not in event_stations:
                                        event_stations[station_key] = {}
                                    
                                    event_stations[station_key][frequency_range] = amplitude
                        else:
                            print(f"Unexpected data format in file: {file_path}")
                    except Exception as e:
                        print(f"Error reading file {file_path}: {e}")
            
            if event_stations:
                events_data[timestamp_dir] = event_stations
    
    return events_data



def Lg_amplitude_calculation(fmin, fmax, Lg_dict, savefile=False,special_site_terms=None):
    '''
    Input: 
        - fmin: float, the minimum frequency
        - fmax: float, the maximum frequency
        - Lg_dict: dict, the dictionary with the Lg amplitude for all events and stations, obtained by calling Lg_amplitude_dict
        - savefile: bool, if True, the output will be saved in a file
        - special_site_terms: dict, if None, the site terms will be read in from the file "../Data1/Site_effects_dict.txt"

    Output:
        - results: dict, with A_kl (the amplitude of each station and event) as well as the distance to the event
        - results_with_coords: dict, with the coordinates of the event and the stations, and their amplitude (A_kl)

    Method:
        - for each event:
        - reads in all available site terms for the given frequency range
        - calculates the A_kl for each station, using the formula:
           A_kl = ln(A_kl) + (5/6)*ln(r) - ln(site_term)
    '''
    
    if special_site_terms:
        site_terms = special_site_terms
    else:   
        with open("../Data1/Site_effects_dict.txt", "r") as f:
            site_terms = json.load(f)

    station_coords = {}
    with open("../Data1/all_stations_coordinates.txt", "r") as f:
        next(f)
        for line in f:
            if line.strip():
                network, station, lat, lon = line.strip().split(",")
                station_coords[f"{network}.{station}"] = (float(lat), float(lon))

    event_data = {}
    with open("../Data/big_box_4.5.csv", "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            event_time = UTCDateTime(row[0])
            event_lat = float(row[1])
            event_lon = float(row[2])
            
            event_time_str = event_time.strftime("%Y_%m_%dT%H_%M_%S")
            event_data[event_time_str] = (event_lat, event_lon)

    results = {}
    Q0_calculation_terms = {}

    for event_time, stations in Lg_dict.items():
        if event_time in ["Dicts", "Metadata"]: 
            continue
        
        formatted_event_time = (UTCDateTime(event_time) + 25).strftime("%Y_%m_%dT%H_%M_%S")
        if formatted_event_time not in event_data:
            continue
        
        event_lat, event_lon = event_data[formatted_event_time]
        event_results = {}
        tmp = {}
        for station, freq_data in stations.items():
            if station not in station_coords:
                continue
            
            lat, lon = station_coords[station]
            r = haversine(event_lat, event_lon, lat, lon)

            for freq_range, amplitude in freq_data.items():
                freq_min, freq_max = map(float, freq_range.replace("Hz", "").split("_"))
                if not (fmin <= freq_min and fmax >= freq_max):
                    continue

                site_effect = None
                for site_freq_band, site_info in site_terms.items():
                    site_freq_min, site_freq_max = map(lambda x: float(x.replace('Hz', '').strip()), site_freq_band.split('-'))
                    if freq_min <= site_freq_min <= freq_max and freq_max >= site_freq_max >= freq_min:
                        if station.split(".")[1] in site_info:
                            site_effect = site_info[station.split(".")[1]].get("median_after", None)
                            break
                
                if site_effect is None or site_effect <= 0:
                    continue 

                log_value = np.log(amplitude) + (5/6)*np.log(r) - np.log(site_effect)
                event_results[station] = {"log_value": log_value, "distance_km": r}
                tmp[station] = {'amplitude': amplitude, 'distance': r, 'site_effect': site_effect}
        
        results[formatted_event_time] = event_results
        Q0_calculation_terms[formatted_event_time] = tmp

    results_with_coords = {}
    for event_time, event_results in results.items():
        event_lat, event_lon = event_data[event_time]
        results_with_coords[event_time] = {
            "coordinates": {"latitude": event_lat, "longitude": event_lon},
            "data": event_results
        }
    if savefile:
        with open("../Data1/processed_event_amplitudes.txt", "w") as f:
            json.dump(results, f, indent=4)
        with open("../Data1/processed_event_amplitudes_coords.txt", "w") as g:
            json.dump(results_with_coords, g, indent=4)

    print("Processing complete.")
    return results, results_with_coords






from scipy.stats import linregress
def Lg_Q0_Sk(fmin, fmax, V_Lg=3.2,plot_Q0=False, plot_Sk=False, plot_Q0_map=False, plot_Sk_map=False,special_site_terms=None,std_thresh=None,give_out_results=False):
    '''
    Input:
        - fmin: float, the minimum frequency
        - fmax: float, the maximum frequency
        - plot options: bool, if True, the plot will be shown
        - special_site_terms: dict, if you want to use a special site term, you can pass it here.
             In case the site term wants to be altered instead of read from disk.
        - std_thresh: float, if you want to set a threshold for the standard deviation of Q0, you can pass it here.
        - give_out_results: bool, if True, the function will return the results of each individual event

    Output:
        - dict with the Q0 and Sk values for all events for a single frequency range

    Method:
        - calculates, using Lg_amplitude_dict all the amplitudes of all stations for all events and frequency ranges
        - then for each event, the points of all available stations are 'plotted' against the distance
        - a linear regression is calculated, and the Q factor is calculated using the slope of the regression line, and Sk is calculated using the intercept
        - the error on Q0 and Sk is calculated using the standard error of the slope and intercept
        - if the option std_thresh is not None, events with a std higher than given threshold are not used in the above calculation
        - the results are stored in a dict, where the keys are the event times and the values are dicts with Q0, Sk and their errors
    '''





    base_directory = "/home/schreinl/Stage/Data1" 
    
    fminlist = [0.5,2,4,6]
    fmaxlist = [1.5,3,6,8]
    Lg_dict = Lg_amplitude_dict(base_directory,fminlist,fmaxlist)
    results, results_with_coords = Lg_amplitude_calculation(fmin, fmax, Lg_dict, savefile=False,special_site_terms=special_site_terms)

    event_data_dict = {}
    eventcounter = 0
    drop1counter = 0
    drop2counter = 0
    for event_time, event_data in results.items():
        eventcounter += 1
        log_amplitudes = [station_data["log_value"] for station_data in event_data.values()]
        distances = [station_data["distance_km"] for station_data in event_data.values()]

        filtered_distances = []
        filtered_log_amplitudes = []

        for d, a in zip(distances, log_amplitudes):
            if d > 200 and np.isfinite(d) and np.isfinite(a):
                filtered_distances.append(d)
                filtered_log_amplitudes.append(a)

        filtered_distances = np.array(filtered_distances)
        filtered_log_amplitudes = np.array(filtered_log_amplitudes)
        #take off outliers with IQR and z score 
        filtered_pairs = [(d, a) for d, a in zip(filtered_distances, filtered_log_amplitudes)]
        if filtered_pairs:
            filtered_distances, filtered_log_amplitudes = zip(*filtered_pairs)
            filtered_distances, filtered_log_amplitudes = remove_outliers_iqr(filtered_distances, filtered_log_amplitudes)
        else:
            filtered_distances, filtered_log_amplitudes = [], []
        if len(filtered_distances) <= 1:
            drop1counter += 1
            continue
        
        slope, intercept, r_value, p_value, std_err_slope = linregress(filtered_distances, filtered_log_amplitudes)
        if np.isnan(slope) or np.isnan(intercept):
            print(f'{event_time} is not able to produce Q')
    #else:
        #drop1counter += 1
        #print(f'{event_time} not enough data points')

        #propagate the errors, maybe for further use
        f_avg = (fmin + fmax) / 2
        Q0 = - (np.pi * f_avg) / (3.2 * slope)
        Q0_error = np.abs((np.pi * f_avg) / (V_Lg * slope**2)) * std_err_slope


        std_err_intercept = std_err_slope * np.sqrt(np.mean(np.square(filtered_distances))) 
        Sk = np.exp(intercept)
        Sk_error = Sk * std_err_intercept

        if std_thresh is not None:
            if Q0_error > std_thresh:
                drop2counter += 1
                #print(f'{event_time} skipped: Q0 error {Q0_error:.2f} exceeds threshold {std_thresh}')
                continue
            else:
                if event_time in results_with_coords:
                    coordinates = results_with_coords[event_time]["coordinates"]
                    event_data_dict[event_time] = {
                        "Q0": Q0,
                        "Q0_error": Q0_error,
                        "Sk": Sk,
                        "Sk_error": Sk_error,
                        "coordinates": coordinates
                    }
        else:
            if event_time in results_with_coords:
                    coordinates = results_with_coords[event_time]["coordinates"]
                    event_data_dict[event_time] = {
                        "Q0": Q0,
                        "Q0_error": Q0_error,
                        "Sk": Sk,
                        "Sk_error": Sk_error,
                        "coordinates": coordinates
                    }




        
    print(f'{eventcounter} events in total')
    print(f'{drop2counter} events dropped due to insufficient std on Q0')
    print(f'{drop1counter} events dropped due to insufficient data points')
    print(f"Processed {len(event_data_dict)} events with Q0 and Sk values.")

    if plot_Q0 and event_data_dict:
        Q0_values = [data["Q0"] for data in event_data_dict.values()]
        Q0_errors = [data["Q0_error"] for data in event_data_dict.values()]
        plt.figure(figsize=(12, 6))
        plt.errorbar(range(len(Q0_values)), Q0_values, yerr=Q0_errors, fmt='o', color='green', ecolor='lightgray', elinewidth=2, capsize=3)
        plt.xlabel('Event Index')
        plt.ylim([0, 1000])
        plt.ylabel('Q Factor')
        plt.title(f'Q Factor for Each Event with Error Bars for {fmin}-{fmax}Hz')
        plt.xticks(range(0, len(Q0_values), max(1, len(Q0_values)//20)), rotation=90)
        plt.tight_layout()
        plt.show()

    if plot_Sk and event_data_dict:
        Sk_values = [data["Sk"] for data in event_data_dict.values()]
        plt.figure(figsize=(12, 6))
        plt.scatter(range(len(Sk_values)), Sk_values, marker='o', color='blue')
        plt.xlabel('Event Index')
        plt.ylabel('Sk')
        plt.title('Source term for Each Event')
        plt.xticks(range(0, len(Sk_values), max(1, len(Sk_values)//20)), rotation=90)
        plt.tight_layout()
        plt.show()

    if plot_Q0_map and event_data_dict:
        latitudes = [event_data['coordinates']["latitude"] for event_data in event_data_dict.values()]
        longitudes = [event_data['coordinates']["longitude"] for event_data in event_data_dict.values()]
        Q_values = [event_data["Q0"] for event_data in event_data_dict.values()]

        unique_Q_values = sorted(set(Q for Q in Q_values if not np.isnan(Q)))
        print(f'lower boundary {np.percentile(unique_Q_values,5)} and upper boundary {np.percentile(unique_Q_values,95)}')
        colormap = cm.LinearColormap(
            colors=["blue", "green", "yellow", "orange", "red"], 
            vmin=np.percentile(unique_Q_values,5), 
            vmax=np.percentile(unique_Q_values,95)
        )
        colormap.caption = "Q Factor"

        event_map = folium.Map(location=[np.mean(latitudes), np.mean(longitudes)], zoom_start=5, tiles="OpenStreetMap")

        counter = 0
        for event_time, event_data in event_data_dict.items():
            Q = event_data["Q0"]
            lat = event_data['coordinates']["latitude"]
            lon = event_data['coordinates']["longitude"]
            
            if not np.isnan(Q):
                counter += 1
                popup_text = f"Event: {event_time}<br>Q: {Q:.2f}"
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=8,
                    tooltip=popup_text,
                    color=colormap(Q),  # Use colormap with sorted values
                    weight=1,
                    fill=True,
                    fill_color=colormap(Q),
                    fill_opacity=0.9
                ).add_to(event_map)

        colormap.add_to(event_map)
        print(f'{counter} events on map')
        event_map
        return event_data_dict, event_map
        

    if plot_Sk_map and event_data_dict:
        latitudes = [event_data['coordinates']["latitude"] for event_data in event_data_dict.values()]
        longitudes = [event_data['coordinates']["longitude"] for event_data in event_data_dict.values()]
        Sk_values = [event_data["Sk"] for event_data in event_data_dict.values()]

        unique_Sk_values = sorted(set(Sk for Sk in Sk_values if np.isfinite(Sk)))
        print(unique_Sk_values)
        print(f'lower boundary {np.percentile(unique_Sk_values,5)} and upper boundary {np.percentile(unique_Sk_values,95)}')
        colormap = cm.LinearColormap(
            colors=["blue", "green", "yellow", "orange", "red"], 
            vmin=np.percentile(unique_Sk_values,5), 
            vmax=np.percentile(unique_Sk_values,95)
        )
        colormap.caption = "Source Term"

        event_map = folium.Map(location=[np.mean(latitudes), np.mean(longitudes)], zoom_start=5, tiles="OpenStreetMap")

        counter = 0
        for event_time, event_data in event_data_dict.items():
            Sk = event_data["Sk"]
            lat = event_data['coordinates']["latitude"]
            lon = event_data['coordinates']["longitude"]
            
            if not np.isnan(Sk):
                counter += 1
                popup_text = f"Event: {event_time}<br>Q: {Sk:.2f}"
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=8,
                    tooltip=popup_text,
                    color=colormap(Sk),
                    weight=1,
                    fill=True,
                    fill_color=colormap(Sk),
                    fill_opacity=0.9
                ).add_to(event_map)

        colormap.add_to(event_map)
        print(f'{counter} events on map')
        event_map
        return event_data_dict, event_map


    if give_out_results:
        print("Returning event data dictionary.")
        return event_data_dict, results
    else:
        return event_data_dict





def remove_outliers_iqr(distances, log_amplitudes):
    distances = np.array(distances)
    log_amplitudes = np.array(log_amplitudes)
    
    Q1 = np.percentile(log_amplitudes, 25)
    Q3 = np.percentile(log_amplitudes, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    mask = (log_amplitudes >= lower_bound) & (log_amplitudes <= upper_bound)
    
    return distances[mask], log_amplitudes[mask]



def extract_frequency_range(data_dict, target_freq):
    result = {}
    for event, stations in data_dict.items():
        for station, freqs in stations.items():
            if target_freq in freqs:
                if event not in result:
                    result[event] = {}
                result[event][station] = freqs[target_freq]
    return result



import csv
def Lg_Q_kl(fmin, fmax, savefile=False,std_thresh=100, envelope_name='new', special_site_terms=None,
            best_bet = ['MLS','GRA1','SSB','BFO','LOR','SENIN','HASLI','BOURR','BNALP','DAVOX','FUORN','MOA','CONA','CLUD','OBKA','MONC','CIMO','ECH'],
            frequenciesmin=[0.5, 2, 4, 6], frequenciesmax=[1.5, 3, 6, 8]):


    #with open("../Data1/Site_effects_dict.txt", "r") as f:
    #    site_terms = json.load(f)
    
    if special_site_terms:
        site_terms = special_site_terms
    else:
        site_terms = site_effect_overall(fmin, fmax, best_bet, frequenciesmin=frequenciesmin, frequenciesmax=frequenciesmax, method='multiple', map_plot=False, envelope_name='new')



    fmins = [0.5, 2, 4, 6]  
    fmaxs = [1.5,3, 6, 8] 

    base_directory = "/home/schreinl/Stage/Data1"
    Lg_dicts = Lg_amplitude_dict(base_directory, fmins, fmaxs)
    Lg_dict = extract_frequency_range(Lg_dicts, f"{fmin}_{fmax}Hz")

    station_coords = {}
    with open("../Data1/all_stations_coordinates.txt", "r") as f:
        next(f)
        for line in f:
            if line.strip():
                network, station, lat, lon = line.strip().split(",")
                station_coords[f"{network}.{station}"] = (float(lat), float(lon))
    
    event_data = {}
    with open("../Data/big_box_4.5.csv", "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            event_time = UTCDateTime(row[0])
            event_lat = float(row[1])
            event_lon = float(row[2])
            event_time_str = event_time.strftime("%Y_%m_%dT%H_%M_%S")
            event_data[event_time_str] = (event_lat, event_lon)
    

    Q0_Sk_dict = Lg_Q0_Sk(fmin, fmax,std_thresh=std_thresh, special_site_terms=special_site_terms)
    Qkl_dict = {}
    v = 3.2
    
    for event_time, stations in Lg_dict.items():
        if event_time in ["Dicts", "Metadata"]: 
            continue
        
        formatted_event_time = (UTCDateTime(event_time) + 25).strftime("%Y_%m_%dT%H_%M_%S")
        if formatted_event_time not in event_data:
            continue
        
        event_lat, event_lon = event_data[formatted_event_time]
        Qkl_dict[formatted_event_time] = {
            "coordinates": {"latitude": event_lat, "longitude": event_lon},
            "stations": {}
        }
        
        for station, amplitude in stations.items():
            if station not in station_coords:
                continue
            
            lat, lon = station_coords[station]
            r = haversine(event_lat, event_lon, lat, lon)
            
            site_effect = None
            for freq_band, site_info in site_terms.items():
                freq_min, freq_max = map(lambda x: float(x.replace('Hz', '').strip()), freq_band.split('-'))
                if fmin <= freq_min <= fmax and fmax >= freq_max >= fmin:
                    if station.split(".")[1] in site_info:
                        site_effect = site_info[station.split(".")[1]].get("median_after", None)
                        break
            
            if site_effect is None or site_effect <= 0 or r == 0:
                Qkl = None
                observable=None
            else:
                f = (fmin + fmax) / 2 

                if formatted_event_time in Q0_Sk_dict:
                    source_term = Q0_Sk_dict[formatted_event_time]['Sk']            
                    Qkl = (-(np.pi * r * f) / v) * (-np.log(site_effect) - np.log(source_term) + np.log(amplitude * r**0.83))**-1
                    observable = r / Qkl
                else:
                    #print(f"Warning: Event time {formatted_event_time} not found in Q0_Sk_dict.")
                    Qkl = None
                    observable = None
            
            Qkl_dict[formatted_event_time]["stations"][station] = {
                "Qkl": Qkl,
                "observable": observable,
                "coordinates": {"latitude": lat, "longitude": lon}
            }
    
    if savefile:
        with open(f"../Data1/Qkl_results_{fmin}_{fmax}Hz.json", "w") as f:
            json.dump(Qkl_dict, f, indent=4)
    
    return Qkl_dict











