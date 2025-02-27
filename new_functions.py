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
        maxlongitude=17,
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
    output_file = f"C:/UGA/Stage/Data/Metadata/{time_string}.txt"

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


from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from obspy.clients.fdsn import RoutingClient
from obspy import Stream
from obspy.geodetics import gps2dist_azimuth
from obspy import signal

import matplotlib.pyplot as plt
import folium
import numpy as np

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


def get_Pn_time(dist_deg) :


    from obspy.taup import TauPyModel

    model = TauPyModel(model='ak135') #crust at 35 (?) , but only 3s difference with 11km crust 
    t_Pn=111.*dist_deg/8.  #default value
    
    arrivals = model.get_travel_times(source_depth_in_km=0,
                                  distance_in_degree=dist_deg,phase_list=["Pn"])


    try:
        t_Pn=arrivals [0].time
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
        t_Sn=arrivals [0].time
    except Exception as e:
        print('no Sn ', dist_deg, e)

    return t_Sn

def get_Pg_time(dist_deg) :

    from obspy.taup import TauPyModel

    model = TauPyModel(model='ak135') 
    t_Pg=111.*dist_deg/6.
    
    arrivals = model.get_travel_times(source_depth_in_km=0,
                                  distance_in_degree=dist_deg,phase_list=["Pg"])


    try:
        t_Pg=arrivals [0].time
    except Exception as e:
        t_Pg = dist_deg/5.7
        #print('no Pg ', dist_deg, e)

    return t_Pg

import os
from obspy import read, UTCDateTime
from time import sleep
import sys
from tqdm.auto import tqdm

def get_data2(client, inventory, start, end, eq_lon, eq_lat, distmin, distmax, directory='C:/UGA/Stage/Data/',datacenter='datacenter'):
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



def plot_record_section(
    st, stations, eq_lat, eq_lon, eq_start, size=(1200, 1000), show=True, outfile=None, tracehodo=True, v_Lg_min=3.1, v_Lg_max=3.5, 
v_Pg=6.,tmincoda=300,tmaxcoda=320):

    if not st or not stations:
        print("Station or stream empty.")
        return
    st2 = Stream()
    
 # create stream corresponding to the stations in matrix stations. 
    station_array=np.array(stations)

    dist_work=station_array[:,5].astype(float)/1000.
    t_Pn_work=station_array[:,7].astype(float)
    t_Sn_work=station_array[:,8].astype(float)
    t_Pg_work=station_array[:,9].astype(float)
    dist_index=np.argsort(dist_work)
    dist_sectplot=dist_work[dist_index]
    t_Pn_plot=t_Pn_work[dist_index]
    t_Sn_plot=t_Sn_work[dist_index]
    t_Lg_min_plot=dist_sectplot/v_Lg_max
    t_Lg_max_plot=dist_sectplot/v_Lg_min
    t_Pg_plot=dist_sectplot/v_Pg
    tmin_coda = [tmincoda] * len(t_Pg_plot)
    tmax_coda = [tmaxcoda] *len(t_Pg_plot)
    #t_Pg_plot_tauP = t_Pg_work[dist_index]

    for tr in st:
        for net, sta, lat, lon, elev , dist, az, t_Pn, t_Sn, t_Pg in stations:
            # We keep traces with a corresponding station only:
            if tr.stats.network == net and tr.stats.station == sta:
                tr.stats.coordinates = {"latitude": lat, "longitude": lon}
                tr.stats.distance = dist
                st2.append(tr)  

    # Plot the section:
    figure = plt.figure(figsize=(size[0] // 100, size[1] // 100))
    if len(st2) < 2:
        print("Cannot build plot section with less than two traces.\n")
        return

    begin = min(tr.stats.starttime for tr in st2)
    st2.trim(starttime=begin, pad=True, fill_value=0)

    st2.plot(type="section", linewidth=0.25, grid_linewidth=0.25, fig=figure, norm_method='trace')
    ax = figure.axes[0]

    ds = [(tr.stats.distance, tr.stats.station) for tr in st2]
    ds.sort()
    for n, (dist, sta) in enumerate(ds):
        # to avoid merged titles
        ycoord = 1.05 if (n + 1) % 2 == 0 else 1.07
        ax.text(dist / 1e3, ycoord * ax.get_ylim()[1], sta, fontsize=7, rotation=45)
    if tracehodo == True :
        plt.plot(dist_sectplot, t_Pn_plot, color='r', linestyle='dashed',linewidth=1,label='Pn')
        plt.plot(dist_sectplot, t_Sn_plot, color='b', linestyle='dashed',linewidth=1,label='Sn')
        plt.plot(dist_sectplot, t_Lg_min_plot, color='g', linestyle='dashed',linewidth=1,label='Lg_min')
        plt.plot(dist_sectplot, t_Lg_max_plot, color='g', linestyle='dashed',linewidth=1,label='Lg_max')
        plt.plot(dist_sectplot, t_Pg_plot, color='orange', linestyle='dashed',linewidth=1,label='Pg')
        plt.plot(dist_sectplot, tmax_coda, color='purple', linestyle='dashed',linewidth=1,label='Coda max')
        plt.plot(dist_sectplot, tmin_coda, color='purple', linestyle='dashed',linewidth=1,label='Coda min')


        #plt.plot(dist_sectplot, t_Pg_plot_tauP, color='k', linestyle='dashed',linewidth=1,label='Pg_tauP')
        plt.legend(loc='upper left')
        st_Lg = Stream()
        for tr in st2:
            dist = tr.stats.distance / 1000.0
            t_Lg_min = dist / v_Lg_max
            t_Lg_max = dist / v_Lg_min
            tr_Lg = tr.copy().trim(starttime=eq_start + t_Lg_min, endtime=eq_start + t_Lg_max)
            st_Lg.append(tr_Lg)
        
            
    
    if outfile:
        plt.savefig(outfile)
    elif show:

        plt.show()
    return figure, st_Lg



def SNR(stations, st, Dtmin_Pn, Dtmax_Pn, Dtmin_Sn, Dtmax_Sn, vLg_min,vLg_max,vPg_min, vPg_max, tmin_Coda, tmax_Coda,
         Dtmin_Noise, Dtmax_Noise,eq_start, method='time_amplitude',signal_window='coda',plot_map=False,dB=False):

    
    print(f'calculating SNR for {signal_window}  phase')
    SNR = []
    if method == 'time_amplitude':
        for net, sta, lat, lon, elev , dist, az, t_Pn, t_Sn ,t_Pg in stations:
            A_Noise=0.
            A_Pn=0.
            A_Sn=0.
            A_Lg=0.
            A_Coda=0.
            A_LgAP=0.
            A_LgACoda=0.
            tmin_Noise=t_Pn+Dtmin_Noise
            tmax_Noise=t_Pn+Dtmax_Noise
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
                    dt=tr.stats.delta
                    nt=tr.stats.npts
                    trace_end=trace_start+dt*(nt-1)
                    tvector=np.arange(trace_start,trace_end+dt,dt)
                    datavector=tr.data
                
                    if signal_window == 'coda':
                        iminCoda=int((tmin_Coda-trace_start)/dt)
                        imaxCoda=int((tmax_Coda-trace_start)/dt)
                        iminNoise=int((tmin_Noise-trace_start)/dt)
                        imaxNoise=int((tmax_Noise-trace_start)/dt)
                        dataselectCoda=(datavector[iminCoda:imaxCoda])
                        dataselectNoise=(datavector[iminNoise:imaxNoise])                        
                        signal_power = np.sum(np.abs(dataselectCoda)**2) 
                        noise_power = np.sum(np.abs(dataselectNoise)**2)
                        if noise_power == 0 or (signal_power/noise_power) == 0:
                            snr = 0
                        if dB:
                            snr = 10 * np.log10(signal_power / noise_power)
                        else:
                            snr = signal_power/noise_power

                    if signal_window == 'Lg':
                        iminLg=int((tminLg-trace_start)/dt)
                        imaxLg=int((tmaxLg-trace_start)/dt)
                        iminNoise=int((tmin_Noise-trace_start)/dt)
                        imaxNoise=int((tmax_Noise-trace_start)/dt)
                        dataselectLg=(datavector[iminLg:imaxLg])
                        dataselectNoise=(datavector[iminNoise:imaxNoise])                        
                        signal_power = np.sum(np.abs(dataselectLg)**2) 
                        noise_power = np.sum(np.abs(dataselectNoise)**2)
                        if noise_power == 0 or (signal_power/noise_power) == 0:
                            snr = 0
                        if dB:
                            snr = 10 * np.log10(signal_power / noise_power)
                        else:
                            snr = signal_power / noise_power

                    if signal_window == 'Pn':
                        iminPn=int((tmin_Pn-trace_start)/dt)
                        imaxPn=int((tmax_Pn-trace_start)/dt)
                        iminNoise=int((tmin_Noise-trace_start)/dt)
                        imaxNoise=int((tmax_Noise-trace_start)/dt)
                        dataselectPn=(datavector[iminPn:imaxPn])
                        dataselectNoise=(datavector[iminNoise:imaxNoise])                        
                        signal_power = np.sum(np.abs(dataselectPn)**2) 
                        noise_power = np.sum(np.abs(dataselectNoise)**2)
                        if noise_power == 0 or (signal_power/noise_power) == 0:
                            snr = 0
                        if dB:
                            snr = 10 * np.log10(signal_power / noise_power)
                        else:
                            snr = signal_power / noise_power
                    
                    if signal_window == 'Sn':
                        iminSn=int((tmin_Sn-trace_start)/dt)
                        imaxSn=int((tmax_Sn-trace_start)/dt)
                        iminNoise=int((tmin_Noise-trace_start)/dt)
                        imaxNoise=int((tmax_Noise-trace_start)/dt)
                        dataselectSn=(datavector[iminSn:imaxSn])
                        dataselectNoise=(datavector[iminNoise:imaxNoise])
                        signal_power = np.sum(np.abs(dataselectSn)**2)
                        noise_power = np.sum(np.abs(dataselectNoise)**2)
                        if noise_power == 0 or (signal_power/noise_power) == 0:
                            snr = 0
                        if dB:
                            snr = 10 * np.log10(signal_power / noise_power)
                        else:
                            snr = signal_power / noise_power

                    if signal_window == 'Pg':
                        iminPg=int((tmin_Pg-trace_start)/dt)
                        imaxPg=int((tmax_Pg-trace_start)/dt)
                        iminNoise=int((tmin_Noise-trace_start)/dt)
                        imaxNoise=int((tmax_Noise-trace_start)/dt)
                        dataselectPg=(datavector[iminPg:imaxPg])
                        dataselectNoise=(datavector[iminNoise:imaxNoise])
                        signal_power = np.sum(np.abs(dataselectPg)**2)
                        noise_power = np.sum(np.abs(dataselectNoise)**2)
                        if noise_power == 0 or (signal_power/noise_power) == 0:
                            snr = 0
                        if dB:
                            snr = 10 * np.log10(signal_power / noise_power)
                        else:
                            snr = signal_power / noise_power
                        
            SNR.append([snr])
        stations_with_SNR=np.append(np.array(stations),np.array(SNR),axis=1)
        
        if plot_map==True:
            Amp_Draw=stations_with_SNR[:,9].astype(float)
            Amp_Draw[np.isnan(Amp_Draw)] = 0    
            Amp_Draw[np.isinf(Amp_Draw)] = 0
            plotit =plot_stations_amps(stations_with_SNR, 1, 0.7, Amp_Draw, origin=[eq_lat,eq_lon], zoom=5, forcescale=False)
            plotit


        return stations_with_SNR






def SNR_all(stations, st, Dtmin_Pn, Dtmax_Pn, Dtmin_Sn, Dtmax_Sn, vLg_min,vLg_max,vPg_min, vPg_max, tmin_Coda, tmax_Coda,
         Dtmin_Noise, Dtmax_Noise,eq_start,eq_lat,eq_lon,snr_threshold=2,plot_SNR=False,plot_amps=False,wavecode="Lg_Pg",dB=False):
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
        for k, station in enumerate(stations):
            net, sta, lat, lon, elev, dist, az, t_Pn, t_Sn, t_Pg = station
            A_Noise=0.
            A_Pn=0.
            A_Sn=0.
            A_Lg=0.
            A_Coda=0.
            A_LgAP=0.
            A_LgACoda=0.
            tmin_Noise=t_Pn+Dtmin_Noise
            tmax_Noise=t_Pn+Dtmax_Noise
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
    print("Reduced from  ", len(stations_with_SNR), " stations to  ", len(filtered_arr), " stations due to insufficient SNR or distance > " ,  dist_mean)
    
    #with the earthquake specific cutoff distance we can now set tmin_coda:
    
    tmin_Coda = 1.3* (dist_Lg/3)
    tmax_Coda = tmin_Coda + 50
    print(f"coda window set from {tmin_Coda}-{tmax_Coda}s")
    phase_distance['tmin_Coda'] = tmin_Coda
    
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
    




def SNR_amplitude(stations, st, Dtmin_Pn, Dtmax_Pn, Dtmin_Sn, Dtmax_Sn, vLg_min,vLg_max,vPg_min, vPg_max, tmin_Coda, tmax_Coda,
         Dtmin_Noise, Dtmax_Noise,eq_start, method='time_amplitude',signal_window='coda',plot_map=False,dB=False):

    
    print(f'calculating SNR for {signal_window}  phase')
    SNR = []
    if method == 'time_amplitude':
        for net, sta, lat, lon, elev , dist, az, t_Pn, t_Sn ,t_Pg in stations:
            A_Noise=0.
            A_Pn=0.
            A_Sn=0.
            A_Lg=0.
            A_Coda=0.
            A_LgAP=0.
            A_LgACoda=0.
            tmin_Noise=t_Pn+Dtmin_Noise
            tmax_Noise=t_Pn+Dtmax_Noise
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
                    dt=tr.stats.delta
                    nt=tr.stats.npts
                    trace_end=trace_start+dt*(nt-1)
                    tvector=np.arange(trace_start,trace_end+dt,dt)
                    datavector=tr.data
                
                    if signal_window == 'coda':
                        iminCoda=int((tmin_Coda-trace_start)/dt)
                        imaxCoda=int((tmax_Coda-trace_start)/dt)
                        iminNoise=int((tmin_Noise-trace_start)/dt)
                        imaxNoise=int((tmax_Noise-trace_start)/dt)
                        dataselectCoda=(datavector[iminCoda:imaxCoda])
                        dataselectNoise=(datavector[iminNoise:imaxNoise])                        
                        signal_power = np.sqrt(np.dot(dataselectCoda,np.transpose(dataselectCoda)))/len(dataselectCoda) 
                        noise_power = np.sqrt(np.dot(dataselectNoise,np.transpose(dataselectNoise)))/len(dataselectNoise)
                        if noise_power == 0 or (signal_power/noise_power) == 0:
                            snr = 0
                        if dB:
                            snr = 10 * np.log10(signal_power / noise_power)
                        else:
                            snr = signal_power/noise_power

                    if signal_window == 'Lg':
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

                    if signal_window == 'Pn':
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
                    
                    if signal_window == 'Sn':
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

                    if signal_window == 'Pg':
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
                        
            SNR.append([snr])
        stations_with_SNR=np.append(np.array(stations),np.array(SNR),axis=1)
        
        if plot_map==True:
            Amp_Draw=stations_with_SNR[:,9].astype(float)
            Amp_Draw[np.isnan(Amp_Draw)] = 0    
            Amp_Draw[np.isinf(Amp_Draw)] = 0
            plotit =plot_stations_amps(stations_with_SNR, 1, 0.7, Amp_Draw, origin=[eq_lat,eq_lon], zoom=5, forcescale=False)
            plotit


        return stations_with_SNR




from scipy import stats
def SNR_distance(stations, st, Dtmin_Pn, Dtmax_Pn, Dtmin_Sn, Dtmax_Sn, vLg_min, vLg_max, vPg_min, vPg_max, tmin_Coda, tmax_Coda,
                 Dtmin_Noise, Dtmax_Noise,eq_start, dB= True):
    phases = ['Lg', 'Pn', 'Sn', 'Pg']
    phase_distance = {}
    
    fig, axs = plt.subplots(2, 2, figsize=(15, 15))
    plt.style.use('seaborn-v0_8')
    for i, phase in enumerate(phases):
        stations_with_SNR = SNR_amplitude(stations, st, Dtmin_Pn, Dtmax_Pn, Dtmin_Sn, Dtmax_Sn, vLg_min, vLg_max, vPg_min, vPg_max,
                                tmin_Coda, tmax_Coda, Dtmin_Noise, Dtmax_Noise,eq_start, method='time_amplitude', signal_window=phase, plot_map=False, dB=dB)
        SNR_vals = stations_with_SNR[:, -1].astype(float)
        dist_vals = stations_with_SNR[:, 5].astype(float) / 1000. 
        #SNR_vals = SNR_vals[np.isfinite(SNR_vals)]
        #dist_vals = dist_vals[np.isfinite(SNR_vals)]
        
        threshold = 2
        snr_threshold = 0.9


        filtered_distances = dist_vals[SNR_vals > snr_threshold]

        if len(filtered_distances) > 0:
            percentile_distance = np.percentile(filtered_distances, 90)
            phase_distance[phase] = percentile_distance
            print(f"Distance where 90% of SNR values are above 2: {percentile_distance}")
        else:
            print("No valid SNR values above 1.5.")
            percentile_distance = None

        if percentile_distance is not None:
            #calculate the slope for for the regression snr = a*dist + b, when dist < percentile_distance, so sufficient SNR
            coef = np.polyfit(dist_vals[dist_vals < percentile_distance],np.nan_to_num(SNR_vals[dist_vals < percentile_distance], nan=0.0, posinf=0.0, neginf=0.0),1)
            coef_quad = np.polyfit(dist_vals[dist_vals < percentile_distance],np.nan_to_num(SNR_vals[dist_vals < percentile_distance], nan=0.0, posinf=0.0, neginf=0.0),2)
            poly1d_fn_quad = np.poly1d(coef_quad)
            poly1d_fn = np.poly1d(coef)
            
            #calculate the slope for the regression snr = a*dist + b, when dist > percentile_distance, so insufficient SNR
            coef1 = np.polyfit(dist_vals[dist_vals > percentile_distance],np.nan_to_num(SNR_vals[dist_vals > percentile_distance], nan=0.0, posinf=0.0, neginf=0.0),1)
            poly1d_fn1 = np.poly1d(coef1)
            phase_distance[phase] = {
                'percentile_distance': percentile_distance,
                'coef_quad': coef[0],
                'coef1': coef1[0]
            }
            ax = axs[i//2, i%2]
            ax.plot(dist_vals, SNR_vals, 'o')
            ax.plot(dist_vals[dist_vals > percentile_distance], poly1d_fn1(dist_vals[dist_vals > percentile_distance]), 'r', color='r', label='insufficient SNR')
            ax.plot(dist_vals[dist_vals < percentile_distance], poly1d_fn(dist_vals[dist_vals < percentile_distance]), 'r', color='g',label='sufficient SNR')
            #ax.plot(dist_vals[dist_vals < percentile_distance], poly1d_fn_quad(dist_vals[dist_vals < percentile_distance]), 'r', color='r', label='sufficient SNR')
            ax.vlines(percentile_distance, ymin=-10, ymax=50, color='r', linestyle='dashed', label=f'90th percentile at {percentile_distance}')
        else:
            ax = axs[i//2, i%2]
            ax.plot(dist_vals, SNR_vals, 'o')
        
        ax.set_xlabel('Distance (km)')
        ax.legend(loc='upper right')
        ax.set_ylim(-20, 70)
        ax.set_ylabel('SNR (dB)')
        ax.set_title(f'{event_name} SNR with {phase} phase')
    plt.tight_layout()
    plt.show()
    return phase_distance 
    


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
        tmin_Noise=float(t_Pn)+Dtmin_Noise
        tmax_Noise=float(t_Pn)+Dtmax_Noise
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
                if (trace_start<tminPg) and (trace_end>tmaxPg) :
                    iminPg=int((tminPg-trace_start)/dt)
                    imaxPg=int((tmaxPg-trace_start)/dt)
                    dataselectPg=(datavector[iminPg:imaxPg])
                    A_Pg=np.sqrt(np.dot(dataselectPg,np.transpose(dataselectPg)))/len(dataselectPg)

    
        stations_amplitudes.append([A_Pn, A_Sn, A_Lg, A_Coda, A_Noise, A_Pg])


    stations_with_amps=np.append(np.array(stations[:,:10]),np.array(stations_amplitudes),axis=1)

    return stations_with_amps
        
def plot_stations_amps(stations_amps, amin, amax, Amp_Draw, origin=[0, 0], zoom=4, color="red", geom=False, 
                       geompower=0.5, normQ=False, Q=1000, f0=1, v=3.4, forcescale=False, outfile=None,amplitudes_or_snr="amplitudes"):
    import branca.colormap as cm

    stations_with_amps_list=stations_amps.tolist()    

    
    plot_amp=Amp_Draw
    distwork=stations_amps[:,5].astype(float)/1000.
    
    if geom == True :
            A0=1./np.power(200.,geompower)
            dist_power=A0*np.power(distwork,geompower)
    else:
            dist_power=1+0.*distwork
            
    if normQ == True:
            multexp=np.pi*(f0/v/Q)
            dist_exp=np.exp(multexp*distwork)
    else:
            dist_exp=1+0.*distwork
    
    plot_amp=np.multiply(plot_amp,dist_power)
    plot_amp=np.multiply(plot_amp,dist_exp)
        
#    linear = cm.linear.RdYlGn_04.scale(amin, amax)
    linear = cm.LinearColormap(["green", "yellow", "red"], vmin=amin*min(plot_amp), vmax=amax*max(plot_amp))
    if forcescale :
            linear = cm.LinearColormap(["green", "yellow", "red"], vmin=amin, vmax=amax)
    carte = folium.Map(location=origin, zoom_start=zoom)
    
    istat=-1
    if amplitudes_or_snr == "amplitudes":
        for net, sta, lat, lon, elev, dist, az, t_Pn, t_Sn, t_Pg, A_Pn, A_Sn, A_Lg, A_Coda, A_Noise, A_Pg in stations_with_amps_list:
            istat=istat+1        
            name = ".".join([net, sta])
            infos = "%s (%s, %s) %s m" % (name, lat, lon, elev)
            folium.CircleMarker(
                location=[lat, lon],
                tooltip=infos,         
                fill=True,
                fill_opacity=1.0,
                color=linear(plot_amp[istat]), 
                radius=4,
            ).add_to(carte)
    elif amplitudes_or_snr == "snr":
          for net, sta, lat, lon, elev, dist, az, t_Pn, t_Sn, t_Pg, SNR in stations_with_amps_list:
            istat=istat+1        
            name = ".".join([net, sta])
            infos = "%s (%s, %s) %s m" % (name, lat, lon, elev)
            folium.CircleMarker(
                location=[lat, lon],
                tooltip=infos,         
                fill=True,
                fill_opacity=1.0,
                color=linear(plot_amp[istat]), 
                radius=4,
            ).add_to(carte)
          

    
    folium.CircleMarker(
        location=origin,
        radius=5,
        color='blue',
        fill=True,
        fill_color="#FF8C00",
        ).add_to(carte)
    
    if outfile:
        carte.save(outfile)
    #        webbrowser.open(outfile, new=2, autoraise=True)
    #        time.sleep(1)
    
    return carte

               




import numpy as np
import matplotlib.pyplot as plt
from obspy import Stream
from matplotlib.colors import LogNorm


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def snr_azimuth(stations_with_snr, column=10, barlabel="SNR", xlabel="Azimuth (°)", 
                ylabel="Distance (km)", title="SNR", event_name="Earthquake",
                savefig=False, show=False):
    
    '''
    - stations_with_snr: unfiltered stations list with the SNR of different phases
    - column: int of the column that will be read in, col 10 for 'Pn', 11 'Pg', 12 'Sn', 13 'Lg'
    - savefig: optional figure save
    - show: optional figure show
    - plots the snr in the dependency of the azimuth
    - returns: None 
    '''
    
    phases = {10: 'Pn', 11: 'Pg', 12: 'Sn', 13: 'Lg'}
    phase = phases.get(column, 'Unknown')

    distDraw = stations_with_snr[:, 5].astype(float) / 1000.
    azDraw = stations_with_snr[:, 6].astype(float)
    SNR = stations_with_snr[:, column].astype(float)

    plt.style.use('seaborn-v0_8')

    theta = np.radians(azDraw)

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': 'polar'})

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    sc = ax.scatter(theta, distDraw, c=SNR, cmap='rainbow', norm=LogNorm())
    cbar = plt.colorbar(sc, ax=ax, pad=0.1)
    cbar.set_label(barlabel)

    ax.set_xlabel(xlabel, labelpad=15)
    ax.set_ylabel(ylabel, labelpad=15)
    
    ax.set_title(f'{event_name} {title} of phase {phase}', va='top', y=1.1)

    fig.tight_layout()

    if savefig:
        plt.savefig(f'../Figures/{event_name}_{phase}_SNR_az.png', format='png')

    if show:
        plt.show()
    elif show==False:
        plt.close(fig)
    return 



def magnitude_cutoff(eq_list, plottype='single', event_name='Earthquake', savefig=False, show=True):
    """
    Reads magnitude and cutoff distances for different seismic phases and plots them.

    Parameters:
    - eq_list: List of event times (first column of query).
    - plottype: 'single' (4 subplots) or 'combined' (1 scatter plot).
    - event_name: Name of the event (default 'Earthquake').
    - savefig: Whether to save the figure (default False).
    - show: Whether to display the plot (default True).

    Returns:
    - None
    """

    mag_dir = '/home/schreinl/Stage/Data/Metadata/'
    cut_dist_dir = '/home/schreinl/Stage/Data/Dicts/'
    
    Pn = []
    Sn = []
    Pg = []
    Lg = []
    mags = []


    for event in eq_list:
        time_string = UTCDateTime.strftime(event, format="%Y_%m_%dT%H_%M_%S")

        with open(f"{cut_dist_dir}{time_string}_dict.txt", "r") as file:
            dist_data = json.load(file)

        with open(f"{mag_dir}{time_string}.txt", "r") as meta:
            for line in meta:
                if line.startswith("Magnitude:"):
                    magnitude = float(line.split(":")[1].strip())
                    mags.append(magnitude)

        for key, values in dist_data.items():
            if isinstance(values, dict) and 'percentile_distance' in values:
                if key == 'Pn':
                    Pn.append(values['percentile_distance'])
                elif key == 'Sn':
                    Sn.append(values['percentile_distance'])
                elif key == 'Pg':
                    Pg.append(values['percentile_distance'])
                elif key == 'Lg':
                    Lg.append(values['percentile_distance'])

    if plottype == 'single':
        fig, axs = plt.subplots(2, 2, figsize=(15, 15))
        plt.style.use('seaborn-v0_8')

        axs[0, 0].plot(Pn, mags, 'o')
        axs[0, 0].set_ylabel('Magnitude')
        axs[0, 0].set_xlabel('Distance (km)')
        axs[0, 0].set_title(f'{event_name} Pn phase')

        axs[0, 1].plot(Sn,mags, 'o')
        axs[0, 1].set_ylabel('Magnitude')
        axs[0, 1].set_xlabel('Distance (km)')
        axs[0, 1].set_title(f'{event_name} Sn phase')

        axs[1, 0].plot(Pg,mags, 'o')
        axs[1, 0].set_ylabel('Magnitude')
        axs[1, 0].set_xlabel('Distance (km)')
        axs[1, 0].set_title(f'{event_name} Pg phase')

        axs[1, 1].plot(Lg,mags, 'o')
        axs[1, 1].set_ylabel('Magnitude')
        axs[1, 1].set_xlabel('Distance (km)')
        axs[1, 1].set_title(f'{event_name} Lg phase')

        plt.tight_layout()
        if savefig:
            plt.savefig(f'/home/schreinl/Stage/Figures/SNR/{event_name}_magnitude_cutoff.png', format='png')
        if show:
            plt.show()
        else:
            plt.close(fig)

    elif plottype == 'combined':
        plt.figure(figsize=(10, 10))
        plt.scatter(Pn, mags, label= 'Pn', alpha=0.7)
        plt.scatter(Sn, mags, label='Sn', alpha=0.7)
        plt.scatter(Pg,mags, label='Pg', alpha=0.7)
        plt.scatter(Lg, mags, label='Lg', alpha=0.7)
        plt.ylabel('Magnitude')
        plt.xlabel('Distance (km)')
        plt.title(f'{event_name} Cutoff Distances')
        plt.legend()

        if savefig:
            plt.savefig(f'/home/schreinl/Stage/Figures/SNR/{event_name}_magnitude_cutoff_combined.png', format='png')
        if show:
            plt.show()
        else:
            plt.close()


def magnitude_cutoff(eq_list, plottype='single', event_name='Earthquake', savefig=False, show=True):
    """
    Reads magnitude and cutoff distances for different seismic phases and plots them.

    Parameters:
    - eq_list: List of event times (first column of query).
    - plottype: 'single' (4 subplots) or 'combined' (1 scatter plot).
    - event_name: Name of the event (default 'Earthquake').
    - savefig: Whether to save the figure (default False).
    - show: Whether to display the plot (default True).

    Returns:
    - None
    """

    mag_dir = '/home/schreinl/Stage/Data/Metadata/'
    cut_dist_dir = '/home/schreinl/Stage/Data/Dicts/'
    
    Pn = []
    Sn = []
    Pg = []
    Lg = []
    Pg_Pn = []
    Pg_Sn = []
    Pg_Lg=[]
    Pn_Sn=[]
    Sn_Lg=[]
    Pn_Lg = []
    mags = []


    for event in eq_list:
        time_string = UTCDateTime.strftime(event, format="%Y_%m_%dT%H_%M_%S")

        with open(f"{cut_dist_dir}{time_string}_dict.txt", "r") as file:
            dist_data = json.load(file)

        with open(f"{mag_dir}{time_string}.txt", "r") as meta:
            for line in meta:
                if line.startswith("Magnitude:"):
                    magnitude = float(line.split(":")[1].strip())
                    mags.append(magnitude)

        for key, values in dist_data.items():
            if isinstance(values, dict) and 'percentile_distance' in values:
                if key == 'Pn':
                    Pn.append(values['percentile_distance'])
                elif key == 'Sn':
                    Sn.append(values['percentile_distance'])
                elif key == 'Pg':
                    Pg.append(values['percentile_distance'])
                elif key == 'Lg':
                    Lg.append(values['percentile_distance'])
    for i in range(len(Pn)):
        Pg_Pn.append(Pg[i] - Pn[i])
        Pg_Sn.append(Pg[i] - Sn[i])
        Pg_Lg.append(Pg[i] - Lg[i])
        Pn_Sn.append(Pn[i] - Sn[i])
        Sn_Lg.append(Sn[i] - Lg[i])
        Pn_Lg.append(Pn[i] - Lg[i])


    if plottype == 'single':
        fig, axs = plt.subplots(2, 2, figsize=(15, 15))
        plt.style.use('seaborn-v0_8')

        axs[0, 0].plot(Pn, mags, 'o')
        axs[0, 0].set_ylabel('Magnitude')
        axs[0, 0].set_xlabel('Distance (km)')
        axs[0, 0].set_title(f'{event_name} Pn phase')

        axs[0, 1].plot(Sn,mags, 'o')
        axs[0, 1].set_ylabel('Magnitude')
        axs[0, 1].set_xlabel('Distance (km)')
        axs[0, 1].set_title(f'{event_name} Sn phase')

        axs[1, 0].plot(Pg,mags, 'o')
        axs[1, 0].set_ylabel('Magnitude')
        axs[1, 0].set_xlabel('Distance (km)')
        axs[1, 0].set_title(f'{event_name} Pg phase')

        axs[1, 1].plot(Lg,mags, 'o')
        axs[1, 1].set_ylabel('Magnitude')
        axs[1, 1].set_xlabel('Distance (km)')
        axs[1, 1].set_title(f'{event_name} Lg phase')

        plt.tight_layout()
        if savefig:
            plt.savefig(f'/home/schreinl/Stage/Figures/SNR/{event_name}_magnitude_cutoff.png', format='png')
        if show:
            plt.show()
        else:
            plt.close(fig)

    elif plottype == 'combined':
        plt.figure(figsize=(10, 10))
        plt.scatter(Pn, mags, label= 'Pn',color="blue")
        plt.scatter(Sn, mags, label='Sn',color="red")
        plt.scatter(Pg,mags, label='Pg',color="green")
        plt.scatter(Lg, mags, label='Lg',color="yellow")
        plt.ylabel('Magnitude')
        plt.xlabel('Distance (km)')
        plt.title(f'{event_name} Cutoff Distances')
        plt.legend()

        if savefig:
            plt.savefig(f'/home/schreinl/Stage/Figures/SNR/{event_name}_magnitude_cutoff_combined.png', format='png')
        if show:
            plt.show()
        else:
            plt.close()


    elif plottype== 'relative':
        plt.figure(figsize=(10,10))
        #plt.scatter(Pn_Lg, mags,label='Pn-Lg')
        #plt.scatter(Sn_Lg, mags,label='Sn-Lg')
        #plt.scatter(Pg_Lg, mags,label='Pg-Lg')
        #plt.scatter(Pg_Pn,mags, label='Pg-Pn')
        plt.scatter(Pg_Sn,mags, label='Pg-Sn')
        plt.legend()
        plt.xlabel('relative Distance (km)')
        plt.ylabel('Magnitude (Mw)')
        if savefig:
            plt.savefig(f'/home/schreinl/Stage/Figures/SNR/{event_name}_magnitude_cutoff_combined.png', format='png')
        if show:
            plt.show()
        else:
            plt.close()




    



def plot_record_section_with_energy(
    st, stations, eq_lat, eq_lon, eq_start, size=(1200, 1000), show=True, outfile=None, 
    tracehodo=True, v_Lg_min=3.1, v_Lg_max=3.5, v_Pg=6., tmincoda=300, tmaxcoda=320, 
    window_length=2.0, overlap=0.5):

    if not st or not stations:
        print("Station or stream empty.")
        return
    
    station_array = np.array(stations)
    dist_work = station_array[:,5].astype(float) / 1000.0
    dist_index = np.argsort(dist_work)
    dist_sectplot = dist_work[dist_index]
    
    st2 = Stream()
    for tr in st:
        for net, sta, lat, lon, elev, dist, az, t_Pn, t_Sn, t_Pg in stations:
            if tr.stats.network == net and tr.stats.station == sta:
                tr.stats.coordinates = {"latitude": lat, "longitude": lon}
                tr.stats.distance = dist
                st2.append(tr)  

    if len(st2) < 2:
        print("Cannot build plot section with less than two traces.")
        return

    begin = min(tr.stats.starttime for tr in st2)
    st2.trim(starttime=begin, pad=True, fill_value=0)

    energy_matrix = []
    time_axis = None
    for tr in st2:
        data = tr.data.astype(float)
        dt = tr.stats.delta
        win_samples = int(window_length / dt)
        step = int(win_samples * (1 - overlap))
        times = np.arange(0, len(data) - win_samples, step) * dt

        energy = [np.sum(data[i:i + win_samples] ** 2) for i in times.astype(int)]
        
        if np.max(energy) > 0:  # Prevent division by zero
            energy /= np.max(energy)
        
        energy_matrix.append(energy)
        if time_axis is None or len(times) > len(time_axis):
            time_axis = times

    # Ensure uniform shape for energy_matrix
    max_len = max(len(energy) for energy in energy_matrix)
    energy_matrix = np.array([np.pad(energy, (0, max_len - len(energy)), constant_values=np.nan) for energy in energy_matrix])

    # Plot Section + Energy Heatmap
    figure, ax = plt.subplots(figsize=(size[0] // 100, size[1] // 100))
    im = ax.imshow(energy_matrix.T, aspect='auto', cmap='hot', extent=[min(dist_sectplot), max(dist_sectplot), max(time_axis), min(time_axis)])

    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Time (s)")
    ax.set_title("Record Section with Energy")
    plt.colorbar(im, label="Normalized Energy")

    if outfile:
        plt.savefig(outfile)
    elif show:
        plt.show()
    
    return figure





import folium
import numpy as np
import branca.colormap as cm




def plot_stations_amps_lines(eq_list, amin, amax, wavecode='Lg_Coda', origin=[0, 0], zoom=4, color="red", geom=False, 
                       geompower=0.5, normQ=False, Q=1000, f0=1, v=3.4, forcescale=False, outfile=None):
    """
    Plots seismic station amplitudes on a folium map.
    """

    dir = '/home/schreinl/Stage/Data/'
    all_maps = []  # Store maps if multiple events exist

    for event in eq_list:
        time_string = UTCDateTime.strftime(event, format="%Y_%m_%dT%H_%M_%S")

        with open(f'{dir}{time_string}/{time_string}_stations_with_amps.txt', "r") as file:
            stations_with_amps = json.load(file)

        stations_with_amps = np.array(stations_with_amps)

        plot_amp = select_ratio(wavecode, stations_with_amps)
        distwork = stations_with_amps[:, 5].astype(float) / 1000. 

        if geom:
            A0 = 1. / np.power(200., geompower)
            dist_power = A0 * np.power(distwork, geompower)
        else:
            dist_power = np.ones_like(distwork)

        if normQ:
            multexp = np.pi * (f0 / v / Q)
            dist_exp = np.exp(multexp * distwork)
        else:
            dist_exp = np.ones_like(distwork)

        plot_amp = np.multiply(plot_amp, dist_power)
        plot_amp = np.multiply(plot_amp, dist_exp)

        vmin = amin if forcescale else amin * np.min(plot_amp)
        vmax = amax if forcescale else amax * np.max(plot_amp)
        linear = cm(["green", "yellow", "red"], vmin=vmin, vmax=vmax)

        carte = folium.Map(location=origin, zoom_start=zoom)

        for istat in range(len(stations_with_amps)):
            lat, lon = stations_with_amps[istat, 2], stations_with_amps[istat, 3]
            color_value = linear(plot_amp[istat])

            folium.PolyLine(
                locations=[origin, [lat, lon]],
                color=color_value,
                weight=1,
                opacity=1.0
            ).add_to(carte)

        folium.CircleMarker(
            location=origin,
            radius=5,
            color='blue',
            fill=True,
            fill_color="#FF8C00",
        ).add_to(carte)

        if outfile:
            carte.save(outfile)

        all_maps.append(carte)

    return all_maps if len(all_maps) > 1 else all_maps[0]




def plot_stations_amps_lines_old(eq_list, amin, amax,wavecode='Lg_Coda', origin=[0, 0], zoom=4, color="red", geom=False, 
                       geompower=0.5, normQ=False, Q=1000, f0=1, v=3.4, forcescale=False, outfile=None):
    '''
    can only be run when station_with_amps is written to file
    call like this:
                        plot =plot_stations_amps_lines(stations_with_amps, 0.8, 0.3, Amp_Draw, origin=[eq_lat,eq_lon], zoom=5, forcescale=False)
                        plot
    reading in, or supporting with stations_with_amps, implement reading in from disk
    
    '''
    dir = '/home/schreinl/Stage/Data/'
    for event in eq_list:
        time_string = UTCDateTime.strftime(event, format="%Y_%m_%dT%H_%M_%S")
        with open(f'{dir}{time_string}_stations_with_amps.txt', "r") as file:
            stations_with_amps = json.load(file)
        
        stations_with_amps_list = stations_with_amps.tolist()    

        plot_amp = select_ratio(wavecode, stations_with_amps)
        distwork = stations_with_amps[:, 5].astype(float) / 1000.  # Distance in kilometers

        # Apply geometric and normQ adjustments to the amplitude
        if geom:
            A0 = 1. / np.power(200., geompower)
            dist_power = A0 * np.power(distwork, geompower)
        else:
            dist_power = 1 + 0. * distwork

        if normQ:
            multexp = np.pi * (f0 / v / Q)
            dist_exp = np.exp(multexp * distwork)
        else:
            dist_exp = 1 + 0. * distwork

        plot_amp = np.multiply(plot_amp, dist_power)
        plot_amp = np.multiply(plot_amp, dist_exp)

        linear = cm.LinearColormap(["green", "yellow", "red"], vmin=amin * min(plot_amp), vmax=amax * max(plot_amp))
        if forcescale:
            linear = cm.LinearColormap(["green", "yellow", "red"], vmin=amin, vmax=amax)

        carte = folium.Map(location=origin, zoom_start=zoom)
        
        for istat, (net, sta, lat, lon, elev, dist, az, t_Pn, t_Sn, t_Pg, A_Pn, A_Sn, A_Lg, A_Coda, A_Noise, A_Pg) in enumerate(stations_with_amps_list):
            color_value = linear(plot_amp[istat])

            folium.PolyLine(
                locations=[origin, [lat, lon]],
                color=color_value,
                weight=1,
                opacity=1.0
            ).add_to(carte) 

        folium.CircleMarker(
            location=origin,
            radius=5,
            color='blue',
            fill=True,
            fill_color="#FF8C00",
        ).add_to(carte)

        if outfile:
            carte.save(outfile)

    return carte




def plot_amplitudes_distance(station_with_amps):
    Lg = select_ratio('Lg', station_with_amps)
    Pg = select_ratio('Pg', station_with_amps)
    Pn = select_ratio('Pn', station_with_amps)
    Sn = select_ratio('Sn', station_with_amps)
    dist = station_with_amps[:, 5].astype(float) / 1000.

    plt.figure(figsize=(10,10))
    plt.scatter(dist, Lg, label='Lg')
    plt.scatter(dist, Pg, label='Pg')
    plt.scatter(dist, Pn, label='Pn')
    plt.scatter(dist, Sn, label='Sn')
    plt.legend()
    plt.show()
    return




import json
import pandas as pd
from obspy.clients.fdsn.header import FDSNNoDataException



def processing(datacenters=['RESIF','ODC','ETH','INGV','GEOFON','BGR', 'IRIS', 'ICGC'], catalogue_file='/home/schreinl/Stage/Scripts/europe_bigger_than_5.csv',
                distmin=1.9, distmax=10.,Dtmin_Noise=-25,Dtmax_Noise=-5,Dtmin_Pn=-5.,Dtmax_Pn=10.,Dtmin_Sn=-5.,Dtmax_Sn=10.,
                vLg_max=3.5, vLg_min=3.1, vPg_max=6.2, snr_threshold = 2 ,vPg_min=5.2, directory='/home/schreinl/Stage/Data/', fmin=3, fmax=6,
                plot_SNR=False,plot_amps=True, wavecode="Lg_Coda",dB=True):
    catalogue = pd.read_csv(catalogue_file) 
    vLg=0.5*(vLg_max+vLg_min)
    vPg=0.5*(vPg_max+vPg_min)
    for i in range(len(catalogue)):
        print(f'Processing earthquake {i+1} out of {len(catalogue)}')

        try:
            start = UTCDateTime(catalogue['time'][i])
            eq_start = start
            end = start + 400
            eq_lon = float(catalogue['longitude'][i])
            eq_lat = float(catalogue['latitude'][i])

            # Start downloading routine
            st_all, stations_all, plot = big_downloader2(datacenters, start, end, eq_lon, eq_lat, distmin, distmax, directory, plot=False)

        except FDSNNoDataException:
            print(f"No data available for earthquake {i+1}, skipping...")
            continue  # Skip to the next earthquakef0=0.5*(fmin+fmax)
        time_string = UTCDateTime.strftime(start, format="%Y_%m_%dT%H_%M_%S")
        st_plot_filt_all=st_all.copy()
        st_plot_filt_all.filter("bandpass", freqmin=fmin, freqmax=fmax)
        #run SNR and station filtering routine
        filtered_stations_with_SNR, stations_with_SNR, distance_dict, tcoda_test, filtered_st, stations_with_amps, amp_plot = SNR_all(stations_all, st_plot_filt_all, Dtmin_Pn, Dtmax_Pn, Dtmin_Sn, Dtmax_Sn, vLg_min,vLg_max,vPg_min, vPg_max, tminCoda, tmaxCoda,
            Dtmin_Noise, Dtmax_Noise,eq_start,eq_lat,eq_lon,snr_threshold=snr_threshold,plot_SNR=plot_SNR,plot_amps=plot_amps, wavecode=wavecode,dB=dB)

        with open(f"{directory}/Dicts/{time_string}_{snr_threshold}thresh_{fmin}_{fmax}Hz_dict.txt", "w") as file:
            json.dump(distance_dict, file, indent=4)
        
        # Create plot of azimuth dependency of SNR
        #10: 'Pn', 11: 'Pg', 12: 'Sn', 13: 'Lg'
        #snr_az_sn = snr_azimuth(stations_with_SNR, column=12, barlabel="SNR", xlabel="Azimuth (°)", ylabel="Distance (km)", title="SNR", event_name=time_string, savefig=True, show=False)
        #snr_az_pn = snr_azimuth(stations_with_SNR, column=10, barlabel="SNR", xlabel="Azimuth (°)", ylabel="Distance (km)", title="SNR", event_name=time_string, savefig=True, show=False)
        #snr_az_pg = snr_azimuth(stations_with_SNR, column=11, barlabel="SNR", xlabel="Azimuth (°)", ylabel="Distance (km)", title="SNR", event_name=time_string, savefig=True, show=False)
        #snr_az_lg = snr_azimuth(stations_with_SNR, column=13, barlabel="SNR", xlabel="Azimuth (°)", ylabel="Distance (km)", title="SNR", event_name=time_string, savefig=True, show=False)

        # Save stations_with_amps to a file
        with open(f"{directory}/{time_string}/{time_string}_{snr_threshold}_thresh_stations_with_amps.txt", "w") as ampls:
            json.dump(stations_with_amps.tolist(), ampls, indent=4)
        
        # Save filtered stations with their corresponding SNR
        with open(f"{directory}/{time_string}/{time_string}_{snr_threshold}thresh_{fmin}_{fmax}Hz_filtered_stations_SNR.txt", "w") as snrfile:
            json.dump(filtered_stations_with_SNR.tolist(), snrfile, indent=4)
        
        # Save the stations with SNR, unfiltered
        with open(f"{directory}/{time_string}/{time_string}_{fmin}_{fmax}Hz_unfiltered_stations_SNR.txt", "w") as unsnrfile:
            json.dump(stations_with_SNR.tolist(), unsnrfile, indent=4)

        




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




def update_event_file(file_path, event_name, station_data):
    """
    Update a CSV file with station-event data.
    
    :param file_path: Path to the CSV file.
    :param event_name: Name of the event (prefix for column names).
    :param station_data: Dictionary with station names as keys and tuples of values for the event.
    """
    # Load existing data if the file exists
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, index_col=0)
    else:
        df = pd.DataFrame()

    col1 = f"{event_name}_coda"
    col2 = f"{event_name}_envelope"
    
    for col in [col1, col2]:
        if col not in df.columns:
            df[col] = 0.0 
    for station in station_data.keys():
        if station not in df.index:
            df.loc[station] = [0.0] * len(df.columns)

    df = df.astype({col1: float, col2: float})

    for station, (value1, value2) in station_data.items():
        df.at[station, col1] = float(value1)
        df.at[station, col2] = float(value2)

    df.to_csv(file_path)
    print(f"Updated {file_path} with event '{event_name}'.")