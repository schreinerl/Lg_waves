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




'''


def get_Pn_time(dist_deg, velocity=6.9):
    t_Pn = 111*dist_deg / velocity +25
    return t_Pn


def get_Sn_time(dist_deg, velocity=4):
    t_Sn = 111 * dist_deg / velocity   + 25
    return t_Sn





def get_Pn_time(dist_deg, velocity=7):
    from obspy.taup import TauPyModel

    model = TauPyModel(model='ak135')
    t_Pn = 111. * dist_deg / 8.0  # Default calculation

    arrivals = model.get_travel_times(source_depth_in_km=0,
                                      distance_in_degree=dist_deg, phase_list=["Pn"])

    try:
        t_Pn = arrivals[0].time
    except IndexError as e:
        if velocity:
            t_Pn = dist_deg / velocity
        else:
            t_Pn = dist_deg / 8.0 

    return t_Pn


def get_Sn_time(dist_deg, velocity=4):
    from obspy.taup import TauPyModel

    model = TauPyModel(model='ak135')
    t_Sn = 111. * dist_deg / 8.0 

    arrivals = model.get_travel_times(source_depth_in_km=0,
                                      distance_in_degree=dist_deg, phase_list=["Sn"])

    try:
        t_Sn = arrivals[0].time
    except IndexError as e:
        if velocity:
            t_Sn = dist_deg / velocity  
        

    return t_Sn

'''

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

import os
from obspy import read, UTCDateTime
from time import sleep
import sys
from tqdm.auto import tqdm

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
    t_Pg_plot=t_Pg_work[dist_index]
    t_Lg_min_plot=dist_sectplot/v_Lg_max +25
    t_Lg_max_plot=dist_sectplot/v_Lg_min + 25
    #t_Pg_plot=dist_sectplot/v_Pg
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



def site_effect_old(eq_file='/home/schreinl/Stage/Data/eq_4_france.csv', codafile='envelope_amps_fac_1.5_dict',sample_size=25,ref_station1 = "SSB", ref_station2 = "ECH", ratio_plot = False):
    '''
    Function calculates the site effect for each station by building the median amplitude ratio of each station to the reference station.
    For each event and for each station the site effect is calculated.
    output should be one large
    Inputs: earthquakes in a list, codafile which is the ending of the files that contain the amplitudes
    Output: a dictionary with the site effect for each station

    '''

    #read in the earthquake list
    eq_list = pd.read_csv(eq_file)

    #create the dictionary
    amplitudes_dict = {}

    #fill the dict with the amplitudes, so for a specific coda window file the amplitudes are read in
    for i in range(len(eq_list)):
        start = UTCDateTime(eq_list["time"][i])
        time_string = UTCDateTime.strftime(start, format="%Y_%m_%dT%H_%M_%S")
        
        try:    #/home/schreinl/Stage/Data/{time_string}/{time_string}_envelope_amps_fac_1.5_dict.txt
            with open(f"/home/schreinl/Stage/Data/{time_string}/{time_string}_{codafile}.txt", "r") as file:
                amp_dict = json.load(file)

            for station, data in amp_dict.items():
                if station not in amplitudes_dict:
                    amplitudes_dict[station] = [None] * i 
                amplitudes_dict[station].append(data["amplitude"])
            
            for station in amplitudes_dict:
                if station not in amp_dict:
                    amplitudes_dict[station].append(None)
        except FileNotFoundError:
            for station in amplitudes_dict:
                amplitudes_dict[station].append(None)

        #so now we have a dictionary with the amplitudes for each station for each event, so dimensions are N number of events given,
        # and M number of stations that have recorded at least one event

    # if we wanted to test what we did, we use a smaller batch size of stations
    sample_size = min(sample_size, len(amplitudes_dict.keys()))
    station_tests = random.sample(list(amplitudes_dict.keys()), sample_size)

    #here we choose the reference stations
    station2 = ref_station1
    fallback_station = ref_station2
    amplitude_ratios = {}

    # Counters for stations with and without valid ratios
    total_stations = len(amplitudes_dict.keys())
    stations_with_no_valid_ratio = 0

    for station1 in amplitudes_dict.keys():
        denom_amplitudes = np.array(amplitudes_dict[station1], dtype=np.float64)
        counter_amplitudes = np.array(amplitudes_dict[station2], dtype=np.float64)

        # Replace None with np.nan
        denom_amplitudes = np.where(denom_amplitudes == None, np.nan, denom_amplitudes)
        counter_amplitudes = np.where(counter_amplitudes == None, np.nan, counter_amplitudes)

        # Check if the reference station has missing values and use fallback if necessary
        
        # Check if the reference station has missing values and use fallback if necessary
        if np.isnan(counter_amplitudes).all():
            print(f"Using fallback station {fallback_station} for station {station1}")
            counter_amplitudes = np.array(amplitudes_dict[fallback_station], dtype=np.float64)
            counter_amplitudes = np.where(counter_amplitudes == None, np.nan, counter_amplitudes)

        # Calculate the ratio
        ratio = np.divide(denom_amplitudes, counter_amplitudes, out=np.full_like(denom_amplitudes, np.nan), where=(counter_amplitudes != 0))
        valid_ratios = ratio[~np.isnan(ratio)]
        
        if len(valid_ratios) > 0:
            amplitude_ratios[station1] = {
                "median": np.nanmedian(valid_ratios),
                "std": np.nanstd(valid_ratios),
            }
        else:
            amplitude_ratios[station1] = {
                "median": np.nan,
                "std": np.nan,
            }
            #print(f"No valid ratio computed for station {station1}")
            stations_with_no_valid_ratio += 1

    # Print the number of stations with no valid ratio and the total number of stations
    print(f"Total number of stations: {total_stations}")
    print(f"Number of stations with no valid ratio: {stations_with_no_valid_ratio}")

    if ratio_plot == True:
        station_tests = random.sample(list(amplitudes_dict.keys()), 25)
        median_values = [amplitude_ratios[st]["median"] for st in station_tests]
        std_values = [amplitude_ratios[st]["std"] for st in station_tests]

        plt.figure(figsize=(8, 5))
        plt.errorbar(station_tests, median_values, yerr=std_values, fmt='o', capsize=5, markersize=8, color="b", label="Median ± Std Dev")

        plt.xlabel("Station")
        plt.ylabel(f"Median Amplitude Ratio ({station2} Reference)")
        plt.title(f"Median Amplitude Ratio with Standard Deviation")
        plt.legend()
        plt.ylim(0, 20)
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.xticks(rotation=45)
        plt.show()

    return amplitude_ratios




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





from scipy.fftpack import hilbert
import obspy
def envelope_calculator(data):
    hilb = hilbert(data)
    data = (data ** 2 + hilb ** 2) ** 0.5
    return data





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



from scipy.signal import savgol_filter
from random import randint
def smooth_plot_envelope(time_string, n_traces,st_envelope, method='Cutoff distance',tmincoda_dist=442, tmaxcoda_dist=462,tmincoda_S = None, tmaxcoda_S= None,plotshow=False, savefig=True):
    testing = [(randint(1, len(st_envelope))) for i in range(n_traces)]

    plt.figure(figsize=(10,10))
    for i in testing:
        if i < len(st_envelope):
            npts = len(st_envelope[i].data)
            samprate = st_envelope[i].stats.sampling_rate #10000/700 
            t = np.arange(0, npts / samprate, 1 / samprate)

            #use a stable window length in s, while the window lenght in samples is dependant on the sample rate
            
            window_length = min(50*samprate, npts) # Ensure window_length is not greater than npts
            if window_length % 2 == 0:
                window_length -= 1
            yhat = savgol_filter(st_envelope[i].data, int(window_length), 3) 
            t = t[:len(yhat)]

            #plt.semilogy(t,st_envelope[i])
            plt.semilogy(t,yhat, color='red')
            plt.ylim([1e-9,1e-4])
            plt.title(f"Vertical component envelope {time_string}")
            plt.ylabel("Energy")
            plt.xlabel("Time (s)")
            #plt.xlim([100,350])
    plt.vlines(tmaxcoda_dist,ymax=1e-4,ymin=1e-9, label=f'coda window calculated with {method}',colors='g',linestyles='--')
    plt.vlines(tmincoda_dist,ymax=1e-4,ymin=1e-9,colors='g',linestyles='--')
    if tmincoda_S is not None and tmaxcoda_S is not None:
        plt.vlines(tmaxcoda_S,ymax=1e-4,ymin=1e-9, label='coda window calculated with S-wave',colors='b',linestyles='--')
        plt.vlines(tmincoda_S,ymax=1e-4,ymin=1e-9,colors='b',linestyles='--')
    #plt.vlines(350,ymax=1e-4,ymin=1e-9, label='coda window similar to Galina&Shapiro 2024',colors='b',linestyles='--')
    #plt.vlines(470,ymax=1e-4,ymin=1e-9,colors='b',linestyles='--')
    plt.legend()
    if savefig:
        mag_dir = '/home/schreinl/Stage/Data/Metadata/'
        with open(f"{mag_dir}{time_string}.txt", "r") as meta:
            for line in meta:
                if line.startswith("Magnitude:"):
                    magnitude = float(line.split(":")[1].strip())
        plt.savefig(f'/home/schreinl/Stage/Figures/SiteEffect/multiple_envelopes_{time_string}_{magnitude}_{method}_filtered.png', format='png')
    if plotshow:
        plt.show()
    
    return




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



def envelope_processing(st,stations,time_string,station_ref,filtered_stations_with_SNR,Dtmin_Pn, Dtmax_Pn, Dtmin_Sn, Dtmax_Sn, vLg_min,vLg_max,vPg_min,vPg_max, tcoda_start, coda_duration, Dtmin_Noise, Dtmax_Noise, eq_start):
    '''

    '''
    
    st_envelope = obspy.Stream()
    smallest = 7000
    for tr in st:
        data_envelope = envelope_calculator(tr.data)
        npts = tr.stats.npts
        if npts >= smallest:
            samprate = tr.stats.sampling_rate
            t = np.arange(0, npts / samprate, 1 / samprate)
            tr_envelope = obspy.Trace(data=data_envelope, header=tr.stats)
            st_envelope.append(tr_envelope)
    print("calculated Envelopes")
    amplitudes_full = calc_amps(stations, st, Dtmin_Pn, Dtmax_Pn, Dtmin_Sn, Dtmax_Sn, vLg_min,vLg_max,vPg_min,vPg_max, tcoda_start, tcoda_start+coda_duration, Dtmin_Noise, Dtmax_Noise, eq_start)
    amplitudes_small = calc_amps(stations, st, Dtmin_Pn, Dtmax_Pn, Dtmin_Sn, Dtmax_Sn, vLg_min,vLg_max,vPg_min,vPg_max, tcoda_start, tcoda_start+coda_duration, Dtmin_Noise, Dtmax_Noise, eq_start)
    SNR_dict = select_ratio_dict("Coda_Noise", amplitudes_full)
    SNR_dict_small = select_ratio_dict("Coda_Noise", amplitudes_small)
    envelopes_amps, st_smooth = envelopes_routine1(time_string, st_envelope, coda_dist_start=tcoda_start, coda_dist_end=tcoda_start + coda_duration, plotting=False, method='cutoff', snr=SNR_dict, snr_window=SNR_dict_small)
    filtered_station_names = set(row[1] for row in filtered_stations_with_SNR)
    filtered_smooth_stream = obspy.Stream()
    filtered_stream = obspy.Stream()
    for trace in st_smooth:
        if trace.stats.station in filtered_station_names:
            filtered_smooth_stream.append(trace)
    for trace in st_envelope:
        if trace.stats.station in filtered_station_names:
            filtered_stream.append(trace) 
    print('filtered out traces with large distance or weak SNR')

    stationref = station_ref
    for i,tr in enumerate(filtered_stream):
        if tr.stats.station == stationref:
            idx_reference_filt = i

    for i,tr in enumerate(st_envelope):
        if tr.stats.station == stationref:
            idx_reference = i

    valid = 0
    invalid = 0
    st_double_filt = obspy.Stream()
    for i in range(len(filtered_stream)):
        station = filtered_stream[i].stats.station
        if station in envelopes_amps:
            if envelopes_amps[station]['snr_coda'] > 2 and envelopes_amps[station]['snr_coda_end'] > 2:
                valid += 1
                st_double_filt.append(filtered_stream[i])
            else:
                invalid += 1
        else:
            invalid += 1
    print('after filtering out stations with weak SNR in the coda window:')
    print(valid, "valid stations")
    print(invalid, "invalid stations")

    return filtered_stream, st_envelope, envelopes_amps, idx_reference,idx_reference_filt, st_double_filt


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









def get_color(value):
    """
    Returns an RGB color based on a linear color scale:
    - Blue to White for [0.1, 1]
    - White to Red for [1, 10]
    """
    import matplotlib.colors as mcolors
    vmin, vmid, vmax = 0.1, 1, 10
    color_low = np.array(mcolors.to_rgb("blue")) 
    color_mid = np.array(mcolors.to_rgb("white"))
    color_high = np.array(mcolors.to_rgb("red")) 
   
    if value <= vmin:
        return mcolors.to_hex(color_low)
    elif vmin < value <= vmid:
        t = (value - vmin) / (vmid - vmin)
        color = (1 - t) * color_low + t * color_mid
    elif vmid < value <= vmax:
        t = (value - vmid) / (vmax - vmid)
        color = (1 - t) * color_mid + t * color_high
    else:
        return mcolors.to_hex(color_high)
    
    return mcolors.to_hex(color)



def get_color_logstep(value):
        import matplotlib.colors as mcolors
        if value <= 0: 
            return "blue"
        
        log_val = np.log10(value) 
        norm_val = (log_val - np.log10(0.1)) / (np.log10(10) - np.log10(0.1)) 

        colormap = mcolors.LinearSegmentedColormap.from_list("log_colormap", ["blue", "white", "red"])
        
        return mcolors.to_hex(colormap(norm_val))

def color_scale(min_value, max_value, num_steps=100):
    return [get_color_logstep(value) for value in np.logspace(np.log10(min_value), np.log10(max_value), num_steps)]






import glob
from branca.colormap import StepColormap

def map_site_effect(fmin, fmax, site_effect_medians, method='multiple'):
    event_map = folium.Map(location=[46.2145, -0.7295], zoom_start=5)
    data_directory = '/home/schreinl/Stage/Data1/'
    metadata_directory = os.path.join(data_directory, "Metadata/")
    event_locations = []
    
    for metadata_file in os.listdir(metadata_directory):
        metadata_path = os.path.join(metadata_directory, metadata_file)
        if os.path.isfile(metadata_path):
            try:
                with open(metadata_path, 'r') as file:
                    lines = file.readlines()
                    lat, lon = None, None
                    
                    for line in lines:
                        if line.startswith("Latitude:"):
                            lat = float(line.split(":")[1].strip())
                        elif line.startswith("Longitude:"):
                            lon = float(line.split(":")[1].strip())
                    
                    if lat is not None and lon is not None:
                        event_locations.append((lat, lon))
            except Exception as e:
                print(f"Error reading {metadata_path}: {e}")
    
    plotted_stations = {}
    
    for event_dir in os.listdir(data_directory):
        event_path = os.path.join(data_directory, event_dir)
        if os.path.isdir(event_path):
            file_pattern = os.path.join(event_path, f'*_thresh_stations_with_amps.txt')
            matching_files = glob.glob(file_pattern)
            
            for file_path in matching_files:
                try:
                    with open(file_path, 'r') as file:
                        data = json.load(file)
                        
                        for station in data:
                            if len(station) >= 4:
                                network = station[0]
                                name = station[1]
                                lat = float(station[2])
                                lon = float(station[3])
                                station_id = (lat, lon)
                                
                                plotted_stations[station_id] = (network, name)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
    
    print('done reading in')
    
    log_values = np.logspace(np.log10(0.1), np.log10(10), num=100)
    step_values = np.logspace(np.log10(0.1), np.log10(10), num=120)  
    step_colors = [get_color_logstep(value) for value in step_values]
    
    colormap = StepColormap(
        step_colors, 
        vmin=0.1, 
        vmax=10, 
        index=step_values
    )
    colormap.add_to(event_map)
    colormap.caption = f"Median Site Effect ({fmin}-{fmax}Hz)"
    
    for (lat, lon), (network, name) in plotted_stations.items():
        if method == 'single':
            station_medians = site_effect_medians
        elif method == 'multiple':
            station_medians = site_effect_medians.get(f'{fmin}-{fmax}Hz', {}).get(name, {})
        
        median_before = station_medians.get('median_after', None)
        std = station_medians.get('std_after', None)
        numpoints = station_medians.get('num_points_after', None)
        
        if numpoints is not None and not (np.isnan(numpoints) or numpoints == 0):
            coloroutside = 'red' if numpoints <= 2 else 'green'
            if median_before is not None:
                color = get_color(median_before)
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=6,
                    tooltip=f"{network}.{name} ({lat}, {lon}), Median: {median_before}, std: {std}, Points: {numpoints}",
                    color=coloroutside,
                    weight=1,
                    fill=True,
                    fill_color=color,
                    fill_opacity=1
                ).add_to(event_map)
    
    return event_map


def map_site_effect_old(fmin,fmax,site_effect_medians,method='multiple'):
    import glob
    event_map = folium.Map(location=[46.2145, -0.7295], zoom_start=5)
    data_directory = '/home/schreinl/Stage/Data1/'
    metadata_directory = os.path.join(data_directory, "Metadata/")
    event_locations = []
    for metadata_file in os.listdir(metadata_directory):
        metadata_path = os.path.join(metadata_directory, metadata_file)
        if os.path.isfile(metadata_path):
            try:
                with open(metadata_path, 'r') as file:
                    lines = file.readlines()
                    lat, lon = None, None
                    
                    for line in lines:
                        if line.startswith("Latitude:"):
                            lat = float(line.split(":")[1].strip())
                        elif line.startswith("Longitude:"):
                            lon = float(line.split(":")[1].strip())
                    
                    if lat is not None and lon is not None:
                        event_locations.append((lat, lon))
            except Exception as e:
                print(f"Error reading {metadata_path}: {e}")


    plotted_stations = {}

    for event_dir in os.listdir(data_directory):
        event_path = os.path.join(data_directory, event_dir)
        if os.path.isdir(event_path):
            file_pattern = os.path.join(event_path, f'*_thresh_stations_with_amps.txt')
            #file_pattern = os.path.join(event_path, f'*_new_cutoff_fac_1.1_{fmin}_{fmax}Hz_dict.txt')
            matching_files = glob.glob(file_pattern)
            
            for file_path in matching_files:
                try:
                    with open(file_path, 'r') as file:
                        data = json.load(file)
                        
                        for station in data:
                            if len(station) >= 4:
                                network = station[0]
                                name = station[1]
                                lat = float(station[2])
                                lon = float(station[3])
                                station_id = (lat, lon)
                                
                                plotted_stations[station_id] = (network, name)

                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

    print('done reading in')

    from branca.colormap import StepColormap

    log_values = np.logspace(np.log10(0.1), np.log10(10), num=100)
    color_list = [get_color_logstep(value) for value in log_values]

    step_values = np.logspace(np.log10(0.1), np.log10(10), num=120)  
    step_colors = [get_color_logstep(value) for value in step_values]

    colormap = StepColormap(
        step_colors, 
        vmin=0.1, 
        vmax=10, 
        index=step_values
    )

    colormap.add_to(event_map)
    colormap.caption = f"Median Site Effect ({fmin}-{fmax}Hz)"

    for (lat, lon), (network, name) in plotted_stations.items():
        if method == 'single':
            station_medians = site_effect_medians
            
        elif method == 'multiple':
            station_medians = site_effect_medians.get(f'{fmin}-{fmax}Hz', {}).get(name, {})
        median_before = station_medians.get('median_after', None)
        std = station_medians.get('std_after', None)
        numpoints = station_medians.get('num_points_after', None)

        if numpoints is not None and numpoints <= 2:
            coloroutside = 'red'
        elif numpoints is not None and numpoints > 2:
            coloroutside = 'green'
        if median_before is not None:
            color = get_color(median_before)
            folium.CircleMarker(
                location=[lat, lon],
                radius=6,
                tooltip=f"{network}.{name} ({lat}, {lon}), Median: {median_before}, std: {std}, Points: {numpoints}",
                color=coloroutside,
                weight=1,
                fill=True,
                fill_color=color,
                fill_opacity=1
            ).add_to(event_map)

    return event_map



