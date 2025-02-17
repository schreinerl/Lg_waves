import numpy as np
import matplotlib.pyplot as plt
import folium
from obspy import UTCDateTime
from obspy import Stream
from obspy import read
from obspy import read_inventory
import os
from obspy.clients.fdsn import Client
from obspy.clients.fdsn import RoutingClient
from obspy.geodetics import gps2dist_azimuth
from obspy import signal

import matplotlib.pyplot as plt
import folium
import numpy as np

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
from obspy import read

def get_data2(client, inventory, start, end, distmin, distmax, directory='/home/schreinl/Stage/Data/'):
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

    for network in inventory:
        for station in network.stations:
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
                        print(f"Reading from disk: {file_path}")
                        st = read(file_path)
                        conv_step = True
                    elif status == "unknown":
                        print(f"Retrying download for {station_channel}")
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
                            print(f"Downloaded: {file_path}")
                            logfile.write(f"{station_channel},downloaded\n")
                        except Exception as e:
                            print(f"Failed to download {station_channel} again: {e}")
                            logfile.write(f"{station_channel},unknown\n")
                            continue
                    else:
                        print(f"Skipping {station_channel} due to previous error: {status}")
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
                        print(f"Downloaded: {file_path}")
                        logfile.write(f"{station_channel},downloaded\n")
                    except Exception as e:
                        if "204" or "404" in str(e):
                            print(f"No data for {station.code}")
                            logfile.write(f"{station_channel},204/404\n")
                        elif "403" in str(e):
                            print(f"Access denied for {station.code}")
                            logfile.write(f"{station_channel},403\n")
                        elif "URLError" in str(e):
                            print(f"URL error for {station.code}")
                            logfile.write(f"{station_channel},URLError\n")
                        else:
                            print(f"Unknown error for {station.code}: {e}")
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
    return st_final_target, stations_target



def big_downloader2(datacenters, start, end, distmin, distmax, directory='/home/schreinl/Stage/Data/',plot=False):
    """
    -input: list of datacenters type str, start and end time of the time window, minimum and maximum distance in degrees
    -function that downloads data from multiple datacenters
    -filters the stations based on their location
    -checks if data is already downloaded and reads it from disk if it is
    -possibility to plot the filtered stations
    -gives back stream with all the data {st_final_target}, as well as a list of stations and their metadata,
      including expected arrival times of Pn and Sn {stations_target}
    """

    #hardcoded for now the box values of stations
    minlat_st=40.
    maxlat_st=52
    minlon_st=-5            
    maxlon_st=16
    st_final_target = None
    logfile_combined = []
    st_final_stations = []
    #create inventory for each datacenter
    for i in range(len(datacenters)):
        client = Client(datacenters[i])
        logfile_center = []
        inventory=client.get_stations(
                        network="*",station="*",channel="HHZ,BHZ,SHZ",
                        location="*",starttime=start,endtime= end,
                        minlatitude=minlat_st,maxlatitude=maxlat_st,
                        minlongitude=minlon_st,maxlongitude=maxlon_st,
                        level="response")
        #get data for each datacenter
        st_final_center, stations_center = get_data2(client, inventory, start, end, distmin, distmax, directory='/home/schreinl/Stage/Data/')
        if st_final_target is None:
            st_final_target = st_final_center
        else:
            st_final_target += st_final_center
        st_final_stations += stations_center
    #plotting option
    if plot:
        filtered_stations = folium.Map(location=[eq_lat,eq_lon],zoom_start=5)
            
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




def plot_record_section(
    st, stations, eq_lat, eq_lon, eq_start, size=(1200, 1000), show=True, outfile=None, tracehodo=True, v_Lg_min=3.1, v_Lg_max=3.5, 
v_Pg=6.):

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
    t_Pg_plot_tauP = t_Pg_work[dist_index]

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
        plt.plot(dist_sectplot, t_Pg_plot_tauP, color='k', linestyle='dashed',linewidth=1,label='Pg_tauP')
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
         Dtmin_Noise, Dtmax_Noise,fmin=3, fmax=6, method='time_amplitude',signal_window='coda',plot_map=False):

    
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
                        snr = 10 * np.log10(signal_power / noise_power)

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
                        snr = 10 * np.log10(signal_power / noise_power)

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
                        snr = 10 * np.log10(signal_power / noise_power)
                    
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
                        snr = 10 * np.log10(signal_power / noise_power)

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
                        snr = 10 * np.log10(signal_power / noise_power)
                        
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
                 Dtmin_Noise, Dtmax_Noise, fmin=3, fmax=6):
    phases = ['coda', 'Pn', 'Sn', 'Lg', 'Pg']
    phase_distance = {}
    
    fig, axs = plt.subplots(3, 2, figsize=(15, 15))
    plt.style.use('seaborn-v0_8')
    for i, phase in enumerate(phases):
        stations_with_SNR = SNR(stations, st, Dtmin_Pn, Dtmax_Pn, Dtmin_Sn, Dtmax_Sn, vLg_min, vLg_max, vPg_min, vPg_max,
                                tmin_Coda, tmax_Coda, Dtmin_Noise, Dtmax_Noise, fmin, fmax, method='time_amplitude', signal_window=phase, plot_map=False)
        SNR_vals = stations_with_SNR[:, -1].astype(float)
        dist_vals = stations_with_SNR[:, 5].astype(float) / 1000. 
        #SNR_vals = SNR_vals[np.isfinite(SNR_vals)]
        #dist_vals = dist_vals[np.isfinite(SNR_vals)]
        
        threshold = 1.5
        snr_threshold = 0.9


        filtered_distances = dist_vals[SNR_vals > snr_threshold]

        if len(filtered_distances) > 0:
            percentile_distance = np.percentile(filtered_distances, 90)
            phase_distance[phase] = percentile_distance
            print(f"Distance where 90% of SNR values are above 1.5: {percentile_distance}")
        else:
            print("No valid SNR values above 1.5.")

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
        ax.set_xlabel('Distance (km)')
        ax.vlines(percentile_distance, ymin=min(SNR_vals), ymax=max(SNR_vals), color='r', linestyle='dashed', label='90th percentile')
        ax.legend(loc='upper right')
        ax.set_ylim(-20, 70)
        ax.set_ylabel('SNR (dB)')
        ax.set_title(f'{event_name} SNR with {SNR_method} method and {phase} phase')
    plt.tight_layout()
    plt.show()
    return phase_distance 
    

def calc_amps(stations, st, Dtmin_Pn, Dtmax_Pn, Dtmin_Sn, Dtmax_Sn, vLg_min,vLg_max,vPg_min,vPg_max, tmin_Coda, tmax_Coda, Dtmin_Noise, Dtmax_Noise):


    stations_amplitudes=[]
    for net, sta, lat, lon, elev , dist, az, t_Pn, t_Sn, t_Pg  in stations:
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
                tminPg = dist/1000/vPg_max
                tmaxPg = dist/1000/vPg_min
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


    stations_with_amps=np.append(np.array(stations),np.array(stations_amplitudes),axis=1)

    return stations_with_amps
        
               

from obspy.signal.filter import bandpass

def spectral_ratios(stations, st, Dtmin_Pn, Dtmax_Pn, Dtmin_Sn, Dtmax_Sn, vLg_min,vLg_max,vPg_min, vPg_max, tmin_Coda, tmax_Coda, Dtmin_Noise, Dtmax_Noise, fmin,fmax):
    stations_amplitudes=[]
    
    for net, sta, lat, lon, elev , dist, az, t_Pn, t_Sn, t_Pg  in stations:
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
                tminPg = dist/1000/vPg_max
                tmaxPg = dist/1000/vPg_min
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
                    filtered_dataselectPn = bandpass(dataselectPn, fmin, fmax, df=1/dt, corners=4, zerophase=True)
                    filteredPn_ft = np.fft.fft(filtered_dataselectPn)
                    A_Pn=np.sqrt(np.dot(filteredPn_ft,np.transpose(filteredPn_ft)))/len(filteredPn_ft)
                if (trace_start<tmin_Sn) and (trace_end>tmax_Sn) :
                    iminSn=int((tmin_Sn-trace_start)/dt)
                    imaxSn=int((tmax_Sn-trace_start)/dt)
                    dataselectSn=(datavector[iminSn:imaxSn])
                    filtered_dataselectSn = bandpass(dataselectSn, fmin, fmax, df=1/dt, corners=4, zerophase=True)
                    filteredSn_ft = np.fft.fft(filtered_dataselectSn)
                    A_Sn=np.sqrt(np.dot(filteredSn_ft,np.transpose(filteredSn_ft)))/len(filteredSn_ft)
                if (trace_start<tminLg) and (trace_end>tmaxLg) :
                    iminLg=int((tminLg-trace_start)/dt)
                    imaxLg=int((tmaxLg-trace_start)/dt)
                    dataselectLg=(datavector[iminLg:imaxLg])
                    filtered_dataselectLg = bandpass(dataselectLg, fmin, fmax, df=1/dt, corners=4, zerophase=True)
                    filteredLg_ft = np.fft.fft(filtered_dataselectLg)
                    A_Lg=np.sqrt(np.dot(filteredLg_ft,np.transpose(filteredLg_ft)))/len(filteredLg_ft)
                if (trace_start<tmin_Coda) and (trace_end>tmax_Coda) :
                    iminCoda=int((tmin_Coda-trace_start)/dt)
                    imaxCoda=int((tmax_Coda-trace_start)/dt)
                    dataselectcoda=(datavector[iminCoda:imaxCoda])
                    filtered_dataselectcoda = bandpass(dataselectcoda, fmin, fmax, df=1/dt, corners=4, zerophase=True)
                    filteredcoda_ft = np.fft.fft(filtered_dataselectcoda)
                    A_Coda=np.sqrt(np.dot(filteredcoda_ft,np.transpose(filteredcoda_ft)))/len(filteredcoda_ft)
                if (trace_start<tmin_Noise) and (trace_end>tmax_Noise) :
                    iminNoise=int((tmin_Noise-trace_start)/dt)
                    imaxNoise=int((tmax_Noise-trace_start)/dt)
                    dataselectNoise=(datavector[iminNoise:imaxNoise])
                    filtered_dataselectNoise = bandpass(dataselectNoise, fmin, fmax, df=1/dt, corners=4, zerophase=True)
                    filteredNoise_ft = np.fft.fft(filtered_dataselectNoise)
                    A_Noise=np.sqrt(np.dot(filteredNoise_ft,np.transpose(filteredNoise_ft)))/len(filteredNoise_ft)
                if (trace_start<tminPg) and (trace_end>tmaxPg) :
                    iminPg=int((tminPg-trace_start)/dt)
                    imaxPg=int((tmaxPg-trace_start)/dt)
                    dataselectPg=(datavector[iminPg:imaxPg])
                    filtered_dataselectPg = bandpass(dataselectPg, fmin, fmax, df=1/dt, corners=4, zerophase=True)
                    filteredPg_ft = np.fft.fft(filtered_dataselectPg)
                    A_Pg=np.sqrt(np.dot(filteredPg_ft,np.transpose(filteredPg_ft)))/len(filteredPg_ft)

    
        stations_amplitudes.append([A_Pn, A_Sn, A_Lg, A_Coda, A_Noise,A_Pg])


    stations_with_amps=np.append(np.array(stations),np.array(stations_amplitudes),axis=1)

    return stations_with_amps



def plot_stations_amps(stations_amps, amin, amax, Amp_Draw, origin=[0, 0], zoom=4, color="red", geom=False, 
                       geompower=0.5, normQ=False, Q=1000, f0=1, v=3.4, forcescale=False, outfile=None):
    import branca.colormap as cm

    stations_with_amps_list=stations_with_amps.tolist()    

    
    plot_amp=Amp_Draw
    distwork=stations_with_amps[:,5].astype(float)/1000.
    
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


def routine(cataloguefile, client, start, end, distmin, distmax,Dtmin_Pn, Dtmax_Pn, Dtmin_Sn, Dtmax_Sn, vLg_min,vLg_max,vPg_min,vPg_max, tminCoda, tmaxCoda,
         Dtmin_Noise, Dtmax_Noise, SNR_window ='coda', directory='/home/schreinl/Stage/Data/',fmin=3, fmax=6):
    #handle each earthquake in the catalogue
    catalogue = pd.read_csv(cataloguefile) #input has to be string with the path to the file, which needs to be in .csv format

    for i in range(len(catalogue)):
        start = catalogue['time']
        end = start + 400
        eq_lon = catalogue['longitude']
        eq_lat = catalogue['latitude']
        #start downloading routine, and filter if needed
        st_all, stations_all, plot = big_downloader2(client, start, end, distmin, distmax, directory='/home/schreinl/Stage/Data/', plot=False)
        #then apply a narrow filter, to filter out almost a "single frequency", then the processing is done for each frequency band
        for freqmin, freqmax in zip(fmin, fmax):
            st_plot_filt_all=st_all.copy()
            st_plot_filt_all.filter("bandpass", freqmin=freqmin, freqmax=freqmax)
            #then calculate the SNR, in order for that specify in the arguments the phase that should be used for calculation
            Stations_with_SNR = SNR(stations_all, st_plot_filt_all, Dtmin_Pn, Dtmax_Pn, Dtmin_Sn, Dtmax_Sn, vLg_min,vLg_max,vPg_min,vPg_max, tminCoda, tmaxCoda,
            Dtmin_Noise, Dtmax_Noise, method='time_amplitude',signal_window=SNR_window,plot_map=False)
            #remove all stations that have a snr lower than 1.5, write in filtered_stations
            for i, snr in enumerate(Stations_with_SNR[:,10].astype(float)):
                if snr < 1.5:
                    filtered_statios = np.delete(stations_all, i, 0)
                    st_plot_filt_all.remove(st_plot_filt_all[i])

            #now calculate the amplitudes
            stations_with_amps = calc_amps(filtered_statios, st_plot_filt_all, Dtmin_Pn, Dtmax_Pn, Dtmin_Sn, Dtmax_Sn, vLg_min,vLg_max,vPg_min,vPg_max, tminCoda, tmaxCoda, Dtmin_Noise, Dtmax_Noise)
            #we divide the amplitude of the Lg wave by the amplitude of the coda, thus getting rid of the site effect of each station
            Lg_Coda = np.divide(stations_with_amps[:,12].astype(float),stations_with_amps[:,13].astype(float))
            #we add the Lg/Coda ratio to filtered_stations list
            #filtered_stations has this layout of columns: network[0], station[1], lat[2], lon[3], elev[4], dist[5], az[6], t_Pn[7], t_Sn[8],t_Pg[9]
            #so at each iteration over the frequencies, a new column is added to the filtered_stations list, the first column filled with 
            #amplitudes for the frst frequency couple is the 11th column, with index [10]
            filtered_statios = np.append(np.array(filtered_statios),np.array(Lg_Coda),axis=1)

        #once filtered_station has be filled with all the Amplitudes for each freq band and each station, we save it in a .csv file
        np.savetxt(f'/home/schreinl/Stage/Data/Amplitudes/{start}.txt', filtered_statios, delimiter=',')
        #alongside with the file containing the amplitudes, we save the frequencies for which the data was filtered
    np.savetxt(f'/home/schreinl/Stage/Data/Amplitudes/{start}_freqs.txt', np.column_stack((freqmin,freqmax)), delimiter=',')

    return

                

        
        