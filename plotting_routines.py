import matplotlib.pyplot as plt
import numpy as np
from obspy import UTCDateTime
from obspy.core import Stream
from processing_routines import *



###########################
#Section 1.)
#Plotting functions for visualisation of seismic data



def plot_record_section(
    st, stations, eq_lat, eq_lon, eq_start, size=(1200, 1000), show=True, outfile=None, tracehodo=True, v_Lg_min=3.1, v_Lg_max=3.5, 
v_Pg=6.,tmincoda=300,tmaxcoda=320):
    
    '''
    plots a record section for a given event, and a list of stations.
    '''

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





import folium
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




###########################
#Section 2.)
#Plotting of SNR



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







import json

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




    
#############################
#Section 3.)
#Plotting of various things and envelopes


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



#############################
#Section 4.)
#Plotting of site effects




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