#!/usr/bin/python
#-----------------------------------------------------------------------------------------------------------------------------------------

# Script Description:
# Driver that runs Salvus to generate synthetic seismograms (Green's functions) for a given model.
# Moved into unshake from python/initial_joint_inv_development/run_salvus_sm_synthetics.py.

# Input variables:

# Output variables:

# Created by Tom Hudson, 27th February 2025

#-----------------------------------------------------------------------------------------------------------------------------------------

# Import neccessary modules:
# (Salvus and its helpers are optional: unshake imports without them, and
# gen_synths_salvus raises an ImportError only when called without Salvus.)
import os
import pathlib
import numpy as np
import matplotlib.pyplot as plt

try:
    import xarray as xr
    import pandas as pd
    import pyproj
    import obspy
    # The Salvus package.
    import salvus.namespace as sn
    from salvus.toolbox.helpers.wavefield_output import (
        WavefieldOutput,
        wavefield_output_to_xarray,
    )
    import salvus.flow.utils
    HAS_SALVUS = True
except ImportError:
    HAS_SALVUS = False

SALVUS_FLOW_SITE_NAME = os.environ.get("SITE_NAME", "local")

# Default simulation box (m), used when domain_dict is None:
DEFAULT_DOMAIN_DICT = {'x0': -100.0, 'x1': 100.0, 'y0': -100.0, 'y1': 100.0, 'z0': -80.0, 'z1': 0.0}




#----------------------------------------------- Define main functions -----------------------------------------------
def gen_synths_salvus(project_path, domain_dict, xyz_src_m, receivers_df, medium_model_dict, centre_freq_Hz, 
                        output_waveforms_settings_dict={'output_fs_Hz': 1000, 'gaussian_noise_dur_s': 0, 'snr_amp': 100.0, 
                        'add_noise_to_signal': False, 'wfs_outdir' : "data_out/example_iq"}, gen_body_waves_only=False,
                        end_time_in_seconds=0.1, wavelet=None, max_frequency_in_hertz=None,
                        elements_per_wavelength=2.0, verbosity=0):
    """Function to generate synthetic seismograms using Salvus.
    
    Args:
    project_path (str): Path to the Salvus project.
    domain_dict (dict): Dictionary containing the volume domain parameters for the synthetic seismograms.
                        (including params: x0, x1, y0, y1, z0, z1). If None, uses DEFAULT_DOMAIN_DICT.
    xyz_src_m (np.array): Array containing the source locations in metres in x,y,z.
    receivers_df (pd.DataFrame): DataFrame containing the receiver locations in x,y,z (x_m, y_m, z_m).
    medium_model_dict (dict): Dictionary containing the medium model parameters (vp, vs, rho, qkappa,
                                qmu, x, y, z).
    centre_freq_Hz (float): Centre frequency of the source wavelet, in Hz.
    output_waveforms_settings_dict (dict): Dictionary containing the output waveform settings. (see defaults
                                            in function args for structure).
    gen_body_waves_only (bool): If True, only generate body waves (by running full-space rather than half-space model).
    end_time_in_seconds (float): Simulation end time, in seconds.
    wavelet (salvus stf object): Source time function used for every MT component. Default (None) is
                                    sn.simple_config.stf.Ricker(center_frequency=centre_freq_Hz).
                                    NOTE: Salvus applies the wavelet s(t) as the moment, M(t) = M * s(t)
                                    (verified against Aki & Richards), so the output synthetics are the
                                    Green's functions (response to a step in moment) convolved with ds/dt. With the default Ricker they are therefore NOT
                                    formal Green's functions (use unshake.Dataset.remove_stf to
                                    deconvolve the Ricker afterwards).
                                    Recommended for formal, band-limited Green's functions G*B:
                                        sn.simple_config.stf.FlatSpectrumWavelet(f0=5, f1=10, f2=140, f3=150,
                                                                                time_shift_in_seconds=0.04)
                                    with max_frequency_in_hertz=150 (Salvus truncates this wavelet to
                                    +/-0.2/f0 s, hence f0=5 Hz and the 0.04 s shift; extend end_time_in_seconds
                                    by the shift). Its spectrum is only approximately flat (ripple ~20 %), so
                                    still pass the GFs through unshake.Dataset.remove_stf(wavelet.get_stf_rate(),
                                    band=(10, 20, 120, 140)): this divides out the exact wavelet and applies
                                    the band filter B = unshake.spectra.flat_band(f, *band). The data are
                                    band-limited by the same B there; B must lie inside the wavelet's band.
    max_frequency_in_hertz (float): Mesh maximum frequency. Default (None) is 2*centre_freq_Hz. Should be
                                    >= the highest frequency in the wavelet (e.g. f3 = 150 Hz above).
    elements_per_wavelength (float): Mesh elements per minimum wavelength. Default 2.0 (coarse; ~3 is
                                    recommended with tensor_order=2 for accurate GFs up to max frequency).
    verbosity (int): Verbosity level of the output.
    
    Returns:
    """  
    if not HAS_SALVUS:
        raise ImportError("Salvus (and xarray, pandas, pyproj, obspy) not available. Install Salvus via "
                          "Mondaic, then `pip install -e \".[salvus]\"`, to generate Green's functions.")
    if wavelet is None:
        wavelet = sn.simple_config.stf.Ricker(center_frequency=float(centre_freq_Hz))
    if max_frequency_in_hertz is None:
        max_frequency_in_hertz = 2*float(centre_freq_Hz)


    # 1. Create a domain and project:
    if domain_dict is None:
        domain_dict = DEFAULT_DOMAIN_DICT
    domain = sn.domain.dim3.BoxDomain(**{k: float(domain_dict[k]) for k in ('x0', 'x1', 'y0', 'y1', 'z0', 'z1')})
    if verbosity > 0:
        domain.plot()
    # if os.path.isdir(project_path):
    try:
        p = sn.Project.from_domain(path=project_path, domain=domain)#, load_if_exists=True)
    except ValueError:
        os.system('rm -rf '+project_path) #(WARNING: POTENTIALLY DANGEROUS! CONSIDER ALTERNATIVE!)
        p = sn.Project.from_domain(path=project_path, domain=domain)



    # 2. Define the sources and receivers:
    # Source and receivers exist for multiple dimensions and coordinate systems.
    source_Mxx = sn.simple_config.source.cartesian.MomentTensorPoint3D(x=xyz_src_m[0]*1e6, y=xyz_src_m[1]*1e6, 
                                                z=xyz_src_m[2]*1e6, mxx=1, myy=0, mzz=0, myz=0, mxz=0, mxy=0)
    source_Myy = sn.simple_config.source.cartesian.MomentTensorPoint3D(x=xyz_src_m[0]*1e6, y=xyz_src_m[1]*1e6, 
                                                z=xyz_src_m[2]*1e6, mxx=0, myy=1, mzz=0, myz=0, mxz=0, mxy=0)
    source_Mzz = sn.simple_config.source.cartesian.MomentTensorPoint3D(x=xyz_src_m[0]*1e6, y=xyz_src_m[1]*1e6, 
                                                z=xyz_src_m[2]*1e6, mxx=0, myy=0, mzz=1, myz=0, mxz=0, mxy=0)
    source_Mxy = sn.simple_config.source.cartesian.MomentTensorPoint3D(x=xyz_src_m[0]*1e6, y=xyz_src_m[1]*1e6, 
                                                z=xyz_src_m[2]*1e6, mxx=0, myy=0, mzz=0, myz=0, mxz=0, mxy=1)
    source_Mxz = sn.simple_config.source.cartesian.MomentTensorPoint3D(x=xyz_src_m[0]*1e6, y=xyz_src_m[1]*1e6, 
                                                z=xyz_src_m[2]*1e6, mxx=0, myy=0, mzz=0, myz=0, mxz=1, mxy=0)
    source_Myz = sn.simple_config.source.cartesian.MomentTensorPoint3D(x=xyz_src_m[0]*1e6, y=xyz_src_m[1]*1e6, 
                                                z=xyz_src_m[2]*1e6, mxx=0, myy=0, mzz=0, myz=1, mxz=0, mxy=0)
    receivers = []
    for idx, row in receivers_df.iterrows():
        if gen_body_waves_only:
            zrec_curr=-10. + (row['z_m']*1e6) # Make sure receivers below absorbing boundary (if no free surface)
        else:
            zrec_curr=row['z_m']*1e6
        receivers.append(sn.simple_config.receiver.cartesian.Point3D(
        y=row['y_m']*1e6,
        x=row['x_m']*1e6,
        z=zrec_curr,
        station_code=row['Name'],
        # Note that one can specify the desired recording field here.
        fields=["velocity", "strain"],
        ))

    # 3. Define events (one for each moment tensor component):
    # (The event is a combination of the source and receivers)
    # Add events with varying sources to project:
    event_Mxx = sn.Event(event_name="event_Mxx", sources=source_Mxx, receivers=receivers)
    event_Myy = sn.Event(event_name="event_Myy", sources=source_Myy, receivers=receivers)
    event_Mzz = sn.Event(event_name="event_Mzz", sources=source_Mzz, receivers=receivers)
    event_Mxy = sn.Event(event_name="event_Mxy", sources=source_Mxy, receivers=receivers)
    event_Mxz = sn.Event(event_name="event_Mxz", sources=source_Mxz, receivers=receivers)
    event_Myz = sn.Event(event_name="event_Myz", sources=source_Myz, receivers=receivers)
    p.add_to_project(event_Mxx)
    p.add_to_project(event_Myy)
    p.add_to_project(event_Mzz)
    p.add_to_project(event_Mxy)
    p.add_to_project(event_Mxz)
    p.add_to_project(event_Myz)
    if verbosity > 0:
        #  visualize project.
        p.visualizations.notebook.domain()
    
    # 4. Specify the velocity model:
    # (Assemble model from medium_model_dict).
    ds = xr.Dataset(
    data_vars={
        "VP": (["x", "y", "z"], medium_model_dict['vp']),
        "VS": (["x", "y", "z"], medium_model_dict['vs']),
        "RHO": (["x", "y", "z"], medium_model_dict['rho']),
        "qkappa": (["x", "y", "z"], medium_model_dict['qkappa']),
        "qmu": (["x", "y", "z"], medium_model_dict['qmu']),
    },
    coords={"x": medium_model_dict['x'], "y": medium_model_dict['y'], "z": medium_model_dict['z']},
    )
    # Once we have an xarray data set, we can add it to the project.
    model = sn.model.volume.cartesian.GenericModel(name="my_volume_model", data=ds)
    p.add_to_project(model, overwrite=True)

    # 5. Setup the simulation:
    # Define free surface parameter:
    if gen_body_waves_only:
        free_surface_config = False # Turns off free surface (and installs absorbing boundary!)
        num_wavelengths_absorb_bc = 1.0 #3.5
        # print("WARNING: gen_body_waves_only=True not yet implemented!!!")
        # absorbing_boundary_config = sn.AbsorbingBoundaryParameters(
        #     reference_frequency=float(round(centre_freq_Hz)),
        #     number_of_wavelengths=1.0,#3.5, #1.0,
        #     reference_velocity=1900.0,
        #     #free_surface=None,
        #     ),
    else:
        free_surface_config = ["z1"]
        num_wavelengths_absorb_bc = 1.0
        # absorbing_boundary_config = sn.AbsorbingBoundaryParameters(
        #     reference_frequency=float(round(centre_freq_Hz)),
        #     number_of_wavelengths=1.0,#1.0,
        #     reference_velocity=1900.0,
        #     free_surface=["z1"],
        #     ),
    # Define simulation:
    sc = sn.SimulationConfiguration(
    name="my_tmp_src_inv_fwd_simulation",
    # Mesh resolution settings.
    max_frequency_in_hertz=float(max_frequency_in_hertz),
    elements_per_wavelength=float(elements_per_wavelength),
    tensor_order=2,
    # Using the volumetric model we specified before.
    model_configuration=sn.ModelConfiguration(
        volume_models=["my_volume_model"],
        # And for attenuation:
        linear_solids=sn.LinearSolids(reference_frequency=1.0),
    ),
    # Event specific configuration.
    event_configuration=sn.EventConfiguration(
        # Source wavelet.
        wavelet=wavelet,
        waveform_simulation_configuration=sn.WaveformSimulationConfiguration(
            end_time_in_seconds=end_time_in_seconds,#0.1,
            # And for attenuation:
            attenuation=True
        ),
    ),
    # Absorbing boundaries everywhere but the top.
    # (now defined above depending on whether generating body waves or surface waves too...)
    absorbing_boundaries=sn.AbsorbingBoundaryParameters(
            reference_frequency=float(round(centre_freq_Hz)),
            number_of_wavelengths=num_wavelengths_absorb_bc, #1.0,#1.0,
            reference_velocity=1900.0,
            free_surface=free_surface_config, #["z1"],
    ), #absorbing_boundary_config,
    )
    # As always, we'll add it to the project, and visualize the full setup.
    p.add_to_project(sc,overwrite=True,)
    if verbosity > 0:
        p.visualizations.notebook.simulation_setup("my_tmp_src_inv_fwd_simulation")
    
    # 6. Run the simulation:
    p.simulations.launch(
    simulation_configuration="my_tmp_src_inv_fwd_simulation",
    events=p.events.list(),
    # Determines where and how to run. Depending on the site
    # it will either directly launch, use a job queuing system, use GPUs, ...
    # The simulations will be automatically load balanced.
    site_name=SALVUS_FLOW_SITE_NAME,
    ranks_per_job=4, #2,
    # For many applications, volumetric output is not necessary, for
    # illustrative purposes we use it here.
    extra_output_configuration={
        "volume_data": {
            "sampling_interval_in_time_steps": 10,
            "fields": ["displacement"],
        },
    },    
    )
    p.simulations.query(block=True)

    # 7. Retrieve the synthetic seismograms:
    # Write waveform data to file:
    for event_uid in p.events.list():
        event_data = p.waveforms.get("my_tmp_src_inv_fwd_simulation", event_uid)[0]
        _write_wfs_to_file(output_waveforms_settings_dict['wfs_outdir'], event_data, receivers_df, output_waveforms_settings_dict['output_fs_Hz'], output_waveforms_settings_dict['gaussian_noise_dur_s'], output_waveforms_settings_dict['snr_amp'], add_noise_to_signal=output_waveforms_settings_dict['add_noise_to_signal'], event_uid=event_uid[-3:], verbosity=verbosity)
    
    return None



def _write_wfs_to_file(wfs_outdir, event_data, receivers_df, output_fs_Hz, gaussian_noise_dur_s, snr_amp, add_noise_to_signal=False, event_uid='', verbosity=0):
    # Check outdir exists:
    pathlib.Path(wfs_outdir).mkdir(parents=True, exist_ok=True)
    new_starttime = obspy.UTCDateTime(year=2000, julday=1, hour=1, minute=5)
    
    # Fibre direction for each DAS channel (unit vector, from the previous channel; the first
    # channel uses the direction to the next one). Assumes DAS rows are ordered along the fibre.
    das_rows = receivers_df[receivers_df['Name'].str[0] == "D"]
    das_xyz = das_rows[['x_m', 'y_m', 'z_m']].to_numpy(dtype=float)
    das_dirs = np.diff(das_xyz, axis=0)
    das_dirs = np.vstack((das_dirs[:1], das_dirs)) if len(das_xyz) > 1 else np.array([[0., 1., 0.]])
    das_dirs /= np.linalg.norm(das_dirs, axis=1, keepdims=True)
    das_dir_lookup = dict(zip(das_rows['Name'], das_dirs))

    # Loop through receivers, writing data out:
    for idx, row in receivers_df.iterrows():
        # Sort DAS from nodes:
        if row['Name'][0] == "D":
            st = event_data.get_receiver_data(receiver_name='XX.'+row['Name'], receiver_field='strain')  
            # Project the strain tensor onto the fibre axis n: eps_nn = n_i n_j eps_ij.
            # Salvus strain channels (verified against Aki & Richards): 0-2 = eps_xx, eps_yy, eps_zz;
            # 3-5 = engineering shear 2*eps_yz, 2*eps_xz, 2*eps_xy (x = east, y = north, z = up).
            nx, ny, nz = das_dir_lookup[row['Name']]
            e = {c: st.select(channel=c)[0].data for c in "012345"}
            das_data_curr_ch = (nx**2 * e["0"] + ny**2 * e["1"] + nz**2 * e["2"]
                                + ny*nz * e["3"] + nx*nz * e["4"] + nx*ny * e["5"])
            st_tmp = obspy.Stream()
            st[0].data = das_data_curr_ch
            st_tmp.append(st[0])
            st = st_tmp.copy()
            del st_tmp
            
        else:
            st = event_data.get_receiver_data(receiver_name='XX.'+row['Name'], receiver_field='velocity')
        for i in range(len(st)):
            st[i].stats.starttime = new_starttime
            st[i].resample(output_fs_Hz)
            if gaussian_noise_dur_s > 0:
                noise_before = np.max(np.abs(st[i].data)) * np.random.normal(0,0.6,(output_fs_Hz*gaussian_noise_dur_s)) / snr_amp
                noise_after = np.max(np.abs(st[i].data)) * np.random.normal(0,0.6,(output_fs_Hz*gaussian_noise_dur_s)) / snr_amp
                if add_noise_to_signal:
                    st[i].data += np.max(np.abs(st[i].data)) * np.random.normal(0,0.6,(len(st[i].data))) / snr_amp
                st[i].data = np.concatenate((noise_before, st[i].data, noise_after), axis=None)
            if st[i].stats.channel == 'X':
                st[i].stats.channel = 'HHE'
            elif st[i].stats.channel == 'Y':
                st[i].stats.channel = 'HHN'
            elif st[i].stats.channel == 'Z':
                st[i].stats.channel = 'HHZ'
            if len(event_uid) > 0:
                st[i].write(os.path.join(wfs_outdir, '2000001_'+st[0].stats.station+'_'+st[i].stats.channel+'_'+event_uid+'.m'), format="MSEED")
            else:
                st[i].write(os.path.join(wfs_outdir, '2000001_'+st[0].stats.station+'_'+st[i].stats.channel+'.m'), format="MSEED")
        if verbosity>0:
            print("Saved data for receiver:", st[0].stats.station)
        if idx == 0:
            print(st)
            st.plot()

    print("Finished writing waveforms to file.")



#----------------------------------------------- End: Define main functions -----------------------------------------------


