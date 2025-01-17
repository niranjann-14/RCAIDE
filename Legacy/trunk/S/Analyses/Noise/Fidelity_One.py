## @ingroup Analyses-Noise
# Fidelity_One.py
#
# Created:  
# Modified: Feb 2016, A. Wendorff
# Modified: Apr 2021, M. Clarke
#           Jul 2021, E. Botero
#           Feb 2022, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import Legacy.trunk.S as SUAVE 
from Legacy.trunk.S.Core import Data , Units
from .Noise     import Noise 

from Legacy.trunk.S.Components.Physical_Component import Container 

# noise imports 
from Legacy.trunk.S.Methods.Noise.Fidelity_One.Airframe.noise_airframe_Fink                   import noise_airframe_Fink
from Legacy.trunk.S.Methods.Noise.Fidelity_One.Engine.noise_SAE                               import noise_SAE  
from Legacy.trunk.S.Methods.Noise.Fidelity_One.Noise_Tools.noise_geometric                    import noise_geometric
from Legacy.trunk.S.Methods.Noise.Fidelity_One.Noise_Tools.decibel_arithmetic                 import SPL_arithmetic
from Legacy.trunk.S.Methods.Noise.Fidelity_One.Noise_Tools.generate_microphone_points         import generate_ground_microphone_points
from Legacy.trunk.S.Methods.Noise.Fidelity_One.Noise_Tools.compute_noise_evaluation_locations import compute_ground_noise_evaluation_locations
from Legacy.trunk.S.Methods.Noise.Fidelity_One.Noise_Tools.compute_noise_evaluation_locations import compute_building_noise_evaluation_locations
from Legacy.trunk.S.Methods.Noise.Fidelity_One.Propeller.propeller_mid_fidelity               import propeller_mid_fidelity 

# package imports
import numpy as np

# ----------------------------------------------------------------------
#  Analysis
# ----------------------------------------------------------------------
## @ingroup Analyses-Noise
class Fidelity_One(Noise):
    
    """ SUAVE.Analyses.Noise.Fidelity_One()
    
        The Fidelity One Noise Analysis Class
        
            Assumptions:
            None
            
            Source:
            N/A
    """
    
    def __defaults__(self):
        
        """ This sets the default values for the analysis.
        
                Assumptions:
                Ground microphone angles start in front of the aircraft (0 deg) and sweep in a lateral direction 
                to the starboard wing and around to the tail (180 deg)
                
                Source:
                N/A
                
                Inputs:
                None
                
                Output:
                None
                
                Properties Used:
                N/A
        """
        
        # Initialize quantities                   
        settings                                      = self.settings
        settings.harmonics                            = np.arange(1,30) 
        settings.flyover                              = False    
        settings.approach                             = False
        settings.sideline                             = False
        settings.print_noise_output                   = False 
        settings.fix_lateral_microphone_distance      = True
        settings.static_microphone_array              = False  
        settings.urban_canyon_microphone_locations    = None  
        settings.urban_canyon_building_dimensions     = []
        settings.urban_canyon_building_locations      = []  
        settings.urban_canyon_microphone_x_resolution = 4 
        settings.urban_canyon_microphone_y_resolution = 4
        settings.urban_canyon_microphone_z_resolution = 16  
        settings.mic_x_position                       = 0       
        settings.level_ground_microphone_min_x        = 1E-6
        settings.level_ground_microphone_max_x        = 5000 
        settings.level_ground_microphone_min_y        = 1E-6
        settings.level_ground_microphone_max_y        = 450   # sideline microphone distance
        settings.level_ground_microphone_x_resolution = 5
        settings.level_ground_microphone_y_resolution = 5
        settings.center_frequencies                   = np.array([16,20,25,31.5,40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, \
                                                                  500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150,
                                                                  4000, 5000, 6300, 8000, 10000])        
        settings.lower_frequencies                    = np.array([14,18,22.4,28,35.5,45,56,71,90,112,140,180,224,280,355,450,560,710,\
                                                                  900,1120,1400,1800,2240,2800,3550,4500,5600,7100,9000 ])
        settings.upper_frequencies                    = np.array([18,22.4,28,35.5,45,56,71,90,112,140,180,224,280,355,450,560,710,900,1120,\
                                                                 1400,1800,2240,2800,3550,4500,5600,7100,9000,11200 ])
        
        return
            
    def evaluate_noise(self,segment):
        """ Process vehicle to setup geometry, condititon and configuration
    
        Assumptions:
        None
    
        Source:
        N/4
    
        Inputs:
        self.settings.
            center_frequencies  - 1/3 octave band frequencies   [unitless]
    
        Outputs:
        None
    
        Properties Used:
        self.geometry
        """         
    
        # unpack 
        config        = segment.analyses.noise.geometry
        analyses      = segment.analyses
        settings      = self.settings 
        print_flag    = settings.print_noise_output  
        conditions    = segment.state.conditions  
        dim_cf        = len(settings.center_frequencies ) 
        ctrl_pts      = int(segment.state.numerics.number_control_points)
        min_x         = settings.level_ground_microphone_min_x         
        max_x         = settings.level_ground_microphone_max_x         
        min_y         = settings.level_ground_microphone_min_y         
        max_y         = settings.level_ground_microphone_max_y         
        x_resolution  = settings.level_ground_microphone_x_resolution  
        y_resolution  = settings.level_ground_microphone_y_resolution 
            
        
        # generate noise valuation points
        settings.ground_microphone_locations = generate_ground_microphone_points(min_x,max_x,min_y,max_y,x_resolution,y_resolution )     
        
        GM_THETA,GM_PHI,GML,num_gm_mic = compute_ground_noise_evaluation_locations(settings,segment)
        
        BM_THETA,BM_PHI,UCML,num_b_mic = compute_building_noise_evaluation_locations(settings,segment) 
        
        mic_locations  = np.concatenate((GML,UCML),axis = 1) 
        THETA          = np.concatenate((GM_THETA,BM_THETA),axis = 1) 
        PHI            = np.concatenate((GM_PHI,BM_PHI),axis = 1)  
        
        num_mic = num_b_mic + num_gm_mic  
        
        # append microphone locations to conditions
        conditions.noise.ground_microphone_theta_angles   = GM_THETA
        conditions.noise.building_microphone_theta_angles = BM_THETA
        conditions.noise.total_microphone_theta_angles    = THETA
        
        conditions.noise.ground_microphone_phi_angles     = GM_PHI
        conditions.noise.building_microphone_phi_angles   = BM_PHI
        conditions.noise.total_microphone_phi_angles      = PHI
        
        conditions.noise.ground_microphone_locations      = GML
        conditions.noise.building_microphone_locations    = UCML
        conditions.noise.total_microphone_locations       = mic_locations
        
        conditions.noise.number_ground_microphones        = num_gm_mic
        conditions.noise.number_building_microphones      = num_b_mic 
        conditions.noise.total_number_of_microphones      = num_mic
         
        
        # create empty arrays for results  
        num_src            = len(config.networks) + 1 
        if ('lift_cruise') in config.networks.keys():
            num_src += 1
        source_SPLs_dBA    = np.zeros((ctrl_pts,num_src,num_mic)) 
        source_SPL_spectra = np.zeros((ctrl_pts,num_src,num_mic,dim_cf ))    
        
        si = 1  
        # iterate through sources 
        for source in conditions.noise.sources.keys():  
            for network in config.networks.keys():                 
                if source  == 'turbofan':
                    geometric = noise_geometric(segment,analyses,config)  
                     
                    # flap noise - only applicable for turbofan aircraft
                    if 'flap' in config.wings.main_wing.control_surfaces:            
                        airframe_noise                = noise_airframe_Fink(segment,analyses,config,settings)  
                        source_SPLs_dBA[:,si,:]       = airframe_noise.SPL_dBA          
                        source_SPL_spectra[:,si,:,5:] = np.repeat(airframe_noise.SPL_spectrum[:,np.newaxis,:], num_mic, axis =1)
                    
                    
                    if bool(conditions.noise.sources[source].fan) and bool(conditions.noise.sources[source].core): 
                                              
                        config.networks[source].fan.rotation             = 0 # FUTURE WORK: NEED TO UPDATE ENGINE MODEL WITH FAN SPEED in RPM
                        config.networks[source].fan_nozzle.noise_speed   = conditions.noise.sources.turbofan.fan.exit_velocity 
                        config.networks[source].core_nozzle.noise_speed  = conditions.noise.sources.turbofan.core.exit_velocity
                        engine_noise                                      = noise_SAE(config.networks[source],segment,analyses,config,settings,ioprint = print_flag)  
                        source_SPLs_dBA[:,si,:]                           = np.repeat(np.atleast_2d(engine_noise.SPL_dBA).T, num_mic, axis =1)     # noise measures at one microphone location in segment
                        source_SPL_spectra[:,si,:,5:]                     = np.repeat(engine_noise.SPL_spectrum[:,np.newaxis,:], num_mic, axis =1) # noise measures at one microphone location in segment
                          
                elif (source  == 'propellers')  or (source   == 'lift_rotors'): 
                    if bool(conditions.noise.sources[source]) == True: 
                        net                          = config.networks[network] 
                        acoustic_data                = conditions.noise.sources[source]
                        
                        if source == 'propellers':
                            rotors        = net.propellers
                            identity_flag = net.identical_propellers
                        else:
                            rotors        = net.lift_rotors 
                            identity_flag = net.identical_lift_rotors
                             
                        if identity_flag:
                            aeroacoustic_data  = acoustic_data[list(acoustic_data.keys())[0]] 
                            propeller_noise    = propeller_mid_fidelity(rotors,aeroacoustic_data,segment,settings) 
                        else:
                            distributed_rotors                       = Container()
                            num_rotors                               = len(rotors)
                            distributed_prop_noise_SPL_dBA           = np.zeros((num_rotors,ctrl_pts,num_mic)) 
                            distributed_prop_noise_SPL_1_3_spectrum  = np.zeros((num_rotors,ctrl_pts,num_mic,dim_cf)) 
                            for r_idx , rotor  in enumerate(rotors): 
                                aeroacoustic_data                               = acoustic_data[rotor.tag]                                    
                                distributed_rotors.append(rotors[rotor.tag])       
                                propeller_noise                                 = propeller_mid_fidelity(distributed_rotors,aeroacoustic_data,segment,settings) 
                                distributed_prop_noise_SPL_dBA[r_idx]           = propeller_noise.SPL_dBA 
                                distributed_prop_noise_SPL_1_3_spectrum[r_idx]  = propeller_noise.SPL_1_3_spectrum     
                            propeller_noise.SPL_dBA          = SPL_arithmetic(distributed_prop_noise_SPL_dBA ,sum_axis=0)
                            propeller_noise.SPL_1_3_spectrum = SPL_arithmetic(distributed_prop_noise_SPL_1_3_spectrum ,sum_axis=0)
                     
                        source_SPLs_dBA[:,si,:]      = propeller_noise.SPL_dBA 
                        source_SPL_spectra[:,si,:,:] = propeller_noise.SPL_1_3_spectrum    
                           
                        si += 1
             
        conditions.noise.total_SPL_dBA = SPL_arithmetic(source_SPLs_dBA,sum_axis=1)
        
        return   

