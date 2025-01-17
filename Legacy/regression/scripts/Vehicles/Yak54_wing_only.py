# Yak54_wing_only.py
# 
# Created:  Nov 2018, Stanislav Karpuk
# Modified: 
"""
    Vehicle set-up used for the V-n diagram test only
"""
# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

# Python Imports
import numpy as np

# SUAVE Imports
import Legacy.trunk.S as SUAVE
from Legacy.trunk.S.Core import Data, Units
from Legacy.trunk.S.Core import (
    Data, Container,
)


def vehicle_setup():
          
    vehicle = SUAVE.Vehicle()
    vehicle.tag = 'Yakovlev_Ya54'  
    
    # ------------------------------------------------------------------
    #   Vehicle-level Properties
    # ------------------------------------------------------------------    
    
    # mass properties
    vehicle.mass_properties.max_takeoff               = 901.0 * Units.kilogram      # aerobatic, one pilot
    vehicle.mass_properties.takeoff                   = 901.0 * Units.kilogram   
  
    
    # envelope properties
    vehicle.envelope.category                = 'acrobatic'
    vehicle.envelope.FAR_part_number         = 23
    vehicle.envelope.limit_loads.positive    = 9
    vehicle.envelope.limit_loads.negative    = -7
    vehicle.envelope.cruise_mach 	     = 0.17

    # aerodynamic properties
    vehicle.maximum_lift_coefficient = 1.26
    vehicle.minimum_lift_coefficient = -1.26

    # basic parameters
    vehicle.reference_area         = 12.89 * Units['meters**2']  
    vehicle.passengers             = 1

    # ------------------------------------------------------------------        
    #   Main Wing
    # ------------------------------------------------------------------        
    
    wing = SUAVE.Components.Wings.Main_Wing()
    wing.tag = 'main_wing'
    
    wing.aspect_ratio            = 5.2
    wing.sweeps.quarter_chord    = 0.0 * Units.deg
    wing.thickness_to_chord      = 0.14
    wing.taper                   = 0.43
    wing.spans.projected         = 8.16 * Units.meter
    wing.chords.mean_aerodynamic = 1.57 * Units.meter
    wing.areas.reference         = 12.89 * Units['meters**2']  
    wing.vertical                = False
    wing.symmetric               = True
    wing.high_lift               = False
    wing.dynamic_pressure_ratio  = 1.0


    # add to vehicle
    vehicle.append_component(wing)
    
    # ------------------------------------------------------------------
    #   Vehicle Definition Complete
    # ------------------------------------------------------------------

    return vehicle

# ----------------------------------------------------------------------
#   Define the Configurations
# ---------------------------------------------------------------------

def configs_setup(vehicle):
    
    # ------------------------------------------------------------------
    #   Initialize Configurations
    # ------------------------------------------------------------------
    configs = SUAVE.Components.Configs.Config.Container()

    base_config = SUAVE.Components.Configs.Config(vehicle)
    base_config.tag = 'base'
    configs.append(base_config)

    # ------------------------------------------------------------------
    #   Cruise Configuration
    # ------------------------------------------------------------------
    config = SUAVE.Components.Configs.Config(base_config)
    config.tag = 'cruise'
    configs.append(config)

    return configs  
