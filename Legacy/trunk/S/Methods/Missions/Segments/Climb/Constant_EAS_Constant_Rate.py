## @ingroup Methods-Missions-Segments-Climb
# Constant_EAS_Constant_Rate.py
# 
# Created:  Aug 2016, T. MacDonald
# Modified: 
#
# Adapted from Constant_Speed_Constant_Rate

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import Legacy.trunk.S as SUAVE
import numpy as np

# ----------------------------------------------------------------------
#  Initialize Conditions
# ----------------------------------------------------------------------
## @ingroup Methods-Missions-Segments-Climb
def initialize_conditions(segment):
    """Sets the specified conditions which are given for the segment type.
    
    Assumptions:
    Constant true airspeed with a constant rate of climb

    Source:
    N/A

    Inputs:
    segment.climb_rate                                  [meters/second]
    segment.equivalent_air_speed                        [meters/second]
    segment.altitude_start                              [meters]
    segment.altitude_end                                [meters]
    segment.state.numerics.dimensionless.control_points [Unitless]
    conditions.freestream.density                       [kilograms/meter^3]

    Outputs:
    conditions.frames.inertial.velocity_vector  [meters/second]
    conditions.frames.inertial.position_vector  [meters]
    conditions.freestream.altitude              [meters]

    Properties Used:
    N/A
    """         
    
    # unpack
    climb_rate = segment.climb_rate
    eas        = segment.equivalent_air_speed   
    alt0       = segment.altitude_start 
    altf       = segment.altitude_end
    t_nondim   = segment.state.numerics.dimensionless.control_points
    conditions = segment.state.conditions  

    # check for initial altitude
    if alt0 is None:
        if not segment.state.initials: raise AttributeError('initial altitude not set')
        alt0 = -1.0 * segment.state.initials.conditions.frames.inertial.position_vector[-1,2]

    # discretize on altitude
    alt = t_nondim * (altf-alt0) + alt0
    
    # pack conditions
    conditions.freestream.altitude[:,0]             =  alt[:,0] # positive altitude in this context
    
    # determine airspeed from equivalent airspeed
    SUAVE.Methods.Missions.Segments.Common.Aerodynamics.update_atmosphere(segment) # get density for airspeed
    density   = conditions.freestream.density[:,0]   
    MSL_data  = segment.analyses.atmosphere.compute_values(0.0,0.0)
    air_speed = eas/np.sqrt(density/MSL_data.density[0])    
    
    # process velocity vector
    v_mag = air_speed
    v_z   = -climb_rate # z points down
    v_x   = np.sqrt( v_mag**2 - v_z**2 )
    
    # pack conditions    
    conditions.frames.inertial.velocity_vector[:,0] = v_x
    conditions.frames.inertial.velocity_vector[:,2] = v_z
    conditions.frames.inertial.position_vector[:,2] = -alt[:,0] # z points down