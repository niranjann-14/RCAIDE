## @ingroup Methods-Missions-Segments-Ground
# RCAIDE/Methods/Missions/Segments/Ground/Landing.py
# 
# 
# Created:  Jul 2023, M. Clarke  
 
import numpy as np 

# ----------------------------------------------------------------------------------------------------------------------
# unpack unknowns
# ----------------------------------------------------------------------------------------------------------------------
## @ingroup Methods-Missions-Segments-Ground
def initialize_conditions(segment):
    """Sets the specified conditions which are given for the segment type.

    Assumptions:
    Builds on the initialize conditions for common

    Source:
    N/A

    Inputs:
    segment.throttle                                         [unitless]
    segment.analyses.weights.vehicle.mass_properties.landing [kilogram]
    
    Outputs:
    conditions.weights.total_mass   [kilogram]
    conditions.propulsion.throttle  [unitless]

    Properties Used:
    N/A
    """      
    
    # use the common initialization
    conditions = segment.state.conditions
    
    # unpack inputs
    alt      = segment.altitude 
    v0       = segment.velocity_start
    vf       = segment.velocity_end 
    throttle = segment.throttle	
    
    # check for initial altitude
    if alt is None:
        if not segment.state.initials: raise AttributeError('altitude not set')
        alt = -1.0 *segment.state.initials.conditions.frames.inertial.position_vector[-1,2]   

    if v0  is None: 
        v0 = np.linalg.norm(segment.state.initials.conditions.frames.inertial.velocity_vector[-1])
        
    # avoid having zero velocity since aero and propulsion models need non-zero Reynolds number
    if v0 == 0.0: v0 = 0.01
    if vf == 0.0: vf = 0.01
    
    # intial and final speed cannot be the same
    if v0 == vf:
        vf = vf + 0.01
        
    # repack
    segment.air_speed_start = v0
    segment.air_speed_end   = vf
    
    initialized_velocity = (vf - v0)*segment.state.numerics.dimensionless.control_points + v0
    
    # Initialize the x velocity unknowns to speed convergence:
    segment.state.unknowns.velocity_x = initialized_velocity[1:,0]    

    # pack conditions 
    conditions = segment.state.conditions    
    conditions.frames.inertial.velocity_vector[:,0] = initialized_velocity[:,0]
    conditions.ground.incline[:,0]                  = segment.ground_incline
    conditions.ground.friction_coefficient[:,0]     = segment.friction_coefficient 
    conditions.freestream.altitude[:,0]             = alt
    conditions.frames.inertial.position_vector[:,2] = -alt      

    for network in segment.analyses.energy.networks:
        if 'busses' in network:
            for bus in network.busses: 
                conditions.energy[bus.tag].throttle[:,0]      = throttle   
        if 'fuel_lines' in network:
            for fuel_line in network.fuel_lines: 
                conditions.energy[fuel_line.tag].throttle[:,0] = throttle     
    # Unpack 
    m_initial = segment.analyses.weights.vehicle.mass_properties.landing
          
    # apply initials
    conditions.weights.total_mass[:,0]  = m_initial 