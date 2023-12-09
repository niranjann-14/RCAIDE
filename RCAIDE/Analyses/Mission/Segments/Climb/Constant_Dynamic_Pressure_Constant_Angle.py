## @ingroup Analyses-Mission-Segments-Climb
# RCAIDE/Analyses/Mission/Segments/Climb/Constant_Dynamic_Pressure_Constant_Angle.py
# 
# 
# Created:  Jul 2023, M. Clarke
 
# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------

# RCAIDE imports  
from RCAIDE.Core                                     import Units 
from RCAIDE.Analyses.Mission.Segments.Evaluate       import Evaluate
from RCAIDE.Methods.Mission                          import Segments

# ----------------------------------------------------------------------------------------------------------------------
# Constant_Dynamic_Pressure_Constant_Angle
# ---------------------------------------------------------------------------------------------------------------------- 
 
## @ingroup Analyses-Mission-Segments-Climb
class Constant_Dynamic_Pressure_Constant_Angle(Evaluate):
    """ Climb at a constant dynamic pressure at a constant angle.This segment takes longer to solve than most because 
        it has extra unknowns and residuals
    
        Assumptions:
        None
        
        Source:
        None
    """       
    
    def __defaults__(self):
        """ This sets the default solver flow. Anything in here can be modified after initializing a segment.
    
            Assumptions:
            None
    
            Source:
            N/A
    
            Inputs:
            None
    
            Outputs:
            None
    
            Properties Used:
            None
        """          
        
        # -------------------------------------------------------------------------------------------------------------- 
        #   User Inputs
        # -------------------------------------------------------------------------------------------------------------- 
        self.altitude_start            = None # Optional
        self.altitude_end              = 10.  * Units.km
        self.climb_angle               = 3.   * Units.degrees
        self.dynamic_pressure          = None
        self.true_course_angle         = 0.0 * Units.degrees
        
        # -------------------------------------------------------------------------------------------------------------- 
        #  Mission Specific Unknowns and Residuals 
        # --------------------------------------------------------------------------------------------------------------    
        ones_row = self.state.ones_row        
        self.state.unknowns.altitudes  = ones_row(1) * 0.0
        self.state.unknowns.body_angle = ones_row(1) * 3.0 * Units.degrees
        self.state.residuals.forces    = ones_row(2) * 0.0  
        self.state.residuals.altitude  = ones_row(1) * 0.0              
        
        # -------------------------------------------------------------------------------------------------------------- 
        #  Mission specific processes 
        # --------------------------------------------------------------------------------------------------------------   
        initialize                         = self.process.initialize
        initialize.conditions              = Segments.Climb.Constant_Dynamic_Pressure_Constant_Angle.initialize_conditions_unpack_unknowns 
        iterate                            = self.process.iterate
        iterate.unknowns.mission           = Segments.Climb.Constant_Dynamic_Pressure_Constant_Angle.initialize_conditions_unpack_unknowns 
        iterate.conditions.differentials   = Segments.Climb.Optimized.update_differentials 
        iterate.residuals.total_forces     = Segments.Climb.Constant_Dynamic_Pressure_Constant_Angle.residual_total_forces    
    
        return
       