## @ingroup Library-Attributes-Propellants
# RCAIDE/Library/Attributes/Propellants/Liquid_Natural_Gas.py
# 
#
# Created:  Mar 2024, M. Clarke

# ---------------------------------------------------------------------------------------------------------------------- 
#  Imports
# ---------------------------------------------------------------------------------------------------------------------- 

from .Propellant import Propellant   

# ---------------------------------------------------------------------------------------------------------------------- 
#  Gaseous_Hydrogen Class
# ----------------------------------------------------------------------------------------------------------------------  
## @ingroup  Library-Attributes-Propellants 
class Liquid_Petroleum_Gas(Propellant):
    """Liquid petroleum gas fuel class,
    """

    def __defaults__(self):
        """This sets the default values. 
    
    Assumptions:
        None
    
    Source:
        None
        """    
        self.tag             = 'Liquid_Petroleum_Gas'
        self.reactant        = 'O2'
        self.density         = 509.3                            # kg/m^3 
        self.specific_energy = 46e6                           # J/kg
        self.energy_density  = 23427.8e6                        # J/m^3