# RCAIDE/Library/Methods/Weights/Correlation_Buildups/General_Aviation/compute_landing_gear_weight.py
# 
# 
# Created:  Sep 2024, M. Clarke 

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------

# RCAIDE 
from RCAIDE.Framework.Core import  Units ,  Data  

# ----------------------------------------------------------------------------------------------------------------------
# Main Wing Weight 
# ----------------------------------------------------------------------------------------------------------------------
def compute_landing_gear_weight(landing_weight, Nult, strut_length_main, strut_length_nose):
    """ 
        Calculate the weight of the landing gear

        Source: Raymer- Aircraft Design: a Conceptual Approach (pg 460 in 4th edition)
        
        Inputs:
            Nult - ultimate landing load factor
            landing_weight- landing weight of the aircraft [kilograms]
           
        Outputs:
            weight - weight of the landing gear            [kilograms]
            
        Assumptions:
            calculating the landing gear weight based on the landing weight, load factor, and strut length 
    """ 

    #unpack
    W_l = landing_weight/Units.lbs
    l_n = strut_length_nose/Units.inches
    l_m = strut_length_main/Units.inches
    main_weight = .095*((Nult*W_l)**.768)*(l_m/12.)**.409
    nose_weight = .125*((Nult*W_l)**.566)*(l_n/12.)**.845

    # pack outputs
    output      = Data()
    output.main = main_weight*Units.lbs
    output.nose = nose_weight*Units.lbs

    return output