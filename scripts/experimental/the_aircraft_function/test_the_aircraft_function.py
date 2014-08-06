# test_the_aircraft_function.py
# 
# Created:  Trent Lukaczyk , Aug 2014
# Modified: 


# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import SUAVE
from SUAVE.Attributes import Units
from SUAVE.Structure import Data

import numpy as np

import copy, time

from full_setup import full_setup
from the_aircraft_function import the_aircraft_function


# ----------------------------------------------------------------------
#   Main
# ----------------------------------------------------------------------

def main():
    
    vehicle,mission = full_setup()
    
    results = the_aircraft_function(vehicle,mission)
    


# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------
if __name__ == '__main__':
    main()