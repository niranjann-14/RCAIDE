# airfoil_import_test.py 

# ----------------------------------------------------------------------------------------------------------------   
#   IMPORTS 
# ----------------------------------------------------------------------------------------------------------------   
  
from RCAIDE.Library.Plots import *  
from RCAIDE.Library.Methods.Geometry.Airfoil   import import_airfoil_geometry, compute_airfoil_properties, convert_airfoil_to_meshgrid

# python imports 
import os
import numpy as np
import matplotlib.pyplot as plt  
# ----------------------------------------------------------------------------------------------------------------    
#   Main
# ----------------------------------------------------------------------------------------------------------------    
def main():    
    # ----------------------------------------------------------------------------------------------------------------
    #  Define airfoil geometry and polar files 
    # ----------------------------------------------------------------------------------------------------------------    
    ospath    = os.path.abspath(__file__)
    separator = os.path.sep 
    rel_path  = ospath.split('geometry_airfoils' + separator + 'airfoil_import_test.py')[0] + '..' + separator + 'Vehicles' + separator + 'Airfoils' + separator
    airfoil_geometry_with_selig =  [rel_path + 'NACA_4412.txt','airfoil_geometry_2.txt', 'airfoil_geometry_2-selig.txt']        
    airfoil_geometry_files      = rel_path + 'NACA_4412.txt'
    airfoil_polar_files         =  [rel_path + 'Polars' + separator + 'NACA_4412_polar_Re_50000.txt',
                                     rel_path + 'Polars' + separator + 'NACA_4412_polar_Re_100000.txt',
                                     rel_path + 'Polars' + separator + 'NACA_4412_polar_Re_200000.txt',
                                     rel_path + 'Polars' + separator + 'NACA_4412_polar_Re_500000.txt',
                                     rel_path + 'Polars' + separator + 'NACA_4412_polar_Re_1000000.txt']  
    
    

    # ----------------------------------------------------------------------------------------------------------------
    #  Plot polar files 
    # ----------------------------------------------------------------------------------------------------------------    
    # plot airfoil polar data with and without surrogate
    airfoil_geometry_1   = import_airfoil_geometry(airfoil_geometry_files)  
    airfoil_polar_data_1 = compute_airfoil_properties(airfoil_geometry_1,airfoil_polar_files) 
    plot_airfoil_polar_files(airfoil_polar_data_1) 

    # ----------------------------------------------------------------------------------------------------------------
    #  
    # ----------------------------------------------------------------------------------------------------------------
    airfoil_geometry_2  = import_airfoil_geometry(airfoil_geometry_with_selig[0])
    airfoil_geometry_3  = import_airfoil_geometry(airfoil_geometry_with_selig[1])
    airfoil_geometry_4  = import_airfoil_geometry(airfoil_geometry_with_selig[2])

    # Actual t/c values  
    airfoil_tc_actual = [0.12031526401402462, 0.11177619218206997, 0.11177619218206997] 

    # Check t/c calculation against previously calculated values  
    assert(np.abs(airfoil_tc_actual[0]-airfoil_geometry_2.thickness_to_chord) < 1E-8 ) 
    assert(np.abs(airfoil_tc_actual[1]-airfoil_geometry_3.thickness_to_chord) < 1E-8 ) 
    assert(np.abs(airfoil_tc_actual[2]-airfoil_geometry_4.thickness_to_chord) < 1E-8 ) 

    # Check that camber line comes back the same for the Lednicer and Selig formats 
    for j in range(0, len(airfoil_geometry_3.camber_coordinates)):
        assert( np.abs(airfoil_geometry_3.camber_coordinates[j] - airfoil_geometry_4.camber_coordinates[j]) < 1E-8 )

    # Multiple meshes use too much memory on AppVeyor
    A_MASK_1 = convert_airfoil_to_meshgrid(airfoil_geometry_1) 

    assert (len(np.where(A_MASK_1)[0]) == 32313) 
    plot_airfoil(airfoil_geometry_with_selig[1])

    return  

if __name__ == '__main__': 
    main() 
    plt.show()
