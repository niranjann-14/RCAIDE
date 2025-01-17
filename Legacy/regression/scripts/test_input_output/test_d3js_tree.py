# test_d3js.py
# 
# Created:  Trent Lukaczyk, Feb 2015
# Modified: 

# ----------------------------------------------------------------------        
#   Imports
# ----------------------------------------------------------------------  

import Legacy.trunk.S as SUAVE.Input_Output as IO
import Legacy.trunk.S as SUAVE.Core.Data as Data


# ----------------------------------------------------------------------        
#   Main
# ----------------------------------------------------------------------  

def main():

    data = Data()
    data.x = 'x'
    data.y = 'y'
    data.sub = Data()
    data.sub.z = 'z'
    data.sub.a = 1

    IO.D3JS.save_tree(data,'tree.json',root_name='data')


# ----------------------------------------------------------------------        
#   Call Main
# ----------------------------------------------------------------------  

if __name__ == '__main__':
    main()




