from cbcdns import config

if config.precision == "single":
    from Cheb_single import *
    from TDMA_single import *
    from LUsolve_single import *
    from Matvec_single import *

else:
    from Cheb_double import *
    from TDMA_double import *
    from LUsolve_double import *
    from Matvec_double import *
    
