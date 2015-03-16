from commandline import *
from create_profile import *
from memoryprofiler import *
try:
    #from weave import weaverhs, weavecross, weavecrossi
    import weave_single, weave_double
    def weaverhs(dU, U_hat, K2, K, P_hat, K_over_K2, dealias, nu, precision="double"):
        if precision == "single":
            weave_single.weaverhs(dU, U_hat, K2, K, P_hat, K_over_K2, dealias, nu)
        else:
            weave_double.weaverhs(dU, U_hat, K2, K, P_hat, K_over_K2, dealias, nu)
            
    def weavecross(a, b, c, precision="double"):
        if precision == "single":
            weave_single.weavecross(a, b, c)
        else:
            weave_double.weavecross(a, b, c)
            
    def weavecrossi(a, b, c, precision="double"):
        if precision == "single":
            weave_single.weavecrossi(a, b, c)
        else:
            weave_double.weavecrossi(a, b, c)

except:
    useweave = False

