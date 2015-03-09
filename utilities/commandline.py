__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2014-11-19"
__copyright__ = "Copyright (C) 2014 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

import json

__all__ = ["parse_command_line"]
# Parse command-line keyword arguments
def parse_command_line(cline):
    commandline_kwargs = {}
    for s in cline:
        if s.count('=') == 1:
            key, value = s.split('=', 1)
        else:
            raise TypeError(s+" Only kwargs separated with '=' sign allowed. See NSdefault_hooks for a range of parameters.")
        try:
            value = json.loads(value)
                        
        except ValueError:
            if value in ("True", "False"): # json understands true/false, but not True/False
                value = eval(value)
                
        commandline_kwargs[key] = value
    return commandline_kwargs
