"""
This module is a combination of this Gist: https://gist.github.com/bfroehle/1944819
and the MPI_Import.py module from: https://github.com/langton/MPI_Import

The purpose is to speed up parallel import of Python modules.

Usage:
from MPI_knee import mpi_import
with mpi_import():
    import numpy
    import scipy
    
"""

import __builtin__
import imp
import sys

from types import ModuleType
from warnings import warn

import types
from mpi4py import MPI

class mpi(object):
    rank = MPI.COMM_WORLD.Get_rank()
    @staticmethod
    def bcast(obj=None,root=0):
        return MPI.COMM_WORLD.bcast(obj,root)
#import mpi

class mpi_import(object):
    def __enter__(self):
        imp.acquire_lock()
        import_module_level.mpi_import = self
        self.__funcs = []
        self.original_import = __builtin__.__import__
        __builtin__.__import__ = import_module_level

    def __exit__(self,type,value,traceback):
        __builtin__.__import__ = self.original_import
        import_module_level.mpi_import = None
        imp.release_lock()
        for f in self.__funcs:
            f()

    def callAfterImport(self,f):
        "Add f to the list of functions to call on exit"
        if type(f) != types.FunctionType:
            raise TypeError("Argument must be a function!")
        self.__funcs.append(f)

def get_parent(globals, level):
    """
    parent, name = get_parent(globals, level)

    Return the package that an import is being performed in.  If globals comes
    from the module foo.bar.bat (not itself a package), this returns the
    sys.modules entry for foo.bar.  If globals is from a package's __init__.py,
    the package's entry in sys.modules is returned.

    If globals doesn't come from a package or a module in a package, or a
    corresponding entry is not found in sys.modules, None is returned.
    """
    orig_level = level

    if not level or not isinstance(globals, dict):
        return None, ''

    pkgname = globals.get('__package__', None)

    if pkgname is not None:
        # __package__ is set, so use it
        if not hasattr(pkgname, 'rindex'):
            raise ValueError('__package__ set to non-string')
        if len(pkgname) == 0:
            if level > 0:
                raise ValueError('Attempted relative import in non-package')
            return None, ''
        name = pkgname
    else:
        # __package__ not set, so figure it out and set it
        if '__name__' not in globals:
            return None, ''
        modname = globals['__name__']

        if '__path__' in globals:
            # __path__ is set, so modname is already the package name
            globals['__package__'] = name = modname
        else:
            # Normal module, so work out the package name if any
            lastdot = modname.rfind('.')
            if lastdot < 0 and level > 0:
                raise ValueError("Attempted relative import in non-package")
            if lastdot < 0:
                globals['__package__'] = None
                return None, ''
            globals['__package__'] = name = modname[:lastdot]

    dot = len(name)
    for x in xrange(level, 1, -1):
        try:
            dot = name.rindex('.', 0, dot)
        except ValueError:
            raise ValueError("attempted relative import beyond top-level "
                             "package")
    name = name[:dot]

    try:
        parent = sys.modules[name]
    except:
        if orig_level < 1:
            warn("Parent module '%.200s' not found while handling absolute "
                 "import" % name)
            parent = None
        else:
            raise SystemError("Parent module '%.200s' not loaded, cannot "
                              "perform relative import" % name)

    # We expect, but can't guarantee, if parent != None, that:
    # - parent.__name__ == name
    # - parent.__dict__ is globals
    # If this is violated...  Who cares?
    return parent, name

def load_next(mod, altmod, name, buf):
    """
    mod, name, buf = load_next(mod, altmod, name, buf)

    altmod is either None or same as mod
    """

    if len(name) == 0:
        # completely empty module name should only happen in
        # 'from . import' (or '__import__("")')
        return mod, None, buf

    dot = name.find('.')
    if dot == 0:
        raise ValueError('Empty module name')

    if dot < 0:
        subname = name
        next = None
    else:
        subname = name[:dot]
        next = name[dot+1:]

    if buf != '':
        buf += '.'
    buf += subname
 
    result = import_submodule(mod, subname, buf)
    if result is None and mod != altmod:
        result = import_submodule(altmod, subname, subname)
        if result is not None:
            buf = subname

    if result is None:
        raise ImportError("No module named %.200s" % name)

    return result, next, buf

def import_submodule(mod, subname, fullname):
    """m = import_submodule(mod, subname, fullname)"""
    # Require:
    # if mod == None: subname == fullname
    # else: mod.__name__ + "." + subname == fullname

    if fullname in sys.modules:
        m = sys.modules[fullname]
    else:
        if mod is None:
            path = None
        elif hasattr(mod, '__path__'):
            path = mod.__path__
        else:
            return None

        fp = None         # module's file
        filename = None   # module's location
        stuff = None      # tuple of (suffix,mode,type) for the module
        ierror = False    # are we propagating an import error from rank 0?

        if mpi.rank == 0:
            try:
                fp, filename, stuff  = imp.find_module(subname, path)
            except ImportError:
                ierror = True
                return None
            finally:
                filename,stuff,ierror = mpi.bcast((filename,stuff,ierror))

        else:
            filename,stuff,ierror = mpi.bcast((filename,stuff,ierror))
            if ierror:
                return None
            # If imp.find_module returned an open file to rank 0, then we should
            # open the corresponding file for this process too.
            if stuff and stuff[1]:
                fp = open(filename,stuff[1])
                                
        try:
            m = imp.load_module(fullname, fp, filename, stuff)
        finally:
            if fp: fp.close()

        add_submodule(mod, m, fullname, subname)

    return m

def add_submodule(mod, submod, fullname, subname):
    """mod.{subname} = submod"""
    if mod is None:
        return #Nothing to do here.

    if submod is None:
        submod = sys.modules[fullname]

    setattr(mod, subname, submod)

    return

def ensure_fromlist(mod, fromlist, buf, recursive):
    """Handle 'from module import a, b, c' imports."""
    if not hasattr(mod, '__path__'):
        return
    for item in fromlist:
        if not hasattr(item, 'rindex'):
            raise TypeError("Item in ``from list'' not a string")
        if item == '*':
            if recursive:
                continue # avoid endless recursion
            try:
                all = mod.__all__
            except AttributeError:
                pass
            else:
                ret = ensure_fromlist(mod, all, buf, 1)
                if not ret:
                    return 0
        elif not hasattr(mod, item):
            import_submodule(mod, item, buf + '.' + item)

def import_module_level(name, globals=None, locals=None, fromlist=None, level=-1):
    """Replacement for __import__()"""
    parent, buf = get_parent(globals, level)

    head, name, buf = load_next(parent, None if level < 0 else parent, name, buf)

    tail = head
    while name:
        tail, name, buf = load_next(tail, tail, name, buf)

    # If tail is None, both get_parent and load_next found
    # an empty module name: someone called __import__("") or
    # doctored faulty bytecode
    if tail is None:
        raise ValueError('Empty module name')

    if not fromlist:
        return head

    ensure_fromlist(tail, fromlist, buf, 0)
    return tail

modules_reloading = {}

def reload_module(m):
    """Replacement for reload()."""
    if not isinstance(m, ModuleType):
        raise TypeError("reload() argument must be module")

    name = m.__name__

    if name not in sys.modules:
        raise ImportError("reload(): module %.200s not in sys.modules" % name)

    global modules_reloading
    try:
        return modules_reloading[name]
    except:
        modules_reloading[name] = m

    dot = name.rfind('.')
    if dot < 0:
        subname = name
        path = None
    else:
        try:
            parent = sys.modules[name[:dot]]
        except KeyError:
            modules_reloading.clear()
            raise ImportError("reload(): parent %.200s not in sys.modules" % name[:dot])
        subname = name[dot+1:]
        path = getattr(parent, "__path__", None)

    try:
        fp, filename, stuff  = imp.find_module(subname, path)
    finally:
        modules_reloading.clear()

    try:
        newm = imp.load_module(name, fp, filename, stuff)
    except:
         # load_module probably removed name from modules because of
         # the error.  Put back the original module object.
        sys.modules[name] = m
        raise
    finally:
        if fp: fp.close()

    modules_reloading.clear()
    return newm

# Save the original hooks
original_import = __builtin__.__import__
try:
    original_reload = __builtin__.reload
except AttributeError:
    original_reload = imp.reload    # Python 3

__builtin__.__import__ = import_module_level
__builtin__.reload = reload_module
