from distutils.core import setup
from Cython.Build import cythonize
#from Cython.Distutils import bu
import numpy

# get the market maker, main sniper, and websocket lib files
files_to_compile = [
    "bitmex_sniper.py",
    "bm15mema_optimize.py",
    "market_maker/bitmex.py",
    "market_maker/ws/ws_thread.py",
    "market_maker/auth/APIKeyAuth.py",
    "market_maker/auth/APIKeyAuthWithExpires.py",
    "market_maker/utils/log.py",
    "websocket/*.py",
    "globalvar.py",
    "get_history_bitmex.py",
    "ucr_classifier.py",
    "motif_matching_knn.pyx"
]

ext_options = {"compiler_directives": {"profile": True}, "annotate": True}
#ext_options = {}
setup(
    ext_modules=cythonize(files_to_compile, include_path=[numpy.get_include()], **ext_options),
)
