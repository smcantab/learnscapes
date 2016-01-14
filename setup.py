import glob
import os
import sys
import subprocess

from numpy.distutils.core import setup
from numpy.distutils.core import Extension
from numpy.distutils.misc_util import has_cxx_sources
import numpy as np

# Numpy header files 
numpy_lib = os.path.split(np.__file__)[0] 
numpy_include = os.path.join(numpy_lib, 'core/include') 

#
# Make the git revision visible.  Most of this is copied from scipy
# 
# Return the git revision as a string
def git_version():
    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ['SYSTEMROOT', 'PATH']:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env['LANGUAGE'] = 'C'
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'
        out = subprocess.Popen(cmd, stdout = subprocess.PIPE, env=env).communicate()[0]
        return out

    try:
        out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
        GIT_REVISION = out.strip().decode('ascii')
    except OSError:
        GIT_REVISION = "Unknown"

    return GIT_REVISION

def write_version_py(filename='learnscapes/version.py'):
    cnt = """
# THIS FILE IS GENERATED FROM SCIPY SETUP.PY
git_revision = '%(git_revision)s'
"""
    GIT_REVISION = git_version()

    a = open(filename, 'w')
    try:
        a.write(cnt % dict(git_revision=GIT_REVISION))
    finally:
        a.close()
write_version_py()



setup(name='learnscapes', 
      version='0.1', 
      description="Learning representation through energy landscapes",
      url='https://smcantab@bitbucket.org/smcantab/learnscapes',
      packages=["learnscapes",
                "learnscapes.pspinspherical",
                "learnscapes.NN",
                ],
      ext_modules=ext_modules,
      # data files needed for the tests
      # data_files=[('pele/potentials/tests', list(glob.glob('pele/potentials/tests/*.xyz'))),
      #            ]
        )
