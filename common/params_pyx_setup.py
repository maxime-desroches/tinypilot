import os
from distutils.core import Extension, setup
from Cython.Build import cythonize
from common.cython_hacks import BuildExtWithoutPlatformSuffix
from common.basedir import BASEDIR

sourcefiles = ['params_wrapper.pyx']
extra_compile_args = ["-std=c++17"]

setup(name='Common',
      cmdclass={'build_ext': BuildExtWithoutPlatformSuffix},
      ext_modules=cythonize(
        Extension(
          "params_pyx",
          language="c++",
          sources=sourcefiles,
          include_dirs=[BASEDIR, os.path.join(BASEDIR, 'selfdrive')],
          extra_compile_args=extra_compile_args
        )
      )
)
