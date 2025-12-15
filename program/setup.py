from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

exts = [
    Extension(
        name="utils.PP.cython_func",           # ← import 名と一致させる
        sources=["utils/PP/cython_func.pyx"],  # ← 実際の .pyx への相対パス
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3"],
        language="c",                          # C++なら "c++"
    )
]

setup(
    name="lda_pp_cython_ext",                 # 適当なパッケージ名でOK
    ext_modules=cythonize(exts, language_level="3"),
)