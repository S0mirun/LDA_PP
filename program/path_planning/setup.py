from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

exts = [
    Extension(
        name="cython_func",          # ← import 名と一致
        sources=["cython_func.pyx"], # 追加の .c/.cpp があればここに列挙
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3"],
        language="c",                # C++なら "c++" に変更
    )
]

setup(
    name="cython_func",
    ext_modules=cythonize(exts, language_level="3"),  # Cython 3 でもOK
)