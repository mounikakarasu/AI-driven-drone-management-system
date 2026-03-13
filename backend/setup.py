from setuptools import setup, Extension
import pybind11

functions_module = Extension(
    'drone_safety',
    sources=['core/safety_engine.cpp'],
    include_dirs=[pybind11.get_include()],
    language='c++'
)

setup(
    name='drone_safety',
    version='0.1',
    ext_modules=[functions_module],
)