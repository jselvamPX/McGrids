from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension
import os
import torch

# Get absolute paths
current_dir = os.path.dirname(os.path.abspath(__file__))
differentiable_mcmt_dir = os.path.join(current_dir, 'differentiable_mcmt')
geogram_lib_dir = '/home/jeeva.selvam/geogram/build/Linux64-gcc-dynamic-Release/lib'
geogram_include_dir = '/home/jeeva.selvam/geogram/src/lib'
torch_lib_dir = os.path.dirname(torch.__file__)

setup(
    name='mcmt',
    ext_modules=[
        CppExtension(
            name='mcmt',
            sources=[
                os.path.join('python', 'mcmt_torch.cpp'),
                os.path.join('differentiable_mcmt', 'fast_mcmt.cpp')
            ],
            include_dirs=[
                'include/mcmt',
                differentiable_mcmt_dir,
                geogram_include_dir,
                os.path.join(geogram_include_dir, 'third_party')
            ],
            library_dirs=[
                geogram_lib_dir,
                os.path.join(torch_lib_dir, 'lib')
            ],
            runtime_library_dirs=[
                geogram_lib_dir,
                os.path.join(torch_lib_dir, 'lib')
            ],
            libraries=[
                'geogram',
                'tbb',
                'c10'
            ],
            extra_compile_args=[
                '-O3',
                '-std=c++17',
                '-DNDEBUG',
                f'-I{differentiable_mcmt_dir}',
                '-DGEO_DYNAMIC_LIBS',
                '-fPIC',
            ],
            extra_link_args=[
                f'-L{geogram_lib_dir}',
                '-Wl,--no-as-needed',
                f'-Wl,-rpath,{geogram_lib_dir}',
                f'-Wl,-rpath,{os.path.join(torch_lib_dir, "lib")}',
                '-lgeogram'
            ],
            define_macros=[
                ('GEO_DYNAMIC_LIBS', None)
            ]
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)