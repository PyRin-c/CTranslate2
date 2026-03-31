import glob
import os
import sys

import pybind11

from pybind11.setup_helpers import ParallelCompile
from setuptools import Extension, setup

include_dirs = [pybind11.get_include()]
library_dirs = []


def _maybe_add_library_root(lib_name):
    if "%s_ROOT" % lib_name in os.environ:
        root = os.environ["%s_ROOT" % lib_name]
        include_dirs.append("%s/include" % root)
        for lib_dir in ("lib", "lib64"):
            path = "%s/%s" % (root, lib_dir)
            if os.path.exists(path):
                library_dirs.append(path)
                break


_maybe_add_library_root("CTRANSLATE2")

# Allow overriding include/lib dirs independently (useful for split source/build trees)
if "CTRANSLATE2_INCLUDE_DIR" in os.environ:
    include_dirs.append(os.environ["CTRANSLATE2_INCLUDE_DIR"])
if "CTRANSLATE2_LIB_DIR" in os.environ:
    library_dirs.append(os.environ["CTRANSLATE2_LIB_DIR"])

cflags = ["-std=c++17", "-fvisibility=hidden"]
ldflags = []
package_data = {}
if sys.platform == "darwin":
    # std::visit requires macOS 10.14
    cflags.append("-mmacosx-version-min=10.14")
    ldflags.append("-Wl,-rpath,/usr/local/lib")
elif sys.platform == "win32":
    cflags = ["/std:c++17", "/d2FH4-"]
    package_data["ctranslate2"] = ["*.dll"]
elif sys.platform == "linux":
    cflags.append("-fPIC")
    ldflags.append("-Wl,-rpath,/usr/local/lib64:/usr/local/lib")

ctranslate2_module = Extension(
    "ctranslate2._ext",
    sources=glob.glob(os.path.join("cpp", "*.cc")),
    extra_compile_args=cflags,
    extra_link_args=ldflags,
    include_dirs=include_dirs,
    library_dirs=library_dirs,
    libraries=["ctranslate2"],
)

ParallelCompile("CMAKE_BUILD_PARALLEL_LEVEL").install()

setup(
    package_data=package_data,
    ext_modules=[ctranslate2_module],
)
