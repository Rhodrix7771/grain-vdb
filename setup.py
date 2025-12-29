import os
import subprocess
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

class MetalBuildExt(build_ext):
    def run(self):
        # 1. Compile Metal Kernel
        print("🔨 Compiling Metal Kernel...")
        src_metal = "src/grain_kernel.metal"
        out_metallib = "grainvdb/gv_kernel.metallib"
        
        # Intermediate .air file
        try:
            subprocess.run(["xcrun", "-sdk", "macosx", "metal", "-c", src_metal, "-o", "gv_kernel.air"], check=True)
            subprocess.run(["xcrun", "-sdk", "macosx", "metallib", "gv_kernel.air", "-o", out_metallib], check=True)
            os.remove("gv_kernel.air")
        except Exception as e:
            print(f"Warning: Failed to compile Metal kernel. Querying will fail on GPU: {e}")

        # 2. Build the Shared Library
        print("🔨 Building Dynamic Library...")
        src_cpp = "src/grainvdb.mm"
        out_dylib = "libgrainvdb.dylib"
        
        cmd = [
            "clang++", "-dynamiclib", "-std=c++17", "-O3",
            "-Iinclude",
            "-framework", "Metal", "-framework", "Foundation",
            src_cpp, "-o", out_dylib
        ]
        try:
            subprocess.run(cmd, check=True)
        except Exception as e:
            print(f"Error: Failed to build native C++ core: {e}")
            raise

        super().run()

setup(
    name="grainvdb",
    version="0.1.0",
    description="Native Metal-Accelerated Vector Engine for Apple Silicon",
    author="Adam Sussman",
    packages=["grainvdb"],
    cmdclass={'build_ext': MetalBuildExt},
    python_requires=">=3.9",
)
