#!/bin/bash
set -e

echo "🏗️  Building GrainVDB Native Core..."

# 1. Compile Metal Kernel to binary library
echo "🌀 Compiling Metal Shaders..."
xcrun -sdk macosx metal -c src/grain_kernel.metal -o gv_kernel.air
mkdir -p grainvdb
xcrun -sdk macosx metallib gv_kernel.air -o grainvdb/gv_kernel.metallib
rm gv_kernel.air

# 2. Compile Objective-C++ Bridge to Dynamic Library
echo "💎 Compiling libgrainvdb.dylib..."
clang++ -dynamiclib -std=c++17 -O3 \
    -Iinclude \
    -framework Metal -framework Foundation \
    src/grainvdb.mm -o libgrainvdb.dylib

echo "✅ Build Complete. Run 'python3 benchmark.py' to verify."
