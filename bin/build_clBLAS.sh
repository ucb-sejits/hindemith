#!/bin/bash

wget https://github.com/clMathLibraries/clBLAS/archive/master.zip
unzip master.zip
cd clBLAS-master
mkdir build && cd build
cmake ../src -DCMAKE_BUILD_TYPE=Release -DOCL_VERSION="1.1" -DBUILD_TEST=OFF -DBUILD_KTEST=OFF -DCMAKE_INSTALL_PREFIX=.
CORES=$(grep -c ^processor /proc/cpuinfo 2>/dev/null || sysctl -n hw.ncpu)
make -j$CORES && make install
cd ../../
if [ "$(uname)" == "Darwin" ]; then
  cp clBLAS-master/build/lib/libclBLAS*.dylib hindemith/clibs
elif [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
  cp clBLAS-master/build/lib64/libclBLAS*.so hindemith/clibs
fi
rm -rf clBLAS-master/
rm master.zip
