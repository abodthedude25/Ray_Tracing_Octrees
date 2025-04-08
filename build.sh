#!/bin/bash

# Configure display for X11 forwarding
export DISPLAY=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}'):0

# Edit CMakeLists.txt to use a different target name
cp CMakeLists.txt CMakeLists.txt.bak
sed -i 's/add_executable(453-skeleton-program/add_executable(my-skeleton-program/' CMakeLists.txt

# Build the project
rm -rf build
mkdir build
cd build
cmake .. -DGLFW_BUILD_WAYLAND=OFF -DGLFW_BUILD_X11=ON
make -j$(nproc)

echo "Build complete. Run with: ./my-skeleton-program"