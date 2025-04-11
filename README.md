# Volumetric Building Renderer

This repository contains an advanced volumetric rendering system that supports multiple rendering techniques for visualizing voxel-based building data.

## Features

- **Multiple Rendering Modes**:
  - **Marching Cubes**: Traditional surface extraction algorithm that approximates the isosurface between filled and empty voxels with triangles.
  - **Dual Contouring**: Advanced algorithm that produces sharper features at corners and edges compared to Marching Cubes, while typically using fewer triangles.
  - **Voxel Blocks**: Direct visualization of the voxel grid using cube faces, showing the raw building blocks with minimal processing.
  - **BVH Ray Tracing**: GPU-accelerated ray tracing using a Bounding Volume Hierarchy for fast intersection tests and high-quality lighting.
  - **Volume Raycasting**: Direct volume rendering with interactive carving capabilities, using ray marching through the 3D volume with adaptive sampling.

- **Efficient Data Structures**:
  - **Octree-based spatial partitioning**: Hierarchical structure that adaptively subdivides space to reduce memory usage and accelerate rendering.
  - **GPU-optimized acceleration structures**: SSBO-based storage of scene data for parallelized GPU computation.
  - **Frustum culling**: Skip rendering parts of the scene that aren't visible to the camera.
  - **Edge caching**: Optimized handling of edge intersections for faster surface extraction.

- **Interactive Features**:
  - **Real-time camera navigation**: Smooth orbit-style camera controls for exploring the 3D model.
  - **Volumetric carving**: In Volume Raycast mode, interactively remove parts of the building to see inside.
  - **Adaptive level of detail**: Automatically adjusts detail level based on distance from the camera.
  - **Temporal coherence**: Techniques to improve frame-to-frame stability and reduce flickering.
  - **Empty space skipping**: Accelerated rendering by quickly passing through empty regions of the volume.

- **Visual Enhancements**:
  - **Edge detection**: Enhanced rendering of building edges and boundaries for better visual clarity.
  - **Lighting models**: Physically-based shading with shadows and ambient occlusion.
  - **Radiation visualization**: Glowing visualization of carved regions in Volume Raycast mode.
  - **Building-to-building boundaries**: Visual differentiation between adjacent buildings.

## Building and Running

### Windows (Visual Studio 2022)

From the root directory of the project:

1. Configure the project:
Now you can build the project with or without GDAL support:

With GDAL (if you have it installed):
```cmake -B build -G "Visual Studio 17 2022" -A x64```

Without GDAL (for systems where GDAL is missing):
```cmake -B build -G "Visual Studio 17 2022" -A x64 -DUSE_GDAL=OFF```

You'll also need to modify any code that uses GDAL to check for the USE_GDAL preprocessor definition

2. Build the project:
   ```
   cmake --build build
   ```

3. Run the executable:
   ```
   .\build\Debug\453-skeleton-program.exe
   ```

### Linux

A build script is included in the repository to handle CMake configuration issues on Linux. Simply run:

```bash
chmod +x build.sh
./build.sh
```

After building, run the executable:

```bash
cd build
./my-skeleton-program
```

The build script handles setting the display for X11 forwarding, modifying the target name to avoid conflicts, and configuring CMake with the appropriate options for Linux.

**Note for WSL users**: An X server like VcXsrv must be installed on Windows and configured properly to display the OpenGL window.

## Controls

### General Controls
- **R** - Cycle through rendering modes (Marching Cubes → Voxel Blocks → Dual Contouring → Volume Raycast → BVH Ray Trace)
- **W** - Toggle wireframe mode for geometry
- **S** - Toggle octree wireframe visualization
- **F** - Force update of frustum culling
- **G** - Toggle forced regeneration of Dual Contouring triangles
- **C** - Center camera on building (in BVH Ray Trace mode)

### Navigation
- **Right Mouse Button + Drag** - Rotate camera view
- **Left Mouse Button + Drag** - Pan camera
- **Mouse Wheel** - Zoom in/out

### Volume Raycasting Mode
- **Left Mouse Button** - Carve the volume at the clicked location
- **O** - Toggle octree-based ray skipping for improved performance
- **M** - Toggle MIP-mapped skipping

### BVH Ray Tracing Mode
- **V** - Toggle volume measurement

### Visualization
- **Up/Down Arrow Keys** - Adjust peeling plane position
- **X** - Toggle render mode visualization

## Data Loading

The application can load building data from the City of Calgary GDB format or generate a test sphere volume. The data loading option is controlled in the code.

The GDB loader extracts building footprints and converts them to a 3D voxel representation automatically. For test purposes, a multi-shell sphere can be generated as an alternative dataset.

## Performance Notes

- **Frustum culling** is enabled by default to improve performance by skipping invisible parts of the model
- The application uses various acceleration techniques:
  - **Adaptive sampling**: Adjusts ray step size based on scene complexity
  - **Multiple acceleration structures**: Octree, BVH, and distance field optimizations
  - **GPU computation**: GLSL compute shaders for parallel processing
  - **Ray skipping**: Multiple techniques to avoid sampling empty space
  - **Temporal caching**: Reuses results between frames when possible
  - **View-dependent level of detail**: Simplifies distant geometry

## Troubleshooting

### WSL Graphics Issues
When running in WSL:
1. Install VcXsrv or another X server on Windows
2. Launch it with "Disable access control" checked
3. Set the DISPLAY environment variable in WSL
4. Install necessary OpenGL libraries with `sudo apt install libgl1-mesa-glx libglu1-mesa mesa-utils`

### Build Issues
- If experiencing CMake configuration errors, check if GLFW is trying to use Wayland instead of X11
- Explicitly specify `-DGLFW_BUILD_WAYLAND=OFF -DGLFW_BUILD_X11=ON` when configuring with CMake
