
# README

This repository demonstrates **voxelization** of either a **test sphere** volume **or** building footprints loaded from the **City of Calgary** `.gdb` folder, and then **renders** the resulting voxel data via an **octree** and two volumetric surface extraction methods:
1. **Marching Cubes**
2. **Dual Contouring** (Adaptive)

NOTE: make sure to upload the city of Calgary .gdb folder under the gdb_folder 

## 1. **Overview**

1. **Load or Generate** a voxel volume:
   - **Test Sphere**  
     The code can generate a synthetic sphere by filling a 3D volume with `VoxelState::FILLED` inside a radius, and `VoxelState::EMPTY` outside.
   - **.GDB Building**  
     Alternatively, the code reads a `.gdb` geodatabase containing building polygons for the City of Calgary, finds the bounding box, voxelizes them into a 3D grid, and optionally re-centers around the origin.

2. **Build Octree**  
   A function \(`createOctreeFromVoxelGrid()`\) constructs an octree from the voxel grid by recursively subdividing until regions are uniform or the node is at minimum size.

3. **Render** in **Real-Time**  
   Depending on user input:
   - **Marching Cubes (MC)**: Recursively extracts triangles from each leaf node in the octree to approximate the surface. 
   - **Dual Contouring (DC)**: An alternative volumetric surface approach that can reduce polygon count or produce sharper edges.

4. **Wireframe Debug**  
   A toggle (`S`) allows you to see the **octree wireframe** bounding each leaf node. Another toggle (`W`) sets fill vs. wireframe polygon modes for the surface mesh.

5. **Profiling**  
   The code prints **FPS** and other profiling details (e.g. triangle count) once per second in the console.

## 2. **Repository Layout**

- **CMakeLists.txt**  
  Root build file using [CMake](https://cmake.org/).
- **453-skeleton/**  
  Folder containing:
  - **main.cpp**  
    Entry point where you choose either test sphere or `.gdb` building. 
    Handles toggling between MC/DC.  
  - **BuildingLoader.cpp/.h**  
    Code that voxelizes building footprints from `.gdb`.  
  - **OctreeVoxel.cpp/.h**  
    Functions to build and manage the octree, plus local MC code.  
  - **Renderer.cpp/.h**  
    Classes for Marching Cubes or Dual Contouring. 
- **shaders/**  
  GLSL vertex + fragment shaders for rendering geometry.
- **third-party/Window**, **Camera**, etc.  
  Utility classes for an OpenGL-based window, camera controls, etc.

## 3. **Usage**

1. **Clone** or download this repo.  
2. **Install** [Vcpkg](https://github.com/microsoft/vcpkg) if you don’t already have it. You’ll also need [GDAL](https://gdal.org/) for `.gdb` support, plus the usual OpenGL libraries.
3. **Adjust** any paths in `BuildingLoader.cpp` if needed to find `STRUCT_ID` or other fields.

### A) **Choose** between Sphere or GDB
Inside `main.cpp`, there is a toggle like:
```cpp
bool useGDB = true; 
std::string gdbPath = "./textures/Buildings_3D.gdb";
```
- If `useGDB = false;` -> We generate a **test sphere**.
- If `useGDB = true;`  -> We load building footprints from the `.gdb` file at `gdbPath`.

### B) **Voxelizing** the Building (if `useGDB = true`)
- `BuildingLoader.cpp` finds bounding box from building polygons.
- A voxel array is allocated (`dimX, dimY, dimZ`).
- Each polygon is rasterized in **XY** plane for each Z layer (if desired).  
- The code sets `grid.data[idx] = FILLED` if inside the polygon, else `EMPTY`.

### C) **Re-Center** (optional)
If you want the building around the origin, you can re-center by calling:
```cpp
recenterFilledVoxels(grid);
```
*before* building the octree.

### D) **Build Octree**
```cpp
OctreeNode* root = createOctreeFromVoxelGrid(grid);
```
This subdivides until uniform or minimum size is reached.

### E) **Render** in Real-Time
- Press **R** to toggle **Marching Cubes** vs. **Dual Contouring**.
- Press **W** to toggle fill vs. wireframe rendering of the **surface**.
- Press **S** to toggle **octree wireframe** bounding each leaf node.

## 4. **Building the Project**

From the **root** of the repository:

1. **CMake Configure**  
   ```bash
   cmake -B build -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake
   ```
   Adjust the path for your local vcpkg install.

2. **CMake Build**  
   ```bash
   cmake --build build
   ```
   This compiles all sources and places the executable in `build/Debug/` or `build/Release`.

3. **Run**  
   ```bash
   ./build/Debug/453-skeleton-program.exe
   ```
   or on Windows:
   ```powershell
   .\build\Debug\453-skeleton-program.exe
   ```
   This launches the **OpenGL** window with the chosen mode (sphere or GDB).

## 5. **Keyboard Controls**

- **R** : Toggle rendering method (Marching Cubes / Dual Contouring).  
- **W** : Toggle fill vs. wireframe polygon mode for the extracted surface.  
- **S** : Toggle octree bounding-wire display in red.  
- Right-drag + mouse** : Rotate camera angles (\(\theta\) and \(\phi\)).  
- Mouse wheel** : Zoom.  

