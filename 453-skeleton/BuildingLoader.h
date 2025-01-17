#pragma once
#include <string>
#include "OctreeVoxel.h"

// Load buildings from a .gdb file and fill a voxel grid
VoxelGrid loadBuildingsFromGDB(const std::string& gdbPath, float voxelSize, const std::vector<int>& structureIDs);
