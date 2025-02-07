#pragma once
#include <vector>
#include <string>
#include <fstream>
#include "BuildingLoader.h"  // for VoxelGrid definition

bool saveVoxelGrid(const std::string& filename, const VoxelGrid& grid);
bool loadVoxelGrid(const std::string& filename, VoxelGrid& grid);
bool loadVoxelGridPartial(const std::string& filename, VoxelGrid& grid, int startLayer, int numLayers);
