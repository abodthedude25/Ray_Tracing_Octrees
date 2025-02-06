#pragma once
#include <string>
#include "OctreeVoxel.h"


// A simple 2D bounding box structure.
struct BoundingBox2D {
	float minX, minY, maxX, maxY;
};

// Load buildings from a .gdb file and fill a voxel grid
VoxelGrid loadBuildingsFromGDB(const std::string& gdbPath,
	float voxelSize,
	float minXUser,
	float minYUser,
	float maxXUser,
	float maxYUser,
	int   maxBuildings);
VoxelGrid loadBuildingsFromGDB(const std::string& gdbPath,
	float voxelSize,
	const std::vector<int>& structureIDs);

BoundingBox2D detectDenseArea(const std::string& gdbPath, int maxFeatures, float cellSize);

BoundingBox2D findSkyscraperArea(const std::string& gdbPath, int maxFeatures, float cellSize);

BoundingBox2D detectTallestArea(const std::string& gdbPath, int maxFeatures, int topCount);
