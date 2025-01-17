#pragma once

#include <array>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <glm/glm.hpp>

// Represents each voxel’s state
enum class VoxelState : uint8_t {
	EMPTY = 0,
	FILLED = 1
};

const int edgeToCorner[12][2] = {
	{0,1},{1,2},{2,3},{3,0},
	{4,5},{5,6},{6,7},{7,4},
	{0,4},{1,5},{2,6},{3,7}
};

// For Marching Cubes output
struct MCTriangle {
	glm::vec3 v[3];
	glm::vec3 normal[3]; // optional normals
};

// --------------------------------------------------------------------
// A container for voxel data in a single 3D grid
// --------------------------------------------------------------------
struct VoxelGrid {
	int dimX = 0;
	int dimY = 0;
	int dimZ = 0;
	float minX = 0.f;
	float minY = 0.f;
	float minZ = 0.f;
	float voxelSize = 1.f; // in world units
	std::vector<VoxelState> data;

	// Helper to get a flat index
	int index(int x, int y, int z) const {
		return x + y * dimX + z * (dimX * dimY);
	}
};

// --------------------------------------------------------------------
// Our Octree Node
// --------------------------------------------------------------------
struct OctreeNode {
	int x, y, z;      // voxel-space origin
	int size;         // width in voxels at this octree level
	bool isLeaf = false;
	bool isSolid = false; // if isLeaf, whether it's all FILLED or not
	OctreeNode* parent = nullptr;
	std::array<OctreeNode*, 8> children{ nullptr,nullptr,nullptr,nullptr,
										 nullptr,nullptr,nullptr,nullptr };

	// constructor
	OctreeNode(int _x, int _y, int _z, int _size)
		: x(_x), y(_y), z(_z), size(_size)
	{}
};

// --------------------------------------------------------------------
// Building the Octree from a VoxelGrid
// --------------------------------------------------------------------
OctreeNode* createOctreeFromVoxelGrid(const VoxelGrid& grid);

// Clean up
void freeOctree(OctreeNode* node);

// Helper to safely retrieve a voxel from the grid
VoxelState getVoxelSafe(const VoxelGrid& grid, int x, int y, int z);

// --------------------------------------------------------------------
// Hierarchical Traversal Helpers
// --------------------------------------------------------------------
OctreeNode* getParentCube(OctreeNode* node);

// Return the sub-cube index (which of the 8 children) for a coordinate
int getSubcubeIndex(int x, int y, int z,
	int halfSize,
	int x0, int y0, int z0);

// For adjacency in voxel-space: find up to 6 face-neighbors
std::vector<OctreeNode*> getNeighbors(OctreeNode* node,
	const std::unordered_map<long long, OctreeNode*>& nodeMap);

std::vector<MCTriangle> localMC(const VoxelGrid& grid, int x0, int y0, int z0, int size);
