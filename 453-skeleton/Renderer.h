#pragma once

#include "OctreeVoxel.h"
#include <vector>
#include <unordered_map>
#include <glm/glm.hpp>


// ----------------------------
// Generic Renderer interface
// ----------------------------
class Renderer {
public:
	virtual std::vector<MCTriangle> render(const OctreeNode* node,
		const VoxelGrid& grid,
		int x0, int y0, int z0, int size) = 0;
	virtual ~Renderer() = default;
};

// ----------------------------
// Marching Cubes (unchanged)
// ----------------------------
class MarchingCubesRenderer : public Renderer {
public:
	std::vector<MCTriangle> render(const OctreeNode* node,
		const VoxelGrid& grid,
		int x0, int y0, int z0, int size) override;
};

// ----------------------------
// DC data per cell
// ----------------------------
struct DCCell {
	bool isMixed = false;
	glm::vec3 dcVertex{ 0.f };
	glm::vec3 dcNormal{ 0.f };
};

// For globally identifying a cell coordinate (plus the “lod” or size)
struct DCCellKey {
	int lod;   // which level-of-detail (or which 'size')
	int x, y, z;
	bool operator==(const DCCellKey& o) const {
		return (lod == o.lod && x == o.x && y == o.y && z == o.z);
	}
};
struct DCCellKeyHash {
	std::size_t operator()(const DCCellKey& k) const {
		auto h1 = std::hash<int>()(k.lod);
		auto h2 = std::hash<int>()(k.x);
		auto h3 = std::hash<int>()(k.y);
		auto h4 = std::hash<int>()(k.z);
		// Combine them
		h1 ^= (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2));
		h1 ^= (h3 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2));
		h1 ^= (h4 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2));
		return h1;
	}
};

// ----------------------------
// The Adaptive Dual Contouring
// ----------------------------
class AdaptiveDualContouringRenderer : public Renderer {
public:
	std::vector<MCTriangle> render(const OctreeNode* node,
		const VoxelGrid& grid,
		int x0, int y0, int z0, int size) override;

private:
	// 1) Uniform DC for a leaf (if no finer neighbor)
	void buildUniformDCCells(const VoxelGrid& grid,
		int x0, int y0, int z0,
		int size,
		std::vector<DCCell>& dcCells,
		int subDimX, int subDimY, int subDimZ,
		int lodLevel);

	std::vector<MCTriangle> buildUniformDCMesh(const VoxelGrid& grid,
		int x0, int y0, int z0,
		int subDimX, int subDimY, int subDimZ,
		const std::vector<DCCell>& dcCells,
		int lodLevel);

	// 2) Boundary stitching: key function
	void stitchBoundaryFace(const VoxelGrid& coarseGrid, const OctreeNode* coarseNode,
		int cx0, int cy0, int cz0, int cSize, int cLod,
		const VoxelGrid& fineGrid, const OctreeNode* fineNode,
		int fx0, int fy0, int fz0, int fSize, int fLod,
		std::vector<MCTriangle>& out);

	// Actually subdivides one coarse cell for bridging
	void subdivideCoarseCell(const VoxelGrid& coarseGrid,
		int cX, int cY, int cZ,  // one coarse cell origin
		int cLod,
		const VoxelGrid& fineGrid,
		int ratio,   // fSize / cSize
		// The difference in voxel coords
		std::vector<MCTriangle>& out);

	// 3) QEF, sampling, normals
	glm::vec3 solveQEF(const std::vector<glm::vec3>& points,
		const std::vector<glm::vec3>& normals);
	glm::vec3 computeNormal(const VoxelGrid& grid, int gx, int gy, int gz);
	glm::vec3 intersectEdge(const glm::vec3& p1, const glm::vec3& p2, float v1, float v2);
	float sampleVolume(const VoxelGrid& grid, int gx, int gy, int gz);

	// 4) For building geometry
	void addTriangle(const glm::vec3& v0, const glm::vec3& v1, const glm::vec3& v2,
		const glm::vec3& n0, const glm::vec3& n1, const glm::vec3& n2,
		std::vector<MCTriangle>& out);

	void addQuad(const glm::vec3& v0, const glm::vec3& v1,
		const glm::vec3& v2, const glm::vec3& v3,
		const glm::vec3& n0, const glm::vec3& n1,
		const glm::vec3& n2, const glm::vec3& n3,
		std::vector<MCTriangle>& out);

	std::vector<OctreeNode*> findLODNeighbors(
		const OctreeNode* node,
		int x0, int y0, int z0,
		int size);

private:
	// A global map storing the cell solutions: (lod, x, y, z) -> DCCell
	std::unordered_map<DCCellKey, DCCell, DCCellKeyHash> globalCellMap;
};
