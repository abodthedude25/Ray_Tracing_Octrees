#pragma once

#include "OctreeVoxel.h"
#include <vector>
#include <unordered_map>
#include <glm/glm.hpp>
#include "Camera.h" 

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
// Ray Tracer
// This renderer traverses the octree and generates a cube for each solid leaf,
// but only emits faces on the boundary (where adjacent voxels are empty).
// ============================================================
class VoxelCubeRenderer : public Renderer {
public:
	virtual std::vector<MCTriangle> render(const OctreeNode* node,
		const VoxelGrid& grid,
		int x0, int y0, int z0,
		int size) override;
private:
	// This function builds cube face geometry for a solid octree leaf (of any size)
	// and appends exposed faces (only if the neighboring voxels in the grid are empty).
	void addBlockFaces(const VoxelGrid& grid,
		int x0, int y0, int z0, int size,
		std::vector<MCTriangle>& out);

	// Helpers: add one face (a quad split into two triangles) for each face.
	void addFacePosX(const glm::vec3& minC, const glm::vec3& maxC, std::vector<MCTriangle>& out);
	void addFaceNegX(const glm::vec3& minC, const glm::vec3& maxC, std::vector<MCTriangle>& out);
	void addFacePosY(const glm::vec3& minC, const glm::vec3& maxC, std::vector<MCTriangle>& out);
	void addFaceNegY(const glm::vec3& minC, const glm::vec3& maxC, std::vector<MCTriangle>& out);
	void addFacePosZ(const glm::vec3& minC, const glm::vec3& maxC, std::vector<MCTriangle>& out);
	void addFaceNegZ(const glm::vec3& minC, const glm::vec3& maxC, std::vector<MCTriangle>& out);

	// A helper to add a quad as two triangles given four vertices and a face normal.
	void addQuad(const glm::vec3& v0, const glm::vec3& v1,
		const glm::vec3& v2, const glm::vec3& v3,
		const glm::vec3& normal,
		std::vector<MCTriangle>& out);
};

enum class StitchFace {
	POS_X,
	NEG_X,
	POS_Y,
	NEG_Y,
	POS_Z,
	NEG_Z
};

