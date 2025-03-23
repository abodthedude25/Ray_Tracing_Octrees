#include "Renderer.h"
#include "OctreeVoxel.h"
#include <glm/gtx/norm.hpp>
#include <cmath>
#include <array>
#include <limits>
#include <iostream>
#include <unordered_map>
#include <vector>
#include <cstdint>
std::unordered_map<long long, OctreeNode*> g_octreeMap;

// --------------------------------------------------------
// MARCHING CUBES
// --------------------------------------------------------
std::vector<MCTriangle> MarchingCubesRenderer::render(const OctreeNode* node,
	const VoxelGrid& grid,
	int x0, int y0, int z0, int size)
{
	if (!node) return {};
	std::vector<MCTriangle> results;
	if (node->isLeaf) {
		// local uniform MC
		auto mcTris = localMC(grid, x0, y0, z0, size);
		results.insert(results.end(), mcTris.begin(), mcTris.end());
	}
	else {
		int half = size / 2;
		for (int i = 0; i < 8; i++) {
			int ox = x0 + ((i & 1) ? half : 0);
			int oy = y0 + ((i & 2) ? half : 0);
			int oz = z0 + ((i & 4) ? half : 0);
			auto childTris = render(node->children[i], grid, ox, oy, oz, half);
			results.insert(results.end(), childTris.begin(), childTris.end());
		}
	}
	return results;
}


// --------------------------------------------------------
// NEW: VOXEL CUBE RENDERER (Block-based Voxel Geometry)
// --------------------------------------------------------
std::vector<MCTriangle> VoxelCubeRenderer::render(const OctreeNode* node,
	const VoxelGrid& grid,
	int x0, int y0, int z0, int size)
{
	std::vector<MCTriangle> out;
	if (!node) return out;
	if (node->isLeaf) {
		if (node->isSolid) {
			addBlockFaces(grid, x0, y0, z0, size, out);
		}
	}
	else {
		int half = size / 2;
		for (int i = 0; i < 8; i++) {
			int ox = x0 + ((i & 1) ? half : 0);
			int oy = y0 + ((i & 2) ? half : 0);
			int oz = z0 + ((i & 4) ? half : 0);
			auto childTris = render(node->children[i], grid, ox, oy, oz, half);
			out.insert(out.end(), childTris.begin(), childTris.end());
		}
	}
	return out;
}

void VoxelCubeRenderer::addBlockFaces(const VoxelGrid& grid,
	int x0, int y0, int z0, int size,
	std::vector<MCTriangle>& out)
{
	float vx = grid.voxelSize;
	glm::vec3 minCorner(grid.minX + x0 * vx,
		grid.minY + y0 * vx,
		grid.minZ + z0 * vx);
	glm::vec3 ext(size * vx);
	glm::vec3 maxCorner = minCorner + ext;

	// A simple check at the center of each face is used to decide if the face is
	// exposed. (More elaborate methods could check over the entire face.)
	auto checkFace = [&](int testX, int testY, int testZ) -> bool {
		if (testX < 0 || testY < 0 || testZ < 0 ||
			testX >= grid.dimX || testY >= grid.dimY || testZ >= grid.dimZ)
			return true;
		return (grid.data[grid.index(testX, testY, testZ)] == VoxelState::EMPTY);
		};

	bool posXExposed = checkFace(x0 + size, y0 + size / 2, z0 + size / 2);
	if (posXExposed) addFacePosX(minCorner, maxCorner, out);
	bool negXExposed = checkFace(x0 - 1, y0 + size / 2, z0 + size / 2);
	if (negXExposed) addFaceNegX(minCorner, maxCorner, out);

	bool posYExposed = checkFace(x0 + size / 2, y0 + size, z0 + size / 2);
	if (posYExposed) addFacePosY(minCorner, maxCorner, out);
	bool negYExposed = checkFace(x0 + size / 2, y0 - 1, z0 + size / 2);
	if (negYExposed) addFaceNegY(minCorner, maxCorner, out);

	bool posZExposed = checkFace(x0 + size / 2, y0 + size / 2, z0 + size);
	if (posZExposed) addFacePosZ(minCorner, maxCorner, out);
	bool negZExposed = checkFace(x0 + size / 2, y0 + size / 2, z0 - 1);
	if (negZExposed) addFaceNegZ(minCorner, maxCorner, out);
}

void VoxelCubeRenderer::addFacePosX(const glm::vec3& minC, const glm::vec3& maxC, std::vector<MCTriangle>& out)
{
	glm::vec3 v0(maxC.x, minC.y, minC.z);
	glm::vec3 v1(maxC.x, maxC.y, minC.z);
	glm::vec3 v2(maxC.x, maxC.y, maxC.z);
	glm::vec3 v3(maxC.x, minC.y, maxC.z);
	glm::vec3 normal(1, 0, 0);
	addQuad(v0, v1, v3, v2, normal, out);
}
void VoxelCubeRenderer::addFaceNegX(const glm::vec3& minC, const glm::vec3& maxC, std::vector<MCTriangle>& out)
{
	glm::vec3 v0(minC.x, minC.y, minC.z);
	glm::vec3 v1(minC.x, minC.y, maxC.z);
	glm::vec3 v2(minC.x, maxC.y, maxC.z);
	glm::vec3 v3(minC.x, maxC.y, minC.z);
	glm::vec3 normal(-1, 0, 0);
	addQuad(v0, v1, v3, v2, normal, out);
}
void VoxelCubeRenderer::addFacePosY(const glm::vec3& minC, const glm::vec3& maxC, std::vector<MCTriangle>& out)
{
	glm::vec3 v0(minC.x, maxC.y, minC.z);
	glm::vec3 v1(minC.x, maxC.y, maxC.z);
	glm::vec3 v2(maxC.x, maxC.y, maxC.z);
	glm::vec3 v3(maxC.x, maxC.y, minC.z);
	glm::vec3 normal(0, 1, 0);
	addQuad(v0, v1, v3, v2, normal, out);
}
void VoxelCubeRenderer::addFaceNegY(const glm::vec3& minC, const glm::vec3& maxC, std::vector<MCTriangle>& out)
{
	glm::vec3 v0(minC.x, minC.y, minC.z);
	glm::vec3 v1(maxC.x, minC.y, minC.z);
	glm::vec3 v2(maxC.x, minC.y, maxC.z);
	glm::vec3 v3(minC.x, minC.y, maxC.z);
	glm::vec3 normal(0, -1, 0);
	addQuad(v0, v1, v3, v2, normal, out);
}
void VoxelCubeRenderer::addFacePosZ(const glm::vec3& minC, const glm::vec3& maxC, std::vector<MCTriangle>& out)
{
	glm::vec3 v0(minC.x, minC.y, maxC.z);
	glm::vec3 v1(minC.x, maxC.y, maxC.z);
	glm::vec3 v2(maxC.x, maxC.y, maxC.z);
	glm::vec3 v3(maxC.x, minC.y, maxC.z);
	glm::vec3 normal(0, 0, 1);
	addQuad(v0, v1, v3, v2, normal, out);
}
void VoxelCubeRenderer::addFaceNegZ(const glm::vec3& minC, const glm::vec3& maxC, std::vector<MCTriangle>& out)
{
	glm::vec3 v0(minC.x, minC.y, minC.z);
	glm::vec3 v1(maxC.x, minC.y, minC.z);
	glm::vec3 v2(maxC.x, maxC.y, minC.z);
	glm::vec3 v3(minC.x, maxC.y, minC.z);
	glm::vec3 normal(0, 0, -1);
	addQuad(v0, v1, v3, v2, normal, out);
}

void VoxelCubeRenderer::addQuad(const glm::vec3& v0, const glm::vec3& v1,
	const glm::vec3& v2, const glm::vec3& v3,
	const glm::vec3& normal,
	std::vector<MCTriangle>& out)
{
	MCTriangle tri1;
	tri1.v[0] = v0; tri1.v[1] = v1; tri1.v[2] = v2;
	tri1.normal[0] = normal; tri1.normal[1] = normal; tri1.normal[2] = normal;
	out.push_back(tri1);
	MCTriangle tri2;
	tri2.v[0] = v2; tri2.v[1] = v1; tri2.v[2] = v3;
	tri2.normal[0] = normal; tri2.normal[1] = normal; tri2.normal[2] = normal;
	out.push_back(tri2);
}
