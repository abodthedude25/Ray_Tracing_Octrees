#pragma once
#include "OctreeVoxel.h"
#include <vector>
#include <glm/glm.hpp>
#include "Camera.h"

// Forward declarations
class OctreeNode;
class VoxelGrid;

// Ray structure for ray tracing
struct Ray {
	glm::vec3 origin;
	glm::vec3 direction;
};

class RayTracerBVH {
public:
	RayTracerBVH();
	~RayTracerBVH();

	// Set the octree and grid
	void setOctree(OctreeNode* root, const VoxelGrid& grid);

	// Render the scene
	std::vector<MCTriangle> renderScene(const Camera& camera, const glm::mat4& view,
		const glm::mat4& proj, int width, int height);

private:
	// Ray intersection helpers
	bool intersectAABB(const Ray& ray, const glm::vec3& bmin, const glm::vec3& bmax,
		float& tNear, float& tFar);
	bool intersectOctree(const Ray& ray, OctreeNode* node, float tMin, float tMax,
		glm::vec3& hitPoint, glm::vec3& hitNormal);

	// Shading and ray generation
	glm::vec3 shade(const glm::vec3& hitPoint, const glm::vec3& normal, const glm::vec3& cameraPos);
	Ray generateRay(int x, int y, int width, int height, const glm::mat4& invVP);

private:
	// Scene data
	OctreeNode* m_octreeRoot;
	VoxelGrid m_grid;
};
