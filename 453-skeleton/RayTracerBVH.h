#pragma once

#include <vector>
#include <glm/glm.hpp>
#include "OctreeVoxel.h"

// A simple Ray structure.
struct Ray {
	glm::vec3 origin;
	glm::vec3 direction;
};

class RayTracerBVH {
public:
	RayTracerBVH();
	~RayTracerBVH();

	// Set the octree that serves as our BVH accelerator.
	void setOctree(OctreeNode* root, const VoxelGrid& grid);

	// Render the scene by ray tracing. Returns a list of triangles (or other geometry)
	// that can be rendered with your existing pipeline.
	std::vector<MCTriangle> renderScene(const glm::mat4& view, const glm::mat4& proj, int width, int height);

private:
	// The recursive function that tests a ray against the BVH (octree).
	bool intersectOctree(const Ray& ray, OctreeNode* node, float tMin, float tMax, glm::vec3& hitPoint, glm::vec3& hitNormal);

	// A helper to intersect a ray with an axis-aligned bounding box.
	bool intersectAABB(const Ray& ray, const glm::vec3& bmin, const glm::vec3& bmax, float& tNear, float& tFar);

	// A simple function to “shade” an intersection point.
	glm::vec3 shade(const glm::vec3& hitPoint, const glm::vec3& normal);

	// Generate a ray from camera parameters and pixel coordinates.
	Ray generateRay(int x, int y, int width, int height, const glm::mat4& invVP);

	// The BVH (octree) data:
	OctreeNode* m_octreeRoot;
	VoxelGrid m_grid;
};

