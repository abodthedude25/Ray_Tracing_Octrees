#include "RayTracerBVH.h"
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <limits>
#include <cmath>
#include <iostream>

RayTracerBVH::RayTracerBVH() : m_octreeRoot(nullptr) {}

RayTracerBVH::~RayTracerBVH() {}

// Set the octree and grid from which to “trace” the scene.
void RayTracerBVH::setOctree(OctreeNode* root, const VoxelGrid& grid) {
	m_octreeRoot = root;
	m_grid = grid;
}

// A helper to intersect a ray with an axis-aligned bounding box given by bmin and bmax.
// Returns true if there is an intersection and sets tNear and tFar.
bool RayTracerBVH::intersectAABB(const Ray& ray, const glm::vec3& bmin, const glm::vec3& bmax, float& tNear, float& tFar) {
	tNear = -std::numeric_limits<float>::max();
	tFar = std::numeric_limits<float>::max();

	for (int i = 0; i < 3; i++) {
		if (std::fabs(ray.direction[i]) < 1e-6f) {
			// Ray is nearly parallel to slab. No hit if origin not within slab
			if (ray.origin[i] < bmin[i] || ray.origin[i] > bmax[i]) return false;
		}
		else {
			float t1 = (bmin[i] - ray.origin[i]) / ray.direction[i];
			float t2 = (bmax[i] - ray.origin[i]) / ray.direction[i];
			if (t1 > t2) std::swap(t1, t2);
			if (t1 > tNear) tNear = t1;
			if (t2 < tFar) tFar = t2;
			if (tNear > tFar) return false;
			if (tFar < 0) return false;
		}
	}
	return true;
}

// Given a ray, traverse the octree recursively and find the closest hit that represents an interface.
// For simplicity we assume that a leaf (isLeaf==true) and isSolid==true is “occupied” (hit).
bool RayTracerBVH::intersectOctree(const Ray& ray, OctreeNode* node, float tMin, float tMax, glm::vec3& hitPoint, glm::vec3& hitNormal) {
	if (!node) return false;

	// Compute the bounding box of this node.
	// We assume that each node's box is given by:
	//   min: (grid.minX + node->x * grid.voxelSize, ...)
	//   max: min + (node->size * grid.voxelSize)
	float vx = m_grid.voxelSize;
	glm::vec3 boxMin(m_grid.minX + node->x * vx,
		m_grid.minY + node->y * vx,
		m_grid.minZ + node->z * vx);
	glm::vec3 boxMax = boxMin + glm::vec3(node->size * vx);

	float tNear, tFar;
	if (!intersectAABB(ray, boxMin, boxMax, tNear, tFar))
		return false;

	// Update tMin and tMax from the bounding box hit
	tMin = std::max(tMin, tNear);
	tMax = std::min(tMax, tFar);
	if (tMin > tMax)
		return false;

	// If this node is a leaf, then we have reached our finest resolution.
	if (node->isLeaf) {
		// For a simple example we assume that if node->isSolid is true,
		// then the surface is at the midpoint of the cell and the normal is computed from the gradient of the voxel grid.
		if (node->isSolid) {
			hitPoint = boxMin + 0.5f * (boxMax - boxMin);
			// Compute a simple normal by sampling the volume around the cell center.
			// (You could improve this by computing a gradient from the voxel data.)
			glm::vec3 sampleOffsets[6] = {
				glm::vec3(vx, 0, 0),
				glm::vec3(-vx, 0, 0),
				glm::vec3(0,  vx, 0),
				glm::vec3(0, -vx, 0),
				glm::vec3(0, 0,  vx),
				glm::vec3(0, 0, -vx)
			};
			glm::vec3 grad(0.0f);
			for (int i = 0; i < 6; i++) {
				// Here we “sample” the voxel state. In our grid,
				// the value is -1 for filled, and +1 for empty.
				// We use this difference to form a gradient.
				float samplePosValue = 0.f;
				glm::vec3 pos = hitPoint + sampleOffsets[i];
				// Compute indices from pos:
				int ix = (int)((pos.x - m_grid.minX) / m_grid.voxelSize);
				int iy = (int)((pos.y - m_grid.minY) / m_grid.voxelSize);
				int iz = (int)((pos.z - m_grid.minZ) / m_grid.voxelSize);
				if (ix < 0 || iy < 0 || iz < 0 ||
					ix >= m_grid.dimX || iy >= m_grid.dimY || iz >= m_grid.dimZ)
				{
					samplePosValue = 1.0f;
				}
				else {
					samplePosValue = (m_grid.data[m_grid.index(ix, iy, iz)] == VoxelState::FILLED) ? -1.f : +1.f;
				}
				// Use +/- offset to approximate derivative:
				grad += sampleOffsets[i] * samplePosValue;
			}
			if (glm::length(grad) > 1e-6f)
				hitNormal = glm::normalize(grad);
			else
				hitNormal = glm::vec3(0, 1, 0);
			return true;
		}
		return false;
	}

	// For an internal node, check all children and return the nearest hit.
	bool hitFound = false;
	float closestT = std::numeric_limits<float>::max();
	glm::vec3 tmpHit, tmpNormal;
	for (int i = 0; i < 8; i++) {
		if (node->children[i]) {
			if (intersectOctree(ray, node->children[i], tMin, tMax, tmpHit, tmpNormal)) {
				float tCandidate = glm::length(tmpHit - ray.origin);
				if (tCandidate < closestT) {
					closestT = tCandidate;
					hitPoint = tmpHit;
					hitNormal = tmpNormal;
					hitFound = true;
				}
			}
		}
	}
	return hitFound;
}

// A simple Lambertian shading function (white diffuse light from direction [-1,-1,-1])
glm::vec3 RayTracerBVH::shade(const glm::vec3& hitPoint, const glm::vec3& normal) {
	glm::vec3 lightDir = glm::normalize(glm::vec3(-1, -1, -1));
	float diff = std::max(glm::dot(normal, -lightDir), 0.f);
	// simple ambient plus diffuse shading
	return glm::vec3(0.2f) + diff * glm::vec3(0.8f);
}

// Generate a ray from the camera for a given pixel coordinate.
// We assume that invVP is the inverse of the view*projection matrix.
Ray RayTracerBVH::generateRay(int x, int y, int width, int height, const glm::mat4& invVP) {
	// Convert pixel coordinate to normalized device coordinates [-1, 1]
	float ndcX = (2.0f * x) / width - 1.0f;
	float ndcY = 1.0f - (2.0f * y) / height; // flip Y if needed

	glm::vec4 nearPoint = invVP * glm::vec4(ndcX, ndcY, -1.0f, 1.0f);
	glm::vec4 farPoint = invVP * glm::vec4(ndcX, ndcY, 1.0f, 1.0f);
	nearPoint /= nearPoint.w;
	farPoint /= farPoint.w;

	Ray ray;
	ray.origin = glm::vec3(nearPoint);
	ray.direction = glm::normalize(glm::vec3(farPoint - nearPoint));
	return ray;
}

// The main render function for the ray tracer. It loops over all pixels,
// casts rays through the scene, and builds a collection of tiny triangles (e.g. quads)
// that represent the computed radiance (you could also render to an image buffer).
// Here we “build” triangles for each hit pixel that can later be uploaded to GPU buffers.
std::vector<MCTriangle> RayTracerBVH::renderScene(const glm::mat4& view, const glm::mat4& proj, int width, int height) {
	std::vector<MCTriangle> triList;

	// Compute the inverse view-projection matrix for generating rays.
	glm::mat4 vp = proj * view;
	glm::mat4 invVP = glm::inverse(vp);

	// Loop over a subset (or every) pixel. (For performance you might want to
	// use multi-threading or lower resolution.)
	for (int y = 0; y < height; y += 2) {       // sample every other pixel
		for (int x = 0; x < width; x += 2) {
			Ray ray = generateRay(x, y, width, height, invVP);
			glm::vec3 hitPoint, hitNormal;
			if (intersectOctree(ray, m_octreeRoot, 0.f, std::numeric_limits<float>::max(), hitPoint, hitNormal)) {
				// For demonstration, generate a small quad centered at hitPoint, 
				// oriented perpendicular to hitNormal.
				glm::vec3 color = shade(hitPoint, hitNormal);

				// Create an orthonormal basis for the quad:
				glm::vec3 up = glm::abs(hitNormal.y) < 0.99f ? glm::vec3(0, 1, 0) : glm::vec3(1, 0, 0);
				glm::vec3 right = glm::normalize(glm::cross(hitNormal, up));
				glm::vec3 tangent = glm::normalize(glm::cross(right, hitNormal));

				float size = 0.01f; // size of the quad
				glm::vec3 v0 = hitPoint + (-right - tangent) * size;
				glm::vec3 v1 = hitPoint + (right - tangent) * size;
				glm::vec3 v2 = hitPoint + (-right + tangent) * size;
				glm::vec3 v3 = hitPoint + (right + tangent) * size;

				// Create two triangles for the quad.
				MCTriangle tri1, tri2;
				tri1.v[0] = v0; tri1.v[1] = v1; tri1.v[2] = v2;
				tri2.v[0] = v2; tri2.v[1] = v1; tri2.v[2] = v3;
				// For simplicity assign the same normal for each vertex.
				for (int i = 0; i < 3; i++) {
					tri1.normal[i] = hitNormal;
					tri2.normal[i] = hitNormal;
				}
				triList.push_back(tri1);
				triList.push_back(tri2);
			}
		}
	}
	return triList;
}
