#include "RayTracerBVH.h"
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <limits>
#include <cmath>
#include <iostream>

RayTracerBVH::RayTracerBVH() : m_octreeRoot(nullptr) {}

RayTracerBVH::~RayTracerBVH() {}

void RayTracerBVH::setOctree(OctreeNode* root, const VoxelGrid& grid) {
	m_octreeRoot = root;
	m_grid = grid;
}

// A helper to intersect a ray with an axis-aligned bounding box given by bmin and bmax.
// Returns true if there is an intersection and sets tNear and tFar.
bool RayTracerBVH::intersectAABB(const Ray& ray, const glm::vec3& bmin, const glm::vec3& bmax,
	float& tNear, float& tFar) {
	glm::vec3 invDir = 1.0f / ray.direction;
	glm::vec3 t1 = (bmin - ray.origin) * invDir;
	glm::vec3 t2 = (bmax - ray.origin) * invDir;

	glm::vec3 tMin = glm::min(t1, t2);
	glm::vec3 tMax = glm::max(t1, t2);

	tNear = std::max(std::max(tMin.x, tMin.y), tMin.z);
	tFar = std::min(std::min(tMax.x, tMax.y), tMax.z);

	return tNear <= tFar && tFar > 0;
}

// Given a ray, traverse the octree recursively and find the closest hit that represents an interface.
// For simplicity we assume that a leaf (isLeaf==true) and isSolid==true is “occupied” (hit).

bool RayTracerBVH::intersectOctree(const Ray& ray, OctreeNode* node, float tMin, float tMax,
	glm::vec3& hitPoint, glm::vec3& hitNormal) {
	if (!node) return false;

	float vx = m_grid.voxelSize;
	glm::vec3 boxMin(m_grid.minX + node->x * vx,
		m_grid.minY + node->y * vx,
		m_grid.minZ + node->z * vx);
	glm::vec3 boxMax = boxMin + glm::vec3(node->size * vx);

	float tNear, tFar;
	if (!intersectAABB(ray, boxMin, boxMax, tNear, tFar)) {
		return false;
	}

	if (node->isLeaf && node->isSolid) {
		// Use intersection point as hit point
		hitPoint = ray.origin + ray.direction * tNear;

		// Compute normal based on hit face
		glm::vec3 center = (boxMin + boxMax) * 0.5f;
		hitNormal = glm::normalize(hitPoint - center);

		return true;
	}

	if (!node->isLeaf) {
		float closest = std::numeric_limits<float>::max();
		bool hit = false;
		glm::vec3 tempHit, tempNormal;

		for (int i = 0; i < 8; i++) {
			if (node->children[i] &&
				intersectOctree(ray, node->children[i], tMin, tMax, tempHit, tempNormal)) {
				float dist = glm::length(tempHit - ray.origin);
				if (dist < closest) {
					closest = dist;
					hitPoint = tempHit;
					hitNormal = tempNormal;
					hit = true;
				}
			}
		}
		return hit;
	}

	return false;
}

// A simple Lambertian shading function (white diffuse light from direction [-1,-1,-1])
glm::vec3 RayTracerBVH::shade(const glm::vec3& hitPoint, const glm::vec3& normal, const glm::vec3& cameraPos) {
	// Simplified shading for debugging
	return glm::vec3(1.0f, 0.8f, 0.6f); // Just return a constant color for now
}

// Generate a ray from the camera for a given pixel coordinate.
// We assume that invVP is the inverse of the view*projection matrix.
Ray RayTracerBVH::generateRay(int x, int y, int width, int height, const glm::mat4& invVP) {
	// Convert pixel to NDC space
	float ndcX = (2.0f * x) / width - 1.0f;
	float ndcY = 1.0f - (2.0f * y) / height;  // Flip Y for OpenGL

	// Unproject near and far points
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

std::vector<MCTriangle> RayTracerBVH::renderScene(const Camera& camera, const glm::mat4& view,
	const glm::mat4& proj, int width, int height) {
	std::vector<MCTriangle> triList;
	triList.reserve(width * height / 2); // Pre-allocate for better performance

	glm::mat4 vp = proj * view;
	glm::mat4 invVP = glm::inverse(vp);

	// Calculate scene scale
	glm::vec3 sceneCenter = camera.getTarget();
	float sceneDist = glm::length(camera.getPos() - sceneCenter);
	float baseQuadSize = sceneDist * 0.001f;

	// Smaller step size for denser coverage
	const int stepSize = 1; // Sample every pixel for better coverage

#pragma omp parallel for collapse(2) // Enable OpenMP parallelization if available
	for (int y = 0; y < height; y += stepSize) {
		for (int x = 0; x < width; x += stepSize) {
			Ray ray = generateRay(x, y, width, height, invVP);
			glm::vec3 hitPoint, hitNormal;

			if (intersectOctree(ray, m_octreeRoot, 0.f, std::numeric_limits<float>::max(),
				hitPoint, hitNormal)) {

				// Distance-based quad sizing
				float dist = glm::length(hitPoint - camera.getPos());
				float quadSize = baseQuadSize * (dist / sceneDist);
				quadSize = std::max(quadSize, 0.001f); // Ensure minimum size

				// Create basis vectors for quad
				glm::vec3 viewDir = glm::normalize(camera.getPos() - hitPoint);
				glm::vec3 right = glm::normalize(glm::cross(hitNormal, glm::vec3(0, 1, 0)));
				if (glm::length(right) < 0.001f) {
					right = glm::normalize(glm::cross(hitNormal, glm::vec3(1, 0, 0)));
				}
				glm::vec3 up = glm::normalize(glm::cross(right, hitNormal));

				// Generate quad corners
				glm::vec3 v0 = hitPoint + (-right - up) * quadSize;
				glm::vec3 v1 = hitPoint + (right - up) * quadSize;
				glm::vec3 v2 = hitPoint + (-right + up) * quadSize;
				glm::vec3 v3 = hitPoint + (right + up) * quadSize;

				// Create triangles
				MCTriangle tri1, tri2;
				tri1.v[0] = v0; tri1.v[1] = v1; tri1.v[2] = v2;
				tri2.v[0] = v2; tri2.v[1] = v1; tri2.v[2] = v3;

				for (int i = 0; i < 3; i++) {
					tri1.normal[i] = hitNormal;
					tri2.normal[i] = hitNormal;
				}

				// Thread-safe insertion
#pragma omp critical
				{
					triList.push_back(tri1);
					triList.push_back(tri2);
				}
			}
		}
	}

	return triList;
}
