#include "BVH.h"
#include <algorithm>
#include <cmath>

// Compute the axis–aligned bounding box for a triangle.
static AABB computeAABB(const Triangle* tri) {
	AABB box;
	box.expand(tri->v0);
	box.expand(tri->v1);
	box.expand(tri->v2);
	return box;
}

// Compute the centroid of a triangle.
static glm::vec3 centroid(const Triangle* tri) {
	return (tri->v0 + tri->v1 + tri->v2) / 3.0f;
}

// ------------------ BVH Construction ------------------
BVH::BVH(const std::vector<Triangle>& triangles) {
	// Create a vector of pointers to the triangles.
	std::vector<const Triangle*> tris;
	tris.reserve(triangles.size());
	for (size_t i = 0; i < triangles.size(); i++) {
		tris.push_back(&triangles[i]);
	}
	root = build(tris, 0);
}

BVH::~BVH() {
	freeNode(root);
}

BVHNode* BVH::build(std::vector<const Triangle*>& tris, int depth) {
	BVHNode* node = new BVHNode();
	node->left = node->right = nullptr;

	// Compute the bounding box for all triangles.
	AABB bounds;
	for (const auto& tri : tris) {
		AABB triBox = computeAABB(tri);
		bounds.expand(triBox);
	}
	node->bounds = bounds;

	// If few triangles, create a leaf.
	if (tris.size() <= 2) {
		node->triangles = tris;
		return node;
	}

	// Choose axis based on the extent.
	glm::vec3 extent = bounds.max - bounds.min;
	int axis = 0;
	if (extent.y > extent.x) axis = 1;
	if (extent.z > extent[axis]) axis = 2;

	// Sort triangles by centroid along the chosen axis.
	std::sort(tris.begin(), tris.end(), [axis](const Triangle* a, const Triangle* b) {
		return centroid(a)[axis] < centroid(b)[axis];
		});

	// Split the list at the median.
	size_t mid = tris.size() / 2;
	std::vector<const Triangle*> leftTris(tris.begin(), tris.begin() + mid);
	std::vector<const Triangle*> rightTris(tris.begin() + mid, tris.end());

	node->left = build(leftTris, depth + 1);
	node->right = build(rightTris, depth + 1);

	return node;
}

// ------------------ Ray-AABB Intersection ------------------
// We use the slab method. 'invDir' is the component–wise inverse of the ray direction.
// 'dirIsNeg' is a vector of three integers (0 or 1) indicating if each component of the ray's
// direction is negative.
static bool intersectAABB(const AABB& box, const glm::vec3& origin, const glm::vec3& invDir,
	const std::vector<int>& dirIsNeg, float tmin, float tmax) {
	for (int i = 0; i < 3; i++) {
		float t0 = ((dirIsNeg[i] ? box.max[i] : box.min[i]) - origin[i]) * invDir[i];
		float t1 = ((dirIsNeg[i] ? box.min[i] : box.max[i]) - origin[i]) * invDir[i];
		tmin = t0 > tmin ? t0 : tmin;
		tmax = t1 < tmax ? t1 : tmax;
		if (tmax < tmin)
			return false;
	}
	return true;
}

// ------------------ BVH Query ------------------
void BVH::queryNode(BVHNode* node, const glm::vec3& origin, const glm::vec3& invDir,
	const std::vector<int>& dirIsNeg, float tmin, float tmax,
	std::vector<const Triangle*>& outCandidates) const {
	if (!node)
		return;
	if (!intersectAABB(node->bounds, origin, invDir, dirIsNeg, tmin, tmax))
		return;
	// If leaf, add all triangles.
	if (!node->left && !node->right) {
		for (const auto tri : node->triangles) {
			outCandidates.push_back(tri);
		}
		return;
	}
	queryNode(node->left, origin, invDir, dirIsNeg, tmin, tmax, outCandidates);
	queryNode(node->right, origin, invDir, dirIsNeg, tmin, tmax, outCandidates);
}

void BVH::query(const glm::vec3& origin, const glm::vec3& direction,
	std::vector<const Triangle*>& outCandidates) const {
	// Precompute inverse direction.
	glm::vec3 invDir(1.0f / direction.x, 1.0f / direction.y, 1.0f / direction.z);
	std::vector<int> dirIsNeg = { (invDir.x < 0), (invDir.y < 0), (invDir.z < 0) };
	queryNode(root, origin, invDir, dirIsNeg, 0.0f, std::numeric_limits<float>::max(), outCandidates);
}

// ------------------ BVH Node Freeing ------------------
void BVH::freeNode(BVHNode* node) {
	if (!node)
		return;
	freeNode(node->left);
	freeNode(node->right);
	delete node;
}
