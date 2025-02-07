#pragma once
#include <glm/glm.hpp>
#include <vector>
#include <limits>

// The Triangle structure (make sure this matches your own definition)
struct Triangle {
	glm::vec3 v0;
	glm::vec3 v1;
	glm::vec3 v2;
};

// Axis-aligned bounding box structure.
struct AABB {
	glm::vec3 min;
	glm::vec3 max;

	AABB() {
		min = glm::vec3(std::numeric_limits<float>::max());
		max = glm::vec3(-std::numeric_limits<float>::max());
	}

	// Expand the box to include point p.
	void expand(const glm::vec3& p) {
		min = glm::min(min, p);
		max = glm::max(max, p);
	}

	// Expand the box to include another box.
	void expand(const AABB& other) {
		expand(other.min);
		expand(other.max);
	}
};

// BVH node structure.
struct BVHNode {
	AABB bounds;
	BVHNode* left;
	BVHNode* right;
	std::vector<const Triangle*> triangles; // Non-empty for leaf nodes.
};

class BVH {
public:
	// Build the BVH from a list of triangles.
	BVH(const std::vector<Triangle>& triangles);
	~BVH();

	// Query the BVH with a ray; returns candidate triangles that intersect the ray's AABB.
	void query(const glm::vec3& origin, const glm::vec3& direction, std::vector<const Triangle*>& outCandidates) const;

private:
	BVHNode* root;

	// Recursively build the BVH. 'tris' is a list of triangle pointers.
	BVHNode* build(std::vector<const Triangle*>& tris, int depth);
	// Recursively query a node.
	void queryNode(BVHNode* node, const glm::vec3& origin, const glm::vec3& invDir, const std::vector<int>& dirIsNeg,
		float tmin, float tmax, std::vector<const Triangle*>& outCandidates) const;
	// Recursively free a node.
	void freeNode(BVHNode* node);
};

