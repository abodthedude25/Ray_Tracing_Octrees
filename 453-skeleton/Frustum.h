#pragma once

#include <array>
#include <glm/glm.hpp>

class Frustum {
public:
	enum Planes {
		LEFT = 0,
		RIGHT,
		TOP,
		BOTTOM,
		NEAR,
		FAR,
		COUNT
	};

	// Constructor that builds the frustum from view and projection matrices
	Frustum(const glm::mat4& viewProj);

	// Test if a bounding box (AABB) is inside, outside, or intersecting the frustum
	// Returns: 1 = fully inside, 0 = intersecting, -1 = fully outside
	int testAABB(const glm::vec3& min, const glm::vec3& max, float extraMargin) const;

private:
	// The six planes of the frustum
	std::array<glm::vec4, COUNT> m_planes;
};
