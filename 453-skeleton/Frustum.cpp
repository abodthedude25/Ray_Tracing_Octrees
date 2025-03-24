#include "Frustum.h"
#include <glm/gtc/matrix_access.hpp>


// Constructor that builds the frustum from view and projection matrices
Frustum::Frustum(const glm::mat4& viewProj) {
	// Extract planes from the combined view-projection matrix
	// Left plane
	m_planes[LEFT].x = viewProj[0][3] + viewProj[0][0];
	m_planes[LEFT].y = viewProj[1][3] + viewProj[1][0];
	m_planes[LEFT].z = viewProj[2][3] + viewProj[2][0];
	m_planes[LEFT].w = viewProj[3][3] + viewProj[3][0];

	// Right plane
	m_planes[RIGHT].x = viewProj[0][3] - viewProj[0][0];
	m_planes[RIGHT].y = viewProj[1][3] - viewProj[1][0];
	m_planes[RIGHT].z = viewProj[2][3] - viewProj[2][0];
	m_planes[RIGHT].w = viewProj[3][3] - viewProj[3][0];

	// Bottom plane
	m_planes[BOTTOM].x = viewProj[0][3] + viewProj[0][1];
	m_planes[BOTTOM].y = viewProj[1][3] + viewProj[1][1];
	m_planes[BOTTOM].z = viewProj[2][3] + viewProj[2][1];
	m_planes[BOTTOM].w = viewProj[3][3] + viewProj[3][1];

	// Top plane
	m_planes[TOP].x = viewProj[0][3] - viewProj[0][1];
	m_planes[TOP].y = viewProj[1][3] - viewProj[1][1];
	m_planes[TOP].z = viewProj[2][3] - viewProj[2][1];
	m_planes[TOP].w = viewProj[3][3] - viewProj[3][1];

	// Near plane
	m_planes[NEAR].x = viewProj[0][3] + viewProj[0][2];
	m_planes[NEAR].y = viewProj[1][3] + viewProj[1][2];
	m_planes[NEAR].z = viewProj[2][3] + viewProj[2][2];
	m_planes[NEAR].w = viewProj[3][3] + viewProj[3][2];

	// Far plane
	m_planes[FAR].x = viewProj[0][3] - viewProj[0][2];
	m_planes[FAR].y = viewProj[1][3] - viewProj[1][2];
	m_planes[FAR].z = viewProj[2][3] - viewProj[2][2];
	m_planes[FAR].w = viewProj[3][3] - viewProj[3][2];

	// Normalize all plane normals
	for (int i = 0; i < COUNT; i++) {
		float len = glm::length(glm::vec3(m_planes[i]));
		m_planes[i] /= len;
	}
}

// Test if a bounding box (AABB) is inside, outside, or intersecting the frustum
// Returns: 1 = fully inside, 0 = intersecting, -1 = fully outside
int Frustum::testAABB(const glm::vec3& min, const glm::vec3& max, float extraMargin) const {
	// For debugging
	//std::cout << "Testing AABB: (" << min.x << "," << min.y << "," << min.z << ") to ("
	//          << max.x << "," << max.y << "," << max.z << ")\n";

	// Expand the box by the extra margin to prevent popping
	glm::vec3 expandedMin = min - glm::vec3(extraMargin);
	glm::vec3 expandedMax = max + glm::vec3(extraMargin);

	int result = 1; // Assume inside

	for (int i = 0; i < COUNT; i++) {
		// Find the point furthest in the direction of the plane normal (positive vertex)
		glm::vec3 p(
			m_planes[i].x > 0 ? expandedMax.x : expandedMin.x,
			m_planes[i].y > 0 ? expandedMax.y : expandedMin.y,
			m_planes[i].z > 0 ? expandedMax.z : expandedMin.z
		);

		// If the positive vertex is outside, the box is completely outside the frustum
		if (glm::dot(glm::vec3(m_planes[i]), p) + m_planes[i].w < 0) {
			//std::cout << "  Outside plane " << i << "\n";
			return -1; // Outside
		}

		// Find the point closest to the plane (negative vertex)
		glm::vec3 n(
			m_planes[i].x < 0 ? expandedMax.x : expandedMin.x,
			m_planes[i].y < 0 ? expandedMax.y : expandedMin.y,
			m_planes[i].z < 0 ? expandedMax.z : expandedMin.z
		);

		// If the negative vertex is outside, the box is intersecting the frustum
		if (glm::dot(glm::vec3(m_planes[i]), n) + m_planes[i].w < 0) {
			//std::cout << "  Intersecting plane " << i << "\n";
			result = 0; // Intersecting
		}
	}

	//std::cout << "  Result: " << result << "\n";
	return result;
}
