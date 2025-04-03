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

	Frustum(const glm::mat4& viewProj);
	int testAABB(const glm::vec3& min, const glm::vec3& max, float extraMargin) const;

private:
	// The six planes of the frustum
	std::array<glm::vec4, COUNT> m_planes;
};
