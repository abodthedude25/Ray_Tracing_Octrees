#pragma once

#include <glm/glm.hpp>

//------------------------------------------------------------------------------
// Spherical Camera
//------------------------------------------------------------------------------
class Camera {
public:
	Camera(float t, float p, float r);

	// Mark these methods as const so they can be called on a const Camera.
	glm::mat4 getView() const;
	glm::vec3 getPos() const;

	// Added: A method to return the normalized look direction.
	// Since your camera always looks at the origin, the look direction is -normalize(eye).
	glm::vec3 getLookDir() const;

	void incrementTheta(float dt);
	void incrementPhi(float dp);
	void incrementR(float dr);
	void pan(float dx, float dy);

private:
	float theta;
	float phi;
	float radius;
	const float MIN_RADIUS = 0.1f;
	const float MAX_RADIUS = 1000.0f;
	glm::vec3 target;
};
