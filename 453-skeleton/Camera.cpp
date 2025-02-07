#include "Camera.h"
#define _USE_MATH_DEFINES
#include <math.h>
#include <iostream>
#include "glm/gtc/matrix_transform.hpp"

Camera::Camera(float t, float p, float r)
	: theta(t), phi(p), radius(r), target(0.0f) {}

glm::mat4 Camera::getView() const {
	glm::vec3 eye = radius * glm::vec3(
		std::cos(theta) * std::sin(phi),
		std::sin(theta),
		std::cos(theta) * std::cos(phi)
	) + target;
	return glm::lookAt(eye, target, glm::vec3(0.0f, 1.0f, 0.0f));
}

glm::vec3 Camera::getPos() const {
	return radius * glm::vec3(std::cos(theta) * std::sin(phi),
		std::sin(theta),
		std::cos(theta) * std::cos(phi));
}

glm::vec3 Camera::getLookDir() const {
	// Since the camera always looks at the origin,
	// the look direction is from the eye toward the origin.
	glm::vec3 eye = getPos();
	return glm::normalize(-eye);
}

void Camera::incrementTheta(float dt) {
	if (theta + (dt / 100.0f) < M_PI_2 && theta + (dt / 100.0f) > -M_PI_2) {
		theta += dt / 100.0f;
	}
}

void Camera::incrementPhi(float dp) {
	phi -= dp / 100.0f;
	if (phi > 2.0 * M_PI) {
		phi -= 2.0 * M_PI;
	}
	else if (phi < 0.0f) {
		phi += 2.0 * M_PI;
	}
}

void Camera::incrementR(float dr) {
	radius = std::max(MIN_RADIUS, radius - dr);
}

void Camera::pan(float dx, float dy) {
	glm::vec3 right = glm::normalize(glm::cross(getLookDir(), glm::vec3(0, 1, 0)));
	glm::vec3 up = glm::normalize(glm::cross(right, getLookDir()));
	target += (-dx * right + dy * up) * (radius * 0.001f);
}
