#include "Camera.h"
#define _USE_MATH_DEFINES
#include <math.h>
#include "glm/gtc/matrix_transform.hpp"
#include <algorithm>


Camera::Camera(float t, float p, float r)
	: theta(t), phi(p), radius(r), target(0.0f) {}

glm::mat4 Camera::getView() const {
	// Compute the eye as (radius * directional vector) plus the target offset.
	glm::vec3 eye = radius * glm::vec3(
		std::cos(theta) * std::sin(phi),
		std::sin(theta),
		std::cos(theta) * std::cos(phi)
	) + target;
	return glm::lookAt(eye, target, glm::vec3(0.0f, 1.0f, 0.0f));
}

glm::vec3 Camera::getPos() const {
	// Now getPos() returns the actual eye position (including target offset)
	glm::vec3 eye = radius * glm::vec3(
		std::cos(theta) * std::sin(phi),
		std::sin(theta),
		std::cos(theta) * std::cos(phi)
	) + target;
	return eye;
}

float Camera::getTheta() const {
	return theta;
}

float Camera::getPhi() const {
	return phi;
}

float Camera::getR() const {
	return radius;
}

glm::vec3 Camera::getLookDir() const {
	// Look direction from eye to target
	glm::vec3 eye = getPos();
	return glm::normalize(target - eye);
}

void Camera::incrementTheta(float dt) {
	// Limit theta so the camera does not flip.
	float newTheta = theta + dt / 100.0f;
	if (newTheta < M_PI_2 && newTheta > -M_PI_2) {
		theta = newTheta;
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
	// Compute right and up vectors based on the current look direction.
	glm::vec3 right = glm::normalize(glm::cross(getLookDir(), glm::vec3(0, 1, 0)));
	glm::vec3 up = glm::normalize(glm::cross(right, getLookDir()));
	// Adjust the target by an amount proportional to the current zoom (radius)
	target += (-dx * right + dy * up) * (radius * 0.001f);
}

void Camera::setTarget(const glm::vec3& newTarget) {
	target = newTarget;
}

const glm::vec3& Camera::getTarget() const {
	return target;
}
