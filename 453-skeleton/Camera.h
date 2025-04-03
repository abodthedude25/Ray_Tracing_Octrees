#pragma once

#include <glm/glm.hpp>

class Camera {
public:
    Camera(float t, float p, float r);

    // Return the view matrix (using the updated eye position)
    glm::mat4 getView() const;
    
    // Return the actual camera position (eye) including the pan offset.
    glm::vec3 getPos() const;
    
    // Return the normalized look direction (from the eye to the target)
    glm::vec3 getLookDir() const;

    // Increment (rotate) the camera angles and zoom.
    void incrementTheta(float dt);
    void incrementPhi(float dp);
    void incrementR(float dr);

	float getTheta() const;
	float getPhi() const;
	float getR() const;
	glm::mat4 getProj(float aspect) const;

	glm::vec3 getViewDir() const;
    
    // Pan the camera (this changes the target).
    void pan(float dx, float dy);
    
    // New: set the target (i.e. recenter the camera)
    void setTarget(const glm::vec3& newTarget);
    const glm::vec3& getTarget() const;

    float theta;
    float phi;
    float radius;
    glm::vec3 target; // target point (for panning/recentering)
    
    const float MIN_RADIUS = 0.1f;
    const float MAX_RADIUS = 1000.0f;
};
