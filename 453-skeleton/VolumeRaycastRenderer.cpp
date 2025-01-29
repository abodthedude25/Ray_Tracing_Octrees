// VolumeRaycastRenderer.cpp

#include "VolumeRaycastRenderer.h"
#include <glad/glad.h>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include <vector>

// Structure to hold splat vertex data (not directly used in this integrated approach)
struct SplatVertex {
	glm::vec3 position; // World space position
	glm::vec3 normal;   // Voxel normal for shading
	glm::vec3 color;    // Voxel base color
};

// -----------------------------
// Shader Sources
// -----------------------------

// Vertex Shader: Fullscreen Quad
static const char* raycastVertSrc = R"END(
#version 330 core
layout(location = 0) in vec2 aPos;
out vec2 TexCoord;

void main() {
    // Convert [-1,1]^2 to [0,1]^2
    TexCoord = 0.5 * (aPos + 1.0);
    gl_Position = vec4(aPos, 0.0, 1.0);
}
)END";

// Fragment Shader: Integrated Raycasting + Splatting
static const char* raycastFragSrc = R"END(
#version 330 core
in vec2 TexCoord;
out vec4 FragColor;

// Camera uniforms
uniform mat4 invView;
uniform mat4 invProj;
uniform vec3 camPos;

// Textures
uniform sampler3D uVolumeTex;
uniform sampler3D uMaskTex;

// Raymarching parameters
uniform float stepSize;
uniform float voxelSize;

// Lighting parameters
uniform vec3 lightDir;    // Should be normalized
uniform vec3 lightColor;  // e.g., white
uniform float ambient;

// Constants
const int MAX_STEPS = 1024;
const float OPACITY_THRESHOLD = 0.95;

// Helper function: Ray-box intersection for [-0.5, 0.5]^3
vec2 intersectBox(in vec3 rayOrig, in vec3 rayDir) {
    vec3 minB = vec3(-0.5);
    vec3 maxB = vec3( 0.5);

    vec3 t1 = (minB - rayOrig) / rayDir;
    vec3 t2 = (maxB - rayOrig) / rayDir;

    vec3 tNear = min(t1, t2);
    vec3 tFar  = max(t1, t2);

    float tN = max(tNear.x, max(tNear.y, tNear.z));
    float tF = min(tFar.x,  min(tFar.y,  tFar.z));

    return vec2(tN, tF);
}

// Helper function: Sample the volume density
float sampleVolume(vec3 pos) {
    return texture(uVolumeTex, pos).r;
}

// Helper function: Compute gradient using central differences
vec3 computeGradient(vec3 pos, float delta) {
    // Sample +/- delta in x, y, z
    vec3 ppx = clamp(pos + vec3(delta, 0.0, 0.0), 0.0, 1.0);
    vec3 pmx = clamp(pos - vec3(delta, 0.0, 0.0), 0.0, 1.0);
    vec3 ppy = clamp(pos + vec3(0.0, delta, 0.0), 0.0, 1.0);
    vec3 pmy = clamp(pos - vec3(0.0, delta, 0.0), 0.0, 1.0);
    vec3 ppz = clamp(pos + vec3(0.0, 0.0, delta), 0.0, 1.0);
    vec3 pmz = clamp(pos - vec3(0.0, 0.0, delta), 0.0, 1.0);

    float dx = sampleVolume(ppx) - sampleVolume(pmx);
    float dy = sampleVolume(ppy) - sampleVolume(pmy);
    float dz = sampleVolume(ppz) - sampleVolume(pmz);

    return vec3(dx, dy, dz) / (2.0 * delta);
}

void main()
{
    // Reconstruct NDC coordinates
    vec2 ndc = vec2(2.0 * TexCoord.x - 1.0, 1.0 - 2.0 * TexCoord.y);
    vec4 clipPos = vec4(ndc, 1.0, 1.0);

    // View space
    vec4 viewPos = invProj * clipPos;
    viewPos /= viewPos.w;

    // World space
    vec4 wPos = invView * viewPos;
    vec3 rayDir = normalize(wPos.xyz - camPos);
    vec3 rayOrig = camPos;

    // Ray-box intersection
    vec2 tBox = intersectBox(rayOrig, rayDir);
    float tNear = tBox.x;
    float tFar  = tBox.y;

    // If no intersection or behind camera
    if (tNear > tFar || tFar < 0.0) {
        FragColor = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    // Initialize
    float t = max(tNear, 0.0);
    float tExit = min(tFar, 100.0); // Arbitrary far plane
    float delta = voxelSize; // Step size for gradient calculation

    // Accumulation variables
    vec3 accumColor = vec3(0.0);
    float accumAlpha = 0.0;

    // Main raymarch loop with integrated splatting
    for(int i = 0; i < MAX_STEPS && t <= tExit; i++) {
        vec3 pos = rayOrig + t * rayDir;        // Current position in world space
        vec3 texCoord3 = pos + vec3(0.5);       // Map [-0.5..+0.5] to [0..1]

        // Check if texCoord3 is within [0,1]^3
        if(any(lessThan(texCoord3, vec3(0.0))) || any(greaterThan(texCoord3, vec3(1.0)))) {
            t += stepSize;
            continue;
        }

        // Mask check
        float maskVal = texture(uMaskTex, texCoord3).r;
        if(maskVal > 0.5) {
            t += stepSize;
            continue;
        }

        // Sample volume density
        float density = sampleVolume(texCoord3);
        if(density > 0.0) {
            // Compute gradient for shading
            vec3 grad = computeGradient(texCoord3, delta);

            // Normalize gradient to get normal
            vec3 normal = normalize(length(grad) > 1e-5 ? grad : vec3(0.0, 0.0, 1.0));

            // Lambertian shading
            float diffuse = max(dot(normal, lightDir), 0.0);
            vec3 shadedColor = ambient * lightColor + diffuse * lightColor;

            // Base color (can be modulated based on density or other factors)
            vec3 baseColor = vec3(1.0, 0.5, 0.2); // Example: Orange-like color
            vec3 finalColor = baseColor * shadedColor;

            // Compute opacity based on density
            float alpha = density * 0.1; // Scale opacity as needed

            // Front-to-back compositing
            accumColor += (1.0 - accumAlpha) * finalColor * alpha;
            accumAlpha += (1.0 - accumAlpha) * alpha;

            // Early termination if opacity threshold is reached
            if(accumAlpha >= OPACITY_THRESHOLD) {
                break;
            }
        }

        // Step forward along the ray
        t += stepSize;
    }

    // Final color with accumulated opacity
    FragColor = vec4(accumColor, accumAlpha);
}
)END";

// -----------------------------
// Helper Functions
// -----------------------------

// Helper function: Compile a shader (vertex or fragment)
unsigned int VolumeRaycastRenderer::compileShader(const char* src, GLenum type)
{
	unsigned int shader = glCreateShader(type);
	glShaderSource(shader, 1, &src, nullptr);
	glCompileShader(shader);

	// Check for compilation errors
	int success;
	char infoLog[512];
	glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
	if (!success)
	{
		glGetShaderInfoLog(shader, 512, nullptr, infoLog);
		std::cerr << "Shader compile error ("
			<< (type == GL_VERTEX_SHADER ? "VERTEX" : "FRAGMENT")
			<< "):\n" << infoLog << std::endl;
	}
	return shader;
}

// Helper function: Create the raycast shader program with integrated splatting
unsigned int VolumeRaycastRenderer::createRaycastProgram()
{
	unsigned int vs = compileShader(raycastVertSrc, GL_VERTEX_SHADER);
	unsigned int fs = compileShader(raycastFragSrc, GL_FRAGMENT_SHADER);

	unsigned int prog = glCreateProgram();
	glAttachShader(prog, vs);
	glAttachShader(prog, fs);
	glLinkProgram(prog);

	// Check for linking errors
	int success;
	char infoLog[512];
	glGetProgramiv(prog, GL_LINK_STATUS, &success);
	if (!success)
	{
		glGetProgramInfoLog(prog, 512, nullptr, infoLog);
		std::cerr << "Raycast Program link error:\n" << infoLog << std::endl;
	}

	glDeleteShader(vs);
	glDeleteShader(fs);
	return prog;
}

// -----------------------------
// Constructor & Destructor
// -----------------------------

VolumeRaycastRenderer::VolumeRaycastRenderer()
	: volumeTextureID(0), maskTextureID(0), raycastShaderProg(0),
	quadVAO(0), quadVBO(0),
	volDimX(0), volDimY(0), volDimZ(0),
	gridPtr(nullptr)
{
}

VolumeRaycastRenderer::~VolumeRaycastRenderer()
{
	if (volumeTextureID) {
		glDeleteTextures(1, &volumeTextureID);
	}
	if (maskTextureID) {
		glDeleteTextures(1, &maskTextureID);
	}
	if (raycastShaderProg) {
		glDeleteProgram(raycastShaderProg);
	}
	if (quadVAO) {
		glDeleteVertexArrays(1, &quadVAO);
		glDeleteBuffers(1, &quadVBO);
	}
}

// -----------------------------
// Render Stub (Unused)
// -----------------------------

std::vector<MCTriangle> VolumeRaycastRenderer::render(
	const OctreeNode* node,
	const VoxelGrid& grid,
	int x0, int y0, int z0,
	int size)
{
	// Splatting is integrated into raycasting; no separate render implementation
	return {};
}

// -----------------------------
// Initialization Methods
// -----------------------------

// Initialize 3D texture from the voxel grid
void VolumeRaycastRenderer::initVolume(const VoxelGrid& grid)
{
	volDimX = grid.dimX;
	volDimY = grid.dimY;
	volDimZ = grid.dimZ;

	gridPtr = &grid;

	// Convert voxel states to float data
	std::vector<float> volumeData(volDimX * volDimY * volDimZ, 0.0f);
	for (int z = 0; z < volDimZ; ++z) {
		for (int y = 0; y < volDimY; ++y) {
			for (int x = 0; x < volDimX; ++x) {
				int idx = x + y * volDimX + z * (volDimX * volDimY);
				if (grid.data[idx] == VoxelState::FILLED) {
					volumeData[idx] = 1.0f;
				}
				else {
					volumeData[idx] = 0.0f;
				}
			}
		}
	}

	// Create volume texture
	glGenTextures(1, &volumeTextureID);
	glBindTexture(GL_TEXTURE_3D, volumeTextureID);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

	glTexImage3D(
		GL_TEXTURE_3D,
		0,
		GL_R32F,
		volDimX, volDimY, volDimZ,
		0,
		GL_RED,
		GL_FLOAT,
		volumeData.data()
	);
	glBindTexture(GL_TEXTURE_3D, 0);

	// Initialize mask
	initMaskVolume(grid);

	// Create raycasting shader program with integrated splatting
	raycastShaderProg = createRaycastProgram();

	// Create fullscreen quad
	float fsQuadVerts[8] = {
		-1.0f, -1.0f,
		 1.0f, -1.0f,
		-1.0f,  1.0f,
		 1.0f,  1.0f
	};
	glGenVertexArrays(1, &quadVAO);
	glGenBuffers(1, &quadVBO);

	glBindVertexArray(quadVAO);
	glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(fsQuadVerts), fsQuadVerts, GL_STATIC_DRAW);

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
	glBindVertexArray(0);

	std::cout << "[VolumeRaycastRenderer] initVolume: dimX="
		<< volDimX << " dimY=" << volDimY
		<< " dimZ=" << volDimZ << "\n";
}

// Initialize mask texture based on a peeling plane
void VolumeRaycastRenderer::initMaskVolume(const VoxelGrid& grid)
{
	if (!gridPtr) {
		std::cerr << "[VolumeRaycastRenderer] initMaskVolume: gridPtr is null!\n";
		return;
	}

	std::vector<float> maskData(volDimX * volDimY * volDimZ, 0.0f);

	// Example: Plane-based peeling at z > 0.0
	for (int z = 0; z < volDimZ; z++) {
		float voxelZ = grid.minZ + (z + 0.5f) * grid.voxelSize;
		for (int y = 0; y < volDimY; y++) {
			for (int x = 0; x < volDimX; x++) {
				int idx = x + y * volDimX + z * (volDimX * volDimY);
				if (voxelZ > 0.0f) {
					maskData[idx] = 1.0f; // Carved away
				}
			}
		}
	}

	// Create mask texture
	glGenTextures(1, &maskTextureID);
	glBindTexture(GL_TEXTURE_3D, maskTextureID);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

	glTexImage3D(
		GL_TEXTURE_3D,
		0,
		GL_R32F,
		volDimX, volDimY, volDimZ,
		0,
		GL_RED,
		GL_FLOAT,
		maskData.data()
	);
	glBindTexture(GL_TEXTURE_3D, 0);

	std::cout << "[VolumeRaycastRenderer] initMaskVolume: Initialized mask with plane at z=0.0\n";
}

// -----------------------------
// Rendering Method
// -----------------------------

// Perform integrated raycasting and splatting
void VolumeRaycastRenderer::drawRaycast(const Camera& cam,
	float aspectRatio,
	int screenW, int screenH)
{
	if (!raycastShaderProg || !volumeTextureID || !maskTextureID) return;

	glUseProgram(raycastShaderProg);

	// Build inverse matrices
	glm::mat4 V = cam.getView();
	glm::mat4 P = glm::perspective(glm::radians(45.0f), aspectRatio, 0.01f, 100.0f);
	glm::mat4 invV = glm::inverse(V);
	glm::mat4 invP = glm::inverse(P);
	glm::vec3 camPos = cam.getPos();

	// Send uniforms
	GLint locInvV = glGetUniformLocation(raycastShaderProg, "invView");
	GLint locInvP = glGetUniformLocation(raycastShaderProg, "invProj");
	GLint locCamPos = glGetUniformLocation(raycastShaderProg, "camPos");
	GLint locVoxelSize = glGetUniformLocation(raycastShaderProg, "voxelSize");
	if (locInvV >= 0) glUniformMatrix4fv(locInvV, 1, GL_FALSE, glm::value_ptr(invV));
	if (locInvP >= 0) glUniformMatrix4fv(locInvP, 1, GL_FALSE, glm::value_ptr(invP));
	if (locCamPos >= 0) glUniform3fv(locCamPos, 1, glm::value_ptr(camPos));
	if (locVoxelSize >= 0) glUniform1f(locVoxelSize, gridPtr->voxelSize); // Assuming voxelSize is a member

	// Bind volume texture to texture unit 0
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_3D, volumeTextureID);
	GLint locVol = glGetUniformLocation(raycastShaderProg, "uVolumeTex");
	if (locVol >= 0) glUniform1i(locVol, 0);

	// Bind mask texture to texture unit 1
	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_3D, maskTextureID);
	GLint locMask = glGetUniformLocation(raycastShaderProg, "uMaskTex");
	if (locMask >= 0) glUniform1i(locMask, 1);

	// Set step size
	GLint locStep = glGetUniformLocation(raycastShaderProg, "stepSize");
	if (locStep >= 0) {
		glUniform1f(locStep, 0.005f);
	}

	// Set lighting uniforms
	GLint locLightDir = glGetUniformLocation(raycastShaderProg, "lightDir");
	if (locLightDir >= 0) {
		glm::vec3 lightDir = glm::normalize(glm::vec3(0.6f, 0.6f, 1.0f));
		glUniform3fv(locLightDir, 1, glm::value_ptr(lightDir));
	}

	GLint locLightColor = glGetUniformLocation(raycastShaderProg, "lightColor");
	if (locLightColor >= 0) {
		glm::vec3 lightColor = glm::vec3(1.0f); // White
		glUniform3fv(locLightColor, 1, glm::value_ptr(lightColor));
	}

	GLint locAmbient = glGetUniformLocation(raycastShaderProg, "ambient");
	if (locAmbient >= 0) {
		glUniform1f(locAmbient, 0.2f); // Small ambient factor
	}

	// Render fullscreen quad for integrated raycasting + splatting
	glBindVertexArray(quadVAO);
	glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
	glBindVertexArray(0);

	glUseProgram(0);
}

// -----------------------------
// Update Peeling Plane Method
// -----------------------------

// Update the peeling plane and corresponding mask texture
void VolumeRaycastRenderer::updatePeelPlane(float newZ)
{
	if (!gridPtr) {
		std::cerr << "[VolumeRaycastRenderer] updatePeelPlane: gridPtr is null!\n";
		return;
	}

	std::vector<float> maskData(volDimX * volDimY * volDimZ, 0.0f);
	float worldZ = newZ;

	for (int z = 0; z < volDimZ; z++) {
		float voxelZ = gridPtr->minZ + (z + 0.5f) * gridPtr->voxelSize;
		for (int y = 0; y < volDimY; y++) {
			for (int x = 0; x < volDimX; x++) {
				int idx = x + y * volDimX + z * (volDimX * volDimY);
				if (voxelZ > worldZ) {
					maskData[idx] = 1.0f; // Carved away
				}
			}
		}
	}

	// Update mask texture
	glBindTexture(GL_TEXTURE_3D, maskTextureID);
	glTexSubImage3D(
		GL_TEXTURE_3D,
		0,
		0, 0, 0,
		volDimX, volDimY, volDimZ,
		GL_RED,
		GL_FLOAT,
		maskData.data()
	);
	glBindTexture(GL_TEXTURE_3D, 0);

	std::cout << "[VolumeRaycastRenderer] Updated peeling plane to z = " << newZ << "\n";
}
