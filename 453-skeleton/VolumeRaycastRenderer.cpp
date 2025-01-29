// VolumeRaycastRenderer.cpp

#include "VolumeRaycastRenderer.h"
#include <glad/glad.h>
#include <string>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>

// -----------------------------
// Vertex Shader: Fullscreen Quad
// -----------------------------
static const char* raycastVertSrc = R"(
#version 330 core
layout(location = 0) in vec2 aPos;
out vec2 TexCoord;
void main() {
    // Convert [-1,1]^2 => [0,1]^2
    TexCoord = 0.5 * (aPos + 1.0);
    gl_Position = vec4(aPos, 0.0, 1.0);
}
)";

// ------------------------------------
// Fragment Shader: Raymarch + Lighting
// ------------------------------------
static const char* raycastFragSrc = R"(
#version 330 core
in vec2 TexCoord;
out vec4 FragColor;

// Camera matrices/uniforms
uniform mat4 invView;
uniform mat4 invProj;
uniform vec3 camPos;

// 3D volume & mask textures
uniform sampler3D uVolumeTex;
uniform sampler3D uMaskTex;

// Step size for raymarching
uniform float stepSize;

// Light parameters
uniform vec3 lightDir;    // Directional light direction (should be normalized)
uniform vec3 lightColor;  // Color/intensity of the light (e.g., white)
uniform float ambient;    // Ambient term (small constant light)

// Size of the 3D texture
// (Optional) If you need it for gradient offset calculations, 
// you can define volDimX, volDimY, volDimZ as uniforms, 
// but we'll show a simpler approach.

/////////////////////////////////////////////////////////
// Helper: Ray-box intersection for [-0.5..+0.5]^3 volume
/////////////////////////////////////////////////////////
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

/////////////////////////////////////////////////////////
// Helper: Sample the volume at "pos" in [0,1]^3
// Returns the density (0.0 for empty, >0 for filled)
/////////////////////////////////////////////////////////
float sampleVolume(vec3 pos) {
    return texture(uVolumeTex, pos).r;
}

/////////////////////////////////////////////////////////
// Helper: Compute local gradient using central differences
// We'll do 6 extra texture fetches. 
// If performance is tight, you can optimize further.
/////////////////////////////////////////////////////////
vec3 computeGradient(vec3 pos, float delta) {
    // sample +/- delta in x,y,z
    // clamp pos +/- delta to [0,1]
    vec3 ppx = clamp(pos + vec3(delta, 0, 0), 0.0, 1.0);
    vec3 pmx = clamp(pos - vec3(delta, 0, 0), 0.0, 1.0);
    vec3 ppy = clamp(pos + vec3(0, delta, 0), 0.0, 1.0);
    vec3 pmy = clamp(pos - vec3(0, delta, 0), 0.0, 1.0);
    vec3 ppz = clamp(pos + vec3(0, 0, delta), 0.0, 1.0);
    vec3 pmz = clamp(pos - vec3(0, 0, delta), 0.0, 1.0);

    float dx = sampleVolume(ppx) - sampleVolume(pmx);
    float dy = sampleVolume(ppy) - sampleVolume(pmy);
    float dz = sampleVolume(ppz) - sampleVolume(pmz);

    return vec3(dx, dy, dz);
}

void main()
{
    // Reconstruct NDC
    vec2 ndc = vec2(2.0 * TexCoord.x - 1.0, 1.0 - 2.0 * TexCoord.y);
    vec4 clipPos = vec4(ndc, 1.0, 1.0);

    // View space
    vec4 viewPos = invProj * clipPos;
    viewPos /= viewPos.w;

    // World space
    vec4 wPos = invView * viewPos;
    vec3 rayDir = normalize(wPos.xyz - camPos);
    vec3 rayOrig = camPos;

    // Intersect with bounding box
    vec2 tBox = intersectBox(rayOrig, rayDir);
    float tNear = tBox.x;
    float tFar  = tBox.y;

    // If no intersection or behind camera
    if (tNear > tFar || tFar < 0.0) {
        FragColor = vec4(0,0,0,1);
        return;
    }

    // Initialize
    float t = max(tNear, 0.0);
    float tMax = 100.0;
    float tExit = min(tFar, tMax);
    const int MAX_STEPS = 1024;

    // Accumulation
    vec4 colorAccum = vec4(0.0);
    // We'll keep alpha-based compositing
    // baseColor if we find a voxel

    /////////////////////////////////////////
    // Main Raymarch Loop
    /////////////////////////////////////////
    for(int i=0; i<MAX_STEPS && t<=tExit; i++) {
        vec3 pos = rayOrig + t * rayDir;        // in world space
        vec3 texCoord3 = pos + vec3(0.5);       // map [-0.5..+0.5] -> [0..1]

        // Mask check
        float maskVal = texture(uMaskTex, texCoord3).r;
        if(maskVal > 0.5) {
            t += stepSize;
            continue;
        }

        // Sample main volume
        float density = sampleVolume(texCoord3);
        if(density > 0.0) {
            // Compute gradient for lighting
            // We'll assume a small offset => 1.0/(dim) 
            // or we can use half stepSize if you prefer
            float delta = 1.0 / 64.0;  // For dim=64
            vec3 grad = computeGradient(texCoord3, delta);

            // If gradient is near zero, skip shading
            if (length(grad) < 1e-5) {
                // Just accumulate a constant color
                vec3 baseColor = vec3(1.0, 0.5, 0.2);
                float alpha = 0.1; 
                colorAccum.rgb += (1.0 - colorAccum.a) * baseColor * alpha;
                colorAccum.a   += (1.0 - colorAccum.a) * alpha;
            }
            else {
                // Normal from gradient
                vec3 normal = normalize(grad) * -1.0;
                // Basic Lambertian shading
                float diffuse = max(dot(normal, lightDir), 0.0);

                // finalColor = ambient + diffuseTerm
                vec3 shadedColor = ambient * lightColor + diffuse * lightColor;
                // Tint the shaded color by the voxel "base" color
                vec3 baseColor   = vec3(1.0, 0.5, 0.2);
                vec3 finalColor  = baseColor * shadedColor;

                // Alpha for compositing
                float alpha = 0.1;

                // Composite
                colorAccum.rgb += (1.0 - colorAccum.a) * finalColor * alpha;
                colorAccum.a   += (1.0 - colorAccum.a) * alpha;
            }

            // Early ray termination if fully opaque
            if (colorAccum.a >= 0.95) {
                break;
            }
        }

        // Step forward
        t += stepSize;
    }

    // Final output
    FragColor = vec4(colorAccum.rgb, 1.0);
}
)";

// -----------------------------------------------------
// Helper: Compile a shader (vertex or fragment)
// -----------------------------------------------------
static unsigned int compileShader(const char* src, GLenum type)
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

// -----------------------------------------------------
// Create the raycast shader program
// -----------------------------------------------------
static unsigned int createRaycastProgram()
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
		std::cerr << "Program link error:\n" << infoLog << std::endl;
	}

	glDeleteShader(vs);
	glDeleteShader(fs);
	return prog;
}

// -----------------------------------------------------
// Constructor / Destructor
// -----------------------------------------------------
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

// -----------------------------------------------------
// Splat-based render stub (not used here)
// -----------------------------------------------------
std::vector<MCTriangle> VolumeRaycastRenderer::render(
	const OctreeNode* node,
	const VoxelGrid& grid,
	int x0, int y0, int z0,
	int size)
{
	return {};
}

// -----------------------------------------------------
// initVolume: Create 3D texture from the voxel grid
// -----------------------------------------------------
void VolumeRaycastRenderer::initVolume(const VoxelGrid& grid)
{
	volDimX = grid.dimX;
	volDimY = grid.dimY;
	volDimZ = grid.dimZ;

	gridPtr = &grid;

	// Convert voxel states to float data
	std::vector<float> volumeData(volDimX * volDimY * volDimZ, 0.f);
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

	// Create shader program
	raycastShaderProg = createRaycastProgram();

	// Fullscreen quad
	float fsQuadVerts[8] = {
		-1.f, -1.f,
		 1.f, -1.f,
		-1.f,  1.f,
		 1.f,  1.f
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

// -----------------------------------------------------
// initMaskVolume: Simple peeling
// -----------------------------------------------------
void VolumeRaycastRenderer::initMaskVolume(const VoxelGrid& grid)
{
	if (!gridPtr) {
		std::cerr << "[VolumeRaycastRenderer] initMaskVolume: gridPtr is null!\n";
		return;
	}

	std::vector<float> maskData(volDimX * volDimY * volDimZ, 0.0f);

	// Example plane-based peeling at z>0
	for (int z = 0; z < volDimZ; z++) {
		float voxelZ = grid.minZ + (z + 0.5f) * grid.voxelSize;
		for (int y = 0; y < volDimY; y++) {
			for (int x = 0; x < volDimX; x++) {
				int idx = x + y * volDimX + z * (volDimX * volDimY);
				if (voxelZ > 0.0f) {
					maskData[idx] = 1.0f; // Carve away
				}
			}
		}
	}

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

// -----------------------------------------------------
// drawRaycast: Perform raymarching + shading
// -----------------------------------------------------
void VolumeRaycastRenderer::drawRaycast(const Camera& cam,
	float aspectRatio,
	int screenW, int screenH)
{
	if (!raycastShaderProg || !volumeTextureID || !maskTextureID) return;

	glUseProgram(raycastShaderProg);

	// Build inverse matrices
	glm::mat4 V = cam.getView();
	glm::mat4 P = glm::perspective(glm::radians(45.f), aspectRatio, 0.01f, 100.f);
	glm::mat4 invV = glm::inverse(V);
	glm::mat4 invP = glm::inverse(P);
	glm::vec3 camPos = cam.getPos();

	// Send uniforms
	GLint locInvV = glGetUniformLocation(raycastShaderProg, "invView");
	GLint locInvP = glGetUniformLocation(raycastShaderProg, "invProj");
	GLint locCamPos = glGetUniformLocation(raycastShaderProg, "camPos");
	if (locInvV >= 0) glUniformMatrix4fv(locInvV, 1, GL_FALSE, glm::value_ptr(invV));
	if (locInvP >= 0) glUniformMatrix4fv(locInvP, 1, GL_FALSE, glm::value_ptr(invP));
	if (locCamPos >= 0) glUniform3fv(locCamPos, 1, glm::value_ptr(camPos));

	// volume texture = unit 0
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_3D, volumeTextureID);
	GLint locVol = glGetUniformLocation(raycastShaderProg, "uVolumeTex");
	if (locVol >= 0) glUniform1i(locVol, 0);

	// mask texture = unit 1
	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_3D, maskTextureID);
	GLint locMask = glGetUniformLocation(raycastShaderProg, "uMaskTex");
	if (locMask >= 0) glUniform1i(locMask, 1);

	// stepSize
	GLint locStep = glGetUniformLocation(raycastShaderProg, "stepSize");
	if (locStep >= 0) {
		glUniform1f(locStep, 0.005f);
	}

	// Set lighting uniforms
	// A directional light from some angle
	GLint locLightDir = glGetUniformLocation(raycastShaderProg, "lightDir");
	if (locLightDir >= 0) {
		// e.g., pointing downward from above
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
		glUniform1f(locAmbient, 0.2f); // small ambient factor
	}

	// Render fullscreen quad
	glBindVertexArray(quadVAO);
	glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
	glBindVertexArray(0);

	glUseProgram(0);
}

// -----------------------------------------------------
// updatePeelPlane: If user modifies peeling logic
// -----------------------------------------------------
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
					maskData[idx] = 1.0f; // Carved
				}
			}
		}
	}

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
