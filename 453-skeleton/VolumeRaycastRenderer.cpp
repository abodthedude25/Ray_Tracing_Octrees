// VolumeRaycastRenderer.cpp
// Integrates point-radiation (compute shader) + thresholded, gradient-based Lambert shading
// with performance optimizations for large volumes.

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <vector>
#include <iostream>
#include <algorithm>
#include <cmath> // for std::min, etc.
#include <fstream>
#include <sstream>

// #include <chrono> // If needed for timing

#include "VolumeRaycastRenderer.h"

// For anisotropic defines:
#ifndef GL_TEXTURE_MAX_ANISOTROPY_EXT
#define GL_TEXTURE_MAX_ANISOTROPY_EXT 0x84FE
#endif

#ifndef GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT
#define GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT 0x84FF
#endif

static int frameCounter = 0;


//----------------------------------------------------------------------------
// Utility to check for GL errors
//----------------------------------------------------------------------------
static void checkGLError(const char* msg = "") {
	GLenum err;
	while ((err = glGetError()) != GL_NO_ERROR) {
		std::cerr << "[GL Error] " << msg << ": " << err << std::endl;
	}
}

// A simple bounding-box intersection returning (tNear, tFar)
static glm::vec2 intersectAABB(
	const glm::vec3& ro, const glm::vec3& rd,
	const glm::vec3& bmin, const glm::vec3& bmax)
{
	glm::vec3 t1 = (bmin - ro) / rd;
	glm::vec3 t2 = (bmax - ro) / rd;
	glm::vec3 tmin = glm::min(t1, t2);
	glm::vec3 tmax = glm::max(t1, t2);
	float nearT = std::max(std::max(tmin.x, tmin.y), tmin.z);
	float farT = std::min(std::min(tmax.x, tmax.y), tmax.z);
	return glm::vec2(nearT, farT);
}

// Recursively find earliest intersection with a non-empty octree node.
static float octreeRaySkip(
	const OctreeNode* node,
	const glm::vec3& ro,
	const glm::vec3& rd,
	float tMin,
	float tMax,
	const VoxelGrid& grid)
{
	if (!node) {
		return 1e30f; // no node, skip
	}

	// Convert node->(x,y,z,size) from voxel coords -> world bounding box
	float vx = grid.voxelSize;
	float wx0 = grid.minX + node->x * vx;
	float wy0 = grid.minY + node->y * vx;
	float wz0 = grid.minZ + node->z * vx;
	float wSize = node->size * vx;

	glm::vec3 bmin(wx0, wy0, wz0);
	glm::vec3 bmax(wx0 + wSize, wy0 + wSize, wz0 + wSize);

	// Intersect bounding box
	glm::vec2 tHit = intersectAABB(ro, rd, bmin, bmax);
	float enterT = std::max(tMin, tHit.x);
	float exitT = std::min(tMax, tHit.y);

	// If no intersection, skip
	if (enterT > exitT) {
		return 1e30f;
	}

	// If leaf node
	if (node->isLeaf) {
		// If empty => skip
		if (!node->isSolid) {
			return 1e30f;
		}
		// If solid => earliest intersection is 'enterT'
		return enterT;
	}

	// Otherwise, it's an internal node => check children
	float bestT = 1e30f;
	for (int i = 0; i < 8; i++) {
		const OctreeNode* child = node->children[i];
		if (!child) continue;
		float childT = octreeRaySkip(child, ro, rd, enterT, exitT, grid);
		if (childT < bestT) {
			bestT = childT;
		}
	}
	return bestT;
}

//==================== CONSTRUCTOR / DESTRUCTOR ====================
VolumeRaycastRenderer::VolumeRaycastRenderer()
	: m_volumeTex(0)
	, m_radiationTex(0)
	, m_computeProg(0)
	, m_raycastProg(0)
	, m_precomputeProg(0)
	, m_quadVAO(0)
	, m_quadVBO(0)
	, m_gradientMagTex(0)
	, m_gradientDirTex(0)
	, m_edgeFactorTex(0)
	, m_ambientOcclusionTex(0)   
	, m_indirectLightTex(0)
	, m_indirectLightingComputeProg(0)
	, m_dimX(0)
	, m_dimY(0)
	, m_dimZ(0)
	, m_boxMin(0.0f)
	, m_boxMax(1.0f)
	, m_gridPtr(nullptr)
	, m_cameraPtr(nullptr)
	, m_inited(false)
	, m_precomputeNeeded(true)
	, m_timeValue(0.0f)
{}

VolumeRaycastRenderer::~VolumeRaycastRenderer() {
	if (m_volumeTex)        glDeleteTextures(1, &m_volumeTex);
	if (m_radiationTex)     glDeleteTextures(1, &m_radiationTex);
	if (m_computeProg)      glDeleteProgram(m_computeProg);
	if (m_raycastProg)      glDeleteProgram(m_raycastProg);
	if (m_precomputeProg)   glDeleteProgram(m_precomputeProg);

	if (m_quadVBO) glDeleteBuffers(1, &m_quadVBO);
	if (m_quadVAO) glDeleteVertexArrays(1, &m_quadVAO);

	if (m_gradientMagTex) glDeleteTextures(1, &m_gradientMagTex);
	if (m_gradientDirTex) glDeleteTextures(1, &m_gradientDirTex);
	if (m_edgeFactorTex)  glDeleteTextures(1, &m_edgeFactorTex);
	if (m_ambientOcclusionTex) glDeleteTextures(1, &m_ambientOcclusionTex);
	if (m_indirectLightTex) glDeleteTextures(1, &m_indirectLightTex);
	if (m_indirectLightingComputeProg) glDeleteProgram(m_indirectLightingComputeProg);
}

//==================== INIT ====================
void VolumeRaycastRenderer::init(const VoxelGrid& grid) {
	if (m_inited) return;

	m_gridPtr = &grid;
	m_dimX = grid.dimX;
	m_dimY = grid.dimY;
	m_dimZ = grid.dimZ;

	m_boxMin = glm::vec3(grid.minX, grid.minY, grid.minZ);
	m_boxMax = glm::vec3(
		grid.minX + grid.dimX * grid.voxelSize,
		grid.minY + grid.dimY * grid.voxelSize,
		grid.minZ + grid.dimZ * grid.voxelSize
	);

	// 1) Create volume texture from voxel data
	createVolumeTexture(grid);

	// 2) Create radiation texture (for carving masks)
	createRadiationTexture();

	// 3) Create precompute textures
	createPrecomputeTextures();

	// 4) Create ambient occlusion texture
	createAmbientOcclusionTexture();

	// 5) Create indirect lighting texture
	createIndirectLightTexture();

	// 6) Create indirect lighting compute shader
	createIndirectLightingComputeShader();

	// 6) Compile compute shader for point radiation
	createComputeShader();

	// 7) Compile precompute shader for gradient/edge factor
	createPrecomputeShader();

	// 8) Build the raycasting shader program
	createRaycastProgram();

	// 9) Create fullscreen quad for final pass
	createFullscreenQuad();

	m_precomputeNeeded = true;
	m_inited = true;
}

//==================== createVolumeTexture ====================
void VolumeRaycastRenderer::createVolumeTexture(const VoxelGrid& grid) {
	// If your volume is initially 8 or 16 bits, consider GL_R16 or GL_R8 for memory savings:
	// e.g. internalFormat = GL_R16F or GL_R8
	// For now, we keep GL_R32F as requested logic

	// Build a CPU buffer of floats
	std::vector<float> volumeData(m_dimX * m_dimY * m_dimZ, 0.0f);

	for (int z = 0; z < m_dimZ; z++) {
		for (int y = 0; y < m_dimY; y++) {
			for (int x = 0; x < m_dimX; x++) {
				int idx = x + y * m_dimX + z * (m_dimX * m_dimY);
				if (grid.data[idx] == VoxelState::FILLED) {
					volumeData[idx] = 1.0f;
				}
			}
		}
	}

	glGenTextures(1, &m_volumeTex);
	glBindTexture(GL_TEXTURE_3D, m_volumeTex);

	// Create the texture with float data
	glTexImage3D(GL_TEXTURE_3D, 0, GL_R32F, m_dimX, m_dimY, m_dimZ,
		0, GL_RED, GL_FLOAT, volumeData.data());

	// Use trilinear+mip+anisotropy for better sampling
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

	// Generate MIPmaps so the hardware can quickly step down for distant views
	glGenerateMipmap(GL_TEXTURE_3D);

	// Possibly enable anisotropic filtering if supported
	GLfloat maxAniso = 0.0f;
	glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT, &maxAniso);
	if (maxAniso > 0.0f) {
		glTexParameterf(GL_TEXTURE_3D, GL_TEXTURE_MAX_ANISOTROPY_EXT, std::min(4.0f, maxAniso));
	}

	glBindTexture(GL_TEXTURE_3D, 0);
	checkGLError("createVolumeTexture");
}

//==================== createRadiationTexture ====================
void VolumeRaycastRenderer::createRadiationTexture() {
	glGenTextures(1, &m_radiationTex);
	glBindTexture(GL_TEXTURE_3D, m_radiationTex);

	std::vector<float> zeroData(m_dimX * m_dimY * m_dimZ, 0.0f);

	// We store the "carved" or "radiation" values in R32F too
	glTexImage3D(GL_TEXTURE_3D, 0, GL_R32F,
		m_dimX, m_dimY, m_dimZ,
		0, GL_RED, GL_FLOAT, zeroData.data());

	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

	glBindTexture(GL_TEXTURE_3D, 0);
	checkGLError("createRadiationTexture");
}

//==================== clearRadiationVolume ====================
void VolumeRaycastRenderer::clearRadiationVolume() {
	if (!m_radiationTex) return;
	float clr[4] = { 0.f, 0.f, 0.f, 0.f };
	glClearTexImage(m_radiationTex, 0, GL_RED, GL_FLOAT, clr);
	checkGLError("clearRadiationVolume");
}

//==================== updateSplatPoints ====================
void VolumeRaycastRenderer::updateSplatPoints(const std::vector<RadiationPoint>& pts) {
	m_splatPoints = pts;
}

//==================== Compute Shader (Point Radiation) =====================
static const char* pointRadComputeSrc = R"COMPUTE(
#version 430 core
struct RadiationPoint {
    vec3 worldPos;
    float radius;
};

layout(std430, binding=0) buffer SplatBuffer {
   RadiationPoint splats[];
};

layout(r32f, binding=1) uniform image3D radiationVol;
layout(r32f, binding=2) uniform image3D volumeTex; // optional

uniform vec3 boxMin;
uniform vec3 boxMax;
uniform int dimX;
uniform int dimY;
uniform int dimZ;
uint seed;

// A simple random generator
float randFloat(inout uint n) {
    n = 1664525u * n + 1013904223u;
    uint bits = (n >> 9u) | 0x3F800000u;
    float f = uintBitsToFloat(bits) - 1.0;
    return f;
}

// Sharper cubic B-spline
float bspline1D(float x) {
    x = abs(x);
    if(x < 0.7){
        return (2.0/3.0) + 0.7*x*x*(x-2.0);
    } else if(x < 1.6){
        float t = 1.6 - x;
        return (t*t*t)/5.0;
    }
    return 0.0;
}

// We'll try a tile-based approach or skip it. For now, skip shared memory usage.

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;
void main() {
    uint gid = gl_GlobalInvocationID.x
             + gl_GlobalInvocationID.y * gl_NumWorkGroups.x * gl_WorkGroupSize.x;

    if(gid >= splats.length()) return;

    RadiationPoint rp = splats[gid];
    if(rp.radius <= 0.0) return;

    vec3 size = boxMax - boxMin;
    vec3 voxelCoordF = (rp.worldPos - boxMin) / size * vec3(dimX, dimY, dimZ);
    ivec3 center = ivec3(floor(voxelCoordF));

    float maxSupport = 0.4 * rp.radius;
    int iRad = int(ceil(maxSupport));
    uint rng = seed + gid;

    for(int dz = -iRad; dz <= iRad; dz++){
        for(int dy = -iRad; dy <= iRad; dy++){
            for(int dx = -iRad; dx <= iRad; dx++){
                ivec3 samplePos = center + ivec3(dx, dy, dz);

                if(samplePos.x < 0 || samplePos.x >= dimX ||
                   samplePos.y < 0 || samplePos.y >= dimY ||
                   samplePos.z < 0 || samplePos.z >= dimZ) {
                   continue;
                }

                vec3 d = vec3(samplePos) - voxelCoordF;
                vec3 nd = d / rp.radius;

                // Evaluate bspline with random jitter
                float w = bspline1D(nd.x)*bspline1D(nd.y)*bspline1D(nd.z);

                float rx = (randFloat(rng)-0.5)*0.05;
                float ry = (randFloat(rng)-0.5)*0.05;
                float rz = (randFloat(rng)-0.5)*0.05;
                float w2 = bspline1D(nd.x + rx)*bspline1D(nd.y + ry)*bspline1D(nd.z + rz);

                float finalW = 0.5*(w + w2);

                if(finalW > 1e-4) {
                    float oldVal = imageLoad(radiationVol, samplePos).r;
                    float newVal = oldVal + finalW;
                    imageStore(radiationVol, samplePos, vec4(newVal,0,0,0));
                }
            }
        }
    }
}
)COMPUTE";

void VolumeRaycastRenderer::createComputeShader() {
	GLuint cs = glCreateShader(GL_COMPUTE_SHADER);
	glShaderSource(cs, 1, &pointRadComputeSrc, nullptr);
	glCompileShader(cs);

	GLint success;
	glGetShaderiv(cs, GL_COMPILE_STATUS, &success);
	if (!success) {
		char log[1024];
		glGetShaderInfoLog(cs, 1024, nullptr, log);
		std::cerr << "ComputeShader compile error:\n" << log << std::endl;
		glDeleteShader(cs);
		return;
	}

	m_computeProg = glCreateProgram();
	glAttachShader(m_computeProg, cs);
	glLinkProgram(m_computeProg);
	glDeleteShader(cs);

	glGetProgramiv(m_computeProg, GL_LINK_STATUS, &success);
	if (!success) {
		char log[1024];
		glGetProgramInfoLog(m_computeProg, 1024, nullptr, log);
		std::cerr << "ComputeShader link error:\n" << log << std::endl;
		glDeleteProgram(m_computeProg);
		m_computeProg = 0;
	}
	checkGLError("createComputeShader");
}

//==================== dispatchRadiationCompute ====================
void VolumeRaycastRenderer::dispatchRadiationCompute() {
	if (!m_computeProg || m_splatPoints.empty()) return;

	std::cout << "Dispatching radiation compute with " << m_splatPoints.size() << " points\n";

	// Validate radiation points
	for (auto& pt : m_splatPoints) {
		pt.radius = std::min(pt.radius, 15.0f);
		glm::vec3 normalizedPos = (pt.worldPos - m_boxMin) / (m_boxMax - m_boxMin);
		// If outside volume + margin
		if (normalizedPos.x < -0.1f || normalizedPos.x > 1.1f ||
			normalizedPos.y < -0.1f || normalizedPos.y > 1.1f ||
			normalizedPos.z < -0.1f || normalizedPos.z > 1.1f) {
			std::cout << "Warning: Radiation point outside volume: "
				<< pt.worldPos.x << ", " << pt.worldPos.y << ", " << pt.worldPos.z << std::endl;
			// We can skip or clamp
		}
	}

	// Create SSBO
	GLuint ssbo = 0;
	glGenBuffers(1, &ssbo);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);

	size_t dataSize = sizeof(RadiationPoint) * m_splatPoints.size();
	glBufferData(GL_SHADER_STORAGE_BUFFER, dataSize, m_splatPoints.data(), GL_STATIC_DRAW);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo);

	glUseProgram(m_computeProg);

	// Set uniforms
	glUniform3fv(glGetUniformLocation(m_computeProg, "boxMin"), 1, glm::value_ptr(m_boxMin));
	glUniform3fv(glGetUniformLocation(m_computeProg, "boxMax"), 1, glm::value_ptr(m_boxMax));
	glUniform1i(glGetUniformLocation(m_computeProg, "dimX"), m_dimX);
	glUniform1i(glGetUniformLocation(m_computeProg, "dimY"), m_dimY);
	glUniform1i(glGetUniformLocation(m_computeProg, "dimZ"), m_dimZ);
	glUniform1ui(glGetUniformLocation(m_computeProg, "seed"), 12345u);

	// Bind images
	glBindImageTexture(1, m_radiationTex, 0, GL_TRUE, 0, GL_READ_WRITE, GL_R32F);
	glBindImageTexture(2, m_volumeTex, 0, GL_TRUE, 0, GL_READ_ONLY, GL_R32F);

	// Dispatch
	const GLuint localSizeX = 16;
	const GLuint localSizeY = 16;
	GLuint wgSize = localSizeX * localSizeY;
	GLuint totalPoints = static_cast<GLuint>(m_splatPoints.size());
	GLuint numGroups = (totalPoints + wgSize - 1) / wgSize;

	if (numGroups > 0) {
		std::cout << "Compute dispatch: " << numGroups << " groups.\n";
		glDispatchCompute(numGroups, 1, 1);
	}

	// Single barrier after dispatch
	glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT);

	// Cleanup
	glBindImageTexture(1, 0, 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32F);
	glBindImageTexture(2, 0, 0, GL_FALSE, 0, GL_READ_ONLY, GL_R32F);

	glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
	glDeleteBuffers(1, &ssbo);

	checkGLError("dispatchRadiationCompute");

	// Force re-run precompute for updated radiation
	m_precomputeNeeded = true;
}

//==================== Precompute Shader =====================
static const char* precomputeShaderSrc = R"COMPUTE(
#version 430 core
layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;

// We'll read from volumeTex and output to gradientMagTex, gradientDirTex, edgeFactorTex
layout(binding = 0) uniform sampler3D volumeTex;
layout(r32f, binding = 0) uniform writeonly image3D gradientMagTex;
layout(rgba32f, binding = 1) uniform writeonly image3D gradientDirTex;
layout(r32f, binding = 2) uniform writeonly image3D edgeFactorTex;

uniform vec3 boxMin;
uniform vec3 boxMax;
uniform ivec3 volumeSize;

float safeSampleVolume(vec3 uvw) {
    return texture(volumeTex, clamp(uvw, vec3(0.0), vec3(1.0))).r;
}

float cubicFilter(float x) {
    x = abs(x);
    if (x < 1.0)
        return (2.0/3.0) + (0.5*x*x*(x-2.0));
    else if (x < 2.0) {
        float t = 2.0 - x;
        return (t*t*t)/6.0;
    }
    return 0.0;
}

vec3 computeGradient(vec3 uvw, float eps) {
    float x1 = safeSampleVolume(uvw + vec3(eps,0,0));
    float x2 = safeSampleVolume(uvw - vec3(eps,0,0));
    float y1 = safeSampleVolume(uvw + vec3(0,eps,0));
    float y2 = safeSampleVolume(uvw - vec3(0,eps,0));
    float z1 = safeSampleVolume(uvw + vec3(0,0,eps));
    float z2 = safeSampleVolume(uvw - vec3(0,0,eps));
    return 0.5 * vec3(x1 - x2, y1 - y2, z1 - z2);
}

float sampleVolumeCubic(vec3 uvw) {
    // We skip advanced cubic approach for brevity here. If you want a real cubic approach, add code
    return safeSampleVolume(uvw);
}

void main() {
    ivec3 voxelCoord = ivec3(gl_GlobalInvocationID.xyz);
    if(any(greaterThanEqual(voxelCoord, volumeSize))) return;

    vec3 uvw = (vec3(voxelCoord) + 0.5) / vec3(volumeSize);
    float den = safeSampleVolume(uvw);

    float isoValue = 0.9;
    float isoRange = 0.3;
    float isoFactor = smoothstep(isoValue - isoRange, isoValue + isoRange, den);

    bool isEdge = abs(den - isoValue) < isoRange;
    if(isEdge) {
        den = sampleVolumeCubic(uvw);
    }

    float eps = 1.0 / float(min(min(volumeSize.x, volumeSize.y), volumeSize.z));
    if(isEdge) eps *= 0.5;

    vec3 grad = computeGradient(uvw, eps);
    float gradMag = length(grad);

    vec3 normal = (gradMag > 1e-5) ? (grad / gradMag) : vec3(0.0,1.0,0.0);

    imageStore(gradientMagTex, voxelCoord, vec4(gradMag));
    imageStore(gradientDirTex, voxelCoord, vec4(normal, 0.0));
    imageStore(edgeFactorTex, voxelCoord, vec4(isoFactor));
}
)COMPUTE";

void VolumeRaycastRenderer::createPrecomputeShader() {
	GLuint cs = glCreateShader(GL_COMPUTE_SHADER);
	glShaderSource(cs, 1, &precomputeShaderSrc, nullptr);
	glCompileShader(cs);

	GLint success;
	glGetShaderiv(cs, GL_COMPILE_STATUS, &success);
	if (!success) {
		char log[1024];
		glGetShaderInfoLog(cs, 1024, nullptr, log);
		std::cerr << "PrecomputeShader compile error:\n" << log << std::endl;
		glDeleteShader(cs);
		return;
	}

	m_precomputeProg = glCreateProgram();
	glAttachShader(m_precomputeProg, cs);
	glLinkProgram(m_precomputeProg);
	glDeleteShader(cs);

	glGetProgramiv(m_precomputeProg, GL_LINK_STATUS, &success);
	if (!success) {
		char log[1024];
		glGetProgramInfoLog(m_precomputeProg, 1024, nullptr, log);
		std::cerr << "PrecomputeShader link error:\n" << log << std::endl;
		glDeleteProgram(m_precomputeProg);
		m_precomputeProg = 0;
	}
	checkGLError("createPrecomputeShader");
}

void VolumeRaycastRenderer::createPrecomputeTextures() {
	// Gradient magnitude
	glGenTextures(1, &m_gradientMagTex);
	glBindTexture(GL_TEXTURE_3D, m_gradientMagTex);
	glTexImage3D(GL_TEXTURE_3D, 0, GL_R32F, m_dimX, m_dimY, m_dimZ,
		0, GL_RED, GL_FLOAT, nullptr);

	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

	// Gradient direction
	glGenTextures(1, &m_gradientDirTex);
	glBindTexture(GL_TEXTURE_3D, m_gradientDirTex);
	glTexImage3D(GL_TEXTURE_3D, 0, GL_RGBA32F, m_dimX, m_dimY, m_dimZ,
		0, GL_RGBA, GL_FLOAT, nullptr);

	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

	// Edge factor
	glGenTextures(1, &m_edgeFactorTex);
	glBindTexture(GL_TEXTURE_3D, m_edgeFactorTex);
	glTexImage3D(GL_TEXTURE_3D, 0, GL_R32F, m_dimX, m_dimY, m_dimZ,
		0, GL_RED, GL_FLOAT, nullptr);

	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

	glBindTexture(GL_TEXTURE_3D, 0);
	checkGLError("createPrecomputeTextures");
}

void VolumeRaycastRenderer::dispatchPrecompute() {
	if (!m_precomputeProg) return;

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_3D, m_volumeTex);

	// Bind output images
	glBindImageTexture(0, m_gradientMagTex, 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_R32F);
	glBindImageTexture(1, m_gradientDirTex, 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_RGBA32F);
	glBindImageTexture(2, m_edgeFactorTex, 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_R32F);

	glUseProgram(m_precomputeProg);

	glUniform3fv(glGetUniformLocation(m_precomputeProg, "boxMin"), 1, glm::value_ptr(m_boxMin));
	glUniform3fv(glGetUniformLocation(m_precomputeProg, "boxMax"), 1, glm::value_ptr(m_boxMax));
	glUniform3i(glGetUniformLocation(m_precomputeProg, "volumeSize"), m_dimX, m_dimY, m_dimZ);
	glUniform1i(glGetUniformLocation(m_precomputeProg, "volumeTex"), 0);

	int groupsX = (m_dimX + 7) / 8;
	int groupsY = (m_dimY + 7) / 8;
	int groupsZ = (m_dimZ + 7) / 8;

	glDispatchCompute(groupsX, groupsY, groupsZ);

	glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT);

	// Unbind
	glBindImageTexture(0, 0, 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32F);
	glBindImageTexture(1, 0, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F);
	glBindImageTexture(2, 0, 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32F);

	m_precomputeNeeded = false;
	checkGLError("dispatchPrecompute");
}

//==================== Raycast Program =====================
static const char* raycastVS = R"VS(
#version 430 core
layout(location=0) in vec2 aPos;
out vec2 vTexCoord;
void main(){
    // Fullscreen quad, map -1..1 -> 0..1
    vTexCoord = 0.5*(aPos + vec2(1.0));
    gl_Position = vec4(aPos, 0.0, 1.0);
}
)VS";

// We place the improved stepping logic in a function in the FS code.
static const char* raycastFS = R"FS(

)FS";

std::string loadShaderFromFile(const std::string& filePath) {
	std::ifstream file(filePath);
	if (!file.is_open()) {
		std::cerr << "Failed to open shader file: " << filePath << std::endl;
		return "";
	}

	std::stringstream buffer;
	buffer << file.rdbuf();
	return buffer.str();
}

void VolumeRaycastRenderer::createRaycastProgram() {
	// Load shader sources from files
	std::string vsSource = raycastVS;	
	std::string fsSource = loadShaderFromFile("453-skeleton/shaders/raycastFS.glsl");

	const char* vsSourcePtr = vsSource.c_str();
	const char* fsSourcePtr = fsSource.c_str();

	GLuint vs = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vs, 1, &vsSourcePtr, nullptr);
	glCompileShader(vs);

	GLint success;
	glGetShaderiv(vs, GL_COMPILE_STATUS, &success);
	if (!success) {
		char log[512];
		glGetShaderInfoLog(vs, 512, nullptr, log);
		std::cerr << "RaycastVS compile error:\n" << log << std::endl;
	}

	GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fs, 1, &fsSourcePtr, nullptr);
	glCompileShader(fs);
	glGetShaderiv(fs, GL_COMPILE_STATUS, &success);
	if (!success) {
		char log[512];
		glGetShaderInfoLog(fs, 512, nullptr, log);
		std::cerr << "RaycastFS compile error:\n" << log << std::endl;
	}

	m_raycastProg = glCreateProgram();
	glAttachShader(m_raycastProg, vs);
	glAttachShader(m_raycastProg, fs);
	glLinkProgram(m_raycastProg);

	glDeleteShader(vs);
	glDeleteShader(fs);

	glGetProgramiv(m_raycastProg, GL_LINK_STATUS, &success);
	if (!success) {
		char log[512];
		glGetProgramInfoLog(m_raycastProg, 512, nullptr, log);
		std::cerr << "Raycast program link error:\n" << log << std::endl;
		glDeleteProgram(m_raycastProg);
		m_raycastProg = 0;
	}
	checkGLError("createRaycastProgram");
}

//==================== createFullscreenQuad ====================
void VolumeRaycastRenderer::createFullscreenQuad() {
	float fsQuad[8] = {
	   -1.f, -1.f,
		1.f, -1.f,
	   -1.f,  1.f,
		1.f,  1.f
	};

	glGenVertexArrays(1, &m_quadVAO);
	glGenBuffers(1, &m_quadVBO);

	glBindVertexArray(m_quadVAO);
	glBindBuffer(GL_ARRAY_BUFFER, m_quadVBO);

	glBufferData(GL_ARRAY_BUFFER, sizeof(fsQuad), fsQuad, GL_STATIC_DRAW);

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, (void*)0);

	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	checkGLError("createFullscreenQuad");
}

//==================== bindRaycastUniforms ====================
void VolumeRaycastRenderer::bindRaycastUniforms(float aspect) {
	glUseProgram(m_raycastProg);

	// Build inverse view & inverse projection
	glm::mat4 V = (m_cameraPtr ? m_cameraPtr->getView() : glm::mat4(1.0));
	glm::mat4 P = glm::perspective(glm::radians(45.0f), aspect, 0.1f, 5000.0f);

	glm::mat4 invV = glm::inverse(V);
	glm::mat4 invP = glm::inverse(P);

	GLint locInvView = glGetUniformLocation(m_raycastProg, "invView");
	GLint locInvProj = glGetUniformLocation(m_raycastProg, "invProj");
	glUniformMatrix4fv(locInvView, 1, GL_FALSE, glm::value_ptr(invV));
	glUniformMatrix4fv(locInvProj, 1, GL_FALSE, glm::value_ptr(invP));

	if (m_cameraPtr) {
		glm::vec3 cpos = m_cameraPtr->getPos();
		glUniform3fv(glGetUniformLocation(m_raycastProg, "camPos"), 1, glm::value_ptr(cpos));
	}

	glUniform3fv(glGetUniformLocation(m_raycastProg, "boxMin"), 1, glm::value_ptr(m_boxMin));
	glUniform3fv(glGetUniformLocation(m_raycastProg, "boxMax"), 1, glm::value_ptr(m_boxMax));

	// Bind volume texture
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_3D, m_volumeTex);
	glUniform1i(glGetUniformLocation(m_raycastProg, "volumeTex"), 0);

	// Bind radiation
	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_3D, m_radiationTex);
	glUniform1i(glGetUniformLocation(m_raycastProg, "radiationTex"), 1);

	// precomputed
	glActiveTexture(GL_TEXTURE2);
	glBindTexture(GL_TEXTURE_3D, m_gradientMagTex);
	glUniform1i(glGetUniformLocation(m_raycastProg, "gradientMagTex"), 2);

	glActiveTexture(GL_TEXTURE3);
	glBindTexture(GL_TEXTURE_3D, m_gradientDirTex);
	glUniform1i(glGetUniformLocation(m_raycastProg, "gradientDirTex"), 3);

	glActiveTexture(GL_TEXTURE4);
	glBindTexture(GL_TEXTURE_3D, m_edgeFactorTex);
	glUniform1i(glGetUniformLocation(m_raycastProg, "edgeFactorTex"), 4);

	// New textures for enhanced lighting
	glActiveTexture(GL_TEXTURE5);
	glBindTexture(GL_TEXTURE_3D, m_ambientOcclusionTex);
	glUniform1i(glGetUniformLocation(m_raycastProg, "ambientOcclusionTex"), 5);

	glActiveTexture(GL_TEXTURE6);
	glBindTexture(GL_TEXTURE_3D, m_indirectLightTex);
	glUniform1i(glGetUniformLocation(m_raycastProg, "indirectLightTex"), 6);

	// incremental time for jitter
	m_timeValue += 0.01f;
	glUniform1f(glGetUniformLocation(m_raycastProg, "timeValue"), m_timeValue);

	checkGLError("bindRaycastUniforms");
}

//==================== drawRaycast ====================
void VolumeRaycastRenderer::drawRaycast(float aspect) {
	if (!m_raycastProg) return;

	// Possibly re-run gradient precompute if needed
	if (m_precomputeNeeded) {
		dispatchPrecompute();
	}

	// Update indirect lighting every N frames for performance
	if (frameCounter % 10 == 0) { // Update every 10 frames
		updateIndirectLighting();
	}
	frameCounter++;

	// Modified octree skipping logic to avoid disappearing buildings
	float skipDistance = 0.0f;
	if (m_enableOctreeSkip && m_octreeRoot && m_cameraPtr) {
		// Instead of just a center ray, use multiple rays for more stability
		float maxSkipDistance = 0.0f;
		const int numRays = 5; // Use multiple rays instead of just one

		for (int i = 0; i < numRays; i++) {
			// Create rays at different points on the screen (center and corners)
			float ndcX = (i == 0) ? 0.0f : ((i == 1) ? -0.75f : (i == 2) ? 0.75f : (i == 3) ? -0.75f : 0.75f);
			float ndcY = (i == 0) ? 0.0f : ((i == 1) ? -0.75f : (i == 2) ? -0.75f : (i == 3) ? 0.75f : 0.75f);

			glm::vec4 clipPos(ndcX, ndcY, 1.f, 1.f);
			glm::mat4 V = m_cameraPtr->getView();
			glm::mat4 P = glm::perspective(glm::radians(45.0f), aspect, 0.1f, 5000.0f);
			glm::mat4 invV = glm::inverse(V);
			glm::mat4 invP = glm::inverse(P);

			glm::vec4 viewPos = invP * clipPos;
			viewPos /= viewPos.w;
			glm::vec4 worldPos4 = invV * viewPos;
			glm::vec3 ro = m_cameraPtr->getPos();
			glm::vec3 rd = glm::normalize(glm::vec3(worldPos4) - ro);

			// Use a large tMax
			float raySkip = octreeRaySkip(m_octreeRoot, ro, rd, 0.0f, 1e30f, *m_gridPtr);

			// Take the maximum from all rays for stability
			if (raySkip < 1e30f && (i == 0 || raySkip > maxSkipDistance)) {
				maxSkipDistance = raySkip;
			}
		}

		// Only use skip if we found a valid distance, and apply a safety margin
		if (maxSkipDistance > 0.0f) {
			// Add a small safety margin to avoid skipping too far
			skipDistance = maxSkipDistance * 0.95f; // 5% safety margin
		}
	}

	// Standard rendering code continues as before
	glDisable(GL_DEPTH_TEST);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glClearColor(0, 0, 0, 1);
	glClear(GL_COLOR_BUFFER_BIT);

	glUseProgram(m_raycastProg);
	bindRaycastUniforms(aspect);

	// Set the octreeSkipT uniform
	GLint locSkip = glGetUniformLocation(m_raycastProg, "octreeSkipT");
	glUniform1f(locSkip, skipDistance);

	glBindVertexArray(m_quadVAO);
	glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
	glBindVertexArray(0);

	glDisable(GL_BLEND);
	glEnable(GL_DEPTH_TEST);
	checkGLError("drawRaycast");
}

static const char* indirectLightingComputeSrc = R"COMPUTE(
#version 430 core

layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;

// Input volumes
layout(r32f, binding = 0) readonly uniform image3D volumeTex;         // Building volume
layout(rgba32f, binding = 1) readonly uniform image3D gradientDirTex; // Normals
layout(r32f, binding = 2) readonly uniform image3D radiationTex;      // Carved areas

// Output volume
layout(rgba16f, binding = 3) writeonly uniform image3D indirectLightTex;

// Uniforms
uniform vec3 lightDir;        // Main light direction
uniform vec3 lightColor;      // Light color
uniform ivec3 volumeSize;     // Volume dimensions
uniform float indirectStrength; // Strength multiplier for indirect light

// Check if a voxel is directly lit by sun
bool isDirectlyLit(ivec3 pos, vec3 normal) {
    // Check if normal faces the light
    float NdotL = dot(normal, lightDir);
    if (NdotL <= 0.0)
        return false;
        
    // Check if it's a solid voxel that's not carved
    float density = imageLoad(volumeTex, pos).r;
    float radiation = imageLoad(radiationTex, pos).r;
    
    return (density > 0.5 && radiation < 0.1);
}

void main() {
    ivec3 pos = ivec3(gl_GlobalInvocationID.xyz);
    
    // Skip if outside volume
    if (any(greaterThanEqual(pos, volumeSize)))
        return;
        
    // Initialize indirect light to zero
    vec3 indirectLight = vec3(0.0);
    
    // Check if this is a solid voxel
    float density = imageLoad(volumeTex, pos).r;
    float radiation = imageLoad(radiationTex, pos).r;
    
    // If it's empty or carved, calculate light bounced from nearby surfaces
    if (density < 0.5 || radiation > 0.1) {
        // Define search radius for light gathering (in voxels)
        const int radius = 12;
        
        // Accumulate light from nearby lit voxels
        for (int dz = -radius; dz <= radius; dz++) {
            for (int dy = -radius; dy <= radius; dy++) {
                for (int dx = -radius; dx <= radius; dx++) {
                    ivec3 neighborPos = pos + ivec3(dx, dy, dz);
                    
                    // Skip if outside volume
                    if (any(lessThan(neighborPos, ivec3(0))) || 
                        any(greaterThanEqual(neighborPos, volumeSize)))
                        continue;
                        
                    // Calculate distance to neighbor
                    float dist = length(vec3(dx, dy, dz));
                    if (dist > float(radius))
                        continue;
                        
                    // Get normal of neighbor
                    vec3 neighborNormal = imageLoad(gradientDirTex, neighborPos).xyz;
                    
                    // Check if neighbor is directly lit by light
                    if (isDirectlyLit(neighborPos, neighborNormal)) {
                        // Calculate falloff based on distance
                        float falloff = 1.0 / (1.0 + dist*dist);
                        
                        // Direction from neighbor to this voxel
                        vec3 bounceDir = normalize(vec3(pos - neighborPos));
                        
                        // Check if light actually bounces toward us (dot product with normal)
                        float bounceFactor = max(0.0, dot(neighborNormal, -bounceDir));
                        
                        // Add contribution, factoring in the bounce direction
                        indirectLight += lightColor * falloff * bounceFactor;
                    }
                }
            }
        }
        
        // Scale indirect light
        indirectLight *= indirectStrength;
    }
    
    // Store the result
    imageStore(indirectLightTex, pos, vec4(indirectLight, 0.0));
}
)COMPUTE";

void VolumeRaycastRenderer::createIndirectLightingComputeShader() {
	GLuint cs = glCreateShader(GL_COMPUTE_SHADER);
	glShaderSource(cs, 1, &indirectLightingComputeSrc, nullptr);
	glCompileShader(cs);

	GLint success;
	glGetShaderiv(cs, GL_COMPILE_STATUS, &success);
	if (!success) {
		char log[1024];
		glGetShaderInfoLog(cs, 1024, nullptr, log);
		std::cerr << "IndirectLightingComputeShader compile error:\n" << log << std::endl;
		glDeleteShader(cs);
		return;
	}

	m_indirectLightingComputeProg = glCreateProgram();
	glAttachShader(m_indirectLightingComputeProg, cs);
	glLinkProgram(m_indirectLightingComputeProg);
	glDeleteShader(cs);

	glGetProgramiv(m_indirectLightingComputeProg, GL_LINK_STATUS, &success);
	if (!success) {
		char log[1024];
		glGetProgramInfoLog(m_indirectLightingComputeProg, 1024, nullptr, log);
		std::cerr << "IndirectLightingComputeShader link error:\n" << log << std::endl;
		glDeleteProgram(m_indirectLightingComputeProg);
		m_indirectLightingComputeProg = 0;
	}
	checkGLError("createIndirectLightingComputeShader");
}

void VolumeRaycastRenderer::createAmbientOcclusionTexture() {
	glGenTextures(1, &m_ambientOcclusionTex);
	glBindTexture(GL_TEXTURE_3D, m_ambientOcclusionTex);

	// Initialize with all zeros (no occlusion)
	std::vector<float> aoData(m_dimX * m_dimY * m_dimZ, 0.0f);

	// Pre-compute basic AO by checking neighborhood density in original volume
	// This is a simple approach - could be refined with more advanced algorithms
	for (int z = 1; z < m_dimZ - 1; z++) {
		for (int y = 1; y < m_dimY - 1; y++) {
			for (int x = 1; x < m_dimX - 1; x++) {
				int idx = x + y * m_dimX + z * (m_dimX * m_dimY);

				// Check 6-connected neighborhood
				float neighborhoodDensity = 0.0f;
				for (int dz = -1; dz <= 1; dz++) {
					for (int dy = -1; dy <= 1; dy++) {
						for (int dx = -1; dx <= 1; dx++) {
							if (dx == 0 && dy == 0 && dz == 0) continue;

							int nx = x + dx;
							int ny = y + dy;
							int nz = z + dz;

							if (nx >= 0 && nx < m_dimX &&
								ny >= 0 && ny < m_dimY &&
								nz >= 0 && nz < m_dimZ) {
								int nidx = nx + ny * m_dimX + nz * (m_dimX * m_dimY);
								if (m_gridPtr->data[nidx] == VoxelState::FILLED) {
									neighborhoodDensity += 1.0f;
								}
							}
						}
					}
				}

				// Normalize and set AO value (more neighbors = more occlusion)
				neighborhoodDensity /= 26.0f; // 26 possible neighbors in 3x3x3 grid
				aoData[idx] = neighborhoodDensity * 0.7f; // Scale factor for AO strength
			}
		}
	}

	glTexImage3D(GL_TEXTURE_3D, 0, GL_R32F, m_dimX, m_dimY, m_dimZ,
		0, GL_RED, GL_FLOAT, aoData.data());

	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

	glBindTexture(GL_TEXTURE_3D, 0);
	checkGLError("createAmbientOcclusionTexture");
}

void VolumeRaycastRenderer::createIndirectLightTexture() {
	glGenTextures(1, &m_indirectLightTex);
	glBindTexture(GL_TEXTURE_3D, m_indirectLightTex);

	// Initialize with a neutral indirect light value
	std::vector<float> indirectData(m_dimX * m_dimY * m_dimZ * 4, 0.0f);

	// We'll fill this with actual light propagation later in updateIndirectLighting

	glTexImage3D(GL_TEXTURE_3D, 0, GL_RGBA16F, m_dimX, m_dimY, m_dimZ,
		0, GL_RGB, GL_FLOAT, indirectData.data());

	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

	glBindTexture(GL_TEXTURE_3D, 0);
	checkGLError("createIndirectLightTexture");
}

void VolumeRaycastRenderer::updateIndirectLighting() {
	// Skip if textures not created
	if (!m_volumeTex || !m_indirectLightTex || !m_indirectLightingComputeProg) return;

	// Bind the textures to image units
	glBindImageTexture(0, m_volumeTex, 0, GL_TRUE, 0, GL_READ_ONLY, GL_R32F);
	glBindImageTexture(1, m_gradientDirTex, 0, GL_TRUE, 0, GL_READ_ONLY, GL_RGBA32F);
	glBindImageTexture(2, m_radiationTex, 0, GL_TRUE, 0, GL_READ_ONLY, GL_R32F);
	glBindImageTexture(3, m_indirectLightTex, 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_RGBA16F);

	// Use the compute shader
	glUseProgram(m_indirectLightingComputeProg);

	// Set uniforms
	const glm::vec3 mainLightDir = glm::normalize(glm::vec3(0.5, 0.9, 0.4)); // Same as in fragment shader
	const glm::vec3 mainLightColor = glm::vec3(1.0f, 0.98f, 0.9f) * 1.3f;       // Same as in fragment shader

	glUniform3fv(glGetUniformLocation(m_indirectLightingComputeProg, "lightDir"), 1, glm::value_ptr(mainLightDir));
	glUniform3fv(glGetUniformLocation(m_indirectLightingComputeProg, "lightColor"), 1, glm::value_ptr(mainLightColor));
	glUniform3i(glGetUniformLocation(m_indirectLightingComputeProg, "volumeSize"), m_dimX, m_dimY, m_dimZ);
	glUniform1f(glGetUniformLocation(m_indirectLightingComputeProg, "indirectStrength"), 1.0f); // Adjust as needed

	// Dispatch the compute shader
	int groupsX = (m_dimX + 7) / 8;
	int groupsY = (m_dimY + 7) / 8;
	int groupsZ = (m_dimZ + 7) / 8;
	glDispatchCompute(groupsX, groupsY, groupsZ);

	// Memory barrier to make sure the compute shader writes are visible
	glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

	// Unbind textures
	glBindImageTexture(0, 0, 0, GL_FALSE, 0, GL_READ_ONLY, GL_R32F);
	glBindImageTexture(1, 0, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F);
	glBindImageTexture(2, 0, 0, GL_FALSE, 0, GL_READ_ONLY, GL_R32F);
	glBindImageTexture(3, 0, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA16F);

	checkGLError("updateIndirectLighting");
}
