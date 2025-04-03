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
	const VoxelGrid& grid,
	const std::unordered_map<const OctreeNode*, bool>* visibility = nullptr,
	const GLuint radiationTex = 0)
{
	if (!node) {
		return 1e30f; // no node, skip
	}

	// Check visibility if provided (frustum culling optimization)
	if (visibility && visibility->count(node) && !visibility->at(node)) {
		return 1e30f; // Node is not visible, skip
	}

	// Calculate world-space bounds with precomputed approach
	float vx = grid.voxelSize;
	float wx0 = grid.minX + node->x * vx;
	float wy0 = grid.minY + node->y * vx;
	float wz0 = grid.minZ + node->z * vx;
	float wSize = node->size * vx;

	glm::vec3 bmin(wx0, wy0, wz0);
	glm::vec3 bmax(wx0 + wSize, wy0 + wSize, wz0 + wSize);

	// Calculate ray-box intersection with SIMD-friendly code
	// Using precomputed reciprocals for better vectorization
	glm::vec3 invRd = 1.0f / rd;

	// Handle near-zero components to avoid division by zero
	const float smallValue = 1e-10f;
	if (std::abs(rd.x) < smallValue) invRd.x = rd.x >= 0 ? 1e10f : -1e10f;
	if (std::abs(rd.y) < smallValue) invRd.y = rd.y >= 0 ? 1e10f : -1e10f;
	if (std::abs(rd.z) < smallValue) invRd.z = rd.z >= 0 ? 1e10f : -1e10f;

	// Fast ray-box intersection
	glm::vec3 t1 = (bmin - ro) * invRd;
	glm::vec3 t2 = (bmax - ro) * invRd;

	glm::vec3 tNear = glm::min(t1, t2);
	glm::vec3 tFar = glm::max(t1, t2);

	float enterT = std::max(std::max(tNear.x, tNear.y), std::max(tNear.z, tMin));
	float exitT = std::min(std::min(tFar.x, tFar.y), std::min(tFar.z, tMax));

	// If no intersection or behind current best, skip node
	if (enterT > exitT) {
		return 1e30f;
	}

	// Early termination for leaf nodes
	if (node->isLeaf) {
		if (!node->isSolid) {
			return 1e30f; // Empty leaf node, skip
		}
		return enterT; // Solid leaf node, return entry point
	}

	// Direction-based traversal for internal nodes
	// Calculate ray direction octant (which of the 8 children to check first)
	int dirMask = ((rd.x > 0) ? 1 : 0) |
		((rd.y > 0) ? 2 : 0) |
		((rd.z > 0) ? 4 : 0);

	// Process children using hamming distance for optimal front-to-back ordering
	float bestT = 1e30f;

	// Visit children in order of increasing hamming distance from ray direction
	for (int dist = 0; dist <= 3; dist++) {
		// Check all children with current hamming distance
		for (int octant = 0; octant < 8; octant++) {
			// Count bits different between this octant and the ray direction
			int bitDiff = 0;
			int diff = octant ^ dirMask;
			while (diff) {
				bitDiff += diff & 1;
				diff >>= 1;
			}

			// Skip if not at current distance level
			if (bitDiff != dist) continue;

			const OctreeNode* child = node->children[octant];
			if (!child) continue;

			// Recursive ray check with tight bounds
			float childT = octreeRaySkip(child, ro, rd, enterT, exitT, grid, visibility, radiationTex);

			// Update best result with early termination
			if (childT < bestT) {
				bestT = childT;

				// Early exit optimization - found a hit in this branch
				if (childT < 1e30f) {
					return childT;
				}
			}
		}
	}

	return bestT;
}

//==================== CONSTRUCTOR / DESTRUCTOR ====================
VolumeRaycastRenderer::VolumeRaycastRenderer()
    : m_volumeTex(0)
    , m_workingVolumeTex(0)
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
    , m_useFrustumCulling(true)
    , m_updateFrustumRequested(true)
    , m_needsInitialFrustumCulling(true)
    , m_frustumMargin(150.0f)
    , m_octreeRoot(nullptr)
    , m_enableOctreeSkip(false)
{}

VolumeRaycastRenderer::~VolumeRaycastRenderer() {
    if (m_volumeTex)        glDeleteTextures(1, &m_volumeTex);
    if (m_workingVolumeTex) glDeleteTextures(1, &m_workingVolumeTex);
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

// Setter for octree root
void VolumeRaycastRenderer::setOctreeRoot(OctreeNode* root) {
    m_octreeRoot = root;
}

// Getter/Setter for update frustum flag
void VolumeRaycastRenderer::setUpdateFrustumRequested(bool update) {
    m_updateFrustumRequested = update;
}

bool VolumeRaycastRenderer::getUpdateFrustumRequested() const {
    return m_updateFrustumRequested;
}

// Toggle frustum culling
void VolumeRaycastRenderer::toggleFrustumCulling() {
    m_useFrustumCulling = !m_useFrustumCulling;
    
    // Force an update if enabling
    if (m_useFrustumCulling) {
        m_updateFrustumRequested = true;
    }
    
    std::cout << "Frustum culling " << (m_useFrustumCulling ? "enabled" : "disabled") << "\n";
}

// In your initialization
void VolumeRaycastRenderer::createVolumeTexture(const VoxelGrid& grid) {
	// Generate texture
	glGenTextures(1, &m_volumeTex);
	glBindTexture(GL_TEXTURE_3D, m_volumeTex);

	// Simple texture parameters WITHOUT mip-mapping
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

	// Store dimensions
	m_dimX = grid.dimX;
	m_dimY = grid.dimY;
	m_dimZ = grid.dimZ;

	// Fill volume data
	std::vector<float> volumeData(m_dimX * m_dimY * m_dimZ);
	for (int z = 0; z < m_dimZ; z++) {
		for (int y = 0; y < m_dimY; y++) {
			for (int x = 0; x < m_dimX; x++) {
				int idx = x + y * m_dimX + z * (m_dimX * m_dimY);
				volumeData[idx] = (grid.data[idx] == VoxelState::FILLED) ? 1.0f : 0.0f;
			}
		}
	}

	// Upload data
	glTexImage3D(GL_TEXTURE_3D, 0, GL_R8, m_dimX, m_dimY, m_dimZ, 0, GL_RED, GL_FLOAT, volumeData.data());

	// Set bounds
	m_boxMin = glm::vec3(grid.minX, grid.minY, grid.minZ);
	m_boxMax = glm::vec3(
		grid.minX + grid.dimX * grid.voxelSize,
		grid.minY + grid.dimY * grid.voxelSize,
		grid.minZ + grid.dimZ * grid.voxelSize
	);
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
uniform uint seed;

// Pre-computed jitter offsets for better cache coherency
const vec3 jitterOffsets[16] = vec3[16](
    vec3(-0.4, -0.4, -0.4), vec3(0.4, -0.4, -0.4),
    vec3(-0.4, 0.4, -0.4), vec3(0.4, 0.4, -0.4),
    vec3(-0.4, -0.4, 0.4), vec3(0.4, -0.4, 0.4),
    vec3(-0.4, 0.4, 0.4), vec3(0.4, 0.4, 0.4),
    vec3(-0.2, -0.2, -0.2), vec3(0.2, -0.2, -0.2),
    vec3(-0.2, 0.2, -0.2), vec3(0.2, 0.2, -0.2),
    vec3(-0.2, -0.2, 0.2), vec3(0.2, -0.2, 0.2),
    vec3(-0.2, 0.2, 0.2), vec3(0.2, 0.2, 0.2)
);

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

// Tile size for shared memory aggregation
#define TILE_SIZE 8

// Shared memory buffer for accumulating radiation within a tile
shared float radiationTile[TILE_SIZE][TILE_SIZE][TILE_SIZE];

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
void main() {
    // Get global work group info
    ivec3 tileStart = ivec3(
        gl_WorkGroupID.x * TILE_SIZE,
        gl_WorkGroupID.y * TILE_SIZE,
        gl_WorkGroupID.z * TILE_SIZE
    );
    
    // Initialize shared memory for this workgroup
    for (int z = 0; z < TILE_SIZE; z++) {
        if (gl_LocalInvocationID.z == 0) { // Only the first layer does initialization
            int y = int(gl_LocalInvocationID.y);
            int x = int(gl_LocalInvocationID.x);
            radiationTile[x][y][z] = 0.0;
        }
    }
    barrier();
    
    // Only process point 0 for simplicity and performance
    if (splats.length() == 0) return;
    RadiationPoint rp = splats[0];
    
    // Calculate point position in voxel coordinates
    vec3 size = boxMax - boxMin;
    vec3 voxelCoordF = (rp.worldPos - boxMin) / size * vec3(dimX, dimY, dimZ);
    ivec3 centerVoxel = ivec3(floor(voxelCoordF));
    
    // Calculate distance from the radiation point to this tile
    vec3 tileCenter = vec3(tileStart) + vec3(TILE_SIZE/2);
    float tileDist = distance(tileCenter, voxelCoordF);
    
    // Early exit if this tile is too far from the radiation point
    float maxEffect = rp.radius * 1.6; // Maximum distance of effect
    if (tileDist > maxEffect + float(TILE_SIZE)) {
        return; // Skip this tile completely
    }
    
    // Process voxels in this workgroup's section of the tile
    for (int z = 0; z < TILE_SIZE; z++) {
        // Calculate global voxel coordinates
        ivec3 voxelPos = tileStart + ivec3(gl_LocalInvocationID.xy, z);
        
        // Skip if outside volume bounds
        if (any(greaterThanEqual(voxelPos, ivec3(dimX, dimY, dimZ))) || 
            any(lessThan(voxelPos, ivec3(0)))) {
            continue;
        }
        
        // Calculate distance from voxel to radiation point
        vec3 voxelToPoint = vec3(voxelPos) - voxelCoordF;
        vec3 nd = voxelToPoint / rp.radius;
        float dist = length(nd);
        
        // Skip if too far
        if (dist > 1.6) {
            continue;
        }
        
        // Calculate weight using b-spline
        float w = bspline1D(nd.x) * bspline1D(nd.y) * bspline1D(nd.z);
        
        // Use jitter for better visual quality
        uint jitterIdx = (voxelPos.x + voxelPos.y * 4 + voxelPos.z * 16) % 16;
        vec3 jitter = jitterOffsets[jitterIdx] * 0.05;
        
        float w2 = bspline1D(nd.x + jitter.x) * bspline1D(nd.y + jitter.y) * bspline1D(nd.z + jitter.z);
        float finalW = 0.5 * (w + w2);
        
        // Accumulate in shared memory if significant
        if (finalW > 1e-4) {
            // Map global voxel position to local tile position
            ivec3 localPos = voxelPos - tileStart;
            if (all(lessThan(localPos, ivec3(TILE_SIZE))) && all(greaterThanEqual(localPos, ivec3(0)))) {
                // Since we don't have atomicAdd for floats in GLSL 430, we'll use manual addition
                // This is safe since we ensure each thread writes to a different location
                radiationTile[localPos.x][localPos.y][localPos.z] += finalW;
            }
        }
    }
    
    // Synchronize to ensure all threads have finished accumulating
    barrier();
    
    // Now write back to global memory
    for (int z = 0; z < TILE_SIZE; z++) {
        // Only proceed if this thread handles valid coordinates
        ivec3 voxelPos = tileStart + ivec3(gl_LocalInvocationID.xy, z);
        if (any(greaterThanEqual(voxelPos, ivec3(dimX, dimY, dimZ))) || 
            any(lessThan(voxelPos, ivec3(0)))) {
            continue;
        }
        
        // Map global voxel position to local tile position
        ivec3 localPos = voxelPos - tileStart;
        
        // Only write non-zero values to reduce memory traffic
        if (localPos.x < TILE_SIZE && localPos.y < TILE_SIZE && localPos.z < TILE_SIZE &&
            radiationTile[localPos.x][localPos.y][localPos.z] > 0.0) {
            float oldVal = imageLoad(radiationVol, voxelPos).r;
            float newVal = oldVal + radiationTile[localPos.x][localPos.y][localPos.z];
            imageStore(radiationVol, voxelPos, vec4(newVal, 0, 0, 0));
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

	// Validate and limit radiation points for performance
	for (auto& pt : m_splatPoints) {
		pt.radius = std::min(pt.radius, 6.0f); // Limit radius for performance
	}

	// Create SSBO for the radiation points
	GLuint ssbo = 0;
	glGenBuffers(1, &ssbo);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
	glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(RadiationPoint) * m_splatPoints.size(),
		m_splatPoints.data(), GL_STATIC_DRAW);
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

	// Process one point at a time in small batches
	for (size_t i = 0; i < std::min(size_t(1), m_splatPoints.size()); i++) {
		const RadiationPoint& point = m_splatPoints[i];

		// Calculate voxel position of the radiation point
		glm::vec3 normPos = (point.worldPos - m_boxMin) / (m_boxMax - m_boxMin);
		glm::ivec3 voxelPos = glm::ivec3(normPos * glm::vec3(m_dimX, m_dimY, m_dimZ));

		// Calculate effective radius in voxels (smaller for performance)
		int radiusVoxels = std::min(int(point.radius * 1.6), 12);

		// Calculate the tile range to cover this radius
		int tileRadius = (radiusVoxels + 7) / 8 + 1; // Round up to whole tiles

		// Calculate tile start and end with bounds checking
		int minTileX = std::max(0, voxelPos.x / 8 - tileRadius);
		int maxTileX = std::min(m_dimX / 8, voxelPos.x / 8 + tileRadius);
		int minTileY = std::max(0, voxelPos.y / 8 - tileRadius);
		int maxTileY = std::min(m_dimY / 8, voxelPos.y / 8 + tileRadius);
		int minTileZ = std::max(0, voxelPos.z / 8 - tileRadius);
		int maxTileZ = std::min(m_dimZ / 8, voxelPos.z / 8 + tileRadius);

		// Calculate number of tiles to process
		int numTilesX = maxTileX - minTileX + 1;
		int numTilesY = maxTileY - minTileY + 1;
		int numTilesZ = maxTileZ - minTileZ + 1;
		int totalTiles = numTilesX * numTilesY * numTilesZ;

		// Use much smaller batches for smoother performance
		const int BATCH_SIZE = 4; // Process just a few tiles at once

		std::cout << "  Processing point in " << totalTiles << " tiles with "
			<< (totalTiles + BATCH_SIZE - 1) / BATCH_SIZE << " batches\n";

		for (int z = minTileZ; z <= maxTileZ; z += BATCH_SIZE) {
			// Immediately synchronize after each small batch
			glFinish();

			int batchSizeZ = std::min(maxTileZ - z + 1, BATCH_SIZE);
			glDispatchCompute(numTilesX, numTilesY, batchSizeZ);

			// Add memory barrier between batches
			glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
		}
	}

	// Final barrier
	glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT);

	// Clean up
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
	glDeleteBuffers(1, &ssbo);

	checkGLError("dispatchRadiationCompute");

	// Force re-run precompute for updated radiation
	m_precomputeNeeded = true;

	// Clear the splat points to prevent accumulation
	m_splatPoints.clear();
}

//==================== Precompute Shader =====================
static const char* precomputeShaderSrc = R"COMPUTE(
#version 430 core
layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;

// We'll read from volumeTex and output to gradientMagTex, gradientDirTex, edgeFactorTex
layout(binding = 0) uniform sampler3D volumeTex;
layout(binding = 1) uniform sampler3D radiationTex;
layout(r32f, binding = 0) uniform writeonly image3D gradientMagTex;
layout(rgba16f, binding = 1) uniform writeonly image3D gradientDirTex;
layout(r32f, binding = 2) uniform writeonly image3D edgeFactorTex;

uniform vec3 boxMin;
uniform vec3 boxMax;
uniform ivec3 volumeSize;

// Higher quality sampling with trilinear interpolation
float sampleVolume(vec3 pos) {
    vec3 uvw = (pos - boxMin) / (boxMax - boxMin);
    if(any(lessThan(uvw, vec3(0.0))) || any(greaterThan(uvw, vec3(1.0)))) 
        return 0.0;
    return texture(volumeTex, uvw).r;
}

// Check if a voxel has been carved by radiation
float sampleRadiation(vec3 pos) {
    vec3 uvw = (pos - boxMin) / (boxMax - boxMin);
    if(any(lessThan(uvw, vec3(0.0))) || any(greaterThan(uvw, vec3(1.0)))) 
        return 0.0;
    return texture(radiationTex, uvw).r;
}

// Higher quality gradient computation using Sobel operator
vec3 computeSobelGradient(vec3 pos) {
    vec3 voxelSize = (boxMax - boxMin) / vec3(volumeSize);
    
    // Sobel operator masks for 3D
    float s[3] = float[3](-1.0, 0.0, 1.0);
    float w[3] = float[3](1.0, 2.0, 1.0);
    
    vec3 gradient = vec3(0.0);
    
    // Apply 3D Sobel
    for (int z = 0; z < 3; z++) {
        for (int y = 0; y < 3; y++) {
            for (int x = 0; x < 3; x++) {
                vec3 offset = vec3(s[x], s[y], s[z]);
                float weight = w[x] * w[y] * w[z];
                
                // Check for carved areas - don't compute gradients across radiation boundaries
                float radValue = sampleRadiation(pos + offset * voxelSize);
                if (radValue > 0.5) {
                    // Skip or reduce contribution from carved voxels
                    weight *= max(0.0, 1.0 - radValue);
                }
                
                float sampleValue = sampleVolume(pos + offset * voxelSize); // Changed 'sample' to 'sampleValue'
                gradient.x += sampleValue * s[x] * weight;
                gradient.y += sampleValue * s[y] * weight;
                gradient.z += sampleValue * s[z] * weight;
            }
        }
    }
    
    // Invert gradient direction to point outward from solid to empty
    return -gradient;
}

// Edge detection based on density changes
float detectEdges(vec3 pos, vec3 gradientDir, float gradientMag) {
    vec3 voxelSize = (boxMax - boxMin) / vec3(volumeSize);
    float centerValue = sampleVolume(pos);
    
    // Thresholds for edge detection
    const float isoValue = 0.5;
    const float edgeThreshold = 0.1;
    
    // Primary edge detection: check if voxel is near the isosurface
    float distToIso = abs(centerValue - isoValue);
    float edgeFactor = 1.0 - smoothstep(0.0, edgeThreshold, distToIso);
    
    // Secondary edge enhancement: magnitude of gradient
    // Normalize gradient magnitude to a reasonable range
    float normGradMag = min(1.0, gradientMag / 10.0);
    
    // Combine with curvature-based edge detection
    // Sample in both directions along the normal
    vec3 tangent1 = normalize(cross(gradientDir, vec3(0.0, 1.0, 0.0)));
    if (length(tangent1) < 0.1) tangent1 = normalize(cross(gradientDir, vec3(1.0, 0.0, 0.0)));
    vec3 tangent2 = cross(gradientDir, tangent1);
    
    // Sample density along tangents to detect curvature
    float s1 = sampleVolume(pos + tangent1 * voxelSize);
    float s2 = sampleVolume(pos - tangent1 * voxelSize);
    float s3 = sampleVolume(pos + tangent2 * voxelSize);
    float s4 = sampleVolume(pos - tangent2 * voxelSize);
    
    // Approximate curvature by density change along tangents
    float curvature = abs(s1 - centerValue) + abs(s2 - centerValue) + 
                     abs(s3 - centerValue) + abs(s4 - centerValue);
    curvature = curvature / 4.0;  // Normalize
    
    // Check for carved edges too
    float r1 = sampleRadiation(pos + gradientDir * voxelSize);
    float r0 = sampleRadiation(pos);
    if (r1 > 0.1 || r0 > 0.1) {
        // Enhanced edges around radiation carved areas
        edgeFactor = max(edgeFactor, smoothstep(0.0, 0.3, max(r0, r1)));
    }
    
    // Combine various edge factors - weight them as desired
    return edgeFactor * 0.7 + normGradMag * 0.2 + curvature * 0.1;
}

void main() {
    ivec3 voxelCoord = ivec3(gl_GlobalInvocationID.xyz);
    if(any(greaterThanEqual(voxelCoord, volumeSize))) return;

    // Get voxel position in world space
    vec3 voxelSize = (boxMax - boxMin) / vec3(volumeSize);
    vec3 voxelPos = boxMin + (vec3(voxelCoord) + 0.5) * voxelSize;
    
    // Compute gradient using Sobel
    vec3 gradient = computeSobelGradient(voxelPos);
    float gradMag = length(gradient);
    
    // Compute normalized gradient direction (normal)
    vec3 normal = (gradMag > 0.001) ? normalize(gradient) : vec3(0.0, 1.0, 0.0);
    
    // Detect edges
    float edgeFactor = detectEdges(voxelPos, normal, gradMag);
    
    // Store results in output textures
    imageStore(gradientMagTex, voxelCoord, vec4(gradMag));
    imageStore(gradientDirTex, voxelCoord, vec4(normal, 0.0));
    imageStore(edgeFactorTex, voxelCoord, vec4(edgeFactor));
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

	// Gradient direction - using RGBA16F for better precision and more compact storage
	glGenTextures(1, &m_gradientDirTex);
	glBindTexture(GL_TEXTURE_3D, m_gradientDirTex);
	glTexImage3D(GL_TEXTURE_3D, 0, GL_RGBA16F, m_dimX, m_dimY, m_dimZ,
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

	// Use the actual rendering texture (working volume) for precomputation
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_3D, m_workingVolumeTex);

	// Add radiation texture for better edge detection around carved areas
	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_3D, m_radiationTex);

	// Bind output images
	glBindImageTexture(0, m_gradientMagTex, 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_R32F);
	glBindImageTexture(1, m_gradientDirTex, 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_RGBA16F); // Changed to RGBA16F
	glBindImageTexture(2, m_edgeFactorTex, 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_R32F);

	glUseProgram(m_precomputeProg);

	glUniform3fv(glGetUniformLocation(m_precomputeProg, "boxMin"), 1, glm::value_ptr(m_boxMin));
	glUniform3fv(glGetUniformLocation(m_precomputeProg, "boxMax"), 1, glm::value_ptr(m_boxMax));
	glUniform3i(glGetUniformLocation(m_precomputeProg, "volumeSize"), m_dimX, m_dimY, m_dimZ);
	glUniform1i(glGetUniformLocation(m_precomputeProg, "volumeTex"), 0);
	glUniform1i(glGetUniformLocation(m_precomputeProg, "radiationTex"), 1);

	int groupsX = (m_dimX + 7) / 8;
	int groupsY = (m_dimY + 7) / 8;
	int groupsZ = (m_dimZ + 7) / 8;

	glDispatchCompute(groupsX, groupsY, groupsZ);

	glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT);

	// Unbind
	glBindImageTexture(0, 0, 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32F);
	glBindImageTexture(1, 0, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA16F);
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
		glm::vec3 cdir = m_cameraPtr->getLookDir(); // Get the camera direction

		glUniform3fv(glGetUniformLocation(m_raycastProg, "camPos"), 1, glm::value_ptr(cpos));
		glUniform3fv(glGetUniformLocation(m_raycastProg, "previousCamPos"), 1, glm::value_ptr(m_previousCamPos));
		glUniform3fv(glGetUniformLocation(m_raycastProg, "previousViewDir"), 1, glm::value_ptr(m_previousViewDir));

		// Store current values for next frame
		m_previousCamPos = cpos;
		m_previousViewDir = cdir;
    }

    glUniform3fv(glGetUniformLocation(m_raycastProg, "boxMin"), 1, glm::value_ptr(m_boxMin));
    glUniform3fv(glGetUniformLocation(m_raycastProg, "boxMax"), 1, glm::value_ptr(m_boxMax));

    // Bind working volume texture (includes frustum culling)
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_3D, m_workingVolumeTex);
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

    // Flag for shader to know if frustum culling is enabled
    GLint locUseFrustumCulling = glGetUniformLocation(m_raycastProg, "useFrustumCulling");
    glUniform1i(locUseFrustumCulling, m_useFrustumCulling ? 1 : 0);

	GLint uOctreeSkipTex = glGetUniformLocation(m_raycastProg, "octreeSkipTex");
	glUniform1i(uOctreeSkipTex, 7); // Use texture unit 7
	glActiveTexture(GL_TEXTURE7);
	glBindTexture(GL_TEXTURE_3D, m_octreeSkipTex);

	GLint uMaxMipLevel = glGetUniformLocation(m_raycastProg, "maxMipLevel");
	glUniform1i(uMaxMipLevel, m_maxMipLevel);

	GLint uUseMipMappedSkipping = glGetUniformLocation(m_raycastProg, "useMipMappedSkipping");
	glUniform1i(uUseMipMappedSkipping, m_useMipMappedSkipping ? 1 : 0);


    checkGLError("bindRaycastUniforms");
}

// Method 1: Enhanced frustum culling implementation
void VolumeRaycastRenderer::optimizedFrustumCulling(float aspect) {
	if (!m_octreeRoot || !m_cameraPtr) return;

	// Store previous camera position and direction for temporal coherence
	m_previousCamPos = m_cameraPtr->getPos();
	m_previousViewDir = -glm::normalize(glm::mat3(m_cameraPtr->getView())[2]);

	// Clear previous visibility state
	m_nodeVisibility.clear();

	// Create view frustum with a slightly wider FOV for stability
	glm::mat4 V = m_cameraPtr->getView();
	glm::mat4 P = glm::perspective(glm::radians(48.0f), aspect, 0.01f, 5000.f);
	Frustum frustum(P * V);

	// Start hierarchical traversal - only process potentially visible nodes
	markVisibleNodesOnly(m_octreeRoot, frustum, 0, 0, 0, m_dimX, m_frustumMargin);

	// Update the working volume texture with visibility data
	updateWorkingVolumeWithVisibility();

	// Reset the update request flag
	m_updateFrustumRequested = false;
	m_needsInitialFrustumCulling = false;
}

// Method 2: Efficient frustum test
bool VolumeRaycastRenderer::isNodeInFrustum(const OctreeNode* node, const Frustum& frustum,
	int x0, int y0, int z0, int size, float extraMargin) {
	if (!node) return false;

	// Calculate world-space bounds with margin
	float voxelSize = m_gridPtr->voxelSize;
	glm::vec3 minPoint(
		m_gridPtr->minX + x0 * voxelSize - extraMargin,
		m_gridPtr->minY + y0 * voxelSize - extraMargin,
		m_gridPtr->minZ + z0 * voxelSize - extraMargin
	);
	glm::vec3 maxPoint = minPoint + glm::vec3((size * voxelSize) + 2.0f * extraMargin);

	// Perform frustum test - returns -1 if outside, 0 if partially inside, 1 if fully inside
	return frustum.testAABB(minPoint, maxPoint, 0.0f) != -1;
}

// Method 3: Optimized node marking that only tracks visible nodes
void VolumeRaycastRenderer::markVisibleNodesOnly(const OctreeNode* node, const Frustum& frustum,
	int x0, int y0, int z0, int size, float extraMargin) {
	if (!node) return;

	// Early out test - if not in frustum, entire subtree is invisible
	if (!isNodeInFrustum(node, frustum, x0, y0, z0, size, extraMargin)) {
		return;
	}

	// Node is at least partially visible, mark it
	m_nodeVisibility[node] = true;

	// If it's a leaf node, we're done
	if (node->isLeaf) {
		return;
	}

	// Process children recursively
	int half = size / 2;
	for (int i = 0; i < 8; i++) {
		int ox = x0 + ((i & 1) ? half : 0);
		int oy = y0 + ((i & 2) ? half : 0);
		int oz = z0 + ((i & 4) ? half : 0);

		markVisibleNodesOnly(node->children[i], frustum, ox, oy, oz, half, extraMargin);
	}
}

// Method 4: Create MIP-mapped volume texture for hierarchical skipping
void VolumeRaycastRenderer::createMipMappedVolumeTexture(const VoxelGrid& grid) {
	// Generate texture for the volume data
	glGenTextures(1, &m_volumeTex);
	glBindTexture(GL_TEXTURE_3D, m_volumeTex);

	// Set texture parameters for MIP mapping
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

	// Store dimensions
	m_dimX = grid.dimX;
	m_dimY = grid.dimY;
	m_dimZ = grid.dimZ;

	// Allocate storage for volume data
	std::vector<float> volumeData(m_dimX * m_dimY * m_dimZ);

	// Fill volume data from grid
	for (int z = 0; z < m_dimZ; z++) {
		for (int y = 0; y < m_dimY; y++) {
			for (int x = 0; x < m_dimX; x++) {
				int idx = x + y * m_dimX + z * (m_dimX * m_dimY);
				volumeData[idx] = (grid.data[idx] == VoxelState::FILLED) ? 1.0f : 0.0f;
			}
		}
	}

	// Upload volume data
	glTexImage3D(GL_TEXTURE_3D, 0, GL_R8, m_dimX, m_dimY, m_dimZ, 0, GL_RED, GL_FLOAT, volumeData.data());

	// Generate MIP maps
	glGenerateMipmap(GL_TEXTURE_3D);

	// Calculate maximum MIP level
	m_maxMipLevel = std::floor(std::log2(std::max({ m_dimX, m_dimY, m_dimZ })));

	// Set bounds for raymarching
	m_boxMin = glm::vec3(grid.minX, grid.minY, grid.minZ);
	m_boxMax = glm::vec3(
		grid.minX + grid.dimX * grid.voxelSize,
		grid.minY + grid.dimY * grid.voxelSize,
		grid.minZ + grid.dimZ * grid.voxelSize
	);

	// Create working volume texture as well
	glGenTextures(1, &m_workingVolumeTex);
	glBindTexture(GL_TEXTURE_3D, m_workingVolumeTex);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
	glTexImage3D(GL_TEXTURE_3D, 0, GL_R8, m_dimX, m_dimY, m_dimZ, 0, GL_RED, GL_FLOAT, volumeData.data());
	glGenerateMipmap(GL_TEXTURE_3D);
}

void VolumeRaycastRenderer::buildSkipDistanceTexture() {
	// Create a 3D texture for storing skip distances
	glGenTextures(1, &m_octreeSkipTex);
	glBindTexture(GL_TEXTURE_3D, m_octreeSkipTex);

	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

	// Use a much lower resolution for skip texture - this is a critical optimization
	// Since we're skipping empty space, we don't need high resolution
	int skipTexDimX = std::max(m_dimX / 8, 16);
	int skipTexDimY = std::max(m_dimY / 8, 16);
	int skipTexDimZ = std::max(m_dimZ / 8, 16);

	std::vector<float> skipData(skipTexDimX * skipTexDimY * skipTexDimZ, 0.0f);

	// Cache octree node state to avoid redundant calculations
	std::unordered_map<OctreeNode*, float> nodeSkipCache;

	// Build a heightmap-like array for maximum empty space in each column
	// This is much faster than traversing the octree for every voxel
	std::vector<int> maxEmptyHeight(skipTexDimX * skipTexDimZ, 0);

	// First pass - calculate empty regions from a 2D top-down view
	// This dramatically reduces the number of octree traversals needed
	if (m_octreeRoot) {
		// Pre-process for faster lookups
		// Start from the top of the volume and move down
		for (int z = 0; z < skipTexDimZ; z++) {
			for (int x = 0; x < skipTexDimX; x++) {
				int maxHeight = 0;
				for (int y = 0; y < skipTexDimY; y++) {
					// Map to original grid coordinates
					int origX = (x * m_dimX) / skipTexDimX;
					int origY = (y * m_dimY) / skipTexDimY;
					int origZ = (z * m_dimZ) / skipTexDimZ;

					// Check if this region is solid
					if (origX >= 0 && origX < m_dimX &&
						origY >= 0 && origY < m_dimY &&
						origZ >= 0 && origZ < m_dimZ) {
						int idx = origX + origY * m_dimX + origZ * (m_dimX * m_dimY);
						if (idx >= 0 && idx < m_gridPtr->data.size()) {
							if (m_gridPtr->data[idx] == VoxelState::FILLED) {
								// Found solid voxel - store height and break
								maxHeight = y;
								break;
							}
						}
					}
				}
				maxEmptyHeight[x + z * skipTexDimX] = maxHeight;
			}
		}
	}

	// Second pass - use the heightmap and fill the skip texture
	for (int z = 0; z < skipTexDimZ; z++) {
		for (int y = 0; y < skipTexDimY; y++) {
			for (int x = 0; x < skipTexDimX; x++) {
				float skipDistance = 0.0f;

				// Map to original grid coordinates
				int origX = (x * m_dimX) / skipTexDimX;
				int origY = (y * m_dimY) / skipTexDimY;
				int origZ = (z * m_dimZ) / skipTexDimZ;

				// Quick empty space check using heightmap
				if (y < maxEmptyHeight[x + z * skipTexDimX]) {
					// Estimate skip distance based on empty vertical space
					float voxelSize = m_gridPtr->voxelSize;
					float emptyHeight = (maxEmptyHeight[x + z * skipTexDimX] - y) *
						((float)m_dimY / skipTexDimY) * voxelSize;

					// Apply a conservative factor for safety
					skipDistance = emptyHeight * 0.8f;

					// Normalize by volume size
					skipDistance /= (m_boxMax.y - m_boxMin.y);
				}
				else {
					// Detailed check only if we're in potentially occupied space
					bool isEmpty = true;

					// Simple check - just look at the center voxel
					if (origX >= 0 && origX < m_dimX &&
						origY >= 0 && origY < m_dimY &&
						origZ >= 0 && origZ < m_dimZ) {
						int idx = origX + origY * m_dimX + origZ * (m_dimX * m_dimY);
						if (idx >= 0 && idx < m_gridPtr->data.size()) {
							if (m_gridPtr->data[idx] == VoxelState::FILLED) {
								isEmpty = false;
							}
						}
					}

					if (isEmpty) {
						// Use a fixed value for empty spaces - this is faster than
						// traversing the octree and still gives good performance
						float voxelSize = m_gridPtr->voxelSize;
						float blockSize = voxelSize * (m_dimX / skipTexDimX);

						// Normalize by volume size
						skipDistance = blockSize / std::max({
							m_boxMax.x - m_boxMin.x,
							m_boxMax.y - m_boxMin.y,
							m_boxMax.z - m_boxMin.z
							});
					}
				}

				// Store the skip distance
				int index = x + y * skipTexDimX + z * (skipTexDimX * skipTexDimY);
				skipData[index] = skipDistance;
			}
		}
	}

	// Upload texture data
	glTexImage3D(GL_TEXTURE_3D, 0, GL_R32F, skipTexDimX, skipTexDimY, skipTexDimZ,
		0, GL_RED, GL_FLOAT, skipData.data());

	// Generate mipmaps for multi-resolution sampling
	glGenerateMipmap(GL_TEXTURE_3D);

	std::cout << "Skip distance texture built with dimensions: "
		<< skipTexDimX << "x" << skipTexDimY << "x" << skipTexDimZ << std::endl;
}

// Method 6: Modify your existing init method to use these new functions
void VolumeRaycastRenderer::init(const VoxelGrid& grid) {
	m_inited = true;
	m_gridPtr = &grid;
	m_enableOctreeSkip = true;
	m_useMipMappedSkipping = true;
	m_precomputeNeeded = true;
	m_useFrustumCulling = true;
	m_updateFrustumRequested = true;
	m_needsInitialFrustumCulling = true;
	m_frustumMargin = 20.0f;
	m_timeValue = 0.0f;

	// Use MIP-mapped texture creation
	createMipMappedVolumeTexture(grid);

	// Build skip distance texture for octree skipping
	buildSkipDistanceTexture();

	// Create other textures and shaders
	createRadiationTexture();
	createPrecomputeTextures();
	createAmbientOcclusionTexture();
	createIndirectLightTexture();
	createComputeShader();
	createPrecomputeShader();
	createIndirectLightingComputeShader();
	createRaycastProgram();
	createFullscreenQuad();

	// Initial precompute
	dispatchPrecompute();
}

void VolumeRaycastRenderer::updateFrustumCulling(float aspect) {
	if (!m_useFrustumCulling || !m_cameraPtr) return;

	// Store camera state
	m_previousCamPos = m_cameraPtr->getPos();
	m_previousViewDir = -glm::normalize(glm::mat3(m_cameraPtr->getView())[2]);

	// Create view frustum with a slightly NARROWER FOV for stricter culling
	// Change from 45.f to 42.f
	glm::mat4 V = m_cameraPtr->getView();
	glm::mat4 P = glm::perspective(glm::radians(42.f), aspect, 0.01f, 5000.f);
	Frustum frustum(P * V);

	// Create a simpler, grid-based culling approach
	int cellSize = 8; // Smaller cells for more precise culling (changed from 16)
	std::vector<bool> visibilityGrid((m_dimX / cellSize + 1) * (m_dimY / cellSize + 1) * (m_dimZ / cellSize + 1), false);

	// Mark visible grid cells
	for (int z = 0; z < m_dimZ; z += cellSize) {
		for (int y = 0; y < m_dimY; y += cellSize) {
			for (int x = 0; x < m_dimX; x += cellSize) {
				float voxelSize = m_gridPtr->voxelSize;

				// Calculate world space bounds
				glm::vec3 minPoint(
					m_gridPtr->minX + x * voxelSize,
					m_gridPtr->minY + y * voxelSize,
					m_gridPtr->minZ + z * voxelSize
				);
				glm::vec3 maxPoint = minPoint + glm::vec3(cellSize * voxelSize);

				// Check if cell is visible - use a slightly reduced margin
				int idx = (x / cellSize) + (y / cellSize) * (m_dimX / cellSize + 1) +
					(z / cellSize) * (m_dimX / cellSize + 1) * (m_dimY / cellSize + 1);

				// Reduce the margin by 20% for tighter culling
				float reducedMargin = m_frustumMargin * 0.8f;
				if (frustum.testAABB(minPoint, maxPoint, reducedMargin) != -1) {
					visibilityGrid[idx] = true;
				}
			}
		}
	}

	// Update volume texture based on grid visibility - clear boundary more aggressively
	glBindTexture(GL_TEXTURE_3D, m_workingVolumeTex);
	std::vector<float> modifiedVolume(m_dimX * m_dimY * m_dimZ, 0.0f);

	// Copy only visible regions
	for (int z = 0; z < m_dimZ; z++) {
		for (int y = 0; y < m_dimY; y++) {
			for (int x = 0; x < m_dimX; x++) {
				int voxelIdx = x + y * m_dimX + z * (m_dimX * m_dimY);
				int gridIdx = (x / cellSize) + (y / cellSize) * (m_dimX / cellSize + 1) +
					(z / cellSize) * (m_dimX / cellSize + 1) * (m_dimY / cellSize + 1);

				if (visibilityGrid[gridIdx]) {
					bool isFilled = m_gridPtr->data[voxelIdx] == VoxelState::FILLED;
					modifiedVolume[voxelIdx] = isFilled ? 1.0f : 0.0f;
				}
			}
		}
	}

	// Add a definitive boundary at frustum edges
	// This adds a clear "zero" border around the frustum to prevent mipmapping beyond
	int borderSize = 2;
	for (int z = 0; z < m_dimZ; z++) {
		for (int y = 0; y < m_dimY; y++) {
			for (int x = 0; x < m_dimX; x++) {
				int voxelIdx = x + y * m_dimX + z * (m_dimX * m_dimY);
				int gridIdx = (x / cellSize) + (y / cellSize) * (m_dimX / cellSize + 1) +
					(z / cellSize) * (m_dimX / cellSize + 1) * (m_dimY / cellSize + 1);

				// If this voxel has a visible neighbor but is itself not visible,
				// set it to zero to create a clear boundary
				if (!visibilityGrid[gridIdx]) {
					bool hasVisibleNeighbor = false;
					for (int dz = -1; dz <= 1 && !hasVisibleNeighbor; dz++) {
						for (int dy = -1; dy <= 1 && !hasVisibleNeighbor; dy++) {
							for (int dx = -1; dx <= 1 && !hasVisibleNeighbor; dx++) {
								int nx = x / cellSize + dx;
								int ny = y / cellSize + dy;
								int nz = z / cellSize + dz;

								if (nx >= 0 && nx < (m_dimX / cellSize + 1) &&
									ny >= 0 && ny < (m_dimY / cellSize + 1) &&
									nz >= 0 && nz < (m_dimZ / cellSize + 1)) {

									int neighborIdx = nx + ny * (m_dimX / cellSize + 1) +
										nz * (m_dimX / cellSize + 1) * (m_dimY / cellSize + 1);

									if (visibilityGrid[neighborIdx]) {
										hasVisibleNeighbor = true;
									}
								}
							}
						}
					}

					// If this is at the boundary of visibility, force it to zero
					if (hasVisibleNeighbor) {
						modifiedVolume[voxelIdx] = 0.0f;
					}
				}
			}
		}
	}

	// Update texture with new visibility
	glTexSubImage3D(GL_TEXTURE_3D, 0, 0, 0, 0, m_dimX, m_dimY, m_dimZ,
		GL_RED, GL_FLOAT, modifiedVolume.data());

	m_updateFrustumRequested = false;
}


void VolumeRaycastRenderer::updateWorkingVolumeWithVisibility() {
	if (!m_volumeTex || !m_workingVolumeTex || !m_gridPtr) return;

	// Create a CPU buffer of zeros
	std::vector<float> workingData(m_dimX * m_dimY * m_dimZ, 0.0f);

	// Track voxel counting details for debugging
	size_t totalFilledVoxels = 0;
	size_t visibleFilledVoxels = 0;

	// Count total filled voxels in the original grid
	for (size_t i = 0; i < m_gridPtr->data.size(); i++) {
		if (m_gridPtr->data[i] == VoxelState::FILLED) {
			totalFilledVoxels++;
		}
	}

	// For each visible node, set the corresponding voxels to 1.0
	for (const auto& pair : m_nodeVisibility) {
		const OctreeNode* node = pair.first;
		bool isVisible = pair.second;

		if (isVisible && node) {
			// IMPORTANT: Check if the node is leaf AND solid before filling voxels
			if (node->isLeaf && node->isSolid) {
				// Bounds of this node in voxel coordinates
				int startX = std::max(0, node->x);
				int startY = std::max(0, node->y);
				int startZ = std::max(0, node->z);
				int endX = std::min(m_dimX, node->x + node->size);
				int endY = std::min(m_dimY, node->y + node->size);
				int endZ = std::min(m_dimZ, node->z + node->size);

				// For each voxel in this node, copy the original volume data
				for (int z = startZ; z < endZ; z++) {
					for (int y = startY; y < endY; y++) {
						for (int x = startX; x < endX; x++) {
							int idx = x + y * m_dimX + z * (m_dimX * m_dimY);
							if (idx >= 0 && idx < workingData.size() &&
								idx < m_gridPtr->data.size() &&
								m_gridPtr->data[idx] == VoxelState::FILLED) {
								workingData[idx] = 1.0f;  // Solid voxel (visible)
								visibleFilledVoxels++;
							}
						}
					}
				}
			}
			// For non-leaf nodes, we need to recursively check their children
			else if (!node->isLeaf) {
				// For internal nodes, we can't directly set voxels
				// They might contain both solid and empty regions
				// This is handled by processing their children separately
			}
		}
	}

	// IMPORTANT: Use glTexImage3D to completely replace the texture
	glBindTexture(GL_TEXTURE_3D, m_workingVolumeTex);
	glTexImage3D(GL_TEXTURE_3D, 0, GL_R32F, m_dimX, m_dimY, m_dimZ,
		0, GL_RED, GL_FLOAT, workingData.data());

	// Reset texture parameters
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

	// Regenerate mipmaps
	glGenerateMipmap(GL_TEXTURE_3D);

	// Apply anisotropic filtering if supported
	GLfloat maxAniso = 0.0f;
	glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT, &maxAniso);
	if (maxAniso > 0.0f) {
		glTexParameterf(GL_TEXTURE_3D, GL_TEXTURE_MAX_ANISOTROPY_EXT, std::min(4.0f, maxAniso));
	}

	glBindTexture(GL_TEXTURE_3D, 0);

	// Enhanced debugging output
	size_t visibleVoxels = 0;
	for (const auto& val : workingData) {
		if (val > 0.0f) visibleVoxels++;
	}

	std::cout << "Working volume: " << visibleVoxels << " of " << m_dimX * m_dimY * m_dimZ
		<< " voxels visible (" << (100.0f * visibleVoxels / (m_dimX * m_dimY * m_dimZ)) << "%)\n";
	std::cout << "Total filled voxels in original grid: " << totalFilledVoxels
		<< " (" << (100.0f * totalFilledVoxels / (m_dimX * m_dimY * m_dimZ)) << "%)\n";
	std::cout << "Visible filled voxels: " << visibleFilledVoxels
		<< " (" << (100.0f * visibleFilledVoxels / totalFilledVoxels) << "% of filled)\n";

	// Verify texture upload
	std::cout << "Texture dimensions: " << m_dimX << "x" << m_dimY << "x" << m_dimZ << std::endl;
}


//==================== drawRaycast ====================
void VolumeRaycastRenderer::drawRaycast(float aspect) {
	if (!m_raycastProg) return;

	// Update frustum culling if needed
	if (m_useFrustumCulling && (m_updateFrustumRequested || m_needsInitialFrustumCulling)) {
		updateFrustumCulling(aspect);
		m_updateFrustumRequested = false;
		m_needsInitialFrustumCulling = false;
	}

	// Possibly re-run gradient precompute if needed
	if (m_precomputeNeeded) {
		dispatchPrecompute();
	}

	// Calculate skip distance using optimized ray casting
	float skipDistance = 0.0f;
	if (m_enableOctreeSkip && m_octreeRoot && m_cameraPtr) {
		// Improved sampling pattern with multiple rays
		const int gridSize = 7; // Increased from 5x5 to 7x7 for better coverage
		const float sampleOffset = 0.2f;

		// Reserve space for skip distances - no reallocation during sampling
		std::vector<float> validSkipDistances;
		validSkipDistances.reserve(gridSize * gridSize);

		// Camera setup for ray generation
		glm::mat4 V = m_cameraPtr->getView();
		glm::mat4 P = glm::perspective(glm::radians(45.0f), aspect, 0.1f, 5000.0f);
		glm::mat4 invV = glm::inverse(V);
		glm::mat4 invP = glm::inverse(P);
		glm::vec3 ro = m_cameraPtr->getPos();

		// Sample grid of rays across viewport
		// This gives us multiple samples to ensure stable skipping
		for (int y = 0; y < gridSize; y++) {
			for (int x = 0; x < gridSize; x++) {
				// Generate ray in screen space
				float ndcX = ((float)x / (gridSize - 1) - 0.5f) * 2.0f * sampleOffset;
				float ndcY = ((float)y / (gridSize - 1) - 0.5f) * 2.0f * sampleOffset;

				// Transform to world space
				glm::vec4 clipPos(ndcX, ndcY, 1.f, 1.f);
				glm::vec4 viewPos = invP * clipPos;
				viewPos /= viewPos.w;
				glm::vec4 worldPos4 = invV * viewPos;
				glm::vec3 rd = glm::normalize(glm::vec3(worldPos4) - ro);

				// Use optimized octree ray skip algorithm
				float raySkip = octreeRaySkip(
					m_octreeRoot,
					ro,
					rd,
					0.0f,
					1e30f,
					*m_gridPtr,
					m_useFrustumCulling ? &m_nodeVisibility : nullptr,
					m_radiationTex);

				// Store valid skip distances
				if (raySkip < 1e30f && raySkip > 0.0f) {
					validSkipDistances.push_back(raySkip);
				}
			}
		}

		// Use conservative percentile (15th) for safe skipping
		if (!validSkipDistances.empty()) {
			std::sort(validSkipDistances.begin(), validSkipDistances.end());
			int safeIndex = std::max(0, (int)(validSkipDistances.size() * 0.15f));
			skipDistance = validSkipDistances[safeIndex];

			// Apply safety margin (75% of distance)
			skipDistance *= 0.75f;
		}

		// Temporal coherence - smooth transition between frames
		static float lastSkipDistance = 0.0f;
		float blendFactor = 0.4f; // Reduced from 0.5f for faster adaptation
		skipDistance = lastSkipDistance * blendFactor + skipDistance * (1.0f - blendFactor);
		lastSkipDistance = skipDistance;
	}

	// Standard rendering setup
	glDisable(GL_DEPTH_TEST);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glClearColor(0, 0, 0, 1);
	glClear(GL_COLOR_BUFFER_BIT);

	// Activate shader program and bind uniforms
	glUseProgram(m_raycastProg);
	bindRaycastUniforms(aspect);

	// Set octree skip distance
	GLint locSkip = glGetUniformLocation(m_raycastProg, "octreeSkipT");
	if (locSkip != -1) {
		glUniform1f(locSkip, skipDistance);
	}

	// Draw fullscreen quad
	glBindVertexArray(m_quadVAO);
	glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
	glBindVertexArray(0);

	// Reset state
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
        const int radius = 6;
        
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
    if (!m_workingVolumeTex || !m_indirectLightTex || !m_indirectLightingComputeProg) return;

    // Bind the textures to image units - use working volume
    glBindImageTexture(0, m_workingVolumeTex, 0, GL_TRUE, 0, GL_READ_ONLY, GL_R32F);
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

// Getter methods
const glm::vec3& VolumeRaycastRenderer::getBoxMin() const {
    return m_boxMin;
}

const glm::vec3& VolumeRaycastRenderer::getBoxMax() const {
    return m_boxMax;
}

const VoxelGrid* VolumeRaycastRenderer::getGridPtr() const {
    return m_gridPtr;
}

void VolumeRaycastRenderer::setCamera(Camera* cam) {
    m_cameraPtr = cam;
}
