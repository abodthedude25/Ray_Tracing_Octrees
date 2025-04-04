#pragma once
#include "OctreeVoxel.h"
#include <vector>
#include <glm/glm.hpp>
#include "Camera.h"
#include <glad/glad.h>
#include <unordered_map>
#include "Frustum.h"

// Forward declarations
class OctreeNode;
class VoxelGrid;

// A small struct for Ray
struct Ray {
	glm::vec3 origin;
	glm::vec3 direction;
};

// GPU-friendly node storage
struct GPUNodes {
	int x, y, z, size;
	int isLeaf, isSolid;
	int isUniform;
	int child[8];
};

class RayTracerBVH {
public:
	RayTracerBVH();
	~RayTracerBVH();

	// Set the octree and grid (builds a flattened GPU array)
	void setOctree(OctreeNode* root, const VoxelGrid& grid);

	// Initialize the compute pipeline if not already done
	void ensureComputeInitialized();

	// Main GPU-based render
	void renderSceneCompute(const Camera& camera,
		int width, int height,
		float aspect,
		float fovDeg);

	// Method to set frustum culling state
	void setFrustumCullingEnabled(bool enabled) { m_frustumCullingEnabled = enabled; }

	// Method to render with explicit frustum culling
	void renderSceneComputeWithCulling(const Camera& camera, int width, int height,
		float aspect, float fovDeg, bool updateFrustum);


private:
	// Scene data
	OctreeNode* m_octreeRoot;
	VoxelGrid   m_grid;

	// Flattened octree for GPU
	std::vector<GPUNodes> m_flatNodes;
	GLuint m_nodeSSBO;
	int    m_numNodes;

	// Compute pipeline
	bool   m_computeInited;
	GLuint m_outputTex;
	GLuint m_fullscreenVAO;
	GLuint m_fullscreenVBO;
	GLuint m_computeProg;
	GLuint m_fsqProg;

	// Flag to manage frustum culling for BVH
	bool m_frustumCullingEnabled;

	// Method to update nodes based on frustum culling
	void updateNodesWithFrustumCulling(const Frustum& frustum, float extraMargin = 150.0f);

	// Storage for visible nodes
	std::vector<GPUNodes> m_visibleNodes;

};
