#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <vector>
#include <unordered_map>
#include "Camera.h"
#include "Frustum.h"
#include "OctreeVoxel.h"

// Radiation point for splatting
struct RadiationPoint {
	glm::vec3 worldPos;
	float radius;
};

class VolumeRaycastRenderer {
public:
	VolumeRaycastRenderer();
	~VolumeRaycastRenderer();

	// Initialize with a voxel grid
	void init(const VoxelGrid& grid);

	// Render the volume with raycasting
	void drawRaycast(float aspect);

	// Setters for camera and octree
	void setCamera(Camera* cam);
	void setOctreeRoot(OctreeNode* root);

	// Getters for bounds and grid
	const glm::vec3& getBoxMin() const;
	const glm::vec3& getBoxMax() const;
	const VoxelGrid* getGridPtr() const;

	// Methods for radiation (carving)
	void clearRadiationVolume();
	void updateSplatPoints(const std::vector<RadiationPoint>& pts);
	void dispatchRadiationCompute();

	// Methods for frustum culling
	void setUpdateFrustumRequested(bool update);
	bool getUpdateFrustumRequested() const;
	void toggleFrustumCulling();
	void updateFrustumCulling(float aspect);

	// For octree-based ray skipping
	bool m_enableOctreeSkip;
	bool m_useMipMappedSkipping;  // Flag for enabling MIP mapped skipping

private:
	// Create/initialize various textures and shaders
	void createVolumeTexture(const VoxelGrid& grid);
	void createRadiationTexture();
	void createPrecomputeTextures();
	void createAmbientOcclusionTexture();
	void createIndirectLightTexture();

	// Create compute shaders
	void createComputeShader();
	void createPrecomputeShader();
	void createIndirectLightingComputeShader();

	// Create final rendering components
	void createRaycastProgram();
	void createFullscreenQuad();

	// Precompute and update operations
	void dispatchPrecompute();
	void updateIndirectLighting();
	void bindRaycastUniforms(float aspect);

	void optimizedFrustumCulling(float aspect);
	bool isNodeInFrustum(const OctreeNode* node, const Frustum& frustum,
		int x0, int y0, int z0, int size, float extraMargin);
	void markVisibleNodesOnly(const OctreeNode* node, const Frustum& frustum,
		int x0, int y0, int z0, int size, float extraMargin);

	// Hierarchical octree skipping
	void createMipMappedVolumeTexture(const VoxelGrid& grid);
	void buildSkipDistanceTexture();

	// New member variables
	GLuint m_octreeSkipTex;       // Texture for storing skip
	int m_maxMipLevel;            // Maximum MIP level for hierarchical skipping

	void updateWorkingVolumeWithVisibility();

	// OpenGL texture handles
	GLuint m_volumeTex;          // Original volume texture
	GLuint m_workingVolumeTex;   // Volume texture with frustum culling applied
	GLuint m_radiationTex;       // For storing carving/radiation values
	GLuint m_gradientMagTex;     // Gradient magnitude texture
	GLuint m_gradientDirTex;     // Gradient direction (normal) texture
	GLuint m_edgeFactorTex;      // Edge factor texture
	GLuint m_ambientOcclusionTex; // Ambient occlusion texture  
	GLuint m_indirectLightTex;    // Indirect lighting texture

	// OpenGL shader programs
	GLuint m_computeProg;        // Compute shader for point radiation
	GLuint m_precomputeProg;     // Compute shader for precomputing gradient
	GLuint m_raycastProg;        // Fragment shader for raycasting
	GLuint m_indirectLightingComputeProg; // Compute shader for indirect lighting

	// Quad for fullscreen rendering
	GLuint m_quadVAO;
	GLuint m_quadVBO;

	// Volume dimensions
	int m_dimX, m_dimY, m_dimZ;

	// Volume bounds
	glm::vec3 m_boxMin;
	glm::vec3 m_boxMax;

	// References to external data
	const VoxelGrid* m_gridPtr;
	Camera* m_cameraPtr;
	OctreeNode* m_octreeRoot;

	// State flags
	bool m_inited;
	bool m_precomputeNeeded;
	bool m_useFrustumCulling;
	bool m_updateFrustumRequested;
	bool m_needsInitialFrustumCulling;

	// Parameters
	float m_timeValue;
	float m_frustumMargin;

	// Radiation splatting points
	std::vector<RadiationPoint> m_splatPoints;

	// Frustum culling data
	std::unordered_map<const OctreeNode*, bool> m_nodeVisibility;

	glm::vec3 m_previousCamPos = glm::vec3(0.0f);
	glm::vec3 m_previousViewDir = glm::vec3(0.0f, 0.0f, -1.0f);
};
