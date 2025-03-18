#pragma once
#include "Renderer.h"
#include <vector>
#include <glm/glm.hpp>
#include "OctreeVoxel.h" // for VoxelGrid
#include "Camera.h"      // for the Camera reference

// Structure for point-radiation (unchanged)
struct RadiationPoint {
	glm::vec3 worldPos;
	float radius;
};

class VolumeRaycastRenderer {
public:
	VolumeRaycastRenderer();
	~VolumeRaycastRenderer();

	void init(const VoxelGrid& grid);
	void setCamera(const Camera* camPtr) { m_cameraPtr = camPtr; }

	// point radiation usage
	void clearRadiationVolume();
	void updateSplatPoints(const std::vector<RadiationPoint>& pts);
	void dispatchRadiationCompute();
	void dispatchPrecompute();

	// final volume raycast
	void drawRaycast(float aspect);

	bool isInitialized() const { return m_inited; }

	glm::vec3 getBoxMin() const { return m_boxMin; }
	glm::vec3 getBoxMax() const { return m_boxMax; }
	// Provide read-only pointer to the voxel grid
	const VoxelGrid* getGridPtr() const { return m_gridPtr; }
	bool m_enableOctreeSkip = false;
	OctreeNode* m_octreeRoot = nullptr;
	void setOctreeRoot(OctreeNode* root) {
		m_octreeRoot = root;
	}

private:
	void createVolumeTexture(const VoxelGrid& grid);
	void createRadiationTexture();
	void createComputeShader();
	void createPrecomputeShader();
	void createPrecomputeTextures();
	void createRaycastProgram();
	void createFullscreenQuad();
	void bindRaycastUniforms(float aspect);

	void createAmbientOcclusionTexture();
	void createIndirectLightTexture();
	void createIndirectLightingComputeShader();
	void updateIndirectLighting();

private:
	float m_timeValue;

	// GL IDs
	GLuint m_volumeTex;
	GLuint m_radiationTex;
	GLuint m_computeProg;
	GLuint m_raycastProg;
	GLuint m_quadVAO, m_quadVBO;

	GLuint m_precomputeProg;  // The precompute shader program
	GLuint m_gradientMagTex;  // Texture for gradient magnitude
	GLuint m_gradientDirTex;  // Texture for gradient direction
	GLuint m_edgeFactorTex;   // Texture for edge factors
	bool m_precomputeNeeded;  // Flag for when precomputation is needed

	GLuint m_ambientOcclusionTex;
	GLuint m_indirectLightTex;
	GLuint m_indirectLightingComputeProg = 0;

	// grid info
	int m_dimX, m_dimY, m_dimZ;
	glm::vec3 m_boxMin, m_boxMax;
	const VoxelGrid* m_gridPtr;

	// for point-splats
	std::vector<RadiationPoint> m_splatPoints;

	// camera pointer
	const Camera* m_cameraPtr = nullptr;

	bool m_inited;
};
