// VolumeRaycastRenderer.h
#pragma once
#include "Camera.h"
#include "Geometry.h"
#include "Renderer.h"
#include <vector>

class VolumeRaycastRenderer : public Renderer {
public:
	VolumeRaycastRenderer();
	~VolumeRaycastRenderer();

	// Override the render method from Renderer
	// Since our volume renderer doesn't generate triangles, we return an empty vector
	std::vector<MCTriangle> render(const OctreeNode* node,
		const VoxelGrid& grid,
		int x0, int y0, int z0, int size) override;

	// Initialize the volume textures from the voxel grid data
	void initVolume(const VoxelGrid& grid);

	// Draw the volume using integrated raycasting and splatting
	void drawRaycast(const Camera& cam,
		float aspectRatio,
		int screenW, int screenH);

	// Update the peeling plane position and corresponding mask texture
	void updatePeelPlane(float newZ);

private:
	// OpenGL texture IDs
	unsigned int volumeTextureID;  // 3D texture for volume density
	unsigned int maskTextureID;    // 3D texture for masking (peeling)
	unsigned int gaussian2DTexID;  // 2D texture for XY-plane Gaussian kernel
	unsigned int gaussian1DTexID;  // 1D texture for Z-axis Gaussian kernel

	glm::vec3 boxMin, boxMax;     // The world-space bounding box corners

	// Shader program ID for raycasting and splatting
	unsigned int raycastShaderProg;

	// Vertex Array Object and Vertex Buffer Object for fullscreen quad
	unsigned int quadVAO, quadVBO;

	// Dimensions of the volume grid
	int volDimX, volDimY, volDimZ;

	// Pointer to the voxel grid data
	const VoxelGrid* gridPtr;
	GLuint tempTexture;     // For compute shader passes
	GLuint tempTexture2;    // Additional temp texture for compute
	bool hasComputeShaders; // Flag to check compute shader support
	GLuint antiAliasComputeProgram;

	bool initComputeShader();
	void checkComputeShaderCapabilities();

	// Initialize the mask volume based on a peeling plane
	void initMaskVolume();

	// Initialize Gaussian kernel textures
	void initGaussianKernels();
	void applyParallelAntiAliasing(GLuint sourceTexture, GLuint destTexture);

	// Generate Gaussian kernel data
	std::vector<float> generateGaussian2DKernel(int size = 64, float sigma = 0.4f);
	std::vector<float> generateGaussian1DKernel(int size = 64, float sigma = 0.4f);

	// Anti-aliasing helpers
	void applyJitteringSlice(std::vector<float>& maskData, int dimX, int dimY, int dimZ, int sliceZ);
	void applyCubicBSplineFilterSlice(std::vector<float>& maskData,
		int dimX,
		int dimY,
		int dimZ,
		int sliceZ);
	// Shader compilation helpers
	unsigned int compileShader(const char* src, GLenum type);
	unsigned int createRaycastProgram();
};
