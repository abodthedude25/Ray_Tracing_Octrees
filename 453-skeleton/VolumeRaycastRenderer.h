#pragma once

#include "Camera.h"
#include "Geometry.h"
#include "Renderer.h"

struct VoxelGrid; // Forward declaration

class VolumeRaycastRenderer : public Renderer {
public:
	VolumeRaycastRenderer();
	~VolumeRaycastRenderer();

	// Since our volume renderer doesn't generate triangles, we can just return empty.
	// But we must override the interface from "Renderer."
	std::vector<MCTriangle> render(const OctreeNode* node,
		const VoxelGrid& grid,
		int x0, int y0, int z0, int size) override;

	// Initialize the volume
	void initVolume(const VoxelGrid& grid);

	// Draw the volume with raycasting
	void drawRaycast(const Camera& cam,
		float aspectRatio,
		int screenW, int screenH);

	// Update the peeling plane (for dynamic peeling)
	void updatePeelPlane(float newZ);

private:
	unsigned int volumeTextureID;
	unsigned int maskTextureID; // Mask texture for peeling
	unsigned int raycastShaderProg;
	unsigned int quadVAO, quadVBO;

	int volDimX, volDimY, volDimZ;
	const VoxelGrid* gridPtr;

	// Initialize the mask volume
	void initMaskVolume(const VoxelGrid& grid);
};
