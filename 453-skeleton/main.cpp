// main.cpp
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <limits>
#include "Window.h"
#include "Camera.h"
#include "ShaderProgram.h"
#include "Geometry.h"
#include "Renderer.h"
#include "BuildingLoader.h"
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/glm.hpp>
#include <array>
#include <functional>
#include "RayTracerBVH.h"
#include "VolumeRaycastRenderer.h"
#include "CacheUtils.h"
#include "AdaptiveDualContouringRenderer.h"
#include "Frustum.h"
#include <fstream>
#include <string>
#include <filesystem>


// Function to save triangles to a binary file
bool saveTriangleCache(const std::string& filename, const std::vector<MCTriangle>& triangles) {
	std::ofstream file(filename, std::ios::binary);
	if (!file.is_open()) {
		std::cerr << "Failed to open file for writing: " << filename << std::endl;
		return false;
	}

	// Write number of triangles
	size_t numTriangles = triangles.size();
	file.write(reinterpret_cast<const char*>(&numTriangles), sizeof(numTriangles));

	// Write all triangles to the file
	file.write(reinterpret_cast<const char*>(triangles.data()),
		numTriangles * sizeof(MCTriangle));

	file.close();
	std::cout << "Saved " << numTriangles << " triangles to " << filename << std::endl;
	return true;
}

// Function to load triangles from a binary file
bool loadTriangleCache(const std::string& filename, std::vector<MCTriangle>& triangles) {
	std::ifstream file(filename, std::ios::binary);
	if (!file.is_open()) {
		std::cerr << "Failed to open file for reading: " << filename << std::endl;
		return false;
	}

	// Read number of triangles
	size_t numTriangles = 0;
	file.read(reinterpret_cast<char*>(&numTriangles), sizeof(numTriangles));

	// Resize vector and read all triangles
	triangles.resize(numTriangles);
	file.read(reinterpret_cast<char*>(triangles.data()),
		numTriangles * sizeof(MCTriangle));

	file.close();
	std::cout << "Loaded " << numTriangles << " triangles from " << filename << std::endl;
	return true;
}

// Function to generate a unique cache filename based on camera position
std::string generateCacheFilename(const Camera& camera, float aspect) {
	std::string cacheDir = "triangle_cache";

	// Create cache directory if it doesn't exist
	if (!std::filesystem::exists(cacheDir)) {
		std::filesystem::create_directory(cacheDir);
	}

	// Generate a hash based on camera parameters
	glm::vec3 pos = camera.getPos();
	float theta = camera.getTheta();
	float phi = camera.getPhi();

	// Simple hash function
	size_t hash = std::hash<float>{}(pos.x) ^
		(std::hash<float>{}(pos.y) << 1) ^
		(std::hash<float>{}(pos.z) << 2) ^
		(std::hash<float>{}(theta) << 3) ^
		(std::hash<float>{}(phi) << 4) ^
		(std::hash<float>{}(aspect) << 5);

	return cacheDir + "/dc_triangles_" + std::to_string(hash) + ".bin";
}

// Now modify the renderOctree function to add the force regenerate parameter
std::vector<MCTriangle> renderOctree(
	const OctreeNode* root,
	const VoxelGrid& grid,
	Renderer& renderer,
	const Camera& camera,
	float aspect,
	float extraMargin = 50.0f,
	const std::vector<MCTriangle>* previousTriangles = nullptr,
	bool forceRegenerate = false)  // Added parameter to force regeneration
{
	std::vector<MCTriangle> result;
	if (!root) return result;

	// For Dual Contouring, check if we can use cached triangles
	auto* dcRenderer = dynamic_cast<AdaptiveDualContouringRenderer*>(&renderer);
	if (dcRenderer && !forceRegenerate) {
		// Generate filename based on camera position
		std::string cacheFilename = generateCacheFilename(camera, aspect);

		// Try to load cached triangles
		if (std::filesystem::exists(cacheFilename)) {
			if (loadTriangleCache(cacheFilename, result)) {
				std::cout << "Using cached triangles from: " << cacheFilename << std::endl;
				return result;
			}
		}
	}

	// Create view and projection matrices
	glm::mat4 V = camera.getView();
	glm::mat4 P = glm::perspective(glm::radians(45.f), aspect, 0.01f, 5000.f);
	glm::mat4 VP = P * V;

	// Create frustum from view-projection matrix
	Frustum frustum(VP);

	// Estimate the typical size of the result vector to avoid reallocations
	if (previousTriangles) {
		result.reserve(previousTriangles->size());
	}
	else {
		// Allocate a reasonable size for initial rendering
		result.reserve(100000);
	}

	// For AdaptiveDualContouringRenderer, make sure it's properly initialized
	if (dcRenderer) {
		// Ensure thread pool is initialized
		// This is safe to call even if already initialized
		dcRenderer->initThreadPool();
		dcRenderer->m_useComputeShader = true;

		// Optionally, force parallel processing settings
		dcRenderer->m_useAdaptiveLOD = true;
		// Set a reasonable detail threshold
		dcRenderer->m_detailThreshold = 0.1f;
	}

	// Recursive function to traverse octree with frustum culling
	std::function<void(const OctreeNode*)> traverse = [&](const OctreeNode* node) {
		if (!node) return;

		// Calculate world-space bounds of this octree node
		float voxelSize = grid.voxelSize;
		glm::vec3 minPoint(
			grid.minX + node->x * voxelSize,
			grid.minY + node->y * voxelSize,
			grid.minZ + node->z * voxelSize
		);
		glm::vec3 maxPoint = minPoint + glm::vec3(node->size * voxelSize);

		// Test if this node is visible in the frustum
		int frustumTest = frustum.testAABB(minPoint, maxPoint, extraMargin);

		if (frustumTest == -1) {
			// Completely outside frustum, skip this node and its children
			return;
		}

		if (node->isLeaf) {
			// Leaf node inside or intersecting the frustum, render it
			auto tris = renderer.render(node, grid, node->x, node->y, node->z, node->size);

			// Merge triangles in batches to reduce contention
			if (!tris.empty()) {
				result.insert(result.end(), tris.begin(), tris.end());
			}
		}
		else {
			// Internal node, recurse to children
			for (auto child : node->children) {
				traverse(child);
			}
		}
		};

	// Start traversal from the root
	std::cout << "Starting octree traversal with frustum culling..." << std::endl;

	auto start = std::chrono::high_resolution_clock::now();
	traverse(root);
	auto end = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	std::cout << "Octree traversal completed in " << duration << " ms, triangles: " << result.size() << std::endl;

	// Save the triangles for Dual Contouring
	if (dcRenderer && result.size() > 0) {
		std::string cacheFilename = generateCacheFilename(camera, aspect);
		saveTriangleCache(cacheFilename, result);
	}

	return result;
}
static bool intersectBuildingVoxel(
	const Camera& cam,
	float screenX, float screenY,
	int windowWidth, int windowHeight,
	const glm::vec3& boxMin,
	const glm::vec3& boxMax,
	const VoxelGrid& grid,
	glm::vec3& outPosWorld,
	float aspect
)
{
	// 1) Convert screen coords -> NDC
	float ndcX = (screenX / float(windowWidth)) * 2.f - 1.f;
	float ndcY = 1.f - (screenY / float(windowHeight)) * 2.f;

	// 2) Build ray from camera through click point
	glm::mat4 V = cam.getView();
	glm::mat4 P = cam.getProj(aspect);
	glm::mat4 invV = glm::inverse(V);
	glm::mat4 invP = glm::inverse(P);

	glm::vec4 clipPos(ndcX, ndcY, 1.f, 1.f);
	glm::vec4 viewPos = invP * clipPos;
	viewPos /= viewPos.w;
	glm::vec4 worldPos4 = invV * viewPos;
	glm::vec3 rayDir = glm::normalize(glm::vec3(worldPos4) - cam.getPos());
	glm::vec3 rayOrigin = cam.getPos();

	// 3) Intersect with volume bounding box
	auto intersectBox = [](const glm::vec3& ro, const glm::vec3& rd,
		const glm::vec3& bmin, const glm::vec3& bmax) -> glm::vec2
		{
			glm::vec3 t1 = (bmin - ro) / rd;
			glm::vec3 t2 = (bmax - ro) / rd;
			glm::vec3 tmin = glm::min(t1, t2);
			glm::vec3 tmax = glm::max(t1, t2);
			float tN = std::max(std::max(tmin.x, tmin.y), tmin.z);
			float tF = std::min(std::min(tmax.x, tmax.y), tmax.z);
			return glm::vec2(tN, tF);
		};

	glm::vec2 tHit = intersectBox(rayOrigin, rayDir, boxMin, boxMax);
	float tNear = std::max(tHit.x, 0.f);
	float tFar = tHit.y;

	if (tNear > tFar) {
		return false; // No intersection with volume bounds
	}

	// 4) More precise ray marching to find first building voxel
	float stepSize = grid.voxelSize * 0.5f; // Smaller step size for accuracy
	float T = tNear; // Start at the box entry point

	// Calculate voxel dimensions in world units
	glm::vec3 voxelSize = (boxMax - boxMin) / glm::vec3(grid.dimX, grid.dimY, grid.dimZ);

	for (int i = 0; i < 8000; i++) { // Increased max iterations for precision
		if (T > tFar) break;

		// Current position along ray
		glm::vec3 posWorld = rayOrigin + rayDir * T;

		// Convert to normalized volume coordinates [0,1]
		glm::vec3 uvw = (posWorld - boxMin) / (boxMax - boxMin);

		// Check if within volume bounds
		if (uvw.x < 0.0f || uvw.x >= 1.0f ||
			uvw.y < 0.0f || uvw.y >= 1.0f ||
			uvw.z < 0.0f || uvw.z >= 1.0f)
		{
			T += stepSize;
			continue;
		}

		// Convert to voxel indices
		int vx = int(uvw.x * grid.dimX);
		int vy = int(uvw.y * grid.dimY);
		int vz = int(uvw.z * grid.dimZ);

		// Clamp to valid indices (should be redundant with previous check)
		vx = std::max(0, std::min(vx, grid.dimX - 1));
		vy = std::max(0, std::min(vy, grid.dimY - 1));
		vz = std::max(0, std::min(vz, grid.dimZ - 1));

		// Look up voxel state
		int idx = vx + vy * grid.dimX + vz * (grid.dimX * grid.dimY);

		if (idx >= 0 && idx < grid.data.size() && grid.data[idx] == VoxelState::FILLED) {
			// Found a filled voxel - adjust position to be exactly at the voxel surface

			// Calculate the position slightly before the hit point to place the radiation
			// point just at the surface rather than inside the voxel
			outPosWorld = posWorld - rayDir * (stepSize * 0.1f);

			return true;
		}

		// Adaptive step size - smaller near potential surfaces
		// Check neighbors to see if we're approaching a surface
		bool nearSurface = false;
		for (int dz = -1; dz <= 1 && !nearSurface; dz++) {
			for (int dy = -1; dy <= 1 && !nearSurface; dy++) {
				for (int dx = -1; dx <= 1 && !nearSurface; dx++) {
					int nx = vx + dx;
					int ny = vy + dy;
					int nz = vz + dz;

					if (nx >= 0 && nx < grid.dimX &&
						ny >= 0 && ny < grid.dimY &&
						nz >= 0 && nz < grid.dimZ)
					{
						int nidx = nx + ny * grid.dimX + nz * (grid.dimX * grid.dimY);
						if (nidx >= 0 && nidx < grid.data.size() && grid.data[nidx] == VoxelState::FILLED) {
							nearSurface = true;
						}
					}
				}
			}
		}

		// Adaptive step size - use smaller steps near surfaces
		T += nearSurface ? stepSize * 0.25f : stepSize;
	}

	return false; // No building voxel found along the ray
}

// Enhanced generateTestVolume: Multi-shell Sphere
static std::vector<float> generateTestVolume(int dimX, int dimY, int dimZ) {
	// Multi-shell sphere: outer shell and inner hollow core
	printf("Generating test volume: Multi-shell Sphere\n");
	std::vector<float> volume(dimX * dimY * dimZ, 0.f);

	// Center of the volume
	float cx = 0.5f * (dimX - 1);
	float cy = 0.5f * (dimY - 1);
	float cz = 0.5f * (dimZ - 1);

	// Radii for outer shell and inner core
	float rOuter = 0.4f * std::min({ float(dimX), float(dimY), float(dimZ) });
	float rInner = 0.2f * std::min({ float(dimX), float(dimY), float(dimZ) });

	for (int z = 0; z < dimZ; z++) {
		for (int y = 0; y < dimY; y++) {
			for (int x = 0; x < dimX; x++) {
				float dx = x - cx;
				float dy = y - cy;
				float dz = z - cz;
				float dist = std::sqrt(dx * dx + dy * dy + dz * dz);
				int idx = x + y * dimX + z * (dimX * dimY);

				if (dist < rInner || dist > rOuter) {
					// Outside the outer shell or inside the inner core: Empty
					volume[idx] = -1.0f; // Density <= 0
				}
				else {
					// Within the outer shell: Filled
					volume[idx] = 1.0f; // Density > 0
				}
			}
		}
	}
	return volume;
}


// A function that re-centers the voxel grid around its filled region
static void recenterFilledVoxels(VoxelGrid& grid) {
	// Find bounding box of all FILLED voxels
	float filledMinX = std::numeric_limits<float>::max();
	float filledMinY = std::numeric_limits<float>::max();
	float filledMinZ = std::numeric_limits<float>::max();
	float filledMaxX = -std::numeric_limits<float>::max();
	float filledMaxY = -std::numeric_limits<float>::max();
	float filledMaxZ = -std::numeric_limits<float>::max();

	for (int z = 0; z < grid.dimZ; ++z) {
		for (int y = 0; y < grid.dimY; ++y) {
			for (int x = 0; x < grid.dimX; ++x) {
				int idx = x + y * grid.dimX + z * (grid.dimX * grid.dimY);
				if (grid.data[idx] == VoxelState::FILLED) {
					float cx = grid.minX + (x + 0.5f) * grid.voxelSize;
					float cy = grid.minY + (y + 0.5f) * grid.voxelSize;
					float cz = grid.minZ + (z + 0.5f) * grid.voxelSize;

					if (cx < filledMinX) filledMinX = cx;
					if (cy < filledMinY) filledMinY = cy;
					if (cz < filledMinZ) filledMinZ = cz;
					if (cx > filledMaxX) filledMaxX = cx;
					if (cy > filledMaxY) filledMaxY = cy;
					if (cz > filledMaxZ) filledMaxZ = cz;
				}
			}
		}
	}

	if (filledMinX > filledMaxX) {
		// No filled voxels
		std::cout << "(Recenter) No filled voxels. Skipping.\n";
		return;
	}

	float centerX = 0.5f * (filledMinX + filledMaxX);
	float centerY = 0.5f * (filledMinY + filledMaxY);
	float centerZ = 0.5f * (filledMinZ + filledMaxZ);

	// Shift min coordinates to center the grid around the origin
	grid.minX -= centerX;
	grid.minY -= centerY;
	grid.minZ -= centerZ;

	std::cout << "Recentered grid around origin. Moved by ("
		<< centerX << ", " << centerY << ", " << centerZ << ")\n";
}

// ========== Functions to Gather Octree Wireframe Lines ==========
static void getCubeCorners(const VoxelGrid& grid, int x0, int y0, int z0,
	int size, std::array<glm::vec3, 8>& corners) {
	float vx = grid.voxelSize;
	glm::vec3 base(grid.minX + x0 * vx,
		grid.minY + y0 * vx,
		grid.minZ + z0 * vx);
	float w = size * vx;

	corners[0] = base;
	corners[1] = base + glm::vec3(w, 0, 0);
	corners[2] = base + glm::vec3(w, w, 0);
	corners[3] = base + glm::vec3(0, w, 0);
	corners[4] = base + glm::vec3(0, 0, w);
	corners[5] = base + glm::vec3(w, 0, w);
	corners[6] = base + glm::vec3(w, w, w);
	corners[7] = base + glm::vec3(0, w, w);
}

static void generateOctreeWireframe(const VoxelGrid& grid,
	const OctreeNode* node,
	int x0, int y0, int z0,
	int size,
	std::vector<glm::vec3>& lines,
	const Frustum& frustum,
	float extraMargin = 50.0f) {

	if (!node) return;

	// Calculate world-space bounds
	float voxelSize = grid.voxelSize;
	glm::vec3 minPoint(
		grid.minX + x0 * voxelSize,
		grid.minY + y0 * voxelSize,
		grid.minZ + z0 * voxelSize
	);
	glm::vec3 maxPoint = minPoint + glm::vec3(size * voxelSize);

	// Test if this node is visible in the frustum
	int frustumTest = frustum.testAABB(minPoint, maxPoint, extraMargin);

	if (frustumTest == -1) {
		// Completely outside frustum, skip this node and its children
		return;
	}

	if (node->isLeaf) {
		std::array<glm::vec3, 8> corners;
		getCubeCorners(grid, x0, y0, z0, size, corners);
		static const int E[12][2] = {
			{0,1},{1,2},{2,3},{3,0},
			{4,5},{5,6},{6,7},{7,4},
			{0,4},{1,5},{2,6},{3,7}
		};
		for (int i = 0; i < 12; i++) {
			lines.push_back(corners[E[i][0]]);
			lines.push_back(corners[E[i][1]]);
		}
		return;
	}

	int half = size / 2;
	for (int i = 0; i < 8; i++) {
		int ox = x0 + ((i & 1) ? half : 0);
		int oy = y0 + ((i & 2) ? half : 0);
		int oz = z0 + ((i & 4) ? half : 0);
		generateOctreeWireframe(grid, node->children[i],
			ox, oy, oz, half,
			lines, frustum, extraMargin);
	}
}

enum class RenderMode {
	MarchingCubes,
	DualContouring,
	VoxelBlocks,
	BVHRayTrace,
	VolumeRaycast
};

struct Assignment4 : public CallbackInterface {
	Assignment4()
		: wireframeMode(false),
		showOctreeWire(false),
		currentMode(RenderMode::DualContouring),
		oldMode(RenderMode::DualContouring),
		camera(glm::radians(90.0f), glm::radians(0.f), 500.f),
		aspect(1.f),
		rightMouseDown(false),
		leftMouseDown(false),
		mouseOldX(0.0), mouseOldY(0.0),
		buildingCenter(0.f),
		peelPlaneZ(0.0f),
		renderModeToggle(0),
		cameraChanged(true),
		updateFrustumRequested(true)

	{
		camera.pan(0.f, 100.f);
		lastViewMatrix = camera.getView(); // Initialize last view matrix
	}

	// ----------- KEY CALLBACK -------------
	void keyCallback(int key, int scancode, int action, int mods) override {
		if (action == GLFW_PRESS) {
			if (key == GLFW_KEY_W) {
				wireframeMode = !wireframeMode;
			}
			else if (key == GLFW_KEY_S) {
				showOctreeWire = !showOctreeWire;
			}
			if (key == GLFW_KEY_F) {
				updateFrustumRequested = true;
			}
			else if (key == GLFW_KEY_R) {
				cameraChanged = true;
				updateFrustumRequested = true;
				// First store the old mode
				oldMode = currentMode;

				// Clear geometry caches when switching modes
				cpuGeom.verts.clear();
				cpuGeom.normals.clear();
				cpuGeom.cols.clear();

				// Cycle through modes in order
				switch (currentMode) {
				case RenderMode::MarchingCubes:
					currentMode = RenderMode::VoxelBlocks;
					break;
				case RenderMode::VoxelBlocks:
					currentMode = RenderMode::DualContouring;
					break;
				case RenderMode::DualContouring:
					currentMode = RenderMode::VolumeRaycast;
					break;
				case RenderMode::BVHRayTrace:
					currentMode = RenderMode::MarchingCubes;
					break;
				case RenderMode::VolumeRaycast:
					currentMode = RenderMode::BVHRayTrace;
					break;
				}

				std::cout << "Switched render mode from ";
				switch (oldMode) {
				case RenderMode::MarchingCubes:    std::cout << "MC"; break;
				case RenderMode::DualContouring:   std::cout << "DC"; break;
				case RenderMode::VoxelBlocks:      std::cout << "Blocks"; break;
				case RenderMode::BVHRayTrace:      std::cout << "RayTrace"; break;
				case RenderMode::VolumeRaycast:    std::cout << "Volume"; break;
				}
				std::cout << " to ";
				switch (currentMode) {
				case RenderMode::MarchingCubes:    std::cout << "MC\n"; break;
				case RenderMode::DualContouring:   std::cout << "DC\n"; break;
				case RenderMode::VoxelBlocks:      std::cout << "Blocks\n"; break;
				case RenderMode::BVHRayTrace:      std::cout << "RayTrace\n"; break;
				case RenderMode::VolumeRaycast:    std::cout << "Volume\n"; break;
				}
			}
			else if (key == GLFW_KEY_C && currentMode == RenderMode::BVHRayTrace) {
				camera.setTarget(buildingCenter);
				std::cout << "Camera target set to building center: "
					<< buildingCenter.x << ", "
					<< buildingCenter.y << ", "
					<< buildingCenter.z << "\n";
			}
			else if (key == GLFW_KEY_UP) {
				peelPlaneZ += 0.05f;
			}
			else if (key == GLFW_KEY_DOWN) {
				peelPlaneZ -= 0.05f;
			}
			else if (key == GLFW_KEY_X) {
				renderModeToggle = renderModeToggle == 0 ? 1 : 0;
			}
			else if (key == GLFW_KEY_O && currentMode == RenderMode::VolumeRaycast) {
				octreeSkipEnabled = !octreeSkipEnabled;
				if (raycastRendererPtr) {
					raycastRendererPtr->m_enableOctreeSkip = octreeSkipEnabled;
				}
				std::cout << "Octree Skip " << (octreeSkipEnabled ? "Enabled" : "Disabled") << std::endl;
			}
			else if (key == GLFW_KEY_M && currentMode == RenderMode::VolumeRaycast) {
				mipmappedSkippingEnabled = !mipmappedSkippingEnabled;
				if (raycastRendererPtr) {
					raycastRendererPtr->m_useMipMappedSkipping = mipmappedSkippingEnabled;
				}
				std::cout << "Mipmapped Skipping " << (mipmappedSkippingEnabled ? "Enabled" : "Disabled") << std::endl;
			}
			else if (key == GLFW_KEY_G) {
				forceRegenerateDC = !forceRegenerateDC;
				std::cout << "Force regenerate DC triangles: " << (forceRegenerateDC ? "ON" : "OFF") << std::endl;
				if (forceRegenerateDC) {
					cameraChanged = true;
					updateFrustumRequested = true;
				}
			}
		}
	}

	// Mouse
	void cursorPosCallback(double xpos, double ypos) override {
		if (rightMouseDown) {
			camera.incrementTheta(ypos - mouseOldY);
			camera.incrementPhi(xpos - mouseOldX);
			cameraChanged = true; // Mark camera as changed
		}
		if (leftMouseDown) {
			camera.pan(xpos - mouseOldX, ypos - mouseOldY);
			cameraChanged = true; // Mark camera as changed
		}
		mouseOldX = xpos;
		mouseOldY = ypos;
	}

	void mouseButtonCallback(int button, int action, int mods) override {
		if (button == GLFW_MOUSE_BUTTON_RIGHT) {
			rightMouseDown = (action == GLFW_PRESS);
		}
		if (button == GLFW_MOUSE_BUTTON_LEFT) {
			leftMouseDown = (action == GLFW_PRESS);

			if (action == GLFW_PRESS && currentMode == RenderMode::VolumeRaycast) {
				// Let's do a carve or paint operation:
				// 1) Raycast to find building voxel
				if (raycastRendererPtr) {
					// call our helper
					glm::vec3 hitPos;
					bool foundVoxel = intersectBuildingVoxel(
						camera,
						float(mouseOldX), // the last known cursor pos
						float(mouseOldY),
						lastWindowWidth, lastWindowHeight,
						raycastRendererPtr->getBoxMin(),
						raycastRendererPtr->getBoxMax(),
						*(raycastRendererPtr->getGridPtr()), // deref the pointer
						hitPos,
						aspect
					);
					// print the value of intersectBuildingVoxel
					std::cout << "intersectBuildingVoxel: " << foundVoxel << std::endl;

					if (foundVoxel) {
						std::cout << "Carve at (" << hitPos.x << ", "
							<< hitPos.y << ", " << hitPos.z << ")\n";

						try {
							std::cout << "Step 1: Creating RadiationPoint" << std::endl;
							RadiationPoint rp;
							rp.worldPos = hitPos;
							rp.radius = 0.5f; // Even smaller radius for testing

							std::cout << "Step 2: Clearing radiation volume" << std::endl;
							//raycastRendererPtr->clearRadiationVolume();

							std::cout << "Step 3: Creating vector" << std::endl;
							std::vector<RadiationPoint> singlePoint;
							singlePoint.push_back(rp);

							std::cout << "Step 4: Updating splat points" << std::endl;
							raycastRendererPtr->updateSplatPoints(singlePoint);

							std::cout << "Step 5: Dispatching compute" << std::endl;
							raycastRendererPtr->dispatchRadiationCompute();

							std::cout << "All steps completed successfully" << std::endl;
						}
						catch (const std::exception& e) {
							std::cerr << "Exception: " << e.what() << std::endl;
						}
						catch (...) {
							std::cerr << "Unknown exception caught" << std::endl;
						}
					}
					else {
						std::cout << "No building found under cursor.\n";
					}
				}
			}
		}
	}

	void scrollCallback(double xoffset, double yoffset) override {
		camera.incrementR(yoffset);
		cameraChanged = true; // Mark camera as changed
	}

	void windowSizeCallback(int width, int height) override {
		if (height < 1) height = 1;

		// Track aspect ratio changes
		float oldAspect = aspect;
		aspect = float(width) / float(height);

		lastWindowWidth = width;
		lastWindowHeight = height;

		// Update viewport
		glViewport(0, 0, width, height);

		// Resize framebuffer textures without full recreation
		if (framebufferInitialized) {
			resizeFramebufferTextures();
		}

		// Only mark camera as changed if aspect ratio changed significantly
		if (std::abs(aspect - oldAspect) > 0.001f) {
			updateFrustumRequested = true;
			cameraChanged = true;

			// Force a full redraw after resize
			if (currentMode == RenderMode::VolumeRaycast && raycastRendererPtr) {
				raycastRendererPtr->setUpdateFrustumRequested(true);
			}
		}
	}

	void viewPipeline(ShaderProgram& sp) {
		glm::mat4 M(1.f);
		glm::mat4 V = camera.getView();
		glm::mat4 P = glm::perspective(glm::radians(45.f), aspect, 0.01f, 5000.f);

		GLint uM = glGetUniformLocation(sp, "M");
		GLint uV = glGetUniformLocation(sp, "V");
		GLint uP = glGetUniformLocation(sp, "P");

		glUniformMatrix4fv(uM, 1, GL_FALSE, &M[0][0]);
		glUniformMatrix4fv(uV, 1, GL_FALSE, &V[0][0]);
		glUniformMatrix4fv(uP, 1, GL_FALSE, &P[0][0]);
	}

	bool hasCameraChanged() {
		glm::mat4 currentView = camera.getView();

		// Check if view matrix has changed significantly
		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				/*printf("currentView[%d][%d]: %f\n", i, j, currentView[i][j]);
				printf("lastViewMatrix[%d][%d]: %f\n", i, j, lastViewMatrix[i][j]);*/
				if (std::abs(currentView[i][j] - lastViewMatrix[i][j]) > 0.0001f) {
					lastViewMatrix = currentView;
					return true;
				}
			}
		}

		return false;
	}

	// Keep the original initialization function for first-time setup
	void initializeFramebufferCache() {
		if (framebufferInitialized) return;

		// Create a framebuffer
		glGenFramebuffers(1, &framebufferCache);
		glBindFramebuffer(GL_FRAMEBUFFER, framebufferCache);

		// Create texture to render to
		glGenTextures(1, &frameTexture);
		glBindTexture(GL_TEXTURE_2D, frameTexture);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, lastWindowWidth, lastWindowHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

		// Create depth buffer
		glGenTextures(1, &frameDepthTexture);
		glBindTexture(GL_TEXTURE_2D, frameDepthTexture);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, lastWindowWidth, lastWindowHeight, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

		// Attach textures to framebuffer
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, frameTexture, 0);
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, frameDepthTexture, 0);

		// Check if framebuffer is complete
		if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
			std::cerr << "Framebuffer not complete!" << std::endl;
		}

		// Unbind framebuffer
		glBindFramebuffer(GL_FRAMEBUFFER, 0);

		framebufferInitialized = true;
	}

	// Add a new function specifically for resizing framebuffer textures
	void resizeFramebufferTextures() {
		if (!framebufferInitialized) {
			initializeFramebufferCache();
			return;
		}

		// Just resize the existing textures without recreating the framebuffer
		glBindTexture(GL_TEXTURE_2D, frameTexture);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, lastWindowWidth, lastWindowHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

		glBindTexture(GL_TEXTURE_2D, frameDepthTexture);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, lastWindowWidth, lastWindowHeight, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);

		// No need to rebind them to the framebuffer as the textures are the same objects

		// Make sure everything is OK
		glBindFramebuffer(GL_FRAMEBUFFER, framebufferCache);
		if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
			std::cerr << "Framebuffer incomplete after resize!" << std::endl;
		}
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
	}

	// Add this method to draw the cached frame
	void drawCachedFrame() {
		// Ensure we're drawing to the default framebuffer
		glBindFramebuffer(GL_FRAMEBUFFER, 0);

		// Set up a fullscreen draw
		glDisable(GL_DEPTH_TEST);

		// Use a simple shader to draw the texture
		static bool shaderInitialized = false;
		static GLuint frameCopyProgram = 0;
		static GLuint quadVAO = 0;

		if (!shaderInitialized) {
			// Create a simple shader for copying the texture
			const char* vertSrc =
				"#version 330 core\n"
				"layout(location=0) in vec2 aPos;\n"
				"out vec2 texCoord;\n"
				"void main() {\n"
				"    texCoord = aPos * 0.5 + 0.5;\n"
				"    gl_Position = vec4(aPos, 0.0, 1.0);\n"
				"}\n";

			const char* fragSrc =
				"#version 330 core\n"
				"in vec2 texCoord;\n"
				"out vec4 FragColor;\n"
				"uniform sampler2D screenTexture;\n"
				"void main() {\n"
				"    FragColor = texture(screenTexture, texCoord);\n"
				"}\n";

			// Compile vertex shader
			GLuint vertShader = glCreateShader(GL_VERTEX_SHADER);
			glShaderSource(vertShader, 1, &vertSrc, NULL);
			glCompileShader(vertShader);

			// Compile fragment shader
			GLuint fragShader = glCreateShader(GL_FRAGMENT_SHADER);
			glShaderSource(fragShader, 1, &fragSrc, NULL);
			glCompileShader(fragShader);

			// Link shaders
			frameCopyProgram = glCreateProgram();
			glAttachShader(frameCopyProgram, vertShader);
			glAttachShader(frameCopyProgram, fragShader);
			glLinkProgram(frameCopyProgram);

			// Delete shaders
			glDeleteShader(vertShader);
			glDeleteShader(fragShader);

			// Create a quad
			float quadVertices[] = {
				-1.0f,  1.0f,
				-1.0f, -1.0f,
				 1.0f, -1.0f,
				 1.0f, -1.0f,
				 1.0f,  1.0f,
				-1.0f,  1.0f
			};

			glGenVertexArrays(1, &quadVAO);
			GLuint quadVBO;
			glGenBuffers(1, &quadVBO);

			glBindVertexArray(quadVAO);
			glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
			glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);
			glEnableVertexAttribArray(0);
			glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);

			shaderInitialized = true;
		}

		// Draw the texture
		glUseProgram(frameCopyProgram);
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, frameTexture);
		glUniform1i(glGetUniformLocation(frameCopyProgram, "screenTexture"), 0);

		glBindVertexArray(quadVAO);
		glDrawArrays(GL_TRIANGLES, 0, 6);

		// Reset state
		glBindVertexArray(0);
		glUseProgram(0);
		glEnable(GL_DEPTH_TEST);
	}


	GLuint framebufferCache = 0;
	GLuint frameTexture = 0;
	GLuint frameDepthTexture = 0;
	bool framebufferInitialized = false;

	bool wireframeMode;
	bool showOctreeWire;
	RenderMode currentMode;
	RenderMode oldMode;
	std::vector<RadiationPoint> tmp;

	Camera camera;
	float aspect;
	bool rightMouseDown;
	bool leftMouseDown;
	double mouseOldX, mouseOldY;

	glm::vec3 buildingCenter;

	float peelPlaneZ;
	int renderModeToggle;

	// Store the current window size for re-render calls
	int lastWindowWidth = 1300;
	int lastWindowHeight = 1300;

	// Some geometry for CPU-based modes:
	CPU_Geometry cpuGeom;
	GPU_Geometry gpuGeom;

	VolumeRaycastRenderer* raycastRendererPtr = nullptr;  // Pointer to the volume raycast

	// Possibly store a "carve radius"
	float carveRadius = 2.0f;

	glm::mat4 lastViewMatrix;
	bool cameraChanged;
	bool updateFrustumRequested;
	bool octreeSkipEnabled = false;  
	bool mipmappedSkippingEnabled = true;

	bool forceRegenerateDC = false;

};

int main() {
	// Initialize GLFW
	if (!glfwInit()) {
		std::cerr << "Failed to initialize GLFW.\n";
		return -1;
	}

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

#ifdef _DEBUG
	glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GL_TRUE);
#endif

	// Create window
	Window window(1300, 1300, "Octree + DC + Volume Raycast + Compute BVH");
	if (!window.getWindow()) {
		std::cerr << "Failed to create GLFW window. Check if your GPU supports OpenGL 4.3\n";
		glfwTerminate();
		return -1;
	}

	// After window creation, verify OpenGL version
	const GLubyte* renderer = glGetString(GL_RENDERER);
	const GLubyte* version = glGetString(GL_VERSION);
	GLint major, minor;
	glGetIntegerv(GL_MAJOR_VERSION, &major);
	glGetIntegerv(GL_MINOR_VERSION, &minor);

	std::cout << "Renderer: " << renderer << "\n";
	std::cout << "OpenGL version supported: " << version << "\n";
	std::cout << "OpenGL version requested: " << major << "." << minor << "\n";

	if (major < 4 || (major == 4 && minor < 3)) {
		std::cerr << "Your system only supports OpenGL " << major << "." << minor << "\n";
		std::cerr << "Please update your graphics drivers to get OpenGL 4.3+ support\n";
		glfwTerminate();
		return -1;
	}

	// CPU/GPU geometry
	CPU_Geometry cpuGeom;
	GPU_Geometry gpuGeom;

	auto app = std::make_shared<Assignment4>();
	window.setCallbacks(app);

	bool useGDB = true;
	int dim = 256;
	float voxelSize = 2.0f;
	std::string cacheFilename = "sceneCache.bin";

	VoxelGrid grid;
	// Load or generate voxel data
	if (useGDB) {
		// Attempt to load from your own caching mechanism, or fallback
		if (!loadVoxelGrid(cacheFilename, grid)) {
			grid = loadCSVDataIntoVoxelGrid("./DT/DTVerts.csv", "./DT/DTFaces.csv", voxelSize);
			if (grid.data.empty()) {
				std::cerr << "Voxel grid is empty. Exiting." << std::endl;
				return -1;
			}

			// Re-center after load, *before* building the octree
			recenterFilledVoxels(grid);

			if (grid.data.empty()) {
				std::cerr << "Voxel grid is empty. Exiting.\n";
				return -1;
			}

			// Save the processed grid to a cache file.
			saveVoxelGrid(cacheFilename, grid);
		}
	}
	else {
		// fallback multi-shell sphere
		auto vol = generateTestVolume(dim, dim, dim);
		grid.dimX = dim; grid.dimY = dim; grid.dimZ = dim;
		grid.minX = -0.5f; grid.minY = -0.5f; grid.minZ = -0.5f;
		grid.voxelSize = 1.f / dim;
		grid.data.resize(dim * dim * dim, VoxelState::EMPTY);

		for (int z = 0; z < dim; z++) {
			for (int y = 0; y < dim; y++) {
				for (int x = 0; x < dim; x++) {
					int idx = x + y * dim + z * (dim * dim);
					if (vol[idx] > 0.0f) {
						grid.data[idx] = VoxelState::FILLED;
					}
					else {
						grid.data[idx] = VoxelState::EMPTY;
					}
				}
			}
		}
	}

	// Recenter the voxel grid if necessary
	recenterFilledVoxels(grid);

	// Build the octree
	OctreeNode* root = createOctreeFromVoxelGrid(grid);

	// Find approximate center
	{
		float minX = 1e30f, minY = 1e30f, minZ = 1e30f;
		float maxX = -1e30f, maxY = -1e30f, maxZ = -1e30f;
		for (int z = 0; z < grid.dimZ; z++) {
			for (int y = 0; y < grid.dimY; y++) {
				for (int x = 0; x < grid.dimX; x++) {
					if (grid.data[grid.index(x, y, z)] == VoxelState::FILLED) {
						float wx = grid.minX + (x + 0.5f) * grid.voxelSize;
						float wy = grid.minY + (y + 0.5f) * grid.voxelSize;
						float wz = grid.minZ + (z + 0.5f) * grid.voxelSize;
						if (wx < minX) minX = wx;
						if (wy < minY) minY = wy;
						if (wz < minZ) minZ = wz;
						if (wx > maxX) maxX = wx;
						if (wy > maxY) maxY = wy;
						if (wz > maxZ) maxZ = wz;
					}
				}
			}
		}
		float cx = 0.5f * (minX + maxX);
		float cy = 0.5f * (minY + maxY);
		float cz = 0.5f * (minZ + maxZ);
		app->buildingCenter = glm::vec3(cx, cy, cz);
		std::cout << "Building center: " << cx << ", " << cy << ", " << cz << "\n";
	}

	// Prepare other renderers
	MarchingCubesRenderer      mcRenderer;
	AdaptiveDualContouringRenderer dcRenderer;
	// Initialize the thread pool
	dcRenderer.initThreadPool();
	dcRenderer.m_useComputeShader = false;  // Enable compute shader acceleration

	VoxelCubeRenderer          blockRenderer;
	static VolumeRaycastRenderer pointRadRenderer;
	std::cout << "Before createComputeShader()" << std::endl;
	pointRadRenderer.init(grid);
	pointRadRenderer.setOctreeRoot(root);
	pointRadRenderer.m_enableOctreeSkip = app->octreeSkipEnabled;     
	pointRadRenderer.m_useMipMappedSkipping = true;
	app->raycastRendererPtr = &pointRadRenderer;
	pointRadRenderer.setCamera(&app->camera);
	pointRadRenderer.updateFrustumCulling(app->aspect);
	std::cout << "After createComputeShader()" << std::endl;

	std::vector<MCTriangle> triCache;
	RayTracerBVH bvhRayTracer;
	bvhRayTracer.ensureComputeInitialized();

	// Initialize the BVH Ray tracer (GPU)
	bvhRayTracer.setOctree(root, grid);

	// Wireframe for octree
	CPU_Geometry cpuWire;
	GPU_Geometry gpuWireGeom;
	std::vector<glm::vec3> wireLines;
	if (!wireLines.empty()) {
		cpuWire.verts = wireLines;
		gpuWireGeom.bind();
		gpuWireGeom.setVerts(cpuWire.verts);
	}

	// Shader for wireframe/other geometry
	ShaderProgram shader("453-skeleton/shaders/test.vert", "453-skeleton/shaders/test.frag");

	RenderMode oldMode = app->currentMode;

	using clock_t = std::chrono::high_resolution_clock;
	auto lastTime = clock_t::now();
	int frameCount = 0;
	int rayCastCounter = 0;
	int rayTraceCounter = 0;

	std::cout << "Entering main loop...\n";

	while (!window.shouldClose()) {
		glfwPollEvents();

		// Always clear both buffers at the start
		glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		// Handle mode changes first
		if (app->currentMode != app->oldMode) {
			// Clear all geometry caches
			triCache.clear();
			app->cpuGeom.verts.clear();
			app->cpuGeom.normals.clear();
			app->cpuGeom.cols.clear();

			// Reset OpenGL state
			glDisable(GL_BLEND);
			glDisable(GL_DEPTH_TEST);
			glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

			app->oldMode = app->currentMode;
		}

		// Set appropriate global state for each mode
		if (app->currentMode == RenderMode::VolumeRaycast) {
			glDisable(GL_DEPTH_TEST);
			glEnable(GL_BLEND);
			glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		}
		else {
			glDisable(GL_BLEND);
			glEnable(GL_DEPTH_TEST);
		}

		if (app->currentMode == RenderMode::VolumeRaycast) {
			// Apply frustum update if requested
			if (app->updateFrustumRequested && app->cameraChanged) {
				pointRadRenderer.setUpdateFrustumRequested(true);
				app->updateFrustumRequested = false;
				app->cameraChanged = false;
			}

			// Initialize framebuffer if needed - only once
			if (!app->framebufferInitialized) {
				app->initializeFramebufferCache();
			}

			// Only render to the framebuffer on select frames
			if (rayCastCounter % 7 == 0) {
				// Render to framebuffer
				glBindFramebuffer(GL_FRAMEBUFFER, app->framebufferCache);

				// Standard setup
				glDisable(GL_DEPTH_TEST);
				glEnable(GL_BLEND);
				glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

				// Clear the framebuffer
				glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
				glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

				// Render to the framebuffer
				pointRadRenderer.drawRaycast(app->aspect);

				// Restore state
				glDisable(GL_BLEND);
				glEnable(GL_DEPTH_TEST);
				glBindFramebuffer(GL_FRAMEBUFFER, 0);
			}

			// Always draw the cached frame to the screen
			app->drawCachedFrame();

			rayCastCounter++; // Fixed the syntax error here
		}
		else if (app->currentMode == RenderMode::MarchingCubes) {
			if ((triCache.empty() || app->cameraChanged) && app->updateFrustumRequested) {
				app->cameraChanged = false; // Reset the flag
				app->updateFrustumRequested = false;

				triCache = renderOctree(root, grid, mcRenderer, app->camera, app->aspect);
				cpuGeom.verts.clear();
				cpuGeom.normals.clear();
				cpuGeom.cols.clear();
				for (auto& t : triCache) {
					for (int i = 0; i < 3; i++) {
						cpuGeom.verts.push_back(t.v[i]);
						cpuGeom.normals.push_back(t.normal[i]);
						cpuGeom.cols.push_back({ 0.8f, 0.8f, 0.8f });
					}
				}
				gpuGeom.bind();
				gpuGeom.setVerts(cpuGeom.verts);
				gpuGeom.setNormals(cpuGeom.normals);
				gpuGeom.setCols(cpuGeom.cols);
			}
			shader.use();
			app->viewPipeline(shader);

			if (app->wireframeMode) {
				glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
			}
			gpuGeom.bind();
			glDrawArrays(GL_TRIANGLES, 0, (GLsizei)cpuGeom.verts.size());
			glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

		}
		else if (app->currentMode == RenderMode::DualContouring) {
			if (app->updateFrustumRequested || app->forceRegenerateDC || triCache.empty()) {
				app->cameraChanged = false; // Reset the flag
				app->updateFrustumRequested = false;

				// Force clear the triangle cache to ensure the renderer is called
				triCache.clear();

				// Force rendering with parallel processing
				auto start = std::chrono::high_resolution_clock::now();

				// Pass the forceRegenerateDC flag to renderOctree
				triCache = renderOctree(root, grid, dcRenderer, app->camera, app->aspect, 50.0f, nullptr, app->forceRegenerateDC);

				auto end = std::chrono::high_resolution_clock::now();
				auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
				std::cout << "DC rendering took: " << duration << " ms" << std::endl;

				cpuGeom.verts.clear();
				cpuGeom.normals.clear();
				cpuGeom.cols.clear();
				for (auto& t : triCache) {
					for (int i = 0; i < 3; i++) {
						cpuGeom.verts.push_back(t.v[i]);
						cpuGeom.normals.push_back(t.normal[i]);
						cpuGeom.cols.push_back(glm::vec3(0.7f, 0.7f, 0.9f)); // Slightly different color for DC
					}
				}
				gpuGeom.bind();
				gpuGeom.setVerts(cpuGeom.verts);
				gpuGeom.setNormals(cpuGeom.normals);
				gpuGeom.setCols(cpuGeom.cols);

				// Reset the flag after rendering
				if (app->forceRegenerateDC) {
					app->forceRegenerateDC = false;
				}
			}
			shader.use();
			app->viewPipeline(shader);

			if (app->wireframeMode) {
				glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
			}
			gpuGeom.bind();
			glDrawArrays(GL_TRIANGLES, 0, (GLsizei)cpuGeom.verts.size());
			glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
		}
		else if (app->currentMode == RenderMode::VoxelBlocks) {
			if ((triCache.empty() || app->cameraChanged) && app->updateFrustumRequested) {
				app->cameraChanged = false; // Reset the flag
				app->updateFrustumRequested = false;

				triCache = renderOctree(root, grid, blockRenderer, app->camera, app->aspect);
				cpuGeom.verts.clear();
				cpuGeom.normals.clear();
				cpuGeom.cols.clear();
				for (auto& t : triCache) {
					for (int i = 0; i < 3; i++) {
						cpuGeom.verts.push_back(t.v[i]);
						cpuGeom.normals.push_back(t.normal[i]);
						cpuGeom.cols.push_back(glm::vec3(0.6f, 0.6f, 0.6f));
					}
				}
				gpuGeom.bind();
				gpuGeom.setVerts(cpuGeom.verts);
				gpuGeom.setNormals(cpuGeom.normals);
				gpuGeom.setCols(cpuGeom.cols);
			}
			shader.use();
			app->viewPipeline(shader);

			if (app->wireframeMode) {
				glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
			}
			gpuGeom.bind();
			glDrawArrays(GL_TRIANGLES, 0, (GLsizei)cpuGeom.verts.size());
			glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

		}
		else if (app->currentMode == RenderMode::BVHRayTrace) {
			// Initialize framebuffer if needed
			app->initializeFramebufferCache();

			// Only render to the framebuffer on select frames
			if (rayTraceCounter % 6 == 0 || (app->updateFrustumRequested && app->cameraChanged)) {
				// Render to framebuffer
				glBindFramebuffer(GL_FRAMEBUFFER, app->framebufferCache);

				// Clear the framebuffer
				glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
				glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

				// Render with frustum culling
				bvhRayTracer.renderSceneComputeWithCulling(
					app->camera,
					app->lastWindowWidth,
					app->lastWindowHeight,
					app->aspect,
					/*fov=*/45.0f,
					(app->updateFrustumRequested && app->cameraChanged));

				// Reset flags after they've been used
				if (app->updateFrustumRequested && app->cameraChanged) {
					app->updateFrustumRequested = false;
					app->cameraChanged = false;
				}

				// Restore default framebuffer
				glBindFramebuffer(GL_FRAMEBUFFER, 0);
			}

			// Always draw the cached frame to the screen
			app->drawCachedFrame();

			rayTraceCounter++; // Increment the counter
		}
		// Possibly draw octree wireframe
		if (app->showOctreeWire) {
			if (wireLines.empty() || app->updateFrustumRequested) {

				// Create frustum for culling
				glm::mat4 V = app->camera.getView();
				glm::mat4 P = glm::perspective(glm::radians(45.f), app->aspect, 0.01f, 5000.f);
				Frustum frustum(P * V);

				// Clear previous wireframe
				wireLines.clear();

				// Generate wireframe with frustum culling
				generateOctreeWireframe(grid, root, 0, 0, 0, grid.dimX, wireLines, frustum);

				if (!wireLines.empty()) {
					cpuWire.verts = wireLines;
					gpuWireGeom.bind();
					gpuWireGeom.setVerts(cpuWire.verts);
				}
			}

			shader.use();
			app->viewPipeline(shader);
			GLint uColor = glGetUniformLocation(shader, "overrideColor");
			glUniform3f(uColor, 1.f, 0.f, 0.f); // red
			gpuWireGeom.bind();
			glDrawArrays(GL_LINES, 0, (GLsizei)cpuWire.verts.size());
			glUniform3f(uColor, 1.f, 1.f, 1.f); // reset
		}

		app->cameraChanged = app->cameraChanged || app->hasCameraChanged();
		// Swap buffers
		window.swapBuffers();

		// FPS counter
		frameCount++;
		auto now = clock_t::now();
		double dt = std::chrono::duration<double>(now - lastTime).count();
		if (dt >= 1.0) {
			double fps = (double)frameCount / dt;
			std::cout << "FPS: " << fps << "  Mode: ";
			switch (app->currentMode) {
			case RenderMode::MarchingCubes:    std::cout << "MC\n"; break;
			case RenderMode::DualContouring:   std::cout << "DC\n"; break;
			case RenderMode::VoxelBlocks:      std::cout << "Blocks\n"; break;
			case RenderMode::BVHRayTrace:      std::cout << "RayTrace\n"; break;
			case RenderMode::VolumeRaycast:    std::cout << "Volume\n"; break;
			}
			frameCount = 0;
			lastTime = now;
		}
	}
	std::cout << "Exiting main loop...\n";

	glfwTerminate();
	return 0;
}
