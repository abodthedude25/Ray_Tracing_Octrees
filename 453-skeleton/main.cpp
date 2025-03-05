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

// A helper that traverses an octree and collects triangles from a given renderer
std::vector<MCTriangle> renderOctree(const OctreeNode* root,
	const VoxelGrid& grid,
	Renderer& renderer) {
	std::vector<MCTriangle> result;
	if (!root) return result;

	std::function<void(const OctreeNode*)> traverse = [&](const OctreeNode* node) {
		if (!node) return;
		if (node->isLeaf) {
			auto tris = renderer.render(node, grid, node->x, node->y, node->z, node->size);
			result.insert(result.end(), tris.begin(), tris.end());
		}
		else {
			for (auto child : node->children) {
				traverse(child);
			}
		}
		};
	traverse(root);
	return result;
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
	std::vector<glm::vec3>& lines) {
	if (!node) return;
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
			lines);
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
		currentMode(RenderMode::VolumeRaycast),
		oldMode(RenderMode::VolumeRaycast),
		camera(glm::radians(90.0f), glm::radians(0.f), 500.f),
		aspect(1.f),
		rightMouseDown(false),
		leftMouseDown(false),
		mouseOldX(0.0), mouseOldY(0.0),
		buildingCenter(0.f),
		peelPlaneZ(0.0f),
		renderModeToggle(0)
	{
		camera.pan(0.f, 100.f);
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
			else if (key == GLFW_KEY_R) {
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
		}
	}

	// Mouse
	void cursorPosCallback(double xpos, double ypos) override {
		if (rightMouseDown) {
			camera.incrementTheta(ypos - mouseOldY);
			camera.incrementPhi(xpos - mouseOldX);
		}
		if (leftMouseDown) {
			camera.pan(xpos - mouseOldX, ypos - mouseOldY);
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
							tmp.push_back(rp);

							std::cout << "Step 4: Updating splat points" << std::endl;
							raycastRendererPtr->updateSplatPoints(tmp);

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
	}

	void windowSizeCallback(int width, int height) override {
		if (height < 1) height = 1;
		aspect = float(width) / float(height);
		lastWindowWidth = width;
		lastWindowHeight = height;

		// Tell OpenGL to match the new window size:
		glViewport(0, 0, width, height);
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
	int lastWindowWidth = 800;
	int lastWindowHeight = 800;

	// Some geometry for CPU-based modes:
	CPU_Geometry cpuGeom;
	GPU_Geometry gpuGeom;

	VolumeRaycastRenderer* raycastRendererPtr = nullptr;  // Pointer to the volume raycast

	// Possibly store a "carve radius"
	float carveRadius = 2.0f;
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
	Window window(800, 800, "Octree + DC + Volume Raycast + Compute BVH");
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
	float voxelSize = 0.025f;
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
	VoxelCubeRenderer          blockRenderer;
	static VolumeRaycastRenderer pointRadRenderer;
	std::cout << "Before createComputeShader()" << std::endl;
	pointRadRenderer.init(grid);
	pointRadRenderer.setOctreeRoot(root);
	pointRadRenderer.m_enableOctreeSkip = true;
	app->raycastRendererPtr = &pointRadRenderer;
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
	generateOctreeWireframe(grid, root, 0, 0, 0, grid.dimX, wireLines);
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
			// REPLACED the old volRenderer code with new pointRad usage
			glDisable(GL_DEPTH_TEST);
			glEnable(GL_BLEND);
			glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

			// Clear
			glClearColor(0.2f, 0.f, 0.f, 1.f);
			glClear(GL_COLOR_BUFFER_BIT);

			// If we want to keep camera updates:
			pointRadRenderer.setCamera(&app->camera);
			pointRadRenderer.drawRaycast(app->aspect);

			glDisable(GL_BLEND);
			glEnable(GL_DEPTH_TEST);
		}
		else if (app->currentMode == RenderMode::MarchingCubes) {
			if (triCache.empty()) {
				triCache = renderOctree(root, grid, mcRenderer);
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
			if (triCache.empty()) {
				triCache = renderOctree(root, grid, dcRenderer);
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
		else if (app->currentMode == RenderMode::VoxelBlocks) {
			if (triCache.empty()) {
				triCache = renderOctree(root, grid, blockRenderer);
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
			// --- GPU BVH Ray Tracing via compute shader ---
			bvhRayTracer.renderSceneCompute(app->camera,
				app->lastWindowWidth,
				app->lastWindowHeight,
				app->aspect,
				/*fov=*/45.0f);
		}

		// Possibly draw octree wireframe
		if (app->showOctreeWire) {
			shader.use();
			app->viewPipeline(shader);
			GLint uColor = glGetUniformLocation(shader, "overrideColor");
			glUniform3f(uColor, 1.f, 0.f, 0.f); // red
			gpuWireGeom.bind();
			glDrawArrays(GL_LINES, 0, (GLsizei)cpuWire.verts.size());
			glUniform3f(uColor, 1.f, 1.f, 1.f); // reset
		}

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
