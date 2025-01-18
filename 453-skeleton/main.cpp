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

static std::vector<float> generateTestVolume(int dimX, int dimY, int dimZ) {
	// Sphere
	printf("Generating test volume: Sphere\n");
	std::vector<float> volume(dimX * dimY * dimZ, 0.f);
	float cx = 0.5f * (dimX - 1);
	float cy = 0.5f * (dimY - 1);
	float cz = 0.5f * (dimZ - 1);
	float radius = 0.3f * std::min({ float(dimX), float(dimY), float(dimZ) });
	for (int z = 0; z < dimZ; z++) {
		for (int y = 0; y < dimY; y++) {
			for (int x = 0; x < dimX; x++) {
				float dx = x - cx, dy = y - cy, dz = z - cz;
				float dist = std::sqrt(dx * dx + dy * dy + dz * dz);
				int idx = x + y * dimX + z * (dimX * dimY);
				volume[idx] = dist - radius; // negative => inside sphere
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
		// Means no voxels were filled
		std::cout << "(Recenter) No filled voxels. Skipping.\n";
		return;
	}

	float centerX = 0.5f * (filledMinX + filledMaxX);
	float centerY = 0.5f * (filledMinY + filledMaxY);
	float centerZ = 0.5f * (filledMinZ + filledMaxZ);

	// Shift min coords so the object's center is at world origin
	grid.minX -= centerX;
	grid.minY -= centerY;
	grid.minZ -= centerZ;

	std::cout << "Recentered grid around origin. Moved by ("
		<< centerX << ", " << centerY << ", " << centerZ << ")\n";
}

// ========== A small function to gather octree wireframe lines ========== //
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

// We define our two modes
enum class RenderMode {
	MarchingCubes,
	DualContouring,
	VoxelBlocks
};
struct Assignment4 : public CallbackInterface {
	Assignment4()
		: wireframeMode(false), showOctreeWire(false),
		currentMode(RenderMode::MarchingCubes),
		aspect(1.f), rightMouseDown(false),
		mouseOldX(0), mouseOldY(0),
		camera(glm::radians(45.f), glm::radians(45.f), 3.f)
	{}

	void keyCallback(int key, int scancode, int action, int mods) override {
		if (action == GLFW_PRESS) {
			if (key == GLFW_KEY_W) {
				wireframeMode = !wireframeMode;
			}
			if (key == GLFW_KEY_S) {
				showOctreeWire = !showOctreeWire;
			}
			if (key == GLFW_KEY_R) {
				// Cycle through the three modes
				if (currentMode == RenderMode::MarchingCubes) {
					currentMode = RenderMode::VoxelBlocks;
					std::cout << "Switched to Dual Contouring\n";
				}
				else if (currentMode == RenderMode::VoxelBlocks) {
					currentMode = RenderMode::DualContouring;
					std::cout << "Switched to Voxel Blocks\n";
				}
				else {
					currentMode = RenderMode::MarchingCubes;
					std::cout << "Switched to Marching Cubes\n";
				}
			}
		}
	}
	void cursorPosCallback(double xpos, double ypos) override {
		if (rightMouseDown) {
			camera.incrementTheta(ypos - mouseOldY);
			camera.incrementPhi(xpos - mouseOldX);
		}
		mouseOldX = xpos;
		mouseOldY = ypos;
	}
	void mouseButtonCallback(int button, int action, int mods) override {
		if (button == GLFW_MOUSE_BUTTON_RIGHT) {
			rightMouseDown = (action == GLFW_PRESS);
		}
	}
	void scrollCallback(double xoffset, double yoffset) override {
		camera.incrementR(yoffset);
	}
	void windowSizeCallback(int width, int height) override {
		CallbackInterface::windowSizeCallback(width, height);
		aspect = (float)width / (float)height;
	}
	void viewPipeline(ShaderProgram& sp) {
		glm::mat4 M(1.f);
		glm::mat4 V = camera.getView();
		glm::mat4 P = glm::perspective(glm::radians(45.f), aspect, 0.01f, 1000.f);
		GLint uM = glGetUniformLocation(sp, "M");
		glUniformMatrix4fv(uM, 1, GL_FALSE, &M[0][0]);
		GLint uV = glGetUniformLocation(sp, "V");
		glUniformMatrix4fv(uV, 1, GL_FALSE, &V[0][0]);
		GLint uP = glGetUniformLocation(sp, "P");
		glUniformMatrix4fv(uP, 1, GL_FALSE, &P[0][0]);
	}

	bool wireframeMode;
	bool showOctreeWire;
	RenderMode currentMode;
	Camera camera;
	float aspect;
	bool rightMouseDown;
	double mouseOldX, mouseOldY;
};

int main() {
	glfwInit();
	Window window(800, 800, "Octree + DC Without Holes");
	auto app = std::make_shared<Assignment4>();
	window.setCallbacks(app);

	bool useGDB = false;
	std::string gdbPath = "./gdb_folder/Buildings_3D.gdb";

	VoxelGrid grid;
	if (useGDB) {
		// Possibly an empty vector => first building
		std::vector<int> targetIDs = { };
		grid = loadBuildingsFromGDB(gdbPath, 1.0f, targetIDs);

		// Re-center after load, *before* building the octree
		recenterFilledVoxels(grid);

		if (grid.data.empty()) {
			std::cerr << "Voxel grid is empty. Exiting.\n";
			return -1;
		}
	}
	else {
		// fallback sphere
		int dim = 64;
		auto vol = generateTestVolume(dim, dim, dim);
		grid.dimX = dim; grid.dimY = dim; grid.dimZ = dim;
		grid.minX = -0.5f; grid.minY = -0.5f; grid.minZ = -0.5f;
		grid.voxelSize = 1.f / dim;
		grid.data.resize(dim * dim * dim, VoxelState::EMPTY);

		// *** ADDED an epsilon to ensure borderline volumes classify consistently
		const float EPS = 1e-7f;
		for (int z = 0; z < dim; z++) {
			for (int y = 0; y < dim; y++) {
				for (int x = 0; x < dim; x++) {
					int idx = x + y * dim + z * (dim * dim);
					if (vol[idx] < -EPS) {
						grid.data[idx] = VoxelState::FILLED;
					}
					else {
						grid.data[idx] = VoxelState::EMPTY;
					}
				}
			}
		}
	}

	// (2) Build the octree
	OctreeNode* root = createOctreeFromVoxelGrid(grid);

	// (3) Prepare the three "renderers"
	MarchingCubesRenderer mcRenderer;
	AdaptiveDualContouringRenderer dcRenderer;
	VoxelCubeRenderer blockRenderer;

	// (4) We store geometry in GPU buffers for MC and DC
	CPU_Geometry cpuGeom;
	GPU_Geometry gpuGeom;
	// For wireframe lines
	CPU_Geometry cpuWire;
	GPU_Geometry gpuWireGeom;
	std::vector<glm::vec3> wireLines;
	generateOctreeWireframe(grid, root, 0, 0, 0, grid.dimX, wireLines);
	if (!wireLines.empty()) {
		cpuWire.verts = wireLines;
		gpuWireGeom.bind();
		gpuWireGeom.setVerts(cpuWire.verts);
	}

	ShaderProgram shader("453-skeleton/shaders/test.vert", "453-skeleton/shaders/test.frag");

	// Triangle cache for MC/DC
	std::vector<MCTriangle> triCache;
	RenderMode oldMode = app->currentMode;

	// Basic game loop
	using clock_t = std::chrono::high_resolution_clock;
	auto lastTime = clock_t::now();
	int frameCount = 0;

	while (!window.shouldClose()) {
		glfwPollEvents();

		glClearColor(01.0f, 1.0f, 1.0f, 1);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glEnable(GL_DEPTH_TEST);

		shader.use();
		app->viewPipeline(shader);

		// If user switched mode, clear the caches
		if (app->currentMode != oldMode) {
			triCache.clear();
			oldMode = app->currentMode;
		}

		if (app->currentMode == RenderMode::MarchingCubes) {
			// (a) build triangle mesh once if needed
			if (triCache.empty()) {
				triCache = renderOctree(root, grid, mcRenderer);
				cpuGeom.verts.clear();
				cpuGeom.normals.clear();
				cpuGeom.cols.clear();
				for (auto& t : triCache) {
					for (int i = 0; i < 3; i++) {
						cpuGeom.verts.push_back(t.v[i]);
						cpuGeom.normals.push_back(t.normal[i]);
						cpuGeom.cols.push_back({ 0.8f,0.8f,0.8f });
					}
				}
				gpuGeom.bind();
				gpuGeom.setVerts(cpuGeom.verts);
				gpuGeom.setNormals(cpuGeom.normals);
				gpuGeom.setCols(cpuGeom.cols);
			}
			// (b) draw the mesh
			if (app->wireframeMode) {
				glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
			}
			gpuGeom.bind();
			glDrawArrays(GL_TRIANGLES, 0, (GLsizei)cpuGeom.verts.size());
			glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
		}
		else if (app->currentMode == RenderMode::DualContouring) {
			// (a) build DC triangle mesh once if needed
			if (triCache.empty()) {
				triCache = renderOctree(root, grid, dcRenderer);
				cpuGeom.verts.clear();
				cpuGeom.normals.clear();
				cpuGeom.cols.clear();
				for (auto& t : triCache) {
					for (int i = 0; i < 3; i++) {
						cpuGeom.verts.push_back(t.v[i]);
						cpuGeom.normals.push_back(t.normal[i]);
						cpuGeom.cols.push_back({ 0.8f,0.8f,0.8f });
					}
				}
				gpuGeom.bind();
				gpuGeom.setVerts(cpuGeom.verts);
				gpuGeom.setNormals(cpuGeom.normals);
				gpuGeom.setCols(cpuGeom.cols);
			}
			// (b) draw
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
				// build GPU geometry from triCache
				cpuGeom.verts.clear();
				cpuGeom.normals.clear();
				cpuGeom.cols.clear();
				for (auto& t : triCache) {
					for (int i = 0; i < 3; i++) {
						cpuGeom.verts.push_back(t.v[i]);
						cpuGeom.normals.push_back(t.normal[i]);
						// color the blocks grey or something
						cpuGeom.cols.push_back(glm::vec3(0.6f, 0.6f, 0.6f));
					}
				}
				gpuGeom.bind();
				gpuGeom.setVerts(cpuGeom.verts);
				gpuGeom.setNormals(cpuGeom.normals);
				gpuGeom.setCols(cpuGeom.cols);
			}
			// draw
			if (app->wireframeMode) {
				glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
			}
			gpuGeom.bind();
			glDrawArrays(GL_TRIANGLES, 0, (GLsizei)cpuGeom.verts.size());
			glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
		}


		// Possibly draw wire
		if (app->showOctreeWire) {
			GLint uColor = glGetUniformLocation(shader, "overrideColor");
			glUniform3f(uColor, 1.f, 0.f, 0.f);
			gpuWireGeom.bind();
			glDrawArrays(GL_LINES, 0, (GLsizei)cpuWire.verts.size());
			glUniform3f(uColor, 1.f, 1.f, 1.f);
		}

		window.swapBuffers();

		// FPS
		frameCount++;
		auto now = clock_t::now();
		double dt = std::chrono::duration<double>(now - lastTime).count();
		if (dt >= 1.0) {
			double fps = (double)frameCount / dt;
			std::cout << "FPS: " << fps << "  Mode: ";
			if (app->currentMode == RenderMode::MarchingCubes) std::cout << "MC\n";
			else if (app->currentMode == RenderMode::DualContouring) std::cout << "DC\n";
			else std::cout << "RayTracer\n";
			frameCount = 0;
			lastTime = now;
		}
	}

	glfwTerminate();
	return 0;
}
