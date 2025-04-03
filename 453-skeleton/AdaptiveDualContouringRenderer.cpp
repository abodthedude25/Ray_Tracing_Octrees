#include "AdaptiveDualContouringRenderer.h"
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/norm.hpp>
#include <limits>
#include <algorithm>
#include <cmath>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <chrono>
#include <fstream>
#include <sstream>
#include <string>
#include <iostream>

// Forward declaration of global octree map
extern std::unordered_map<long long, OctreeNode*> g_octreeMap;

extern std::unordered_map<long long, OctreeNode*> g_octreeMap;

// A small helper to read a file into a std::string (for the compute shader).
static inline std::string loadShaderFromFile(const std::string& filePath) {
	std::ifstream file(filePath);
	if (!file.is_open()) {
		std::cerr << "Failed to open shader file: " << filePath << std::endl;
		return "";
	}
	std::stringstream buffer;
	buffer << file.rdbuf();
	return buffer.str();
}

// Simple CPU isSolid to check if a voxel is filled.
static inline bool cpuIsSolid(const VoxelGrid& grid, int x, int y, int z) {
	if (x < 0 || y < 0 || z < 0 ||
		x >= grid.dimX || y >= grid.dimY || z >= grid.dimZ) {
		return false; // out of bounds is empty
	}
	int idx = x + y * grid.dimX + z * (grid.dimX * grid.dimY);
	return (grid.data[idx] == VoxelState::FILLED);
}

// QEFSolver implementation - simplified for better performance
QEFSolver::QEFSolver() : ata(0.0f), atb(0.0f), pointSum(0.0f), numPoints(0) {
}

void QEFSolver::addPoint(const glm::vec3& point, const glm::vec3& normal) {
	// Normalize the normal to ensure consistent weighting
	glm::vec3 n = glm::normalize(normal);

	// Add normal outer product to AtA
	ata[0][0] += n.x * n.x;
	ata[0][1] += n.x * n.y;
	ata[0][2] += n.x * n.z;
	ata[1][0] += n.y * n.x;
	ata[1][1] += n.y * n.y;
	ata[1][2] += n.y * n.z;
	ata[2][0] += n.z * n.x;
	ata[2][1] += n.z * n.y;
	ata[2][2] += n.z * n.z;

	// Compute d = -dot(n, point)
	float d = -glm::dot(n, point);

	// Add to Atb
	atb.x += n.x * d;
	atb.y += n.y * d;
	atb.z += n.z * d;

	// Keep track of point average
	pointSum += point;
	numPoints++;
}

void QEFSolver::clear() {
	ata = glm::mat3(0.0f);
	atb = glm::vec3(0.0f);
	pointSum = glm::vec3(0.0f);
	numPoints = 0;
}

glm::vec3 QEFSolver::solve(const glm::vec3& cellCenter, float cellSize) {
	// Calculate masspoint as fallback
	glm::vec3 masspoint = (numPoints > 0) ? pointSum / static_cast<float>(numPoints) : cellCenter;

	// If very few points, just return masspoint
	if (numPoints <= 2) {
		return masspoint;
	}

	try {
		// Add regularization for stability
		glm::mat3 regularizedA = ata;
		float reg = 0.3f; // Increased regularization for stability
		regularizedA[0][0] += reg;
		regularizedA[1][1] += reg;
		regularizedA[2][2] += reg;

		// Direct solve with regularization
		glm::mat3 invA;
		bool invertible = true;

		// Check determinant first
		float det = glm::determinant(regularizedA);
		if (std::abs(det) < 1e-10) {
			invertible = false;
		}
		else {
			invA = glm::inverse(regularizedA);

			// Check if inverse is valid (no NaNs or extreme values)
			for (int i = 0; i < 3 && invertible; i++) {
				for (int j = 0; j < 3 && invertible; j++) {
					if (std::isnan(invA[i][j]) || std::isinf(invA[i][j]) ||
						std::abs(invA[i][j]) > 1e6) {
						invertible = false;
					}
				}
			}
		}

		if (invertible) {
			glm::vec3 solution = invA * atb;

			// Validation
			if (!std::isnan(solution.x) && !std::isnan(solution.y) && !std::isnan(solution.z)) {
				// Check if solution is reasonably close to the cell
				float distSq = glm::distance2(solution, masspoint);
				const float MAX_DIST_SQ = cellSize * cellSize;

				if (distSq < MAX_DIST_SQ) {
					// Mix with masspoint for stability
					return glm::mix(solution, masspoint, 0.2f);
				}
			}
		}
	}
	catch (...) {
		// Inverse failed, use masspoint
	}

	return masspoint;
}

glm::vec3 QEFSolver::solveConstrained(const glm::vec3& minBound, const glm::vec3& maxBound) {
	// Calculate cell center
	glm::vec3 cellCenter = (minBound + maxBound) * 0.5f;
	float cellSize = maxBound.x - minBound.x;

	// Find an initial solution
	glm::vec3 solution = solve(cellCenter, cellSize);

	// Clamp to cell bounds
	return glm::clamp(solution, minBound, maxBound);
}


AdaptiveDualContouringRenderer::AdaptiveDualContouringRenderer() {
	// Initialize your thresholds, thread counts, etc.
	m_threadCount = std::max(1, (int)std::thread::hardware_concurrency() - 1);
	if (m_threadCount >= 8) {
		m_detailThreshold = 0.05f;
	}
	else {
		m_detailThreshold = 0.03f;
	}
	m_cacheSize = 10000 * m_threadCount;
}

AdaptiveDualContouringRenderer::~AdaptiveDualContouringRenderer() {
	shutdownThreadPool();
	destroyComputeShader();
	clearCaches();
}

void AdaptiveDualContouringRenderer::setThreadCount(int count) {
	bool wasActive = m_threadPoolActive;
	if (wasActive) {
		shutdownThreadPool();
	}
	m_threadCount = count;
	if (wasActive) {
		initThreadPool();
	}
}

void AdaptiveDualContouringRenderer::clearCaches() {
	std::lock_guard<std::mutex> lock(m_cacheMutex);
	edgeIntersectionCache.clear();
	edgeIntersectionCache.reserve(m_cacheSize * 5);

	dualVertexCache.clear();
	dualVertexCache.reserve(m_cacheSize);
}

// GPU: Compile the single-pass compute shader that only does QEF per voxel
bool AdaptiveDualContouringRenderer::compileComputeShader() {
	std::string compSrc = loadShaderFromFile("single_pass_dc.glsl");
	if (compSrc.empty()) {
		std::cerr << "Empty compute shader source!\n";
		return false;
	}
	const char* srcPtr = compSrc.c_str();
	GLuint computeShader = glCreateShader(GL_COMPUTE_SHADER);
	glShaderSource(computeShader, 1, &srcPtr, nullptr);
	glCompileShader(computeShader);

	GLint success;
	glGetShaderiv(computeShader, GL_COMPILE_STATUS, &success);
	if (!success) {
		GLchar infoLog[512];
		glGetShaderInfoLog(computeShader, sizeof(infoLog), NULL, infoLog);
		std::cerr << "Compute shader compilation failed:\n" << infoLog << std::endl;
		glDeleteShader(computeShader);
		return false;
	}

	// Attach and link
	glAttachShader(m_computeProgram, computeShader);
	glLinkProgram(m_computeProgram);

	glGetProgramiv(m_computeProgram, GL_LINK_STATUS, &success);
	if (!success) {
		GLchar infoLog[512];
		glGetProgramInfoLog(m_computeProgram, sizeof(infoLog), NULL, infoLog);
		std::cerr << "Compute program linking failed:\n" << infoLog << std::endl;
		glDeleteShader(computeShader);
		return false;
	}
	glDeleteShader(computeShader);
	return true;
}

// GPU: Initialize + create buffers
bool AdaptiveDualContouringRenderer::initComputeShader() {
	if (m_computeShaderInitialized) {
		return true;
	}
	try {
		m_computeProgram = glCreateProgram();
		if (!compileComputeShader()) {
			std::cerr << "Failed to compile compute shader.\n";
			glDeleteProgram(m_computeProgram);
			return false;
		}
		// create buffers
		glGenBuffers(1, &m_voxelGridBuffer);
		glGenBuffers(1, &m_hermitePointsBuffer);
		glGenBuffers(1, &m_vertexBuffer);
		glGenBuffers(1, &m_triangleBuffer);

		m_computeShaderInitialized = true;
		return true;
	}
	catch (const std::exception& e) {
		std::cerr << "initComputeShader error: " << e.what() << std::endl;
		destroyComputeShader();
		return false;
	}
}

void AdaptiveDualContouringRenderer::destroyComputeShader() {
	if (m_computeShaderInitialized) {
		glDeleteProgram(m_computeProgram);
		glDeleteBuffers(1, &m_voxelGridBuffer);
		glDeleteBuffers(1, &m_hermitePointsBuffer);
		glDeleteBuffers(1, &m_vertexBuffer);
		glDeleteBuffers(1, &m_triangleBuffer);
		m_computeShaderInitialized = false;
	}
}

// SINGLE-PASS GPU approach: For each voxel, gather edges + solve QEF => store vertex
// Return the entire vertex array to the caller so we can do adjacency on CPU
std::vector<glm::vec4>
AdaptiveDualContouringRenderer::executeComputeShaderSinglePass(const VoxelGrid& grid)
{
	if (!m_computeShaderInitialized) {
		if (!initComputeShader()) {
			throw std::runtime_error("Compute shader init failed.");
		}
	}

	// 1) Create + fill the voxel data buffer
	struct VoxelGridData {
		int dimX, dimY, dimZ;
		float minX, minY, minZ;
		float voxelSize;
	};

	VoxelGridData header{
		grid.dimX, grid.dimY, grid.dimZ,
		grid.minX, grid.minY, grid.minZ,
		grid.voxelSize
	};

	glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_voxelGridBuffer);
	glBufferData(GL_SHADER_STORAGE_BUFFER,
		sizeof(VoxelGridData) + grid.data.size() * sizeof(int),
		NULL, GL_DYNAMIC_DRAW);

	glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(VoxelGridData), &header);
	// copy voxel fill data
	std::vector<int> fillArray(grid.data.size());
	for (size_t i = 0; i < grid.data.size(); i++) {
		fillArray[i] = (grid.data[i] == VoxelState::FILLED) ? 1 : 0;
	}
	glBufferSubData(GL_SHADER_STORAGE_BUFFER,
		sizeof(VoxelGridData),
		fillArray.size() * sizeof(int),
		fillArray.data());
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, m_voxelGridBuffer);

	// 2) Hermite points buffer - large enough
	const int MAX_HERMITE = 10000000;
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_hermitePointsBuffer);
	glBufferData(GL_SHADER_STORAGE_BUFFER,
		MAX_HERMITE * sizeof(HermitePoint),
		NULL, GL_DYNAMIC_DRAW);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, m_hermitePointsBuffer);

	// 3) Vertex buffer: one vertex per voxel
	int totalCells = grid.dimX * grid.dimY * grid.dimZ;
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_vertexBuffer);
	glBufferData(GL_SHADER_STORAGE_BUFFER,
		totalCells * sizeof(glm::vec4),
		NULL,
		GL_DYNAMIC_DRAW);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, m_vertexBuffer);

	// 4) Triangle buffer: might not be used in single-pass approach, but let's keep
	struct TmpTriangleBuffer {
		int count;
		MCTriangle triangles[1];
		// if we aren't generating triangles in the shader, this is basically unused
	} tmp;
	tmp.count = 0;

	glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_triangleBuffer);
	glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(TmpTriangleBuffer),
		&tmp, GL_DYNAMIC_DRAW);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, m_triangleBuffer);

	// 5) Dispatch
	glUseProgram(m_computeProgram);
	glUniform3i(glGetUniformLocation(m_computeProgram, "chunkStart"), 0, 0, 0);
	glUniform1i(glGetUniformLocation(m_computeProgram, "chunkSize"), 1);
	glUniform1f(glGetUniformLocation(m_computeProgram, "cellSizeWorld"), 1.0f); // unused

	// Enough threads for [dimX, dimY, dimZ]
	int groupX = (grid.dimX + 7) / 8;
	int groupY = (grid.dimY + 7) / 8;
	int groupZ = (grid.dimZ + 7) / 8;
	glDispatchCompute(groupX, groupY, groupZ);

	// 6) Wait + read back the entire vertex array
	glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

	// read back the GPU side dual vertices
	std::vector<glm::vec4> allVerts(totalCells);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_vertexBuffer);
	glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
		allVerts.size() * sizeof(glm::vec4),
		allVerts.data());

	return allVerts;
}

// CPU adjacency: For each voxel, if sign difference with (x+1,y,z), (x,y+1,z), etc. 
// form two triangles from the 4 voxel vertices
std::vector<MCTriangle>
AdaptiveDualContouringRenderer::buildTrianglesCPU(const VoxelGrid& grid,
	const std::vector<glm::vec4>& allVerts)
{
	std::vector<MCTriangle> triOut;
	triOut.reserve(grid.dimX * grid.dimY * 2); // rough guess

	auto getIndex = [&](int x, int y, int z) {
		return x + y * grid.dimX + z * (grid.dimX * grid.dimY);
		};
	auto getVertex = [&](int x, int y, int z) -> glm::vec3 {
		int idx = getIndex(x, y, z);
		return glm::vec3(allVerts[idx].x, allVerts[idx].y, allVerts[idx].z);
		};

	// A small helper to add the 2 triangles of a quad
	auto addQuad = [&](const glm::vec3& V00,
		const glm::vec3& V01,
		const glm::vec3& V11,
		const glm::vec3& V10,
		bool invertNormal)
		{
			// tri1
			MCTriangle t1;
			t1.v[0] = V00;
			t1.v[1] = V01;
			t1.v[2] = V11;
			glm::vec3 e1 = V01 - V00;
			glm::vec3 e2 = V11 - V00;
			glm::vec3 n = glm::normalize(glm::cross(e1, e2));
			if (invertNormal) n = -n;
			t1.normal[0] = n;
			t1.normal[1] = n;
			t1.normal[2] = n;
			float area1 = 0.5f * glm::length(glm::cross(e1, e2));
			if (area1 > 1e-6) {
				triOut.push_back(t1);
			}

			// tri2
			MCTriangle t2;
			t2.v[0] = V00;
			t2.v[1] = V11;
			t2.v[2] = V10;
			e1 = V11 - V00;
			e2 = V10 - V00;
			n = glm::normalize(glm::cross(e1, e2));
			if (invertNormal) n = -n;
			t2.normal[0] = n;
			t2.normal[1] = n;
			t2.normal[2] = n;
			float area2 = 0.5f * glm::length(glm::cross(e1, e2));
			if (area2 > 1e-6) {
				triOut.push_back(t2);
			}
		};

	// Now loop over each cell [0..dimX-1, 0..dimY-1, 0..dimZ-1],
	// checking sign differences in +X, +Y, +Z
	for (int z = 0; z < grid.dimZ - 1; z++) {
		for (int y = 0; y < grid.dimY - 1; y++) {
			for (int x = 0; x < grid.dimX - 1; x++) {
				bool cFill = cpuIsSolid(grid, x, y, z);

				// +X face
				{
					bool nFill = cpuIsSolid(grid, x + 1, y, z);
					if (cFill != nFill) {
						// gather the 4 vertices:
						// (x,y,z), (x,y+1,z), (x+1,y,z), (x+1,y+1,z)
						glm::vec3 V00 = getVertex(x, y, z);
						glm::vec3 V01 = getVertex(x, y + 1, z);
						glm::vec3 V10 = getVertex(x + 1, y, z);
						glm::vec3 V11 = getVertex(x + 1, y + 1, z);
						// invert normal if cFill is inside
						addQuad(V00, V01, V11, V10, cFill);
					}
				}

				// +Y face
				{
					bool nFill = cpuIsSolid(grid, x, y + 1, z);
					if (cFill != nFill) {
						// (x,y,z), (x+1,y,z), (x,y+1,z), (x+1,y+1,z)
						glm::vec3 V00 = getVertex(x, y, z);
						glm::vec3 V01 = getVertex(x + 1, y, z);
						glm::vec3 V10 = getVertex(x, y + 1, z);
						glm::vec3 V11 = getVertex(x + 1, y + 1, z);
						addQuad(V00, V01, V11, V10, cFill);
					}
				}

				// +Z face
				{
					bool nFill = cpuIsSolid(grid, x, y, z + 1);
					if (cFill != nFill) {
						// (x,y,z), (x,y+1,z), (x,y,z+1), (x,y+1,z+1)
						glm::vec3 V00 = getVertex(x, y, z);
						glm::vec3 V01 = getVertex(x, y + 1, z);
						glm::vec3 V10 = getVertex(x, y, z + 1);
						glm::vec3 V11 = getVertex(x, y + 1, z + 1);
						addQuad(V00, V01, V11, V10, cFill);
					}
				}
			}
		}
	}

	return triOut;
}

// The main "render" function: tries GPU single-pass, then builds adjacency on CPU
std::vector<MCTriangle> AdaptiveDualContouringRenderer::render(
	const OctreeNode* node,
	const VoxelGrid& grid,
	int x0, int y0, int z0,
	int size)
{
	// We'll assume we are always at top-level if x0=0,y0=0,z0=0, size=grid.dimX
	// so let's do everything in one shot
	if (x0 == 0 && y0 == 0 && z0 == 0 && size == grid.dimX) {
		std::cout << "Starting single-pass DC rendering...\n";
		// 1) Clear caches
		clearCaches();

		// 2) If we want GPU, do it, otherwise fallback
		if (m_useComputeShader) {
			try {
				// gather all GPU vertices
				std::vector<glm::vec4> allVerts = executeComputeShaderSinglePass(grid);

				// Then do adjacency on CPU
				std::vector<MCTriangle> triangles = buildTrianglesCPU(grid, allVerts);

				std::cout << "Single-pass DC done. Triangles: " << triangles.size() << "\n";
				return triangles;
			}
			catch (const std::exception& e) {
				std::cerr << "Compute shader error: " << e.what() << std::endl;
				m_useComputeShader = false;
			}
		}
		// If we get here, either not using compute or it failed
		// Fallback to CPU approach
		return createTriangles(grid, node, x0, y0, z0, size);
	}
	// For partial chunk or smaller calls, fallback
	// (Or you can do a subdiv approach if you want.)
	return createTriangles(grid, node, x0, y0, z0, size);
}

std::vector<MCTriangle> AdaptiveDualContouringRenderer::createTriangles(
	const VoxelGrid& grid,
	const OctreeNode* node,
	int x0, int y0, int z0,
	int size)
{
	std::vector<MCTriangle> triangles;

	// Quick exit checks
	if (!node || !node->isLeaf) {
		// Not a leaf => no direct geometry
		return triangles;
	}

	// If no surface inside this cell, skip
	if (!cellContainsSurface(grid, x0, y0, z0, size)) {
		return triangles;
	}

	// Prepare cell info, compute or retrieve dual vertex
	float cellSizeWorld = size * grid.voxelSize;
	glm::vec3 cellCenter = gridToWorld(grid, x0, y0, z0)
		+ glm::vec3(size * 0.5f * grid.voxelSize);

	// A unique key for caching
	long long cellKey = ((long long)x0 << 20)
		| ((long long)y0 << 10)
		| (long long)z0;

	glm::vec3 cellVertex = cellCenter; // default
	bool hasCalculatedVertex = false;

	{
		// See if we already have a dual vertex in the cache
		std::lock_guard<std::mutex> lock(m_cacheMutex);
		auto it = dualVertexCache.find(cellKey);
		if (it != dualVertexCache.end()) {
			cellVertex = it->second;
			hasCalculatedVertex = true;
		}
	}

	// If not in cache, gather Hermite data + generate new dual vertex
	if (!hasCalculatedVertex)
	{
		std::vector<HermitePoint> hermiteData = gatherHermiteData(grid, x0, y0, z0, size);
		if (!hermiteData.empty()) {
			cellVertex = generateDualVertex(hermiteData, cellCenter, cellSizeWorld);
		}
		// Cache it
		{
			std::lock_guard<std::mutex> lock(m_cacheMutex);
			dualVertexCache[cellKey] = cellVertex;
		}
	}

	// Build Triangles from Edge‐Based Sign Changes
	static const int edgeDirections[3][3] = {
		{1, 0, 0},  // +X
		{0, 1, 0},  // +Y
		{0, 0, 1}   // +Z
	};

	for (int dir = 0; dir < 3; dir++)
	{
		// Each cell has 4 edges in this direction
		for (int edge = 0; edge < 4; edge++)
		{
			int ex1 = x0;
			int ey1 = y0;
			int ez1 = z0;

			if (dir == 0) {  // X direction
				ey1 += (edge & 1) ? size : 0;
				ez1 += (edge & 2) ? size : 0;
			}
			else if (dir == 1) { // Y direction
				ex1 += (edge & 1) ? size : 0;
				ez1 += (edge & 2) ? size : 0;
			}
			else { // Z direction
				ex1 += (edge & 1) ? size : 0;
				ey1 += (edge & 2) ? size : 0;
			}

			int ex2 = ex1 + edgeDirections[dir][0] * size;
			int ey2 = ey1 + edgeDirections[dir][1] * size;
			int ez2 = ez1 + edgeDirections[dir][2] * size;

			if (ex1 < 0 || ey1 < 0 || ez1 < 0 ||
				ex2 < 0 || ey2 < 0 || ez2 < 0 ||
				ex1 >= grid.dimX || ey1 >= grid.dimY || ez1 >= grid.dimZ ||
				ex2 >= grid.dimX || ey2 >= grid.dimY || ez2 >= grid.dimZ)
			{
				continue;
			}

			bool isFilled1 = (grid.data[grid.index(ex1, ey1, ez1)] == VoxelState::FILLED);
			bool isFilled2 = (grid.data[grid.index(ex2, ey2, ez2)] == VoxelState::FILLED);
			if (isFilled1 == isFilled2) {
				continue;
			}

			struct AdjCell {
				int x, y, z;
				glm::vec3 vertex;
				bool isValid;
				bool isSolid;
			};

			std::vector<AdjCell> adjacentCells;
			adjacentCells.push_back({
				x0, y0, z0,
				cellVertex,
				true,
				node->isSolid
				});

			const int MAX_ADJ = 4;
			for (int adjIdx = 1; adjIdx < MAX_ADJ; adjIdx++)
			{
				int adjX = x0, adjY = y0, adjZ = z0;
				if (dir == 0) {
					if (adjIdx == 1) adjY = ey1 - size;
					else if (adjIdx == 2) adjZ = ez1 - size;
					else if (adjIdx == 3) { adjY = ey1 - size; adjZ = ez1 - size; }
				}
				else if (dir == 1) {
					if (adjIdx == 1) adjX = ex1 - size;
					else if (adjIdx == 2) adjZ = ez1 - size;
					else if (adjIdx == 3) { adjX = ex1 - size; adjZ = ez1 - size; }
				}
				else {
					if (adjIdx == 1) adjX = ex1 - size;
					else if (adjIdx == 2) adjY = ey1 - size;
					else if (adjIdx == 3) { adjX = ex1 - size; adjY = ey1 - size; }
				}

				if (adjX < 0 || adjY < 0 || adjZ < 0 ||
					adjX >= grid.dimX || adjY >= grid.dimY || adjZ >= grid.dimZ) {
					continue;
				}

				extern std::unordered_map<long long, OctreeNode*> g_octreeMap;
				long long key = ((long long)adjX << 20)
					| ((long long)adjY << 10)
					| (long long)adjZ;

				auto nodeIt = g_octreeMap.find(key);
				if (nodeIt == g_octreeMap.end() || !nodeIt->second->isLeaf) {
					continue;
				}

				int adjSize = nodeIt->second->size;
				int mySize = node->size;
				const int MAX_SIZE_DIFFERENCE = 2;
				if (std::max(mySize, adjSize) > std::min(mySize, adjSize) * MAX_SIZE_DIFFERENCE) {
					continue;
				}

				glm::vec3 adjVertex;
				bool hasAdjVertex = false;
				{
					std::lock_guard<std::mutex> lock(m_cacheMutex);
					auto found = dualVertexCache.find(key);
					if (found != dualVertexCache.end()) {
						adjVertex = found->second;
						hasAdjVertex = true;
					}
				}
				if (!hasAdjVertex) {
					glm::vec3 adjCenter = gridToWorld(grid, adjX, adjY, adjZ)
						+ glm::vec3(size * 0.5f * grid.voxelSize);

					std::vector<HermitePoint> adjHermite =
						gatherHermiteData(grid, adjX, adjY, adjZ, size);

					if (!adjHermite.empty()) {
						adjVertex = generateDualVertex(adjHermite, adjCenter, cellSizeWorld);
					}
					else {
						adjVertex = adjCenter;
					}
					{
						std::lock_guard<std::mutex> lock(m_cacheMutex);
						dualVertexCache[key] = adjVertex;
					}
				}

				adjacentCells.push_back({
					adjX, adjY, adjZ,
					adjVertex,
					true,
					nodeIt->second->isSolid
					});
			}

			bool flipNormals = adjacentCells[0].isSolid;

			if (adjacentCells.size() == 3) {
				MCTriangle tri;
				tri.v[0] = adjacentCells[0].vertex;
				tri.v[1] = adjacentCells[1].vertex;
				tri.v[2] = adjacentCells[2].vertex;

				glm::vec3 e1 = tri.v[1] - tri.v[0];
				glm::vec3 e2 = tri.v[2] - tri.v[0];
				glm::vec3 n = glm::normalize(glm::cross(e1, e2));
				if (flipNormals) n = -n;

				tri.normal[0] = n;
				tri.normal[1] = n;
				tri.normal[2] = n;

				float area = 0.5f * glm::length(glm::cross(e1, e2));
				if (area > 1e-6f) {
					triangles.push_back(tri);
				}
			}
			else if (adjacentCells.size() >= 4) {
				MCTriangle t;
				t.v[0] = adjacentCells[0].vertex;
				t.v[1] = adjacentCells[1].vertex;
				t.v[2] = adjacentCells[2].vertex;

				glm::vec3 e1 = t.v[1] - t.v[0];
				glm::vec3 e2 = t.v[2] - t.v[0];
				glm::vec3 n = glm::normalize(glm::cross(e1, e2));
				if (flipNormals) n = -n;

				t.normal[0] = n;
				t.normal[1] = n;
				t.normal[2] = n;

				float area = 0.5f * glm::length(glm::cross(e1, e2));
				if (area > 1e-6f) {
					triangles.push_back(t);
				}

				MCTriangle t2;
				t2.v[0] = adjacentCells[0].vertex;
				t2.v[1] = adjacentCells[2].vertex;
				t2.v[2] = adjacentCells[3].vertex;

				e1 = t2.v[1] - t2.v[0];
				e2 = t2.v[2] - t2.v[0];
				n = glm::normalize(glm::cross(e1, e2));
				if (flipNormals) n = -n;

				t2.normal[0] = n;
				t2.normal[1] = n;
				t2.normal[2] = n;

				area = 0.5f * glm::length(glm::cross(e1, e2));
				if (area > 1e-6f) {
					triangles.push_back(t2);
				}
			}
		}
	}

	// Fallback for boundary cells: if no triangles were created and this cell
	// touches the grid boundary, try to create face triangles.
	if (triangles.empty()) {
		// Only add fallback if this cell touches a grid boundary
		if (x0 == 0 || y0 == 0 || z0 == 0 ||
			(x0 + size) >= grid.dimX ||
			(y0 + size) >= grid.dimY ||
			(z0 + size) >= grid.dimZ) {
			std::vector<MCTriangle> faceTriangles = createFaceTriangles(grid, node, x0, y0, z0, size);
			triangles.insert(triangles.end(), faceTriangles.begin(), faceTriangles.end());
		}
	}

	return triangles;
}


// Helper function to explicitly handle face triangulation when needed
std::vector<MCTriangle> AdaptiveDualContouringRenderer::createFaceTriangles(
	const VoxelGrid& grid, const OctreeNode* node, int x0, int y0, int z0, int size) {

	std::vector<MCTriangle> triangles;

	// Skip invalid nodes
	if (!node || !node->isLeaf) {
		return triangles;
	}

	// Get this cell's vertex
	long long cellKey = ((long long)x0 << 20) | ((long long)y0 << 10) | (long long)z0;
	glm::vec3 cellVertex;

	{
		std::lock_guard<std::mutex> lock(m_cacheMutex);
		auto it = dualVertexCache.find(cellKey);
		if (it != dualVertexCache.end()) {
			cellVertex = it->second;
		}
		else {
			// Compute cell center if no vertex is available
			cellVertex = gridToWorld(grid, x0, y0, z0) +
				glm::vec3(size * 0.5f * grid.voxelSize);
			dualVertexCache[cellKey] = cellVertex;
		}
	}

	// Define the six face directions
	const int faceDirections[6][3] = {
		{1, 0, 0}, {-1, 0, 0},  // ±X
		{0, 1, 0}, {0, -1, 0},  // ±Y
		{0, 0, 1}, {0, 0, -1}   // ±Z
	};

	// Check each face for a sign change
	for (int face = 0; face < 6; face++) {
		// Calculate adjacent cell position
		int nx = x0 + faceDirections[face][0] * size;
		int ny = y0 + faceDirections[face][1] * size;
		int nz = z0 + faceDirections[face][2] * size;

		// Skip if out of bounds
		if (nx < 0 || ny < 0 || nz < 0 ||
			nx >= grid.dimX || ny >= grid.dimY || nz >= grid.dimZ) {
			continue;
		}

		// Get solid state
		bool currentSolid = node->isSolid;
		bool neighborSolid = false;

		// Check if neighbor exists in octree
		extern std::unordered_map<long long, OctreeNode*> g_octreeMap;
		long long neighborKey = ((long long)nx << 20) | ((long long)ny << 10) | (long long)nz;
		auto nodeIt = g_octreeMap.find(neighborKey);

		const int MAX_SIZE_DIFFERENCE = 2;
		if (nodeIt != g_octreeMap.end() && nodeIt->second->isLeaf) {
			int adjSize = nodeIt->second->size;
			if (std::max(size, adjSize) > std::min(size, adjSize) * MAX_SIZE_DIFFERENCE) {
				continue;  // Skip connections between cells of very different sizes
			}
			neighborSolid = nodeIt->second->isSolid;
		}
		else {
			// Sample from the grid directly (use center point of neighbor)
			int cx = nx + size / 2;
			int cy = ny + size / 2;
			int cz = nz + size / 2;

			// Ensure in bounds
			cx = std::min(std::max(cx, 0), grid.dimX - 1);
			cy = std::min(std::max(cy, 0), grid.dimY - 1);
			cz = std::min(std::max(cz, 0), grid.dimZ - 1);

			neighborSolid = (grid.data[grid.index(cx, cy, cz)] == VoxelState::FILLED);
		}

		// Skip if no sign change
		if (currentSolid == neighborSolid) {
			continue;
		}

		// We have a sign change, so create face triangles

		// Get or create the neighbor vertex
		glm::vec3 neighborVertex;
		bool hasNeighborVertex = false;

		if (nodeIt != g_octreeMap.end() && nodeIt->second->isLeaf) {
			// Get from cache if available
			std::lock_guard<std::mutex> lock(m_cacheMutex);
			auto it = dualVertexCache.find(neighborKey);
			if (it != dualVertexCache.end()) {
				neighborVertex = it->second;
				hasNeighborVertex = true;
			}
		}

		if (!hasNeighborVertex) {
			// Calculate neighbor center
			neighborVertex = gridToWorld(grid, nx, ny, nz) +
				glm::vec3(size * 0.5f * grid.voxelSize);

			// Cache it
			std::lock_guard<std::mutex> lock(m_cacheMutex);
			dualVertexCache[neighborKey] = neighborVertex;
			hasNeighborVertex = true;
		}

		// Create face corners
		float halfSize = size * grid.voxelSize * 0.5f;
		glm::vec3 faceNormal(faceDirections[face][0], faceDirections[face][1], faceDirections[face][2]);

		// Face center is halfway between the two cell centers
		glm::vec3 faceCenter = (cellVertex + neighborVertex) * 0.5f;

		// Get two orthogonal directions to the face normal
		glm::vec3 tangent1, tangent2;

		if (std::abs(faceNormal.x) > 0.5f) {
			// X-aligned face
			tangent1 = glm::vec3(0, 1, 0);
			tangent2 = glm::vec3(0, 0, 1);
		}
		else if (std::abs(faceNormal.y) > 0.5f) {
			// Y-aligned face
			tangent1 = glm::vec3(1, 0, 0);
			tangent2 = glm::vec3(0, 0, 1);
		}
		else {
			// Z-aligned face
			tangent1 = glm::vec3(1, 0, 0);
			tangent2 = glm::vec3(0, 1, 0);
		}

		// Create the four corners of the face
		glm::vec3 corners[4];
		corners[0] = faceCenter - tangent1 * halfSize - tangent2 * halfSize;
		corners[1] = faceCenter + tangent1 * halfSize - tangent2 * halfSize;
		corners[2] = faceCenter + tangent1 * halfSize + tangent2 * halfSize;
		corners[3] = faceCenter - tangent1 * halfSize + tangent2 * halfSize;

		// Create two triangles from this cell to the face
		MCTriangle tri1, tri2;

		// First triangle
		tri1.v[0] = cellVertex;
		tri1.v[1] = corners[0];
		tri1.v[2] = corners[1];

		// Calculate normal (pointing from solid to empty)
		glm::vec3 normal = faceNormal;
		if (currentSolid) {
			normal = normal;  // Already correct
		}
		else {
			normal = -normal; // Flip to point outward
		}

		tri1.normal[0] = normal;
		tri1.normal[1] = normal;
		tri1.normal[2] = normal;

		triangles.push_back(tri1);

		// Second triangle
		tri2.v[0] = cellVertex;
		tri2.v[1] = corners[1];
		tri2.v[2] = corners[2];

		tri2.normal[0] = normal;
		tri2.normal[1] = normal;
		tri2.normal[2] = normal;

		triangles.push_back(tri2);

		// Third triangle
		MCTriangle tri3;
		tri3.v[0] = cellVertex;
		tri3.v[1] = corners[2];
		tri3.v[2] = corners[3];

		tri3.normal[0] = normal;
		tri3.normal[1] = normal;
		tri3.normal[2] = normal;

		triangles.push_back(tri3);

		// Fourth triangle
		MCTriangle tri4;
		tri4.v[0] = cellVertex;
		tri4.v[1] = corners[3];
		tri4.v[2] = corners[0];

		tri4.normal[0] = normal;
		tri4.normal[1] = normal;
		tri4.normal[2] = normal;

		triangles.push_back(tri4);

		// Also create triangles from neighbor to the face (with opposite normals)
		if (hasNeighborVertex) {
			MCTriangle ntri1, ntri2, ntri3, ntri4;

			ntri1.v[0] = neighborVertex;
			ntri1.v[1] = corners[1];
			ntri1.v[2] = corners[0];

			ntri1.normal[0] = -normal;
			ntri1.normal[1] = -normal;
			ntri1.normal[2] = -normal;

			triangles.push_back(ntri1);

			ntri2.v[0] = neighborVertex;
			ntri2.v[1] = corners[2];
			ntri2.v[2] = corners[1];

			ntri2.normal[0] = -normal;
			ntri2.normal[1] = -normal;
			ntri2.normal[2] = -normal;

			triangles.push_back(ntri2);

			ntri3.v[0] = neighborVertex;
			ntri3.v[1] = corners[3];
			ntri3.v[2] = corners[2];

			ntri3.normal[0] = -normal;
			ntri3.normal[1] = -normal;
			ntri3.normal[2] = -normal;

			triangles.push_back(ntri3);

			ntri4.v[0] = neighborVertex;
			ntri4.v[1] = corners[0];
			ntri4.v[2] = corners[3];

			ntri4.normal[0] = -normal;
			ntri4.normal[1] = -normal;
			ntri4.normal[2] = -normal;

			triangles.push_back(ntri4);
		}
	}

	return triangles;
}

// Simplified version of gatherHermiteData that focuses on key surface points
std::vector<HermitePoint> AdaptiveDualContouringRenderer::gatherHermiteData(
	const VoxelGrid& grid, int x0, int y0, int z0, int size) {

	// Boundary checks
	int maxX = std::min(x0 + size, grid.dimX - 1);
	int maxY = std::min(y0 + size, grid.dimY - 1);
	int maxZ = std::min(z0 + size, grid.dimZ - 1);
	int minX = std::max(x0, 0);
	int minY = std::max(y0, 0);
	int minZ = std::max(z0, 0);

	// Use adaptive stride based on cell size, but ensure small cells get checked thoroughly
	int stride = (size > 8) ? 2 : 1;
	if (size <= 4) stride = 1; // Always check every voxel for small cells

	std::vector<HermitePoint> points;
	points.reserve((maxX - minX + 1) * (maxY - minY + 1) * (maxZ - minZ + 1) / 8);

	// Also check the cell boundary explicitly
	for (int z = minZ; z <= maxZ; z += stride) {
		for (int y = minY; y <= maxY; y += stride) {
			for (int x = minX; x <= maxX; x += stride) {
				// Do extra sampling at the cell boundaries
				bool isBoundary = (x == minX || x == maxX || y == minY || y == maxY || z == minZ || z == maxZ);
				int localStride = isBoundary ? 1 : stride; // Always check every voxel on boundaries

				bool currentFilled = (grid.data[grid.index(x, y, z)] == VoxelState::FILLED);

				// Check all three directions
				const int dirs[3][3] = { {1,0,0}, {0,1,0}, {0,0,1} };

				for (int d = 0; d < 3; d++) {
					int nx = x + dirs[d][0];
					int ny = y + dirs[d][1];
					int nz = z + dirs[d][2];

					// Skip if out of bounds
					if (nx < 0 || ny < 0 || nz < 0 ||
						nx >= grid.dimX || ny >= grid.dimY || nz >= grid.dimZ) {
						continue;
					}

					bool nextFilled = (grid.data[grid.index(nx, ny, nz)] == VoxelState::FILLED);

					// If surface crosses here, add a hermite point
					if (currentFilled != nextFilled) {
						points.push_back(calculateIntersection(grid, x, y, z, nx, ny, nz));
					}
				}
			}
		}
	}

	return points;
}

glm::vec3 AdaptiveDualContouringRenderer::generateDualVertex(
	const std::vector<HermitePoint>& hermiteData,
	const glm::vec3& cellCenter, float cellSize) {

	// If no data, return center
	if (hermiteData.empty()) {
		return cellCenter;
	}

	// Define cell bounds
	glm::vec3 halfSize(cellSize * 0.5f);
	glm::vec3 minBound = cellCenter - halfSize;
	glm::vec3 maxBound = cellCenter + halfSize;

	// Small inset to avoid boundary issues
	float inset = cellSize * 0.001f;
	minBound += glm::vec3(inset);
	maxBound -= glm::vec3(inset);

	// Compute mass point first as a fallback
	glm::vec3 massPoint = glm::vec3(0.0f);
	for (const auto& hp : hermiteData) {
		massPoint += hp.position;
	}
	massPoint /= float(hermiteData.size());

	// Calculate average and dominant normal
	glm::vec3 avgNormal(0.0f);
	for (const auto& hp : hermiteData) {
		avgNormal += hp.normal;
	}

	// For architectural meshes, try to snap to dominant axis
	if (glm::length(avgNormal) > 0.0001f) {
		avgNormal = glm::normalize(avgNormal);
		glm::vec3 absNormal = glm::abs(avgNormal);
		float maxComp = std::max(std::max(absNormal.x, absNormal.y), absNormal.z);

		// If one component is clearly dominant (architectural feature)
		if (maxComp > 0.85f) {
			// Snap normal to axis for cleaner results
			if (absNormal.x == maxComp) {
				avgNormal = glm::vec3(avgNormal.x > 0 ? 1.0f : -1.0f, 0, 0);
			}
			else if (absNormal.y == maxComp) {
				avgNormal = glm::vec3(0, avgNormal.y > 0 ? 1.0f : -1.0f, 0);
			}
			else {
				avgNormal = glm::vec3(0, 0, avgNormal.z > 0 ? 1.0f : -1.0f);
			}

			// Find all points on this "plane"
			glm::vec3 planePoint = glm::vec3(0.0f);
			int planePointCount = 0;

			for (const auto& hp : hermiteData) {
				float alignment = glm::dot(glm::normalize(hp.normal), avgNormal);
				if (alignment > 0.7f) {
					planePoint += hp.position;
					planePointCount++;
				}
			}

			if (planePointCount > 0) {
				planePoint /= float(planePointCount);

				// Project cell center onto plane
				float d = -glm::dot(avgNormal, planePoint);
				float t = -(glm::dot(avgNormal, cellCenter) + d);
				glm::vec3 projectedVertex = cellCenter + t * avgNormal;

				// Clamp to cell bounds
				return glm::clamp(projectedVertex, minBound, maxBound);
			}
		}
	}

	// Default QEF solve
	QEFSolver qef;
	for (const auto& hp : hermiteData) {
		qef.addPoint(hp.position, hp.normal);
	}

	// Solve with constraints to guarantee we stay in bounds
	glm::vec3 qefSolution = qef.solveConstrained(minBound, maxBound);

	// Weighted average between QEF solution and mass point for stability
	return glm::mix(qefSolution, massPoint, 0.1f);
}

HermitePoint AdaptiveDualContouringRenderer::calculateIntersection(
	const VoxelGrid& grid, int x1, int y1, int z1, int x2, int y2, int z2) {

	// Create a key for this edge
	EdgeKey key = { x1, y1, z1, x2, y2, z2 };

	// Thread-safe cache lookup
	{
		std::lock_guard<std::mutex> lock(m_cacheMutex);
		auto it = edgeIntersectionCache.find(key);
		if (it != edgeIntersectionCache.end()) {
			return it->second;
		}
	}

	// Get voxel states at endpoints
	bool isFilled1 = (grid.data[grid.index(x1, y1, z1)] == VoxelState::FILLED);
	bool isFilled2 = (grid.data[grid.index(x2, y2, z2)] == VoxelState::FILLED);

	// Convert to scalar field values (-1 inside, +1 outside)
	float v1 = isFilled1 ? -1.0f : 1.0f;
	float v2 = isFilled2 ? -1.0f : 1.0f;

	// Check for sign change (should always be true if this function is called correctly)
	if (v1 * v2 > 0) {
		// No intersection - this is an error case - return midpoint as fallback
		HermitePoint hp;
		hp.position = gridToWorld(grid, (x1 + x2) / 2.0f, (y1 + y2) / 2.0f, (z1 + z2) / 2.0f);
		hp.normal = glm::normalize(glm::vec3(x2 - x1, y2 - y1, z2 - z1));

		// Cache result
		std::lock_guard<std::mutex> lock(m_cacheMutex);
		edgeIntersectionCache[key] = hp;
		return hp;
	}

	// Convert grid indices to world positions
	glm::vec3 p1 = gridToWorld(grid, x1, y1, z1);
	glm::vec3 p2 = gridToWorld(grid, x2, y2, z2);

	// Calculate interpolation parameter t
	float t = v1 / (v1 - v2);
	t = glm::clamp(t, 0.0f, 1.0f);

	// Interpolate the position
	glm::vec3 position = p1 + t * (p2 - p1);

	glm::vec3 normal(0.0f);

	// Determine primary axis of the edge
	int dx = x2 - x1;
	int dy = y2 - y1;
	int dz = z2 - z1;

	if (std::abs(dx) + std::abs(dy) + std::abs(dz) != 1) {
		normal = glm::normalize(glm::vec3(dx, dy, dz));
		// Make normal point from filled to empty
		if (isFilled1) normal = -normal;
	}
	else {
		// Grid-aligned edge - compute normal using central differences if possible

		// Helper function to get voxel scalar value (-1 or +1)
		auto getScalar = [&](int x, int y, int z) -> float {
			if (x < 0 || y < 0 || z < 0 ||
				x >= grid.dimX || y >= grid.dimY || z >= grid.dimZ) {
				return 1.0f; // Outside is empty
			}
			return (grid.data[grid.index(x, y, z)] == VoxelState::FILLED) ? -1.0f : 1.0f;
			};

		// Default to normal along the edge
		normal = glm::vec3(dx, dy, dz);

		// Try to sample perpendicular to the edge for better normal estimation
		if (dx != 0) { // X-aligned edge
			// Sample in Y and Z directions
			float gy = getScalar(x1, y1 + 1, z1) - getScalar(x1, y1 - 1, z1);
			float gz = getScalar(x1, y1, z1 + 1) - getScalar(x1, y1, z1 - 1);
			normal = glm::vec3(0.0f, gy, gz);
		}
		else if (dy != 0) { // Y-aligned edge
			// Sample in X and Z directions
			float gx = getScalar(x1 + 1, y1, z1) - getScalar(x1 - 1, y1, z1);
			float gz = getScalar(x1, y1, z1 + 1) - getScalar(x1, y1, z1 - 1);
			normal = glm::vec3(gx, 0.0f, gz);
		}
		else { // Z-aligned edge
			// Sample in X and Y directions
			float gx = getScalar(x1 + 1, y1, z1) - getScalar(x1 - 1, y1, z1);
			float gy = getScalar(x1, y1 + 1, z1) - getScalar(x1, y1 - 1, z1);
			normal = glm::vec3(gx, gy, 0.0f);
		}

		// If normal is zero or extremely small, just use edge direction
		if (glm::length2(normal) < 1e-10) {
			normal = glm::vec3(dx, dy, dz);
		}
		else {
			normal = glm::normalize(normal);
		}

		// Make sure normal points from filled to empty
		float dotProduct = normal.x * dx + normal.y * dy + normal.z * dz;
		bool normalPointsWithEdge = dotProduct > 0;
		bool edgePointsToFilled = isFilled2; // Edge points from 1 to 2

		// If both true or both false, flip the normal
		if (normalPointsWithEdge == edgePointsToFilled) {
			normal = -normal;
		}
	}

	// Create hermite point
	HermitePoint hp = { position, normal };

	// Cache result
	std::lock_guard<std::mutex> lock(m_cacheMutex);
	edgeIntersectionCache[key] = hp;

	return hp;
}

glm::vec3 AdaptiveDualContouringRenderer::gridToWorld(const VoxelGrid& grid, int x, int y, int z) {
	return glm::vec3(
		grid.minX + x * grid.voxelSize,
		grid.minY + y * grid.voxelSize,
		grid.minZ + z * grid.voxelSize
	);
}

bool AdaptiveDualContouringRenderer::cellContainsSurface(
	const VoxelGrid& grid, int x0, int y0, int z0, int size) {

	// Boundary checking
	int maxX = std::min(x0 + size, grid.dimX);
	int maxY = std::min(y0 + size, grid.dimY);
	int maxZ = std::min(z0 + size, grid.dimZ);
	int minX = std::max(x0, 0);
	int minY = std::max(y0, 0);
	int minZ = std::max(z0, 0);

	// Skip empty cells
	if (minX >= maxX || minY >= maxY || minZ >= maxZ) {
		return false;
	}

	// For performance, first check corners
	bool anyFilled = false;
	bool anyEmpty = false;

	// Check corners - if corners have different states, the cell contains a surface
	const int corners[8][3] = {
		{minX, minY, minZ}, {maxX - 1, minY, minZ}, {maxX - 1, maxY - 1, minZ}, {minX, maxY - 1, minZ},
		{minX, minY, maxZ - 1}, {maxX - 1, minY, maxZ - 1}, {maxX - 1, maxY - 1, maxZ - 1}, {minX, maxY - 1, maxZ - 1}
	};

	for (int i = 0; i < 8; i++) {
		int cx = corners[i][0];
		int cy = corners[i][1];
		int cz = corners[i][2];

		// Skip if corner is out of bounds
		if (cx < 0 || cy < 0 || cz < 0 ||
			cx >= grid.dimX || cy >= grid.dimY || cz >= grid.dimZ) {
			continue;
		}

		bool isFilled = (grid.data[grid.index(cx, cy, cz)] == VoxelState::FILLED);

		if (isFilled) anyFilled = true;
		else anyEmpty = true;

		// Early exit if we found both filled and empty voxels
		if (anyFilled && anyEmpty) {
			return true;
		}
	}

	// If all corners are the same, check the faces
	// Check 3 axis-aligned faces for faster detection
	const int faceDir[3][3] = {
		{1, 0, 0},  // X direction
		{0, 1, 0},  // Y direction
		{0, 0, 1}   // Z direction
	};

	// Don't check every voxel - just check along the center of each face
	// This is a good trade-off between performance and accuracy
	for (int dir = 0; dir < 3; dir++) {
		int stride = std::max(1, size / 4); // Adaptive stride based on cell size

		for (int offset = 0; offset < size; offset += stride) {
			int x1, y1, z1, x2, y2, z2;

			// Set up coordinates for the current direction
			if (dir == 0) { // X direction
				y1 = minY + offset;
				z1 = minZ + offset;
				if (y1 >= maxY || z1 >= maxZ) continue;

				// Check minX-1 and minX
				x1 = minX - 1;
				x2 = minX;

				if (x1 >= 0 && x2 < grid.dimX) {
					bool state1 = (grid.data[grid.index(x1, y1, z1)] == VoxelState::FILLED);
					bool state2 = (grid.data[grid.index(x2, y1, z1)] == VoxelState::FILLED);
					if (state1 != state2) return true;
				}

				// Check maxX-1 and maxX
				x1 = maxX - 1;
				x2 = maxX;

				if (x1 >= 0 && x2 < grid.dimX) {
					bool state1 = (grid.data[grid.index(x1, y1, z1)] == VoxelState::FILLED);
					bool state2 = (grid.data[grid.index(x2, y1, z1)] == VoxelState::FILLED);
					if (state1 != state2) return true;
				}
			}
			else if (dir == 1) { // Y direction
				x1 = minX + offset;
				z1 = minZ + offset;
				if (x1 >= maxX || z1 >= maxZ) continue;

				// Check minY-1 and minY
				y1 = minY - 1;
				y2 = minY;

				if (y1 >= 0 && y2 < grid.dimY) {
					bool state1 = (grid.data[grid.index(x1, y1, z1)] == VoxelState::FILLED);
					bool state2 = (grid.data[grid.index(x1, y2, z1)] == VoxelState::FILLED);
					if (state1 != state2) return true;
				}

				// Check maxY-1 and maxY
				y1 = maxY - 1;
				y2 = maxY;

				if (y1 >= 0 && y2 < grid.dimY) {
					bool state1 = (grid.data[grid.index(x1, y1, z1)] == VoxelState::FILLED);
					bool state2 = (grid.data[grid.index(x1, y2, z1)] == VoxelState::FILLED);
					if (state1 != state2) return true;
				}
			}
			else { // Z direction
				x1 = minX + offset;
				y1 = minY + offset;
				if (x1 >= maxX || y1 >= maxY) continue;

				// Check minZ-1 and minZ
				z1 = minZ - 1;
				z2 = minZ;

				if (z1 >= 0 && z2 < grid.dimZ) {
					bool state1 = (grid.data[grid.index(x1, y1, z1)] == VoxelState::FILLED);
					bool state2 = (grid.data[grid.index(x1, y1, z2)] == VoxelState::FILLED);
					if (state1 != state2) return true;
				}

				// Check maxZ-1 and maxZ
				z1 = maxZ - 1;
				z2 = maxZ;

				if (z1 >= 0 && z2 < grid.dimZ) {
					bool state1 = (grid.data[grid.index(x1, y1, z1)] == VoxelState::FILLED);
					bool state2 = (grid.data[grid.index(x1, y1, z2)] == VoxelState::FILLED);
					if (state1 != state2) return true;
				}
			}
		}
	}

	// For small cells (size <= 4), do a thorough check
	if (size <= 4) {
		// Scan through internal voxels to check for sign changes
		for (int z = minZ; z < maxZ - 1; z++) {
			for (int y = minY; y < maxY - 1; y++) {
				for (int x = minX; x < maxX - 1; x++) {
					bool state = (grid.data[grid.index(x, y, z)] == VoxelState::FILLED);
					bool stateX = (grid.data[grid.index(x + 1, y, z)] == VoxelState::FILLED);
					bool stateY = (grid.data[grid.index(x, y + 1, z)] == VoxelState::FILLED);
					bool stateZ = (grid.data[grid.index(x, y, z + 1)] == VoxelState::FILLED);

					if (state != stateX || state != stateY || state != stateZ) {
						return true;
					}
				}
			}
		}
	}

	return false;
}

void AdaptiveDualContouringRenderer::initThreadPool() {
	if (m_threadPoolActive) return;

	// Auto-detect thread count if not set
	if (m_threadCount <= 0) {
		m_threadCount = std::max(1, (int)std::thread::hardware_concurrency() - 1);
	}

	std::cout << "Initializing thread pool with " << m_threadCount << " workers" << std::endl;

	m_shutdownThreads = false;
	m_threadPoolActive = true;

	// Create worker threads
	for (int i = 0; i < m_threadCount; i++) {
		m_workers.emplace_back(&AdaptiveDualContouringRenderer::workerFunction, this);
	}
}

void AdaptiveDualContouringRenderer::shutdownThreadPool() {
	if (!m_threadPoolActive) return;

	std::cout << "Shutting down thread pool" << std::endl;

	{
		std::unique_lock<std::mutex> lock(m_queueMutex);
		m_shutdownThreads = true;
	}

	m_queueCondition.notify_all();

	for (auto& worker : m_workers) {
		if (worker.joinable()) {
			worker.join();
		}
	}

	m_workers.clear();
	m_threadPoolActive = false;
}

void AdaptiveDualContouringRenderer::workerFunction() {
	while (true) {
		std::shared_ptr<RenderTask> task = getNextTask();

		if (!task) {
			break;  // Exit signal received
		}

		processTask(task);
	}
}

std::shared_ptr<AdaptiveDualContouringRenderer::RenderTask>
AdaptiveDualContouringRenderer::getNextTask() {
	std::unique_lock<std::mutex> lock(m_queueMutex);

	m_queueCondition.wait(lock, [this]() {
		return !m_taskQueue.empty() || m_shutdownThreads;
		});

	if (m_shutdownThreads && m_taskQueue.empty()) {
		return nullptr;
	}

	auto task = m_taskQueue.front();
	m_taskQueue.pop();

	return task;
}

void AdaptiveDualContouringRenderer::processTask(std::shared_ptr<RenderTask> task) {
	try {
		std::vector<MCTriangle> result = render(
			task->node,
			*(task->grid),
			task->x0, task->y0, task->z0,
			task->size
		);

		task->resultPromise.set_value(std::move(result));
	}
	catch (const std::exception& e) {
		try {
			task->resultPromise.set_exception(std::current_exception());
		}
		catch (...) {
			// Promise might be broken
		}
	}
}

std::future<std::vector<MCTriangle>> AdaptiveDualContouringRenderer::submitTask(
	const OctreeNode* node,
	const VoxelGrid* grid,
	int x0, int y0, int z0, int size) {

	auto task = std::make_shared<RenderTask>();
	task->node = node;
	task->grid = grid;
	task->x0 = x0;
	task->y0 = y0;
	task->z0 = z0;
	task->size = size;

	std::future<std::vector<MCTriangle>> future = task->resultPromise.get_future();

	{
		std::unique_lock<std::mutex> lock(m_queueMutex);
		m_taskQueue.push(task);
	}

	m_queueCondition.notify_one();

	return future;
}

std::vector<MCTriangle> AdaptiveDualContouringRenderer::renderParallel(
	const OctreeNode* node,
	const VoxelGrid& grid,
	int x0, int y0, int z0, int size) {

	if (!node) return {};

	// Process leaf nodes directly
	if (node->isLeaf) {
		return createTriangles(grid, node, x0, y0, z0, size);
	}

	// Process children in parallel
	std::vector<std::future<std::vector<MCTriangle>>> futures;
	std::vector<MCTriangle> triangles;

	int halfSize = size / 2;

	// Submit children as parallel tasks
	for (int i = 0; i < 8; i++) {
		if (!node->children[i]) continue;

		int childX = x0 + ((i & 1) ? halfSize : 0);
		int childY = y0 + ((i & 2) ? halfSize : 0);
		int childZ = z0 + ((i & 4) ? halfSize : 0);

		// Submit larger chunks as parallel tasks
		if (halfSize >= 2) {
			futures.push_back(submitTask(
				node->children[i],
				&grid,
				childX, childY, childZ,
				halfSize
			));
		}
		else {
			// Process smaller chunks directly
			std::vector<MCTriangle> childTriangles = render(
				node->children[i],
				grid,
				childX, childY, childZ,
				halfSize
			);

			triangles.insert(triangles.end(), childTriangles.begin(), childTriangles.end());
		}
	}

	// Collect results from parallel tasks
	for (auto& future : futures) {
		auto childTriangles = future.get();
		triangles.insert(triangles.end(), childTriangles.begin(), childTriangles.end());
	}

	return triangles;
}

