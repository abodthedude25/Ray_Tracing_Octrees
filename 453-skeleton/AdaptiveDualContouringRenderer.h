#pragma once

#include "Renderer.h"
#include "OctreeVoxel.h"
#include <vector>
#include <unordered_map>
#include <glm/glm.hpp>
#include <array>
#include <thread>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <future>
#include <queue>

// Forward declaration for OpenGL
typedef unsigned int GLuint;

// Hermite data for an intersection point
struct HermitePoint {
	glm::vec3 position;
	glm::vec3 normal;
};

// EdgeKey for caching intersection results
struct EdgeKey {
	int x1, y1, z1;
	int x2, y2, z2;
	bool operator==(const EdgeKey& other) const {
		return x1 == other.x1 && y1 == other.y1 && z1 == other.z1 &&
			x2 == other.x2 && y2 == other.y2 && z2 == other.z2;
	}
};

namespace std {
	template<>
	struct hash<EdgeKey> {
		size_t operator()(const EdgeKey& key) const {
			return ((((key.x1 * 73856093) ^ (key.y1 * 19349663) ^ (key.z1 * 83492791)) ^
				((key.x2 * 73856093) ^ (key.y2 * 19349663) ^ (key.z2 * 83492791))));
		}
	};
}

// QEF solver
class QEFSolver {
public:
	QEFSolver();
	void addPoint(const glm::vec3& point, const glm::vec3& normal);
	void clear();
	glm::vec3 solve(const glm::vec3& cellCenter, float cellSize);
	glm::vec3 solveConstrained(const glm::vec3& minBound, const glm::vec3& maxBound);
private:
	glm::mat3 ata;
	glm::vec3 atb;
	glm::vec3 pointSum;
	int numPoints;
};

// AdaptiveDualContouringRenderer class
class AdaptiveDualContouringRenderer : public Renderer {
public:
	AdaptiveDualContouringRenderer();
	~AdaptiveDualContouringRenderer();

	// Main rendering method
	std::vector<MCTriangle> render(const OctreeNode* node,
		const VoxelGrid& grid,
		int x0, int y0, int z0, int size) override;

	// Configuration options
	float m_detailThreshold = 0.1f;
	bool m_useAdaptiveLOD = true;
	bool m_useComputeShader = true;

	// Thread management
	void setThreadCount(int count);
	int getThreadCount() const { return m_threadCount; }
	void initThreadPool();
	void shutdownThreadPool();

	// Cache management
	void clearCaches();

	// GPU compute methods
	bool initComputeShader();
	void destroyComputeShader();
	bool compileComputeShader();
	std::vector<glm::vec4> executeComputeShaderSinglePass(const VoxelGrid& grid);
	std::vector<MCTriangle> buildTrianglesCPU(const VoxelGrid& grid, const std::vector<glm::vec4>& allVerts);

	// CPU-based methods
	std::vector<MCTriangle> createTriangles(const VoxelGrid& grid, const OctreeNode* node,
		int x0, int y0, int z0, int size);
	bool cellContainsSurface(const VoxelGrid& grid, int x0, int y0, int z0, int size);
	bool isCellImportant(const VoxelGrid& grid, int x0, int y0, int z0, int size, float cellSizeWorld);
	glm::vec3 gridToWorld(const VoxelGrid& grid, int x, int y, int z);
	std::vector<HermitePoint> gatherHermiteData(const VoxelGrid& grid, int x0, int y0, int z0, int size);
	glm::vec3 generateDualVertex(const std::vector<HermitePoint>& hermiteData,
		const glm::vec3& cellCenter, float cellSize);
	HermitePoint calculateIntersection(const VoxelGrid& grid,
		int x1, int y1, int z1, int x2, int y2, int z2);

private:
	// GPU resources
	unsigned int m_computeProgram = 0;
	unsigned int m_voxelGridBuffer = 0;
	unsigned int m_hermitePointsBuffer = 0;
	unsigned int m_vertexBuffer = 0;
	unsigned int m_triangleBuffer = 0;
	bool m_computeShaderInitialized = false;

	// Caches
	std::unordered_map<EdgeKey, HermitePoint> edgeIntersectionCache;
	std::unordered_map<long long, glm::vec3> dualVertexCache;
	int m_cacheSize = 100000;

	// Thread pool
	struct RenderTask {
		const OctreeNode* node;
		const VoxelGrid* grid;
		int x0, y0, z0, size;
		std::promise<std::vector<MCTriangle>> resultPromise;
	};

	int m_threadCount = 0;
	bool m_threadPoolActive = false;
	std::atomic<bool> m_shutdownThreads{ false };
	std::vector<std::thread> m_workers;
	std::queue<std::shared_ptr<RenderTask>> m_taskQueue;
	std::mutex m_queueMutex;
	std::mutex m_cacheMutex;
	std::condition_variable m_queueCondition;

	void workerFunction();
	std::shared_ptr<RenderTask> getNextTask();
	void processTask(std::shared_ptr<RenderTask> task);
	std::future<std::vector<MCTriangle>> submitTask(const OctreeNode* node,
		const VoxelGrid* grid, int x0, int y0, int z0, int size);
	std::vector<MCTriangle> renderParallel(const OctreeNode* node,
		const VoxelGrid& grid, int x0, int y0, int z0, int size);

	std::vector<MCTriangle> createFaceTriangles(
		const VoxelGrid& grid, const OctreeNode* node, int x0, int y0, int z0, int size);

};
