#pragma once

#include "Renderer.h"
#include "OctreeVoxel.h"
#include <vector>
#include <unordered_map>
#include <glm/glm.hpp>
#include <array>

// Hermite data for an intersection point
struct HermitePoint {
	glm::vec3 position;
	glm::vec3 normal;
};

// Simple key for edge hashing (x1, y1, z1, x2, y2, z2)
struct EdgeKey {
	int x1, y1, z1;
	int x2, y2, z2;

	bool operator==(const EdgeKey& other) const {
		return x1 == other.x1 && y1 == other.y1 && z1 == other.z1 &&
			x2 == other.x2 && y2 == other.y2 && z2 == other.z2;
	}
};

// Structure to hold vertex and edge information for a cell face
struct DualFace {
	glm::vec3 vertex1, vertex2;  // Dual vertices on either side of the face
	std::vector<HermitePoint> hermitePoints; // Hermite points on this face
};

// Hash function for EdgeKey
namespace std {
	template<>
	struct hash<EdgeKey> {
		size_t operator()(const EdgeKey& key) const {
			// Simple hash function for the edge key
			return ((((key.x1 * 73856093) ^ (key.y1 * 19349663) ^ (key.z1 * 83492791)) ^
				((key.x2 * 73856093) ^ (key.y2 * 19349663) ^ (key.z2 * 83492791))));
		}
	};
}

// Quadric Error Function solver
class QEFSolver {
public:
	QEFSolver();

	// Add a point and normal to the QEF
	void addPoint(const glm::vec3& point, const glm::vec3& normal);

	// Clear all points
	void clear();

	// Solve the QEF to find the optimal position
	// with regularization to handle colinear normals
	glm::vec3 solve(const glm::vec3& cellCenter, float cellSize);

	// Solve with constraints to keep the point inside the cell bounds
	glm::vec3 solveConstrained(const glm::vec3& minBound, const glm::vec3& maxBound);

	// Calculate the error for a given position
	float calculateError(const glm::vec3& point);

private:
	// Matrix A and vector b for the QEF
	glm::mat3 ata;       // A^T * A
	glm::vec3 atb;       // A^T * b
	float btb;           // b^T * b
	glm::vec3 pointSum;  // Sum of all points
	int numPoints;       // Number of points added

	// Helper methods
	bool isNonSingular(const glm::mat3& mat);
	// SVD helper methods
	bool svdSolve(const glm::mat3& A, const glm::vec3& b, glm::vec3& x);
	void svdDecompose(const glm::mat3& A, glm::mat3& U, glm::vec3& sigma, glm::mat3& V);
	glm::vec3 solveSVD(const glm::vec3& cellCenter);

};

class AdaptiveDualContouringRenderer : public Renderer {
public:
	AdaptiveDualContouringRenderer();
	~AdaptiveDualContouringRenderer();

	// Override the render method from Renderer base class
	std::vector<MCTriangle> render(const OctreeNode* node,
		const VoxelGrid& grid,
		int x0, int y0, int z0, int size) override;

private:
	// Cache for edge intersections to avoid redundant calculations
	std::unordered_map<EdgeKey, HermitePoint> edgeIntersectionCache;

	// Cache for dual vertices to ensure consistent positions
	std::unordered_map<long long, glm::vec3> dualVertexCache;

	// Helper methods
	HermitePoint calculateIntersection(const VoxelGrid& grid, int x1, int y1, int z1, int x2, int y2, int z2);
	glm::vec3 estimateNormal(const VoxelGrid& grid, const glm::vec3& position, float voxelSize);
	glm::vec3 estimateNormalRobust(const VoxelGrid& grid, const glm::vec3& position,
		int centerX, int centerY, int centerZ);
	std::vector<HermitePoint> gatherHermiteData(const VoxelGrid& grid, int x0, int y0, int z0, int size);
	std::vector<HermitePoint> getFaceHermitePoints(
		const VoxelGrid& grid,
		int faceDir,    // 0=X, 1=Y, 2=Z axis
		int faceSign,   // +1 or -1 for positive/negative direction
		int minX, int minY, int minZ,
		int maxX, int maxY, int maxZ);

	// Generate dual vertex for a cell
	glm::vec3 generateDualVertex(const std::vector<HermitePoint>& hermiteData,
		const glm::vec3& cellCenter, float cellSize);

	// Create triangles for dual cells
	std::vector<MCTriangle> createTriangles(const VoxelGrid& grid,
		const OctreeNode* node,
		int x0, int y0, int z0, int size);

	// Create triangles for a cell face
	std::vector<MCTriangle> createFaceTriangles(const glm::vec3& cellVertex,
		const DualFace& face,
		bool isInsideCell);

	// Check if a cell contains the surface
	bool cellContainsSurface(const VoxelGrid& grid, int x0, int y0, int z0, int size);

	// Clear cached data
	void clearCaches();

	// Convert grid position to world space
	glm::vec3 gridToWorld(const VoxelGrid& grid, int x, int y, int z);

	// Helper struct for point projection
	struct ProjectedPoint {
		float u, v;       // 2D coordinates
		float angle;      // Angle from center
		int index;        // Index in original array
	};

	std::vector<MCTriangle> simplifyMesh(const std::vector<MCTriangle>& input);

};
