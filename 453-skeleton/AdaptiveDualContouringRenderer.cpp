#include "AdaptiveDualContouringRenderer.h"
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/norm.hpp>
#include <iostream>
#include <limits>
#include <unordered_set>
#include <algorithm>
#include <cmath>

// Forward declaration of global octree map (from your existing code)
extern std::unordered_map<long long, OctreeNode*> g_octreeMap;

QEFSolver::QEFSolver() : ata(0.0f), atb(0.0f), btb(0.0f), pointSum(0.0f), numPoints(0) {
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

	// Add to Atb (negative because we're solving for: find p such that n·p + d = 0)
	atb.x += n.x * d;
	atb.y += n.y * d;
	atb.z += n.z * d;

	// Update btb
	btb += d * d;

	// Keep track of point average for masspoint fallback
	pointSum += point;
	numPoints++;
}

void QEFSolver::clear() {
	ata = glm::mat3(0.0f);
	atb = glm::vec3(0.0f);
	btb = 0.0f;
	pointSum = glm::vec3(0.0f);
	numPoints = 0;
}
// SVD decomposition implementation for 3x3 matrices
void QEFSolver::svdDecompose(const glm::mat3& A, glm::mat3& U, glm::vec3& sigma, glm::mat3& V) {
	// This is a simple implementation of SVD for 3x3 matrices
	// For a production environment, consider using a library like Eigen

	// First, compute A^T * A
	glm::mat3 ATA = glm::transpose(A) * A;

	// Find eigenvalues and eigenvectors of A^T * A
	// For a 3x3 symmetric matrix, we can use a specialized algorithm
	// or an iterative method like the power iteration

	// Placeholder: We'll use a simplified approach
	// In a real implementation, you would compute eigenvalues/vectors properly

	// For demonstration, we'll use the Jacobi eigenvalue algorithm
	const int MAX_ITERATIONS = 50;
	const float EPSILON = 1e-10f;

	// Start with V as identity
	V = glm::mat3(1.0f);
	glm::mat3 D = ATA;

	for (int iter = 0; iter < MAX_ITERATIONS; iter++) {
		// Find largest off-diagonal element
		float maxVal = 0.0f;
		int p = 0, q = 1;

		for (int i = 0; i < 3; i++) {
			for (int j = i + 1; j < 3; j++) {
				if (std::abs(D[i][j]) > maxVal) {
					maxVal = std::abs(D[i][j]);
					p = i;
					q = j;
				}
			}
		}

		// If matrix is already diagonal, we're done
		if (maxVal < EPSILON) {
			break;
		}

		// Compute Jacobi rotation
		float theta = 0.5f * std::atan2(2.0f * D[p][q], D[p][p] - D[q][q]);
		float c = std::cos(theta);
		float s = std::sin(theta);

		glm::mat3 J(1.0f);
		J[p][p] = c;
		J[p][q] = -s;
		J[q][p] = s;
		J[q][q] = c;

		// Update D and V
		D = glm::transpose(J) * D * J;
		V = V * J;
	}

	// Extract singular values (eigenvalues of A^T * A are squares of singular values)
	sigma.x = std::sqrt(std::max(0.0f, D[0][0]));
	sigma.y = std::sqrt(std::max(0.0f, D[1][1]));
	sigma.z = std::sqrt(std::max(0.0f, D[2][2]));

	// Handle tiny singular values
	const float MIN_SIGMA = 1e-6f;
	bool hasTinyValues = sigma.x < MIN_SIGMA || sigma.y < MIN_SIGMA || sigma.z < MIN_SIGMA;

	// Sort singular values in descending order
	if (sigma.x < sigma.y) {
		std::swap(sigma.x, sigma.y);
		std::swap(V[0], V[1]);
	}
	if (sigma.y < sigma.z) {
		std::swap(sigma.y, sigma.z);
		std::swap(V[1], V[2]);
	}
	if (sigma.x < sigma.y) {
		std::swap(sigma.x, sigma.y);
		std::swap(V[0], V[1]);
	}

	// Compute U = A * V * Sigma^-1
	U = glm::mat3(0.0f);
	for (int i = 0; i < 3; i++) {
		glm::vec3 AVi = A * glm::vec3(V[i]);
		if (sigma[i] > MIN_SIGMA) {
			U[i] = AVi / sigma[i];
		}
		else {
			// Handle the degenerate case
			U[i] = glm::normalize(glm::cross(V[(i + 1) % 3], V[(i + 2) % 3]));
		}
	}
}

bool QEFSolver::svdSolve(const glm::mat3& A, const glm::vec3& b, glm::vec3& x) {
	glm::mat3 U, V;
	glm::vec3 sigma;

	// Decompose A = U * Sigma * V^T
	svdDecompose(A, U, sigma, V);

	// Solve using SVD: x = V * Sigma^-1 * U^T * b
	glm::vec3 UTb = glm::transpose(U) * b;

	// Apply pseudo-inverse of Sigma
	glm::vec3 SigmaInvUTb(0.0f);
	const float MIN_SIGMA = 1e-6f;

	for (int i = 0; i < 3; i++) {
		if (sigma[i] > MIN_SIGMA) {
			SigmaInvUTb[i] = UTb[i] / sigma[i];
		}
		// else leave as zero
	}

	// Final multiplication with V
	x = V * SigmaInvUTb;

	return true;
}

glm::vec3 QEFSolver::solveSVD(const glm::vec3& cellCenter) {
	// Return center if no data
	if (numPoints == 0) {
		return cellCenter;
	}

	// Calculate masspoint as fallback
	glm::vec3 masspoint = pointSum / static_cast<float>(numPoints);

	// For a single point, return masspoint
	if (numPoints == 1) {
		return masspoint;
	}

	// Approach: Use eigenvalue decomposition of ata and a more stable solver

	// 1. Add regularization to ata to ensure it's positive definite
	const float EPSILON = 1e-6f;
	glm::mat3 regularizedA = ata + glm::mat3(EPSILON);

	// 2. Attempt to solve the regularized system
	try {
		glm::mat3 invA = glm::inverse(regularizedA);
		glm::vec3 solution = invA * atb;

		// Check solution validity
		if (!std::isnan(solution.x) && !std::isnan(solution.y) && !std::isnan(solution.z)) {
			float distSq = glm::distance2(solution, masspoint);
			const float MAX_DIST_SQ = 100.0f; // Larger threshold for initial testing

			if (distSq < MAX_DIST_SQ) {
				return solution;
			}
		}
	}
	catch (...) {
		// Inverse failed, continue to next approach
	}

	// 3. If direct approach fails, use iterative approach (conjugate gradient)
	glm::vec3 x = masspoint; // Initial guess
	glm::vec3 r = atb - regularizedA * x; // Residual
	glm::vec3 p = r; // Search direction
	float rsold = glm::dot(r, r);

	const int MAX_ITERATIONS = 20;
	const float CONVERGENCE_THRESHOLD = 1e-10f;

	for (int i = 0; i < MAX_ITERATIONS; i++) {
		glm::vec3 Ap = regularizedA * p;
		float alpha = rsold / glm::dot(p, Ap);

		x += alpha * p;
		r -= alpha * Ap;

		float rsnew = glm::dot(r, r);
		if (rsnew < CONVERGENCE_THRESHOLD) {
			break;
		}

		p = r + (rsnew / rsold) * p;
		rsold = rsnew;
	}

	// Check solution validity
	if (!std::isnan(x.x) && !std::isnan(x.y) && !std::isnan(x.z)) {
		float distSq = glm::distance2(x, masspoint);
		const float MAX_DIST_SQ = 100.0f;

		if (distSq < MAX_DIST_SQ) {
			return x;
		}
	}

	// If all else fails, return masspoint
	return masspoint;
}


float QEFSolver::calculateError(const glm::vec3& point) {
	float error = 0.0f;

	// p^T * A^T * A * p + 2 * p^T * A^T * b + b^T * b
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			error += point[i] * ata[i][j] * point[j];
		}
		error += 2.0f * point[i] * atb[i];
	}
	error += btb;

	return error;
}

bool QEFSolver::isNonSingular(const glm::mat3& mat) {
	// Check if the matrix is non-singular by evaluating the determinant
	float det = glm::determinant(mat);

	// Also check the condition number via eigenvalues (rough approximation)
	// Get the matrix diagonal and estimate condition
	float trace = mat[0][0] + mat[1][1] + mat[2][2];
	float maxDiag = std::max({ std::abs(mat[0][0]), std::abs(mat[1][1]), std::abs(mat[2][2]) });
	float minDiag = std::min({ std::abs(mat[0][0]), std::abs(mat[1][1]), std::abs(mat[2][2]) });

	// We want matrices with reasonable determinant and not too ill-conditioned
	return std::abs(det) > 1e-10f && (minDiag > 1e-6f * maxDiag);
}

glm::vec3 QEFSolver::solveConstrained(const glm::vec3& minBound, const glm::vec3& maxBound) {
	// Calculate cell center and size
	glm::vec3 cellCenter = (minBound + maxBound) * 0.5f;
	float cellSize = maxBound.x - minBound.x;

	// Try unconstrained solve first
	glm::vec3 solution = solve(cellCenter, cellSize);

	// Check if the solution is within bounds
	bool insideBounds = true;
	for (int i = 0; i < 3; i++) {
		if (solution[i] < minBound[i] || solution[i] > maxBound[i]) {
			insideBounds = false;
			break;
		}
	}

	if (insideBounds) {
		return solution;
	}

	// If the solution is outside the bounds, try a more nuanced approach
	// Project the solution onto the boundary of the volume

	// First approach: clamp to bounds
	glm::vec3 clampedSolution = glm::clamp(solution, minBound, maxBound);

	// Second approach: try the masspoint
	glm::vec3 masspoint = (numPoints > 0) ? pointSum / static_cast<float>(numPoints) : cellCenter;
	glm::vec3 clampedMasspoint = glm::clamp(masspoint, minBound, maxBound);

	// Choose the better of the two approaches by comparing QEF error
	float errorSolution = calculateError(clampedSolution);
	float errorMasspoint = calculateError(clampedMasspoint);

	if (errorMasspoint < errorSolution) {
		return clampedMasspoint;
	}

	return clampedSolution;
}

glm::vec3 QEFSolver::solve(const glm::vec3& cellCenter, float cellSize) {
	// Calculate masspoint as fallback
	glm::vec3 masspoint = (numPoints > 0) ? pointSum / static_cast<float>(numPoints) : cellCenter;

	// If there are no points or just one point, return masspoint
	if (numPoints <= 1) {
		return masspoint;
	}

	// First try with SVD (most accurate for well-conditioned systems)
	glm::vec3 svdSolution = solveSVD(cellCenter);

	// Check if SVD solution is valid and return it if so
	if (!std::isnan(svdSolution.x) && !std::isnan(svdSolution.y) && !std::isnan(svdSolution.z)) {
		float distSq = glm::distance2(svdSolution, masspoint);
		const float MAX_DIST_SQ = cellSize * cellSize * 2.0f;

		if (distSq < MAX_DIST_SQ) {
			return svdSolution;
		}
	}

	// If SVD fails or gives unreasonable results, try regularized approaches
	// Try direct solution with progressive regularization
	float minError = std::numeric_limits<float>::max();
	glm::vec3 bestSolution = masspoint;

	// Try a series of regularization strengths
	for (float lambda = 0.001f; lambda <= 10.0f; lambda *= 3.0f) {
		glm::mat3 regularizedA = ata;
		glm::vec3 regularizedB = atb;

		// Add regularization term (pull toward masspoint)
		regularizedA[0][0] += lambda;
		regularizedA[1][1] += lambda;
		regularizedA[2][2] += lambda;

		regularizedB.x += lambda * masspoint.x;
		regularizedB.y += lambda * masspoint.y;
		regularizedB.z += lambda * masspoint.z;

		if (isNonSingular(regularizedA)) {
			try {
				glm::mat3 invMat = glm::inverse(regularizedA);
				glm::vec3 solution = invMat * regularizedB;

				// Check if solution is valid
				if (!std::isnan(solution.x) && !std::isnan(solution.y) && !std::isnan(solution.z)) {
					// Calculate error and update best solution
					float error = calculateError(solution);
					if (error < minError) {
						minError = error;
						bestSolution = solution;
					}
				}
			}
			catch (...) {
				// Inverse failed, continue with next lambda
				continue;
			}
		}
	}

	// Safety check - if solution is too far from masspoint, use masspoint
	const float MAX_DISTANCE_SQ = cellSize * cellSize * 2.0f;
	if (glm::distance2(bestSolution, masspoint) > MAX_DISTANCE_SQ) {
		return masspoint;
	}

	return bestSolution;
}


// AdaptiveDualContouringRenderer implementation
AdaptiveDualContouringRenderer::AdaptiveDualContouringRenderer() {
}

AdaptiveDualContouringRenderer::~AdaptiveDualContouringRenderer() {
	clearCaches();
}

void AdaptiveDualContouringRenderer::clearCaches() {
	edgeIntersectionCache.clear();
	dualVertexCache.clear();
}

glm::vec3 AdaptiveDualContouringRenderer::gridToWorld(const VoxelGrid& grid, int x, int y, int z) {
	return glm::vec3(
		grid.minX + x * grid.voxelSize,
		grid.minY + y * grid.voxelSize,
		grid.minZ + z * grid.voxelSize
	);
}

HermitePoint AdaptiveDualContouringRenderer::calculateIntersection(
	const VoxelGrid& grid, int x1, int y1, int z1, int x2, int y2, int z2) {

	// Ensure we process edges in a consistent order (smaller index first)
	if (x1 > x2 || (x1 == x2 && y1 > y2) || (x1 == x2 && y1 == y2 && z1 > z2)) {
		std::swap(x1, x2);
		std::swap(y1, y2);
		std::swap(z1, z2);
	}

	// Create a key for this edge
	EdgeKey key = { x1, y1, z1, x2, y2, z2 };

	// Check if we've already computed this intersection
	auto it = edgeIntersectionCache.find(key);
	if (it != edgeIntersectionCache.end()) {
		return it->second;
	}

	// Safety bounds check
	if (x1 < 0 || y1 < 0 || z1 < 0 || x2 < 0 || y2 < 0 || z2 < 0 ||
		x1 >= grid.dimX || y1 >= grid.dimY || z1 >= grid.dimZ ||
		x2 >= grid.dimX || y2 >= grid.dimY || z2 >= grid.dimZ) {

		// Return a default hermite point if out of bounds
		HermitePoint defaultHP;
		defaultHP.position = gridToWorld(grid, (x1 + x2) / 2, (y1 + y2) / 2, (z1 + z2) / 2);
		defaultHP.normal = glm::normalize(glm::vec3(x2 - x1, y2 - y1, z2 - z1));
		return defaultHP;
	}

	// Get density values at endpoints
	bool isFilled1 = (grid.data[grid.index(x1, y1, z1)] == VoxelState::FILLED);
	bool isFilled2 = (grid.data[grid.index(x2, y2, z2)] == VoxelState::FILLED);

	// Convert to scalar field values (-1 inside, +1 outside)
	float v1 = isFilled1 ? -1.0f : 1.0f;
	float v2 = isFilled2 ? -1.0f : 1.0f;

	// Ensure we have a sign change (sanity check)
	if (v1 * v2 > 0) {
		// No intersection - this shouldn't happen if called properly
		// Return midpoint as fallback
		HermitePoint defaultHP;
		defaultHP.position = gridToWorld(grid, (x1 + x2) / 2, (y1 + y2) / 2, (z1 + z2) / 2);
		defaultHP.normal = glm::normalize(glm::vec3(x2 - x1, y2 - y1, z2 - z1));
		return defaultHP;
	}

	// Convert grid indices to world positions
	glm::vec3 p1 = gridToWorld(grid, x1, y1, z1);
	glm::vec3 p2 = gridToWorld(grid, x2, y2, z2);

	// Calculate interpolation parameter t
	float t = v1 / (v1 - v2);
	t = glm::clamp(t, 0.0f, 1.0f);  // Ensure t is between 0 and 1

	// Interpolate the position
	glm::vec3 position = p1 + t * (p2 - p1);

	// Determine the edge direction
	glm::vec3 edgeDir = glm::normalize(p2 - p1);

	// Determine which voxel contains the intersection point
	// This is important for getting a more accurate normal
	int ix, iy, iz;
	if (t <= 0.5f) {
		ix = x1;
		iy = y1;
		iz = z1;
	}
	else {
		ix = x2;
		iy = y2;
		iz = z2;
	}

	// Get normal at the intersection point using central differences
	glm::vec3 normal = estimateNormalRobust(grid, position, ix, iy, iz);

	// Ensure normal is perpendicular to the edge
	normal = normal - glm::dot(normal, edgeDir) * edgeDir;

	// Normalize the normal if it's not degenerate
	float normalLength = glm::length(normal);
	if (normalLength > 1e-6f) {
		normal /= normalLength;
	}
	else {
		// Fallback: create a normal perpendicular to the edge
		if (std::abs(edgeDir.x) < 0.8f) {
			normal = glm::normalize(glm::cross(edgeDir, glm::vec3(1.0f, 0.0f, 0.0f)));
		}
		else {
			normal = glm::normalize(glm::cross(edgeDir, glm::vec3(0.0f, 1.0f, 0.0f)));
		}
	}

	// Make sure normal points from filled to empty
	if ((isFilled1 && !isFilled2 && glm::dot(normal, p2 - p1) < 0) ||
		(!isFilled1 && isFilled2 && glm::dot(normal, p1 - p2) < 0)) {
		normal = -normal;
	}

	// Create and cache the hermite point
	HermitePoint hp = { position, normal };
	edgeIntersectionCache[key] = hp;

	return hp;
}

// An enhanced normal estimation function for better quality
glm::vec3 AdaptiveDualContouringRenderer::estimateNormalRobust(
	const VoxelGrid& grid,
	const glm::vec3& position,
	int centerX, int centerY, int centerZ) {

	// Calculate the grid position
	float invVoxelSize = 1.0f / grid.voxelSize;

	// Create a sampling function (safer than the previous version)
	auto sampleDensity = [&](int x, int y, int z) -> float {
		if (x < 0 || y < 0 || z < 0 || x >= grid.dimX || y >= grid.dimY || z >= grid.dimZ) {
			return 1.0f; // Outside grid is considered empty
		}
		return (grid.data[grid.index(x, y, z)] == VoxelState::FILLED) ? -1.0f : 1.0f;
		};

	// Use Sobel operators for better gradient estimation
	float dx = 0.0f, dy = 0.0f, dz = 0.0f;

	// Sobel X direction
	dx += sampleDensity(centerX + 1, centerY - 1, centerZ - 1) - sampleDensity(centerX - 1, centerY - 1, centerZ - 1);
	dx += 2.0f * (sampleDensity(centerX + 1, centerY, centerZ - 1) - sampleDensity(centerX - 1, centerY, centerZ - 1));
	dx += sampleDensity(centerX + 1, centerY + 1, centerZ - 1) - sampleDensity(centerX - 1, centerY + 1, centerZ - 1);

	dx += 2.0f * (sampleDensity(centerX + 1, centerY - 1, centerZ) - sampleDensity(centerX - 1, centerY - 1, centerZ));
	dx += 4.0f * (sampleDensity(centerX + 1, centerY, centerZ) - sampleDensity(centerX - 1, centerY, centerZ));
	dx += 2.0f * (sampleDensity(centerX + 1, centerY + 1, centerZ) - sampleDensity(centerX - 1, centerY + 1, centerZ));

	dx += sampleDensity(centerX + 1, centerY - 1, centerZ + 1) - sampleDensity(centerX - 1, centerY - 1, centerZ + 1);
	dx += 2.0f * (sampleDensity(centerX + 1, centerY, centerZ + 1) - sampleDensity(centerX - 1, centerY, centerZ + 1));
	dx += sampleDensity(centerX + 1, centerY + 1, centerZ + 1) - sampleDensity(centerX - 1, centerY + 1, centerZ + 1);

	// Sobel Y direction
	dy += sampleDensity(centerX - 1, centerY + 1, centerZ - 1) - sampleDensity(centerX - 1, centerY - 1, centerZ - 1);
	dy += 2.0f * (sampleDensity(centerX, centerY + 1, centerZ - 1) - sampleDensity(centerX, centerY - 1, centerZ - 1));
	dy += sampleDensity(centerX + 1, centerY + 1, centerZ - 1) - sampleDensity(centerX + 1, centerY - 1, centerZ - 1);

	dy += 2.0f * (sampleDensity(centerX - 1, centerY + 1, centerZ) - sampleDensity(centerX - 1, centerY - 1, centerZ));
	dy += 4.0f * (sampleDensity(centerX, centerY + 1, centerZ) - sampleDensity(centerX, centerY - 1, centerZ));
	dy += 2.0f * (sampleDensity(centerX + 1, centerY + 1, centerZ) - sampleDensity(centerX + 1, centerY - 1, centerZ));

	dy += sampleDensity(centerX - 1, centerY + 1, centerZ + 1) - sampleDensity(centerX - 1, centerY - 1, centerZ + 1);
	dy += 2.0f * (sampleDensity(centerX, centerY + 1, centerZ + 1) - sampleDensity(centerX, centerY - 1, centerZ + 1));
	dy += sampleDensity(centerX + 1, centerY + 1, centerZ + 1) - sampleDensity(centerX + 1, centerY - 1, centerZ + 1);

	// Sobel Z direction
	dz += sampleDensity(centerX - 1, centerY - 1, centerZ + 1) - sampleDensity(centerX - 1, centerY - 1, centerZ - 1);
	dz += 2.0f * (sampleDensity(centerX, centerY - 1, centerZ + 1) - sampleDensity(centerX, centerY - 1, centerZ - 1));
	dz += sampleDensity(centerX + 1, centerY - 1, centerZ + 1) - sampleDensity(centerX + 1, centerY - 1, centerZ - 1);

	dz += 2.0f * (sampleDensity(centerX - 1, centerY, centerZ + 1) - sampleDensity(centerX - 1, centerY, centerZ - 1));
	dz += 4.0f * (sampleDensity(centerX, centerY, centerZ + 1) - sampleDensity(centerX, centerY, centerZ - 1));
	dz += 2.0f * (sampleDensity(centerX + 1, centerY, centerZ + 1) - sampleDensity(centerX + 1, centerY, centerZ - 1));

	dz += sampleDensity(centerX - 1, centerY + 1, centerZ + 1) - sampleDensity(centerX - 1, centerY + 1, centerZ - 1);
	dz += 2.0f * (sampleDensity(centerX, centerY + 1, centerZ + 1) - sampleDensity(centerX, centerY + 1, centerZ - 1));
	dz += sampleDensity(centerX + 1, centerY + 1, centerZ + 1) - sampleDensity(centerX + 1, centerY + 1, centerZ - 1);

	// Normalize the weights
	dx /= 32.0f;
	dy /= 32.0f;
	dz /= 32.0f;

	// Create the normal (gradient points from low to high values, but we want normals pointing outward)
	glm::vec3 normal(dx, dy, dz);

	// Handle degenerate case
	if (glm::length2(normal) < 1e-6f) {
		// If we can't estimate a good normal, return a default based on the closest axis
		int axis = 0;
		float maxDist = std::abs(position[0] - (grid.minX + centerX * grid.voxelSize));
		float dist = std::abs(position[1] - (grid.minY + centerY * grid.voxelSize));
		if (dist > maxDist) {
			maxDist = dist;
			axis = 1;
		}
		dist = std::abs(position[2] - (grid.minZ + centerZ * grid.voxelSize));
		if (dist > maxDist) {
			axis = 2;
		}

		normal = glm::vec3(0.0f);
		normal[axis] = 1.0f;

		// Check if we need to invert it
		if (sampleDensity(centerX, centerY, centerZ) < 0) {
			normal = -normal;
		}
	}

	return glm::normalize(normal);
}

glm::vec3 AdaptiveDualContouringRenderer::estimateNormal(
	const VoxelGrid& grid, const glm::vec3& position, float voxelSize) {

	// Calculate the grid position
	float invVoxelSize = 1.0f / voxelSize;
	int gx = static_cast<int>((position.x - grid.minX) * invVoxelSize);
	int gy = static_cast<int>((position.y - grid.minY) * invVoxelSize);
	int gz = static_cast<int>((position.z - grid.minZ) * invVoxelSize);

	// Clamp to valid grid indices
	gx = std::max(0, std::min(gx, grid.dimX - 1));
	gy = std::max(0, std::min(gy, grid.dimY - 1));
	gz = std::max(0, std::min(gz, grid.dimZ - 1));

	// Use central differences to estimate the gradient/normal
	auto sampleDensity = [&](int x, int y, int z) -> float {
		if (x < 0 || y < 0 || z < 0 || x >= grid.dimX || y >= grid.dimY || z >= grid.dimZ) {
			return 1.0f; // Outside grid is considered empty
		}
		return (grid.data[grid.index(x, y, z)] == VoxelState::FILLED) ? -1.0f : 1.0f;
		};

	// Compute gradient using central differences where possible, 
	// falling back to forward/backward differences at boundaries
	float dx, dy, dz;

	// X component
	if (gx > 0 && gx < grid.dimX - 1) {
		dx = sampleDensity(gx + 1, gy, gz) - sampleDensity(gx - 1, gy, gz);
	}
	else if (gx > 0) {
		dx = sampleDensity(gx, gy, gz) - sampleDensity(gx - 1, gy, gz);
	}
	else {
		dx = sampleDensity(gx + 1, gy, gz) - sampleDensity(gx, gy, gz);
	}

	// Y component
	if (gy > 0 && gy < grid.dimY - 1) {
		dy = sampleDensity(gx, gy + 1, gz) - sampleDensity(gx, gy - 1, gz);
	}
	else if (gy > 0) {
		dy = sampleDensity(gx, gy, gz) - sampleDensity(gx, gy - 1, gz);
	}
	else {
		dy = sampleDensity(gx, gy + 1, gz) - sampleDensity(gx, gy, gz);
	}

	// Z component
	if (gz > 0 && gz < grid.dimZ - 1) {
		dz = sampleDensity(gx, gy, gz + 1) - sampleDensity(gx, gy, gz - 1);
	}
	else if (gz > 0) {
		dz = sampleDensity(gx, gy, gz) - sampleDensity(gx, gy, gz - 1);
	}
	else {
		dz = sampleDensity(gx, gy, gz + 1) - sampleDensity(gx, gy, gz);
	}

	// Create the normal
	glm::vec3 normal(dx, dy, dz);

	// Handle degenerate case
	if (glm::length2(normal) < 1e-6f) {
		// If we can't estimate a good normal, use a default based on edge direction
		normal = glm::vec3(1.0f, 1.0f, 1.0f);
	}

	return glm::normalize(normal);
}

glm::vec3 AdaptiveDualContouringRenderer::generateDualVertex(
	const std::vector<HermitePoint>& hermiteData,
	const glm::vec3& cellCenter, float cellSize) {

	// If no hermite data, return cell center
	if (hermiteData.empty()) {
		return cellCenter;
	}

	// For urban buildings, prioritize axis-aligned surfaces and sharp edges
	glm::vec3 avgNormal(0.0f);
	glm::vec3 avgPos(0.0f);

	for (const auto& hp : hermiteData) {
		avgNormal += hp.normal;
		avgPos += hp.position;
	}

	avgPos /= float(hermiteData.size());

	// Check if we have a consistent normal direction (urban buildings typically have flat surfaces)
	if (glm::length(avgNormal) > 1e-6f) {
		avgNormal = glm::normalize(avgNormal);

		// Check if normal is strongly aligned with an axis
		glm::vec3 absNormal = glm::abs(avgNormal);
		float maxComp = std::max({ absNormal.x, absNormal.y, absNormal.z });

		// If this is likely a flat building surface (strongly axis-aligned normal)
		if (maxComp > 0.85f) {
			// For urban buildings, we want vertices to lie on the actual surface
			// Calculate the plane equation: dot(normal, X) + d = 0
			float d = -glm::dot(avgNormal, avgPos);

			// Find intersection of this plane with the center of the cell
			float t = -(glm::dot(avgNormal, cellCenter) + d) / glm::dot(avgNormal, avgNormal);
			glm::vec3 projectedCenter = cellCenter + t * avgNormal;

			// Make sure the projected point is within the cell
			glm::vec3 halfSize(cellSize * 0.5f);
			glm::vec3 minBound = cellCenter - halfSize;
			glm::vec3 maxBound = cellCenter + halfSize;

			// Add a small inset to avoid boundary issues
			float inset = cellSize * 0.001f;
			minBound += glm::vec3(inset);
			maxBound -= glm::vec3(inset);

			// Clamp the projected point to the cell bounds
			glm::vec3 clampedPos = glm::clamp(projectedCenter, minBound, maxBound);

			// For urban buildings, we can slightly move vertices toward cell edges
			// to create sharper building features
			if (maxComp > 0.95f) { // Very strong axis alignment
				// Determine which axis is dominant
				int dominantAxis = 0;
				if (absNormal.y > absNormal.x && absNormal.y > absNormal.z) dominantAxis = 1;
				else if (absNormal.z > absNormal.x && absNormal.z > absNormal.y) dominantAxis = 2;

				// Check which cell edge/face the vertex should be pulled toward
				// Calculate distance to cell boundaries along other two axes
				int axis1 = (dominantAxis + 1) % 3;
				int axis2 = (dominantAxis + 2) % 3;

				float distToMin1 = clampedPos[axis1] - minBound[axis1];
				float distToMax1 = maxBound[axis1] - clampedPos[axis1];
				float distToMin2 = clampedPos[axis2] - minBound[axis2];
				float distToMax2 = maxBound[axis2] - clampedPos[axis2];

				// Find closest boundary in each direction
				float minDist1 = std::min(distToMin1, distToMax1);
				float minDist2 = std::min(distToMin2, distToMax2);

				// If we're close to a boundary, move toward it
				float snapThreshold = cellSize * 0.3f;

				if (minDist1 < snapThreshold) {
					if (distToMin1 < distToMax1) {
						clampedPos[axis1] = minBound[axis1];
					}
					else {
						clampedPos[axis1] = maxBound[axis1];
					}
				}

				if (minDist2 < snapThreshold) {
					if (distToMin2 < distToMax2) {
						clampedPos[axis2] = minBound[axis2];
					}
					else {
						clampedPos[axis2] = maxBound[axis2];
					}
				}
			}

			return clampedPos;
		}
	}

	// For cases that don't match urban building patterns,
	// fall back to QEF solver
	QEFSolver qef;
	for (const auto& hp : hermiteData) {
		qef.addPoint(hp.position, hp.normal);
	}

	// Create cell bounds with inset
	glm::vec3 halfSize(cellSize * 0.5f);
	glm::vec3 minBound = cellCenter - halfSize;
	glm::vec3 maxBound = cellCenter + halfSize;

	float inset = cellSize * 0.001f;
	minBound += glm::vec3(inset);
	maxBound -= glm::vec3(inset);

	// Solve QEF with constraints to keep vertex inside cell
	return qef.solveConstrained(minBound, maxBound);
}

bool AdaptiveDualContouringRenderer::cellContainsSurface(
	const VoxelGrid& grid, int x0, int y0, int z0, int size) {

	// Boundary checking
	int maxX = std::min(x0 + size, grid.dimX - 1);
	int maxY = std::min(y0 + size, grid.dimY - 1);
	int maxZ = std::min(z0 + size, grid.dimZ - 1);

	int minX = std::max(x0, 0);
	int minY = std::max(y0, 0);
	int minZ = std::max(z0, 0);

	// If the boundaries are invalid, return false
	if (minX > maxX || minY > maxY || minZ > maxZ) {
		return false;
	}

	// Look for sign changes within the cell
	bool hasInside = false;
	bool hasOutside = false;

	for (int z = minZ; z <= maxZ; z++) {
		for (int y = minY; y <= maxY; y++) {
			for (int x = minX; x <= maxX; x++) {
				bool isFilled = (grid.data[grid.index(x, y, z)] == VoxelState::FILLED);

				if (isFilled) hasInside = true;
				if (!isFilled) hasOutside = true;

				// Early exit if we found both inside and outside
				if (hasInside && hasOutside) return true;
			}
		}
	}

	return hasInside && hasOutside;
}

// Helper function to calculate triangle area
float calculateTriangleArea(const MCTriangle& tri) {
	glm::vec3 e1 = tri.v[1] - tri.v[0];
	glm::vec3 e2 = tri.v[2] - tri.v[0];
	return 0.5f * glm::length(glm::cross(e1, e2));
}

std::vector<MCTriangle> AdaptiveDualContouringRenderer::createFaceTriangles(
	const glm::vec3& cellVertex,
	const DualFace& face,
	bool isInsideCell) {

	std::vector<MCTriangle> triangles;

	// If there are no hermite points, we can't create meaningful triangles
	if (face.hermitePoints.empty()) {
		return triangles;
	}

	// For urban buildings, we want to create clean quad-like structures

	// 1. Calculate a proper face normal (not just from vertex2 to vertex1)
	glm::vec3 faceNormal(0.0f);
	for (const auto& hp : face.hermitePoints) {
		faceNormal += hp.normal;
	}

	// If we don't have enough information from hermite points, use the vertices
	if (glm::length(faceNormal) < 1e-6f) {
		// Calculate normal from the two dual vertices and an edge
		// This assumes the face is roughly planar
		glm::vec3 edge = face.vertex2 - face.vertex1;
		if (glm::length(edge) > 1e-6f) {
			// Find a perpendicular vector to create a face plane
			glm::vec3 perpVector;
			if (std::abs(edge.x) < std::abs(edge.y) && std::abs(edge.x) < std::abs(edge.z)) {
				perpVector = glm::vec3(1.0f, 0.0f, 0.0f);
			}
			else if (std::abs(edge.y) < std::abs(edge.z)) {
				perpVector = glm::vec3(0.0f, 1.0f, 0.0f);
			}
			else {
				perpVector = glm::vec3(0.0f, 0.0f, 1.0f);
			}

			faceNormal = glm::normalize(glm::cross(edge, perpVector));
		}
		else {
			// Default fallback if we can't determine a proper normal
			faceNormal = glm::vec3(0.0f, 1.0f, 0.0f);
		}
	}
	else {
		faceNormal = glm::normalize(faceNormal);
	}

	// Ensure normal points from inside to outside
	if (isInsideCell) {
		faceNormal = -faceNormal;
	}

	// 2. For urban buildings, let's create a clean quadrilateral if possible

	// First approach: try to identify a clean quad by finding corners
	if (face.hermitePoints.size() >= 3) {
		// Find axis-aligned bounding box of hermite points in face plane
		// Create a coordinate system on the face plane
		glm::vec3 u, v;

		// Find two perpendicular vectors on the plane
		if (std::abs(faceNormal.x) > std::abs(faceNormal.y) &&
			std::abs(faceNormal.x) > std::abs(faceNormal.z)) {
			// X is dominant normal component
			u = glm::normalize(glm::cross(faceNormal, glm::vec3(0, 0, 1)));
		}
		else if (std::abs(faceNormal.y) > std::abs(faceNormal.z)) {
			// Y is dominant normal component
			u = glm::normalize(glm::cross(faceNormal, glm::vec3(1, 0, 0)));
		}
		else {
			// Z is dominant normal component
			u = glm::normalize(glm::cross(faceNormal, glm::vec3(0, 1, 0)));
		}

		v = glm::normalize(glm::cross(faceNormal, u));

		// Calculate center of hermite points
		glm::vec3 center(0.0f);
		for (const auto& hp : face.hermitePoints) {
			center += hp.position;
		}
		center /= float(face.hermitePoints.size());

		// Project points to 2D and find min/max in u and v directions
		float minU = std::numeric_limits<float>::max();
		float maxU = -std::numeric_limits<float>::max();
		float minV = std::numeric_limits<float>::max();
		float maxV = -std::numeric_limits<float>::max();

		for (const auto& hp : face.hermitePoints) {
			glm::vec3 relPos = hp.position - center;
			float projU = glm::dot(relPos, u);
			float projV = glm::dot(relPos, v);

			minU = std::min(minU, projU);
			maxU = std::max(maxU, projU);
			minV = std::min(minV, projV);
			maxV = std::max(maxV, projV);
		}

		// Create four corners from the bounding box
		glm::vec3 corners[4];
		corners[0] = center + minU * u + minV * v; // Bottom left
		corners[1] = center + maxU * u + minV * v; // Bottom right
		corners[2] = center + maxU * u + maxV * v; // Top right
		corners[3] = center + minU * u + maxV * v; // Top left

		// Create clean quad from corners (two triangles)
		MCTriangle tri1, tri2;

		tri1.v[0] = corners[0];
		tri1.v[1] = corners[1];
		tri1.v[2] = corners[2];

		tri2.v[0] = corners[0];
		tri2.v[1] = corners[2];
		tri2.v[2] = corners[3];

		// Set uniform normals for both triangles
		for (int i = 0; i < 3; i++) {
			tri1.normal[i] = faceNormal;
			tri2.normal[i] = faceNormal;
		}

		// Check if triangles are valid (non-degenerate)
		float area1 = calculateTriangleArea(tri1);
		float area2 = calculateTriangleArea(tri2);

		if (area1 > 1e-6f) {
			triangles.push_back(tri1);
		}

		if (area2 > 1e-6f) {
			triangles.push_back(tri2);
		}

		// If we successfully created triangles, return them
		if (!triangles.empty()) {
			return triangles;
		}
	}

	// 3. Fallback: simpler approach for cases where the quad technique fails
	// Just connect cell vertices to form a simple triangulated face

	// Calculate center of hermite points for a better reference point
	glm::vec3 faceCenter(0.0f);
	for (const auto& hp : face.hermitePoints) {
		faceCenter += hp.position;
	}
	faceCenter /= float(face.hermitePoints.size());

	// Create two triangles using the dual vertices and face center
	MCTriangle tri1, tri2;

	tri1.v[0] = cellVertex;
	tri1.v[1] = face.vertex2;
	tri1.v[2] = faceCenter;

	tri2.v[0] = cellVertex;
	tri2.v[1] = faceCenter;
	tri2.v[2] = face.vertex1;

	// Set consistent normals
	for (int i = 0; i < 3; i++) {
		tri1.normal[i] = faceNormal;
		tri2.normal[i] = faceNormal;
	}

	// Check if triangles are valid before adding
	if (calculateTriangleArea(tri1) > 1e-6f) {
		triangles.push_back(tri1);
	}

	if (calculateTriangleArea(tri2) > 1e-6f) {
		triangles.push_back(tri2);
	}

	return triangles;
}

std::vector<MCTriangle> AdaptiveDualContouringRenderer::createTriangles(
	const VoxelGrid& grid, const OctreeNode* node, int x0, int y0, int z0, int size) {

	std::vector<MCTriangle> triangles;

	// Only process leaf nodes
	if (!node || !node->isLeaf) {
		return triangles;
	}

	// If the node doesn't contain the surface, no triangles to generate
	if (!cellContainsSurface(grid, x0, y0, z0, size)) {
		return triangles;
	}

	// Gather hermite data for this cell
	std::vector<HermitePoint> hermiteData = gatherHermiteData(grid, x0, y0, z0, size);

	// If no hermite data, no triangles to create
	if (hermiteData.empty()) {
		return triangles;
	}

	// Cell center in world space
	glm::vec3 cellCenter = gridToWorld(grid, x0, y0, z0) +
		glm::vec3(size * 0.5f * grid.voxelSize);

	// Create or retrieve the dual vertex for this cell
	long long cellKey = ((long long)x0 << 20) | ((long long)y0 << 10) | (long long)z0;

	if (dualVertexCache.find(cellKey) == dualVertexCache.end()) {
		dualVertexCache[cellKey] = generateDualVertex(hermiteData, cellCenter, size * grid.voxelSize);
	}

	glm::vec3 cellVertex = dualVertexCache[cellKey];

	// Enhanced edge processing for dual contouring
	// The 12 edges of a cell are:
	// X-direction edges (4): (0,0,0)-(1,0,0), (0,1,0)-(1,1,0), (0,0,1)-(1,0,1), (0,1,1)-(1,1,1)
	// Y-direction edges (4): (0,0,0)-(0,1,0), (1,0,0)-(1,1,0), (0,0,1)-(0,1,1), (1,0,1)-(1,1,1)
	// Z-direction edges (4): (0,0,0)-(0,0,1), (1,0,0)-(1,0,1), (0,1,0)-(0,1,1), (1,1,0)-(1,1,1)

	// We'll check all edges that belong to this cell
	const int edgeStartX[12] = { 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1 };
	const int edgeStartY[12] = { 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0 };
	const int edgeStartZ[12] = { 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1 };

	const int edgeEndX[12] = { 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1 };
	const int edgeEndY[12] = { 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1 };
	const int edgeEndZ[12] = { 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1 };

	// Direction of each edge (0=X, 1=Y, 2=Z)
	const int edgeDir[12] = { 0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 1, 1 };

	// Define cell offsets for each edge direction (X, Y, Z)
	// These represent the relative positions of the 4 cells that share an edge
	const int cellOffsetsX[4][3] = {
		{0, 0, 0}, // Reference cell
		{0, -1, 0}, // Cell below
		{0, 0, -1}, // Cell behind
		{0, -1, -1} // Cell below and behind
	};

	const int cellOffsetsY[4][3] = {
		{0, 0, 0}, // Reference cell
		{-1, 0, 0}, // Cell to left
		{0, 0, -1}, // Cell behind
		{-1, 0, -1} // Cell to left and behind
	};

	const int cellOffsetsZ[4][3] = {
		{0, 0, 0}, // Reference cell
		{-1, 0, 0}, // Cell to left
		{0, -1, 0}, // Cell below
		{-1, -1, 0} // Cell to left and below
	};

	for (int e = 0; e < 12; e++) {
		// Get actual grid coordinates of the edge endpoints
		int x1 = x0 + edgeStartX[e] * size;
		int y1 = y0 + edgeStartY[e] * size;
		int z1 = z0 + edgeStartZ[e] * size;

		int x2 = x0 + edgeEndX[e] * size;
		int y2 = y0 + edgeEndY[e] * size;
		int z2 = z0 + edgeEndZ[e] * size;

		// Check if edge endpoints are in bounds
		if (x1 < 0 || y1 < 0 || z1 < 0 || x2 < 0 || y2 < 0 || z2 < 0 ||
			x1 >= grid.dimX || y1 >= grid.dimY || z1 >= grid.dimZ ||
			x2 >= grid.dimX || y2 >= grid.dimY || z2 >= grid.dimZ) {
			continue;
		}

		// Check for sign change along this edge
		VoxelState state1 = grid.data[grid.index(x1, y1, z1)];
		VoxelState state2 = grid.data[grid.index(x2, y2, z2)];

		if (state1 == state2) {
			continue; // No sign change, skip this edge
		}

		// For this edge, find the four cells sharing it
		std::vector<glm::vec3> cellVertices;
		std::vector<bool> cellInside; // Track if cell is inside or outside

		const int (*offsets)[3];
		if (edgeDir[e] == 0) offsets = cellOffsetsX;
		else if (edgeDir[e] == 1) offsets = cellOffsetsY;
		else offsets = cellOffsetsZ;

		for (int c = 0; c < 4; c++) {
			int cx = x1 + offsets[c][0] * size;
			int cy = y1 + offsets[c][1] * size;
			int cz = z1 + offsets[c][2] * size;

			// Check if cell is in bounds
			if (cx < 0 || cy < 0 || cz < 0 ||
				cx >= grid.dimX || cy >= grid.dimY || cz >= grid.dimZ) {
				continue;
			}

			long long cellId = ((long long)cx << 20) | ((long long)cy << 10) | (long long)cz;
			auto nodeIt = g_octreeMap.find(cellId);

			if (nodeIt == g_octreeMap.end() || !nodeIt->second->isLeaf) {
				continue;
			}

			// Find or compute the dual vertex
			glm::vec3 vertex;
			if (dualVertexCache.find(cellId) != dualVertexCache.end()) {
				vertex = dualVertexCache[cellId];
			}
			else {
				glm::vec3 center = gridToWorld(grid, cx, cy, cz) +
					glm::vec3(nodeIt->second->size * 0.5f * grid.voxelSize);

				std::vector<HermitePoint> cellHermite = gatherHermiteData(
					grid, cx, cy, cz, nodeIt->second->size);

				vertex = generateDualVertex(cellHermite, center,
					nodeIt->second->size * grid.voxelSize);

				dualVertexCache[cellId] = vertex;
			}

			cellVertices.push_back(vertex);
			cellInside.push_back(nodeIt->second->isSolid);
		}

		// Need at least 3 vertices to form a polygon
		if (cellVertices.size() < 3) {
			continue;
		}

		// For urban scenes, creating a consistent polygon is important
		// Get edge direction vector
		glm::vec3 edgeVector;
		if (edgeDir[e] == 0) edgeVector = glm::vec3(1, 0, 0);
		else if (edgeDir[e] == 1) edgeVector = glm::vec3(0, 1, 0);
		else edgeVector = glm::vec3(0, 0, 1);

		// Find two perpendicular axes to create a consistent ordering
		glm::vec3 perpAxis1, perpAxis2;
		if (edgeDir[e] == 0) {
			perpAxis1 = glm::vec3(0, 1, 0);
			perpAxis2 = glm::vec3(0, 0, 1);
		}
		else if (edgeDir[e] == 1) {
			perpAxis1 = glm::vec3(1, 0, 0);
			perpAxis2 = glm::vec3(0, 0, 1);
		}
		else {
			perpAxis1 = glm::vec3(1, 0, 0);
			perpAxis2 = glm::vec3(0, 1, 0);
		}

		// Calculate edge midpoint
		glm::vec3 midpoint = (
			gridToWorld(grid, x1, y1, z1) +
			gridToWorld(grid, x2, y2, z2)
			) * 0.5f;

		// Create a 2D ordering of vertices around the edge
		std::vector<std::pair<float, int>> vertexAngles;
		for (size_t i = 0; i < cellVertices.size(); i++) {
			// Project to plane perpendicular to edge
			glm::vec3 relPos = cellVertices[i] - midpoint;
			float dot1 = glm::dot(relPos, perpAxis1);
			float dot2 = glm::dot(relPos, perpAxis2);

			// Calculate angle in this plane
			float angle = std::atan2(dot2, dot1);
			vertexAngles.push_back({ angle, i });
		}

		// Sort vertices by angle
		std::sort(vertexAngles.begin(), vertexAngles.end());

		// Create ordered vertices
		std::vector<glm::vec3> orderedVertices;
		std::vector<bool> orderedInside;
		for (const auto& pair : vertexAngles) {
			orderedVertices.push_back(cellVertices[pair.second]);
			orderedInside.push_back(cellInside[pair.second]);
		}

		// Create triangles from ordered vertices
		bool isFirstCellInside = (state1 == VoxelState::FILLED);

		for (size_t i = 1; i < orderedVertices.size() - 1; i++) {
			MCTriangle tri;
			tri.v[0] = orderedVertices[0];
			tri.v[1] = orderedVertices[i];
			tri.v[2] = orderedVertices[i + 1];

			// Compute normal from triangle
			glm::vec3 edge1 = tri.v[1] - tri.v[0];
			glm::vec3 edge2 = tri.v[2] - tri.v[0];
			glm::vec3 normal = glm::normalize(glm::cross(edge1, edge2));

			// Ensure normal points outward
			if (isFirstCellInside) {
				normal = -normal;
			}

			for (int j = 0; j < 3; j++) {
				tri.normal[j] = normal;
			}

			// Before adding triangle, make sure it's not degenerate
			float area = 0.5f * glm::length(glm::cross(edge1, edge2));
			if (area > 1e-6f) {
				triangles.push_back(tri);
			}
		}
	}

	return triangles;
}

std::vector<MCTriangle> AdaptiveDualContouringRenderer::render(
	const OctreeNode* node,
	const VoxelGrid& grid,
	int x0, int y0, int z0, int size) {

	std::vector<MCTriangle> triangles;

	// Clear caches at the top level to avoid memory growth between frames
	if (x0 == 0 && y0 == 0 && z0 == 0 && size == grid.dimX) {
		clearCaches();
	}

	// If null node, return empty result
	if (!node) return triangles;

	// Process leaf nodes
	if (node->isLeaf) {
		// Check for surface intersection
		std::vector<MCTriangle> cellTriangles = createTriangles(grid, node, x0, y0, z0, size);
		triangles.insert(triangles.end(), cellTriangles.begin(), cellTriangles.end());
	}
	// For non-leaf nodes, recursively process children
	else {
		int halfSize = size / 2;
		for (int i = 0; i < 8; i++) {
			if (node->children[i]) {
				int childX = x0 + ((i & 1) ? halfSize : 0);
				int childY = y0 + ((i & 2) ? halfSize : 0);
				int childZ = z0 + ((i & 4) ? halfSize : 0);

				std::vector<MCTriangle> childTriangles = render(
					node->children[i], grid, childX, childY, childZ, halfSize);

				triangles.insert(triangles.end(), childTriangles.begin(), childTriangles.end());
			}
		}
	}

	// at the top level, apply post-processing to improve mesh quality
	/*if (x0 == 0 && y0 == 0 && z0 == 0 && size == grid.dimX) {
		triangles = simplifyMesh(triangles);
	}*/

	return triangles;
}

std::vector<MCTriangle> AdaptiveDualContouringRenderer::simplifyMesh(const std::vector<MCTriangle>& input) {
	if (input.empty()) return input;

	// First pass: filter out tiny and degenerate triangles
	std::vector<MCTriangle> filtered;
	filtered.reserve(input.size());

	const float MIN_AREA = 1e-6f;
	const float MIN_EDGE_LENGTH = 1e-4f;

	for (const auto& tri : input) {
		// Calculate triangle properties
		glm::vec3 e1 = tri.v[1] - tri.v[0];
		glm::vec3 e2 = tri.v[2] - tri.v[0];
		glm::vec3 e3 = tri.v[2] - tri.v[1];

		float area = 0.5f * glm::length(glm::cross(e1, e2));

		// Skip triangles that are too small
		if (area < MIN_AREA) continue;

		// Skip triangles with very short edges
		if (glm::length(e1) < MIN_EDGE_LENGTH ||
			glm::length(e2) < MIN_EDGE_LENGTH ||
			glm::length(e3) < MIN_EDGE_LENGTH) {
			continue;
		}

		// Keep this triangle
		filtered.push_back(tri);
	}

	// Second pass: ensure consistent orientation
	for (auto& tri : filtered) {
		// Make sure normal is unit length
		for (int i = 0; i < 3; i++) {
			tri.normal[i] = glm::normalize(tri.normal[i]);
		}

		// Check winding matches normal
		glm::vec3 e1 = tri.v[1] - tri.v[0];
		glm::vec3 e2 = tri.v[2] - tri.v[0];
		glm::vec3 faceNormal = glm::normalize(glm::cross(e1, e2));

		// If winding is inconsistent with normal, swap vertices
		if (glm::dot(faceNormal, tri.normal[0]) < 0) {
			std::swap(tri.v[1], tri.v[2]);
		}
	}

	// Third pass: subdivide large triangles
	std::vector<MCTriangle> subdivided;
	subdivided.reserve(filtered.size() * 2); // Reserve space for potential growth

	const float MAX_EDGE_LENGTH = 40.0f; // Adjust based on your scene scale
	const float MAX_AREA = 800.0f; // Adjust based on your scene scale

	for (const auto& tri : filtered) {
		glm::vec3 e1 = tri.v[1] - tri.v[0];
		glm::vec3 e2 = tri.v[2] - tri.v[0];
		glm::vec3 e3 = tri.v[2] - tri.v[1];

		float edgeLen1 = glm::length(e1);
		float edgeLen2 = glm::length(e2);
		float edgeLen3 = glm::length(e3);

		float area = 0.5f * glm::length(glm::cross(e1, e2));

		// Check if this triangle needs subdivision
		bool needsSubdivision = (edgeLen1 > MAX_EDGE_LENGTH ||
			edgeLen2 > MAX_EDGE_LENGTH ||
			edgeLen3 > MAX_EDGE_LENGTH ||
			area > MAX_AREA);

		if (needsSubdivision) {
			// Find the longest edge
			int longestEdge = 0;
			float maxLength = edgeLen1;

			if (edgeLen2 > maxLength) {
				longestEdge = 1;
				maxLength = edgeLen2;
			}

			if (edgeLen3 > maxLength) {
				longestEdge = 2;
			}

			// Create midpoint on the longest edge
			glm::vec3 midpoint;
			if (longestEdge == 0) {
				midpoint = (tri.v[0] + tri.v[1]) * 0.5f;
			}
			else if (longestEdge == 1) {
				midpoint = (tri.v[0] + tri.v[2]) * 0.5f;
			}
			else { // longestEdge == 2
				midpoint = (tri.v[1] + tri.v[2]) * 0.5f;
			}

			// Create two new triangles by splitting at midpoint
			MCTriangle newTri1, newTri2;

			if (longestEdge == 0) {
				// Split edge between v0 and v1
				newTri1.v[0] = midpoint;
				newTri1.v[1] = tri.v[1];
				newTri1.v[2] = tri.v[2];

				newTri2.v[0] = midpoint;
				newTri2.v[1] = tri.v[2];
				newTri2.v[2] = tri.v[0];
			}
			else if (longestEdge == 1) {
				// Split edge between v0 and v2
				newTri1.v[0] = tri.v[0];
				newTri1.v[1] = tri.v[1];
				newTri1.v[2] = midpoint;

				newTri2.v[0] = midpoint;
				newTri2.v[1] = tri.v[1];
				newTri2.v[2] = tri.v[2];
			}
			else { // longestEdge == 2
				// Split edge between v1 and v2
				newTri1.v[0] = tri.v[0];
				newTri1.v[1] = tri.v[1];
				newTri1.v[2] = midpoint;

				newTri2.v[0] = tri.v[0];
				newTri2.v[1] = midpoint;
				newTri2.v[2] = tri.v[2];
			}

			// Inherit normals from original triangle
			for (int i = 0; i < 3; i++) {
				newTri1.normal[i] = tri.normal[i];
				newTri2.normal[i] = tri.normal[i];
			}

			// Recursively subdivide these new triangles if needed
			// (For simplicity, limit to a single subdivision here)
			subdivided.push_back(newTri1);
			subdivided.push_back(newTri2);
		}
		else {
			// Triangle doesn't need subdivision
			subdivided.push_back(tri);
		}
	}

	// Optional: Apply additional refinement to urban structures
	// Look for long thin triangles that might be part of building facades
	std::vector<MCTriangle> refined;
	refined.reserve(subdivided.size());

	const float ASPECT_RATIO_THRESHOLD = 8.0f; // High aspect ratio triangles

	for (const auto& tri : subdivided) {
		glm::vec3 e1 = tri.v[1] - tri.v[0];
		glm::vec3 e2 = tri.v[2] - tri.v[0];
		glm::vec3 e3 = tri.v[2] - tri.v[1];

		// Calculate longest and shortest edges
		float len1 = glm::length(e1);
		float len2 = glm::length(e2);
		float len3 = glm::length(e3);

		float maxLen = std::max({ len1, len2, len3 });
		float minLen = std::min({ len1, len2, len3 });

		// Calculate aspect ratio
		float aspectRatio = (minLen > 1e-6f) ? maxLen / minLen : 999.0f;

		// For urban scenes, long thin triangles sometimes need to be split along the long axis
		if (aspectRatio > ASPECT_RATIO_THRESHOLD && maxLen > MAX_EDGE_LENGTH * 0.7f) {
			// Identify the longest edge
			int longestIdx = 0;
			if (len2 > len1 && len2 > len3) longestIdx = 1;
			else if (len3 > len1 && len3 > len2) longestIdx = 2;

			// Find the vertex opposite to the longest edge
			int oppositeVert;
			glm::vec3 midLongest;

			if (longestIdx == 0) { // Edge between v0 and v1
				oppositeVert = 2;
				midLongest = (tri.v[0] + tri.v[1]) * 0.5f;
			}
			else if (longestIdx == 1) { // Edge between v0 and v2
				oppositeVert = 1;
				midLongest = (tri.v[0] + tri.v[2]) * 0.5f;
			}
			else { // Edge between v1 and v2
				oppositeVert = 0;
				midLongest = (tri.v[1] + tri.v[2]) * 0.5f;
			}

			// Create two triangles by splitting along the height
			MCTriangle splitTri1, splitTri2;

			// Configure based on which edge is longest
			if (longestIdx == 0) {
				splitTri1.v[0] = tri.v[0];
				splitTri1.v[1] = midLongest;
				splitTri1.v[2] = tri.v[2];

				splitTri2.v[0] = midLongest;
				splitTri2.v[1] = tri.v[1];
				splitTri2.v[2] = tri.v[2];
			}
			else if (longestIdx == 1) {
				splitTri1.v[0] = tri.v[0];
				splitTri1.v[1] = tri.v[1];
				splitTri1.v[2] = midLongest;

				splitTri2.v[0] = tri.v[1];
				splitTri2.v[1] = tri.v[2];
				splitTri2.v[2] = midLongest;
			}
			else { // longestIdx == 2
				splitTri1.v[0] = tri.v[0];
				splitTri1.v[1] = tri.v[1];
				splitTri1.v[2] = midLongest;

				splitTri2.v[0] = tri.v[0];
				splitTri2.v[1] = midLongest;
				splitTri2.v[2] = tri.v[2];
			}

			// Copy normals
			for (int i = 0; i < 3; i++) {
				splitTri1.normal[i] = tri.normal[i];
				splitTri2.normal[i] = tri.normal[i];
			}

			refined.push_back(splitTri1);
			refined.push_back(splitTri2);
		}
		else {
			refined.push_back(tri);
		}
	}

	return refined;
}

std::vector<HermitePoint> AdaptiveDualContouringRenderer::getFaceHermitePoints(
	const VoxelGrid& grid,
	int faceDir,    // 0=X, 1=Y, 2=Z axis
	int faceSign,   // +1 or -1 for positive/negative direction
	int minX, int minY, int minZ,
	int maxX, int maxY, int maxZ) {

	std::vector<HermitePoint> points;

	// Helper lambda to check if coordinates are within grid bounds
	auto inBounds = [&grid](int x, int y, int z) -> bool {
		return x >= 0 && y >= 0 && z >= 0 &&
			x < grid.dimX && y < grid.dimY && z < grid.dimZ;
		};

	// Process based on face direction
	if (faceDir == 0) { // X-axis face
		// Determine the exact x-coordinate of the face
		int x = faceSign > 0 ? maxX - 1 : minX;
		int xNext = faceSign > 0 ? x + 1 : x - 1;

		// Loop through all points on this face - only check edges crossing the face
		for (int z = minZ; z < maxZ; z++) {
			for (int y = minY; y < maxY; y++) {
				// Check if both endpoints of this edge are in bounds
				if (inBounds(x, y, z) && inBounds(xNext, y, z)) {
					// Check if there's a sign change (surface intersection)
					bool isFilled = (grid.data[grid.index(x, y, z)] == VoxelState::FILLED);
					bool isNextFilled = (grid.data[grid.index(xNext, y, z)] == VoxelState::FILLED);

					if (isFilled != isNextFilled) {
						// Found an intersection, calculate and add hermite point
						points.push_back(calculateIntersection(grid, x, y, z, xNext, y, z));
					}
				}
			}
		}
	}
	else if (faceDir == 1) { // Y-axis face
		// Determine the exact y-coordinate of the face
		int y = faceSign > 0 ? maxY - 1 : minY;
		int yNext = faceSign > 0 ? y + 1 : y - 1;

		// Loop through all points on this face - only check edges crossing the face
		for (int z = minZ; z < maxZ; z++) {
			for (int x = minX; x < maxX; x++) {
				// Check for Y-direction edges (crossing the face)
				if (inBounds(x, y, z) && inBounds(x, yNext, z)) {
					bool isFilled = (grid.data[grid.index(x, y, z)] == VoxelState::FILLED);
					bool isNextFilled = (grid.data[grid.index(x, yNext, z)] == VoxelState::FILLED);

					if (isFilled != isNextFilled) {
						points.push_back(calculateIntersection(grid, x, y, z, x, yNext, z));
					}
				}
			}
		}
	}
	else { // Z-axis face
		// Determine the exact z-coordinate of the face
		int z = faceSign > 0 ? maxZ - 1 : minZ;
		int zNext = faceSign > 0 ? z + 1 : z - 1;

		// Loop through all points on this face - only check edges crossing the face
		for (int y = minY; y < maxY; y++) {
			for (int x = minX; x < maxX; x++) {
				// Check for Z-direction edges (crossing the face)
				if (inBounds(x, y, z) && inBounds(x, y, zNext)) {
					bool isFilled = (grid.data[grid.index(x, y, z)] == VoxelState::FILLED);
					bool isNextFilled = (grid.data[grid.index(x, y, zNext)] == VoxelState::FILLED);

					if (isFilled != isNextFilled) {
						points.push_back(calculateIntersection(grid, x, y, z, x, y, zNext));
					}
				}
			}
		}
	}

	// Remove duplicate points if any (using a small epsilon)
	if (points.size() > 1) {
		std::vector<HermitePoint> uniquePoints;
		const float EPSILON_SQ = 1e-6f;  // Increased for urban structures

		for (const auto& point : points) {
			bool isDuplicate = false;
			for (const auto& existing : uniquePoints) {
				if (glm::distance2(point.position, existing.position) < EPSILON_SQ) {
					isDuplicate = true;
					break;
				}
			}

			if (!isDuplicate) {
				uniquePoints.push_back(point);
			}
		}

		return uniquePoints;
	}

	return points;
}


std::vector<HermitePoint> AdaptiveDualContouringRenderer::gatherHermiteData(
	const VoxelGrid& grid, int x0, int y0, int z0, int size) {

	// Boundary checking
	int maxX = std::min(x0 + size, grid.dimX - 1);
	int maxY = std::min(y0 + size, grid.dimY - 1);
	int maxZ = std::min(z0 + size, grid.dimZ - 1);

	int minX = std::max(x0, 0);
	int minY = std::max(y0, 0);
	int minZ = std::max(z0, 0);

	// First pass: collect all raw hermite points
	std::vector<HermitePoint> rawPoints;

	for (int z = minZ; z <= maxZ; z++) {
		for (int y = minY; y <= maxY; y++) {
			for (int x = minX; x <= maxX; x++) {
				bool isCurrentVoxelFilled = (grid.data[grid.index(x, y, z)] == VoxelState::FILLED);

				// Check all three axis directions
				if (x < maxX) {
					bool isNextXFilled = (grid.data[grid.index(x + 1, y, z)] == VoxelState::FILLED);
					if (isCurrentVoxelFilled != isNextXFilled) {
						rawPoints.push_back(calculateIntersection(grid, x, y, z, x + 1, y, z));
					}
				}

				if (y < maxY) {
					bool isNextYFilled = (grid.data[grid.index(x, y + 1, z)] == VoxelState::FILLED);
					if (isCurrentVoxelFilled != isNextYFilled) {
						rawPoints.push_back(calculateIntersection(grid, x, y, z, x, y + 1, z));
					}
				}

				if (z < maxZ) {
					bool isNextZFilled = (grid.data[grid.index(x, y, z + 1)] == VoxelState::FILLED);
					if (isCurrentVoxelFilled != isNextZFilled) {
						rawPoints.push_back(calculateIntersection(grid, x, y, z, x, y, z + 1));
					}
				}
			}
		}
	}

	// For urban buildings, optimize the hermite data
	std::vector<HermitePoint> processedPoints;

	// If we have too few points, just return them as is
	if (rawPoints.size() <= 4) {
		return rawPoints;
	}

	// For urban buildings, we want to recognize dominant planes
	// Group points by similar normals (within a tolerance)
	std::vector<std::vector<HermitePoint>> normalGroups;

	const float NORMAL_SIMILARITY_THRESHOLD = 0.9f; // cos(about 25 degrees)

	for (const auto& point : rawPoints) {
		bool addedToGroup = false;

		// Try to add to an existing group
		for (auto& group : normalGroups) {
			if (glm::dot(point.normal, group[0].normal) > NORMAL_SIMILARITY_THRESHOLD) {
				group.push_back(point);
				addedToGroup = true;
				break;
			}
		}

		// If not added to any existing group, create a new one
		if (!addedToGroup) {
			normalGroups.push_back({ point });
		}
	}

	// Process each normal group to extract key points
	for (const auto& group : normalGroups) {
		// If the group is small, just add all points
		if (group.size() <= 4) {
			processedPoints.insert(processedPoints.end(), group.begin(), group.end());
			continue;
		}

		// Calculate average normal and position for the group
		glm::vec3 avgNormal(0.0f);
		glm::vec3 avgPos(0.0f);

		for (const auto& p : group) {
			avgNormal += p.normal;
			avgPos += p.position;
		}

		avgNormal = glm::normalize(avgNormal);
		avgPos /= float(group.size());

		// For larger groups, extract key points that define the shape outline
		// Create a coordinate system on the plane defined by avgNormal
		glm::vec3 u, v;

		// Find two perpendicular vectors on the plane
		if (std::abs(avgNormal.x) > std::abs(avgNormal.y) &&
			std::abs(avgNormal.x) > std::abs(avgNormal.z)) {
			// X is dominant normal component
			u = glm::normalize(glm::cross(avgNormal, glm::vec3(0, 0, 1)));
		}
		else if (std::abs(avgNormal.y) > std::abs(avgNormal.z)) {
			// Y is dominant normal component
			u = glm::normalize(glm::cross(avgNormal, glm::vec3(1, 0, 0)));
		}
		else {
			// Z is dominant normal component
			u = glm::normalize(glm::cross(avgNormal, glm::vec3(0, 1, 0)));
		}

		v = glm::normalize(glm::cross(avgNormal, u));

		// For urban buildings, choose key points at extremes
		// This will help capture the corners of buildings
		float minU = std::numeric_limits<float>::max();
		float maxU = -std::numeric_limits<float>::max();
		float minV = std::numeric_limits<float>::max();
		float maxV = -std::numeric_limits<float>::max();

		int minUIdx = -1, maxUIdx = -1, minVIdx = -1, maxVIdx = -1;

		for (size_t i = 0; i < group.size(); i++) {
			glm::vec3 relPos = group[i].position - avgPos;
			float projU = glm::dot(relPos, u);
			float projV = glm::dot(relPos, v);

			if (projU < minU) { minU = projU; minUIdx = i; }
			if (projU > maxU) { maxU = projU; maxUIdx = i; }
			if (projV < minV) { minV = projV; minVIdx = i; }
			if (projV > maxV) { maxV = projV; maxVIdx = i; }
		}

		// Add extreme points to processed points
		std::vector<int> indices = { minUIdx, maxUIdx, minVIdx, maxVIdx };
		std::sort(indices.begin(), indices.end());
		indices.erase(std::unique(indices.begin(), indices.end()), indices.end());

		for (int idx : indices) {
			if (idx >= 0) {
				processedPoints.push_back(group[idx]);
			}
		}

		// Also add the center point with average normal
		HermitePoint centerPoint;
		centerPoint.position = avgPos;
		centerPoint.normal = avgNormal;
		processedPoints.push_back(centerPoint);
	}

	return processedPoints;
}
