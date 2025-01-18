#include "Renderer.h"
#include "OctreeVoxel.h"
#include <glm/gtx/norm.hpp>
#include <cmath>
#include <array>
#include <limits>
#include <iostream>
#include <unordered_map>
#include <vector>
#include <cstdint>
std::unordered_map<long long, OctreeNode*> g_octreeMap;

// --------------------------------------------------------
// MARCHING CUBES
// --------------------------------------------------------
std::vector<MCTriangle> MarchingCubesRenderer::render(const OctreeNode* node,
	const VoxelGrid& grid,
	int x0, int y0, int z0, int size)
{
	if (!node) return {};
	std::vector<MCTriangle> results;
	if (node->isLeaf) {
		// local uniform MC
		auto mcTris = localMC(grid, x0, y0, z0, size);
		results.insert(results.end(), mcTris.begin(), mcTris.end());
	}
	else {
		int half = size / 2;
		for (int i = 0; i < 8; i++) {
			int ox = x0 + ((i & 1) ? half : 0);
			int oy = y0 + ((i & 2) ? half : 0);
			int oz = z0 + ((i & 4) ? half : 0);
			auto childTris = render(node->children[i], grid, ox, oy, oz, half);
			results.insert(results.end(), childTris.begin(), childTris.end());
		}
	}
	return results;
}

std::vector<OctreeNode*> AdaptiveDualContouringRenderer::findLODNeighbors(
	const OctreeNode* node,
	int x0, int y0, int z0,
	int size)
{
	std::vector<OctreeNode*> neighbors;
	if (!node) {
		return neighbors;
	}

	// Helper to build a unique key from (x, y, z).
	// This must match how you built the octree in 'buildOctree(...)'.
	// For example, if you used (x << 20) ^ (y << 10) ^ z as a key:
	auto buildKey = [&](int xx, int yy, int zz) -> long long {
		long long key =
			(static_cast<long long>(xx) << 20) ^
			(static_cast<long long>(yy) << 10) ^
			static_cast<long long>(zz);
		return key;
		};

	// +X neighbor
	{
		int nx = x0 + size;
		int ny = y0;
		int nz = z0;
		long long k = buildKey(nx, ny, nz);
		if (g_octreeMap.count(k)) {
			neighbors.push_back(g_octreeMap[k]);
		}
	}

	// -X neighbor
	{
		int nx = x0 - size;  // or (x0 - size) if you want the block behind
		int ny = y0;
		int nz = z0;
		long long k = buildKey(nx, ny, nz);
		if (g_octreeMap.count(k)) {
			neighbors.push_back(g_octreeMap[k]);
		}
	}

	// +Y neighbor
	{
		int nx = x0;
		int ny = y0 + size;
		int nz = z0;
		long long k = buildKey(nx, ny, nz);
		if (g_octreeMap.count(k)) {
			neighbors.push_back(g_octreeMap[k]);
		}
	}

	// -Y neighbor
	{
		int nx = x0;
		int ny = y0 - size;
		int nz = z0;
		long long k = buildKey(nx, ny, nz);
		if (g_octreeMap.count(k)) {
			neighbors.push_back(g_octreeMap[k]);
		}
	}

	// +Z neighbor
	{
		int nx = x0;
		int ny = y0;
		int nz = z0 + size;
		long long k = buildKey(nx, ny, nz);
		if (g_octreeMap.count(k)) {
			neighbors.push_back(g_octreeMap[k]);
		}
	}

	// -Z neighbor
	{
		int nx = x0;
		int ny = y0;
		int nz = z0 - size;
		long long k = buildKey(nx, ny, nz);
		if (g_octreeMap.count(k)) {
			neighbors.push_back(g_octreeMap[k]);
		}
	}

	return neighbors;
}


// --------------------------------------------------------
// ADAPTIVE DUAL CONTOURING
// --------------------------------------------------------
std::vector<MCTriangle> AdaptiveDualContouringRenderer::render(const OctreeNode* node,
	const VoxelGrid& grid,
	int x0, int y0, int z0, int size)
{
	if (!node) {
		return {};
	}

	std::vector<MCTriangle> out;

	// For demonstration, let's define a function that yields "lodLevel" from size
	// e.g. if top-level size is 64 => lod=0, if half=32 => lod=1, etc.
	// You can do something more robust as needed:
	auto getLodFromSize = [&](int s) {
		// Just a quick hack: bigger 's' => smaller lod number
		// E.g. if the entire grid is sizePow2 => 64 => lod=0, 32 => lod=1, 16 => lod=2, ...
		int lod = 0;
		int tmp = s;
		while (tmp < grid.dimX || tmp < grid.dimY || tmp < grid.dimZ) {
			tmp <<= 1;
			lod--;
		}
		// or we do a while (tmp>1) { tmp/=2; lod++;}
		// The logic depends on how you define LOD. 
		// We'll do something like:
		int ret = 0;
		int p = std::max({ grid.dimX,grid.dimY,grid.dimZ });
		while (p > s) {
			p /= 2;
			ret++;
		}
		return ret;
		};
	int currentLod = getLodFromSize(size);

	if (node->isLeaf) {
		// 1) Do uniform DC for the leaf
		int subDimX = size, subDimY = size, subDimZ = size;
		if (subDimX <= 0 || subDimY <= 0 || subDimZ <= 0) return {};

		std::vector<DCCell> dcCells;
		buildUniformDCCells(grid, x0, y0, z0, size,
			dcCells, subDimX, subDimY, subDimZ,
			currentLod);

		auto localTris = buildUniformDCMesh(grid, x0, y0, z0,
			subDimX, subDimY, subDimZ,
			dcCells, currentLod);
		out.insert(out.end(), localTris.begin(), localTris.end());

		// 2) Attempt to find neighbors that might be finer
		// For each of the 6 faces, see if there's a neighbor octree node 
		// whose 'size' is smaller => finer => we do 'stitchBoundaryFace'.
		// This is a simplified example that only checks +X face, etc.
		// In real code, you'd do it for -X,+Y,-Y,+Z,-Z too.
		auto neighbors = findLODNeighbors(node, x0, y0, z0, size);
		for (auto& n : neighbors) {
			if (!n) continue; // safety check
			if (n->size < node->size) { // or however you detect "finer"
				stitchBoundaryFace(grid, node, x0, y0, z0, size, currentLod,
					grid, n, x0, y0, z0, size, getLodFromSize(n->size),
					out);
			}
			else if (n->size > node->size) {
				// or the reverse, if this node is finer
				stitchBoundaryFace(grid, n, x0, y0, z0, size, getLodFromSize(n->size),
					grid, node, x0, y0, z0, size, currentLod,
					out);
			}
		}

	}
	else {
		// Internal node: recurse, then stitch among children
		int half = size / 2;
		for (int i = 0; i < 8; i++) {
			int ox = x0 + ((i & 1) ? half : 0);
			int oy = y0 + ((i & 2) ? half : 0);
			int oz = z0 + ((i & 4) ? half : 0);
			auto childTris = render(node->children[i], grid, ox, oy, oz, half);
			out.insert(out.end(), childTris.begin(), childTris.end());
		}
		// Optionally, also do any stitching among child boundaries if they differ in size. 
	}

	return out;
}

// -----------------------------------------------------------------------------
// 1) BUILD DC CELLS (Ensures each cell has one consistent vertex)
// -----------------------------------------------------------------------------
void AdaptiveDualContouringRenderer::buildUniformDCCells(
	const VoxelGrid& grid,
	int x0, int y0, int z0,
	int size,
	std::vector<DCCell>& dcCells,
	int subDimX, int subDimY, int subDimZ,
	int lodLevel)
{
	dcCells.resize(subDimX * subDimY * subDimZ);

	auto cellIndex = [&](int lx, int ly, int lz) {
		return lx + subDimX * (ly + subDimY * lz);
		};

	// These define how we track corners in local sub-cell
	static const int cornerBits[8][3] = {
		{0,0,0},{1,0,0},{1,1,0},{0,1,0},
		{0,0,1},{1,0,1},{1,1,1},{0,1,1}
	};
	static const int edgePairs[12][2] = {
		{0,1},{1,2},{2,3},{3,0},
		{4,5},{5,6},{6,7},{7,4},
		{0,4},{1,5},{2,6},{3,7}
	};

	float vx = grid.voxelSize;

	for (int lz = 0; lz < subDimZ; ++lz) {
		for (int ly = 0; ly < subDimY; ++ly) {
			for (int lx = 0; lx < subDimX; ++lx) {
				int idx = cellIndex(lx, ly, lz);
				DCCell& cell = dcCells[idx];
				cell.isMixed = false;

				// Gather corner signs
				float cornerVals[8];
				bool allNeg = true, allPos = true;
				for (int c = 0; c < 8; ++c) {
					int gx = x0 + lx + cornerBits[c][0];
					int gy = y0 + ly + cornerBits[c][1];
					int gz = z0 + lz + cornerBits[c][2];

					float val = sampleVolume(grid, gx, gy, gz);
					cornerVals[c] = val;
					if (val < 0) allPos = false;
					if (val > 0) allNeg = false;
				}
				if (allNeg || allPos) {
					// uniform
					cell.isMixed = false;
					continue;
				}
				cell.isMixed = true;

				// See if we already have a stored DC vertex from globalCellMap
				DCCellKey key{ lodLevel, x0 + lx, y0 + ly, z0 + lz };
				auto it = globalCellMap.find(key);
				if (it != globalCellMap.end()) {
					cell = it->second;  // reuse
					continue;
				}

				// Otherwise, compute intersections on edges -> QEF
				std::vector<glm::vec3> pts, nrms;
				pts.reserve(12);
				nrms.reserve(12);

				for (int e = 0; e < 12; e++) {
					int c1 = edgePairs[e][0];
					int c2 = edgePairs[e][1];
					float v1 = cornerVals[c1];
					float v2 = cornerVals[c2];
					if (v1 * v2 < 0.f) {
						// There's an intersection
						int gx1 = x0 + lx + cornerBits[c1][0];
						int gy1 = y0 + ly + cornerBits[c1][1];
						int gz1 = z0 + lz + cornerBits[c1][2];
						glm::vec3 p1(
							grid.minX + gx1 * vx,
							grid.minY + gy1 * vx,
							grid.minZ + gz1 * vx
						);

						int gx2 = x0 + lx + cornerBits[c2][0];
						int gy2 = y0 + ly + cornerBits[c2][1];
						int gz2 = z0 + lz + cornerBits[c2][2];
						glm::vec3 p2(
							grid.minX + gx2 * vx,
							grid.minY + gy2 * vx,
							grid.minZ + gz2 * vx
						);

						glm::vec3 pi = intersectEdge(p1, p2, v1, v2);
						// Use a normal from e.g. the midpoint or the first corner
						glm::vec3 Ni = computeNormal(grid, gx1, gy1, gz1);
						pts.push_back(pi);
						nrms.push_back(Ni);
					}
				}

				// Solve QEF
				glm::vec3 cellVert = solveQEF(pts, nrms);
				cell.dcVertex = cellVert;
				if (!nrms.empty()) {
					glm::vec3 N(0.f);
					for (auto& nn : nrms) {
						N += glm::normalize(nn);
					}
					cell.dcNormal = glm::normalize(N);
				}
				else {
					cell.dcNormal = glm::vec3(0, 1, 0);
				}

				globalCellMap[key] = cell;
			}
		}
	}
}

// -----------------------------------------------------------------------------
// 2) BUILD MESH from DC cells (only +X, +Y, +Z to avoid double adjacency)
// -----------------------------------------------------------------------------
std::vector<MCTriangle>
AdaptiveDualContouringRenderer::buildUniformDCMesh(
	const VoxelGrid& grid,
	int x0, int y0, int z0,
	int subDimX, int subDimY, int subDimZ,
	const std::vector<DCCell>& dcCells,
	int lodLevel)
{
	std::vector<MCTriangle> out;

	auto cellIndex = [&](int lx, int ly, int lz) {
		return lx + subDimX * (ly + subDimY * lz);
		};
	auto getCellPtr = [&](int lx, int ly, int lz)-> const DCCell* {
		if (lx < 0 || lx >= subDimX || ly < 0 || ly >= subDimY || lz < 0 || lz >= subDimZ) {
			return nullptr;
		}
		const DCCell& c = dcCells[cellIndex(lx, ly, lz)];
		return c.isMixed ? &c : nullptr;
		};

	for (int lz = 0; lz < subDimZ; ++lz) {
		for (int ly = 0; ly < subDimY; ++ly) {
			for (int lx = 0; lx < subDimX; ++lx) {
				const DCCell& c0 = dcCells[cellIndex(lx, ly, lz)];
				if (!c0.isMixed) continue;

				glm::vec3 v0 = c0.dcVertex;
				glm::vec3 n0 = c0.dcNormal;

				// +X face
				if (lx + 1 < subDimX) {
					const DCCell* cx = getCellPtr(lx + 1, ly, lz);
					if (cx) {
						// +Y direction from c0, cx
						if (ly + 1 < subDimY) {
							auto cy = getCellPtr(lx, ly + 1, lz);
							auto cxy = getCellPtr(lx + 1, ly + 1, lz);
							if (cy && cxy) {
								addQuad(
									v0, cx->dcVertex,
									cy->dcVertex, cxy->dcVertex,
									n0, cx->dcNormal,
									cy->dcNormal, cxy->dcNormal,
									out
								);
							}
						}
						// +Z direction from c0, cx
						if (lz + 1 < subDimZ) {
							auto cz = getCellPtr(lx, ly, lz + 1);
							auto cxz = getCellPtr(lx + 1, ly, lz + 1);
							if (cz && cxz) {
								addQuad(
									v0, cx->dcVertex,
									cz->dcVertex, cxz->dcVertex,
									n0, cx->dcNormal,
									cz->dcNormal, cxz->dcNormal,
									out
								);
							}
						}
					}
				}

				// If we skip +X, we still want to connect +Y, +Z
				// +Y face
				if (ly + 1 < subDimY) {
					const DCCell* cy = getCellPtr(lx, ly + 1, lz);
					if (cy && (lz + 1 < subDimZ)) {
						// Connect +Z diagonal
						auto cz = getCellPtr(lx, ly, lz + 1);
						auto cyz = getCellPtr(lx, ly + 1, lz + 1);
						if (cz && cyz) {
							addQuad(
								v0, cy->dcVertex,
								cz->dcVertex, cyz->dcVertex,
								n0, cy->dcNormal,
								cz->dcNormal, cyz->dcNormal,
								out
							);
						}
					}
				}
			}
		}
	}
	return out;
}

// --------------------------------------------------------
// 2) Actual "stitchBoundaryFace" that merges coarse & fine
// --------------------------------------------------------
void AdaptiveDualContouringRenderer::stitchBoundaryFace(const VoxelGrid& coarseGrid,
	const OctreeNode* coarseNode,
	int cx0, int cy0, int cz0, int cSize, int cLod,
	const VoxelGrid& fineGrid,
	const OctreeNode* fineNode,
	int fx0, int fy0, int fz0, int fSize, int fLod,
	std::vector<MCTriangle>& out)
{
	// Example: Suppose we want to stitch the +X face of the coarse leaf 
	// with the -X face of the fine leaf (or some bounding).
	// We'll assume fSize is smaller => ratio = fSize / cSize > 1 => fine is "finer".

	int ratio = cSize / fSize; // or if cSize < fSize => ratio= fSize/cSize
	if (ratio < 2) {
		// If ratio=1 => same LOD => no special stitching needed 
		return;
	}

	// For each voxel in the coarse boundary face:
	// e.g. if the boundary is +X => x in [cx0 + cSize-1], y in [cy0..cy0+cSize-1], z in [cz0..cz0+cSize-1]
	// Then subdiv each coarse cell => unify with fine data. 
	int boundaryX = cx0 + cSize - 1; // for +X face
	for (int z = cz0; z < cz0 + cSize - 1; z++) {
		for (int y = cy0; y < cy0 + cSize - 1; y++) {
			// subdivide one coarse cell
			subdivideCoarseCell(coarseGrid, boundaryX, y, z,
				cLod,
				fineGrid,
				ratio,
				out);
		}
	}
}

// Actually subdivides the coarse cell
void AdaptiveDualContouringRenderer::subdivideCoarseCell(const VoxelGrid& coarseGrid,
	int cX, int cY, int cZ,
	int cLod,
	const VoxelGrid& fineGrid,
	int ratio,
	std::vector<MCTriangle>& out)
{
	// This is a big operation. We’d gather the 8 corners from the coarse cell,
	// then break it into ratio^3 sub-cells. Each sub-cell corresponds to 
	// multiple cells in the fine grid. Then unify the QEF solutions => produce bridging faces.

	// We'll do a minimal snippet (still conceptual):

	for (int sz = 0; sz < ratio; sz++) {
		for (int sy = 0; sy < ratio; sy++) {
			for (int sx = 0; sx < ratio; sx++) {
				// 1) Build a subcell QEF from coarse corner(s) + the relevant fine cells
				//    e.g. the fine coords might be fX = cX*ratio + sx, etc.
				//    or we do a relative offset if the fine leaf is at a different origin. 
				// 2) Solve QEF => a single vertex
				// 3) Build bridging polygons that connect the coarse vertex to subcell vertices, 
				//    ensuring no gaps.

				// Because this function is large, we only show a short structure:

				std::vector<glm::vec3> pts, nrms;
				// gather from coarse corner(s):
				// gather from fine leaf cells (sx,sy,sz)...

				glm::vec3 subVert = solveQEF(pts, nrms);
				glm::vec3 subNorm(0.f);
				for (auto& nn : nrms) {
					subNorm += glm::normalize(nn);
				}
				if (!nrms.empty()) {
					subNorm = glm::normalize(subNorm);
				}
				else {
					subNorm = glm::vec3(0, 1, 0);
				}

				// Build bridging geometry:
				// e.g. connect the coarse cell's main DC vertex with subVert and adjacent subcells
				// for a watertight mesh
				// out.push_back(...some triangles...);
			}
		}
	}
}

// --------------------------------------------------------
// 3) QEF, sampling, normal, etc.
// --------------------------------------------------------
float AdaptiveDualContouringRenderer::sampleVolume(const VoxelGrid& grid,
	int gx, int gy, int gz)
{
	if (gx < 0 || gx >= grid.dimX ||
		gy < 0 || gy >= grid.dimY ||
		gz < 0 || gz >= grid.dimZ) {
		return +1.f;
	}
	return (grid.data[grid.index(gx, gy, gz)] == VoxelState::FILLED) ? -1.f : +1.f;
}

glm::vec3 AdaptiveDualContouringRenderer::computeNormal(const VoxelGrid& grid,
	int gx, int gy, int gz)
{
	auto val = [&](int dx, int dy, int dz) {
		return sampleVolume(grid, gx + dx, gy + dy, gz + dz);
		};
	float nx = val(+1, 0, 0) - val(-1, 0, 0);
	float ny = val(0, +1, 0) - val(0, -1, 0);
	float nz = val(0, 0, +1) - val(0, 0, -1);
	glm::vec3 N(nx, ny, nz);
	if (glm::length2(N) < 1e-12f) {
		return glm::vec3(0, 1, 0);
	}
	return glm::normalize(N);
}

glm::vec3 AdaptiveDualContouringRenderer::intersectEdge(const glm::vec3& p1,
	const glm::vec3& p2,
	float v1, float v2)
{
	float t = v1 / (v1 - v2);
	return p1 + t * (p2 - p1);
}

glm::vec3 AdaptiveDualContouringRenderer::solveQEF(
	const std::vector<glm::vec3>& points,
	const std::vector<glm::vec3>& normals)
{
	if (points.empty()) return glm::vec3(0.f);

	glm::vec3 centroid(0.f);
	for (auto& p : points) {
		centroid += p;
	}
	centroid /= (float)points.size();

	glm::mat3 ATA(0.f);
	glm::vec3 ATb(0.f);
	for (size_t i = 0; i < points.size(); i++) {
		glm::vec3 pi = points[i] - centroid;
		glm::vec3 ni = glm::normalize(normals[i]);
		ATA += glm::outerProduct(ni, ni);
		float d = glm::dot(pi, ni);
		ATb += d * ni;
	}

	float det = glm::determinant(ATA);
	if (std::fabs(det) < 1e-10f) {
		return centroid;
	}
	glm::vec3 r = glm::inverse(ATA) * ATb;
	return centroid + r;
}

// Building geometry
void AdaptiveDualContouringRenderer::addTriangle(const glm::vec3& v0, const glm::vec3& v1, const glm::vec3& v2,
	const glm::vec3& n0, const glm::vec3& n1, const glm::vec3& n2,
	std::vector<MCTriangle>& out)
{
	MCTriangle tri;
	tri.v[0] = v0; tri.v[1] = v1; tri.v[2] = v2;
	tri.normal[0] = n0; tri.normal[1] = n1; tri.normal[2] = n2;
	out.push_back(tri);
}

void AdaptiveDualContouringRenderer::addQuad(const glm::vec3& v0, const glm::vec3& v1,
	const glm::vec3& v2, const glm::vec3& v3,
	const glm::vec3& n0, const glm::vec3& n1,
	const glm::vec3& n2, const glm::vec3& n3,
	std::vector<MCTriangle>& out)
{
	addTriangle(v0, v1, v2, n0, n1, n2, out);
	addTriangle(v2, v1, v3, n2, n1, n3, out);
}


// --------------------------------------------------------
// NEW: VOXEL CUBE RENDERER (Block-based Voxel Geometry)
// --------------------------------------------------------
std::vector<MCTriangle> VoxelCubeRenderer::render(const OctreeNode* node,
	const VoxelGrid& grid,
	int x0, int y0, int z0, int size)
{
	std::vector<MCTriangle> out;
	if (!node) return out;
	if (node->isLeaf) {
		if (node->isSolid) {
			addBlockFaces(grid, x0, y0, z0, size, out);
		}
	}
	else {
		int half = size / 2;
		for (int i = 0; i < 8; i++) {
			int ox = x0 + ((i & 1) ? half : 0);
			int oy = y0 + ((i & 2) ? half : 0);
			int oz = z0 + ((i & 4) ? half : 0);
			auto childTris = render(node->children[i], grid, ox, oy, oz, half);
			out.insert(out.end(), childTris.begin(), childTris.end());
		}
	}
	return out;
}

void VoxelCubeRenderer::addBlockFaces(const VoxelGrid& grid,
	int x0, int y0, int z0, int size,
	std::vector<MCTriangle>& out)
{
	float vx = grid.voxelSize;
	glm::vec3 minCorner(grid.minX + x0 * vx,
		grid.minY + y0 * vx,
		grid.minZ + z0 * vx);
	glm::vec3 ext(size * vx);
	glm::vec3 maxCorner = minCorner + ext;

	// A simple check at the center of each face is used to decide if the face is
	// exposed. (More elaborate methods could check over the entire face.)
	auto checkFace = [&](int testX, int testY, int testZ) -> bool {
		if (testX < 0 || testY < 0 || testZ < 0 ||
			testX >= grid.dimX || testY >= grid.dimY || testZ >= grid.dimZ)
			return true;
		return (grid.data[grid.index(testX, testY, testZ)] == VoxelState::EMPTY);
		};

	bool posXExposed = checkFace(x0 + size, y0 + size / 2, z0 + size / 2);
	if (posXExposed) addFacePosX(minCorner, maxCorner, out);
	bool negXExposed = checkFace(x0 - 1, y0 + size / 2, z0 + size / 2);
	if (negXExposed) addFaceNegX(minCorner, maxCorner, out);

	bool posYExposed = checkFace(x0 + size / 2, y0 + size, z0 + size / 2);
	if (posYExposed) addFacePosY(minCorner, maxCorner, out);
	bool negYExposed = checkFace(x0 + size / 2, y0 - 1, z0 + size / 2);
	if (negYExposed) addFaceNegY(minCorner, maxCorner, out);

	bool posZExposed = checkFace(x0 + size / 2, y0 + size / 2, z0 + size);
	if (posZExposed) addFacePosZ(minCorner, maxCorner, out);
	bool negZExposed = checkFace(x0 + size / 2, y0 + size / 2, z0 - 1);
	if (negZExposed) addFaceNegZ(minCorner, maxCorner, out);
}

void VoxelCubeRenderer::addFacePosX(const glm::vec3& minC, const glm::vec3& maxC, std::vector<MCTriangle>& out)
{
	glm::vec3 v0(maxC.x, minC.y, minC.z);
	glm::vec3 v1(maxC.x, maxC.y, minC.z);
	glm::vec3 v2(maxC.x, maxC.y, maxC.z);
	glm::vec3 v3(maxC.x, minC.y, maxC.z);
	glm::vec3 normal(1, 0, 0);
	addQuad(v0, v1, v3, v2, normal, out);
}
void VoxelCubeRenderer::addFaceNegX(const glm::vec3& minC, const glm::vec3& maxC, std::vector<MCTriangle>& out)
{
	glm::vec3 v0(minC.x, minC.y, minC.z);
	glm::vec3 v1(minC.x, minC.y, maxC.z);
	glm::vec3 v2(minC.x, maxC.y, maxC.z);
	glm::vec3 v3(minC.x, maxC.y, minC.z);
	glm::vec3 normal(-1, 0, 0);
	addQuad(v0, v1, v3, v2, normal, out);
}
void VoxelCubeRenderer::addFacePosY(const glm::vec3& minC, const glm::vec3& maxC, std::vector<MCTriangle>& out)
{
	glm::vec3 v0(minC.x, maxC.y, minC.z);
	glm::vec3 v1(minC.x, maxC.y, maxC.z);
	glm::vec3 v2(maxC.x, maxC.y, maxC.z);
	glm::vec3 v3(maxC.x, maxC.y, minC.z);
	glm::vec3 normal(0, 1, 0);
	addQuad(v0, v1, v3, v2, normal, out);
}
void VoxelCubeRenderer::addFaceNegY(const glm::vec3& minC, const glm::vec3& maxC, std::vector<MCTriangle>& out)
{
	glm::vec3 v0(minC.x, minC.y, minC.z);
	glm::vec3 v1(maxC.x, minC.y, minC.z);
	glm::vec3 v2(maxC.x, minC.y, maxC.z);
	glm::vec3 v3(minC.x, minC.y, maxC.z);
	glm::vec3 normal(0, -1, 0);
	addQuad(v0, v1, v3, v2, normal, out);
}
void VoxelCubeRenderer::addFacePosZ(const glm::vec3& minC, const glm::vec3& maxC, std::vector<MCTriangle>& out)
{
	glm::vec3 v0(minC.x, minC.y, maxC.z);
	glm::vec3 v1(minC.x, maxC.y, maxC.z);
	glm::vec3 v2(maxC.x, maxC.y, maxC.z);
	glm::vec3 v3(maxC.x, minC.y, maxC.z);
	glm::vec3 normal(0, 0, 1);
	addQuad(v0, v1, v3, v2, normal, out);
}
void VoxelCubeRenderer::addFaceNegZ(const glm::vec3& minC, const glm::vec3& maxC, std::vector<MCTriangle>& out)
{
	glm::vec3 v0(minC.x, minC.y, minC.z);
	glm::vec3 v1(maxC.x, minC.y, minC.z);
	glm::vec3 v2(maxC.x, maxC.y, minC.z);
	glm::vec3 v3(minC.x, maxC.y, minC.z);
	glm::vec3 normal(0, 0, -1);
	addQuad(v0, v1, v3, v2, normal, out);
}

void VoxelCubeRenderer::addQuad(const glm::vec3& v0, const glm::vec3& v1,
	const glm::vec3& v2, const glm::vec3& v3,
	const glm::vec3& normal,
	std::vector<MCTriangle>& out)
{
	MCTriangle tri1;
	tri1.v[0] = v0; tri1.v[1] = v1; tri1.v[2] = v2;
	tri1.normal[0] = normal; tri1.normal[1] = normal; tri1.normal[2] = normal;
	out.push_back(tri1);
	MCTriangle tri2;
	tri2.v[0] = v2; tri2.v[1] = v1; tri2.v[2] = v3;
	tri2.normal[0] = normal; tri2.normal[1] = normal; tri2.normal[2] = normal;
	out.push_back(tri2);
}
