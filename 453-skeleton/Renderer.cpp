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

// --------------------------------------------------------
// 1) Uniform DC building
// --------------------------------------------------------
void AdaptiveDualContouringRenderer::buildUniformDCCells(const VoxelGrid& grid,
	int x0, int y0, int z0,
	int size,
	std::vector<DCCell>& dcCells,
	int subDimX, int subDimY, int subDimZ,
	int lodLevel)
{
	dcCells.resize(subDimX * subDimY * subDimZ);
	auto cellIndex = [&](int x, int y, int z) {
		return x + subDimX * (y + subDimY * z);
		};

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
	for (int z = 0; z < subDimZ; z++) {
		for (int y = 0; y < subDimY; y++) {
			for (int x = 0; x < subDimX; x++) {
				int idx = cellIndex(x, y, z);
				float cornerVals[8];
				bool allNeg = true, allPos = true;
				for (int c = 0; c < 8; c++) {
					int gx = x0 + x + cornerBits[c][0];
					int gy = y0 + y + cornerBits[c][1];
					int gz = z0 + z + cornerBits[c][2];
					float val = sampleVolume(grid, gx, gy, gz);
					cornerVals[c] = val;
					if (val < 0) allPos = false;
					if (val > 0) allNeg = false;
				}
				if (allNeg || allPos) {
					dcCells[idx].isMixed = false;
					continue;
				}
				dcCells[idx].isMixed = true;

				// Possibly check globalCellMap for reuse:
				// (lodLevel, x0+x, y0+y, z0+z)
				DCCellKey key{ lodLevel, x0 + x, y0 + y, z0 + z };
				auto it = globalCellMap.find(key);
				if (it != globalCellMap.end()) {
					dcCells[idx] = it->second;
					continue;
				}

				// gather intersections => QEF
				std::vector<glm::vec3> pts, nrms;
				for (int e = 0; e < 12; e++) {
					int c1 = edgePairs[e][0], c2 = edgePairs[e][1];
					float v1 = cornerVals[c1], v2 = cornerVals[c2];
					if (v1 * v2 < 0.f) {
						int gx1 = x0 + x + cornerBits[c1][0];
						int gy1 = y0 + y + cornerBits[c1][1];
						int gz1 = z0 + z + cornerBits[c1][2];
						glm::vec3 p1(grid.minX + gx1 * vx,
							grid.minY + gy1 * vx,
							grid.minZ + gz1 * vx);

						int gx2 = x0 + x + cornerBits[c2][0];
						int gy2 = y0 + y + cornerBits[c2][1];
						int gz2 = z0 + z + cornerBits[c2][2];
						glm::vec3 p2(grid.minX + gx2 * vx,
							grid.minY + gy2 * vx,
							grid.minZ + gz2 * vx);

						glm::vec3 pi = intersectEdge(p1, p2, v1, v2);
						glm::vec3 Ni = computeNormal(grid, gx1, gy1, gz1);
						pts.push_back(pi);
						nrms.push_back(Ni);
					}
				}
				glm::vec3 cellVert = solveQEF(pts, nrms);
				dcCells[idx].dcVertex = cellVert;
				if (!nrms.empty()) {
					glm::vec3 N(0.f);
					for (auto& nn : nrms) {
						N += glm::normalize(nn);
					}
					dcCells[idx].dcNormal = glm::normalize(N);
				}
				else {
					dcCells[idx].dcNormal = glm::vec3(0, 1, 0);
				}

				// store in global map
				globalCellMap[key] = dcCells[idx];
			}
		}
	}
}

std::vector<MCTriangle> AdaptiveDualContouringRenderer::buildUniformDCMesh(
	const VoxelGrid& grid,
	int x0, int y0, int z0,
	int subDimX, int subDimY, int subDimZ,
	const std::vector<DCCell>& dcCells,
	int lodLevel)
{
	std::vector<MCTriangle> out;
	auto cellIndex = [&](int x, int y, int z) {
		return x + subDimX * (y + subDimY * z);
		};
	auto getCellPtr = [&](int x, int y, int z)->const DCCell* {
		if (x < 0 || x >= subDimX || y < 0 || y >= subDimY || z < 0 || z >= subDimZ) return nullptr;
		const DCCell& c = dcCells[cellIndex(x, y, z)];
		return c.isMixed ? &c : nullptr;
		};

	for (int z = 0; z < subDimZ; z++) {
		for (int y = 0; y < subDimY; y++) {
			for (int x = 0; x < subDimX; x++) {
				int idx = cellIndex(x, y, z);
				if (!dcCells[idx].isMixed) continue;
				const DCCell& c0 = dcCells[idx];
				glm::vec3 v0 = c0.dcVertex;
				glm::vec3 n0 = c0.dcNormal;

				// +X +Y
				if (x + 1 < subDimX && y + 1 < subDimY) {
					auto cx = getCellPtr(x + 1, y, z);
					auto cy = getCellPtr(x, y + 1, z);
					auto cxy = getCellPtr(x + 1, y + 1, z);
					if (cx && cy && cxy) {
						addQuad(v0, cx->dcVertex,
							cy->dcVertex, cxy->dcVertex,
							n0, cx->dcNormal,
							cy->dcNormal, cxy->dcNormal,
							out);
					}
				}
				// +X +Z
				if (x + 1 < subDimX && z + 1 < subDimZ) {
					auto cx = getCellPtr(x + 1, y, z);
					auto cz = getCellPtr(x, y, z + 1);
					auto cxz = getCellPtr(x + 1, y, z + 1);
					if (cx && cz && cxz) {
						addQuad(v0, cx->dcVertex,
							cz->dcVertex, cxz->dcVertex,
							n0, cx->dcNormal,
							cz->dcNormal, cxz->dcNormal,
							out);
					}
				}
				// +Y +Z
				if (y + 1 < subDimY && z + 1 < subDimZ) {
					auto cy = getCellPtr(x, y + 1, z);
					auto cz = getCellPtr(x, y, z + 1);
					auto cyz = getCellPtr(x, y + 1, z + 1);
					if (cy && cz && cyz) {
						addQuad(v0, cy->dcVertex,
							cz->dcVertex, cyz->dcVertex,
							n0, cy->dcNormal,
							cz->dcNormal, cyz->dcNormal,
							out);
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
