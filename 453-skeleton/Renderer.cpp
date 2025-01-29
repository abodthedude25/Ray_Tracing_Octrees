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
			(static_cast<long long>(xx) << 20) |
			(static_cast<long long>(yy) << 10) |
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
						if (glm::dot(p2 - p1, Ni) < 0) Ni = -Ni;
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
	int x0, int y0, int z0,    // origin (voxel indices) of this DC cell block
	int subDimX, int subDimY, int subDimZ,
	const std::vector<DCCell>& dcCells,
	int lodLevel)
{
	std::vector<MCTriangle> out;

	// Helper lambda to get the 1D index within the dcCells vector:
	auto cellIndex = [&](int lx, int ly, int lz) -> int {
		return lx + subDimX * (ly + subDimY * lz);
		};

	// Helper lambda that returns a pointer to a mixed cell (or nullptr if out-of-range or not mixed)
	auto getCellPtr = [&](int lx, int ly, int lz) -> const DCCell* {
		if (lx < 0 || lx >= subDimX || ly < 0 || ly >= subDimY || lz < 0 || lz >= subDimZ)
			return nullptr;
		const DCCell& c = dcCells[cellIndex(lx, ly, lz)];
		return c.isMixed ? &c : nullptr;
		};

	// For each cell in the grid, if it is mixed, attempt to build faces on all six directions.
	// (In a production system you might want to restrict face generation to one side to avoid duplicates,
	// but here we show separate faces for both positive and negative directions.)
	for (int lz = 0; lz < subDimZ; ++lz) {
		for (int ly = 0; ly < subDimY; ++ly) {
			for (int lx = 0; lx < subDimX; ++lx) {
				const DCCell* c0 = getCellPtr(lx, ly, lz);
				if (!c0) continue;
				glm::vec3 v0 = c0->dcVertex;
				glm::vec3 n0 = c0->dcNormal;

				// --- +X face: between cell (lx,ly,lz) and (lx+1,ly,lz) ---
				if (const DCCell* cx = getCellPtr(lx + 1, ly, lz)) {
					// Form a quad over the (y,z) subcell
					if (ly + 1 < subDimY && getCellPtr(lx, ly + 1, lz) && getCellPtr(lx + 1, ly + 1, lz)) {
						addQuad(v0, cx->dcVertex,
							getCellPtr(lx, ly + 1, lz)->dcVertex,
							getCellPtr(lx + 1, ly + 1, lz)->dcVertex,
							n0, cx->dcNormal,
							getCellPtr(lx, ly + 1, lz)->dcNormal,
							getCellPtr(lx + 1, ly + 1, lz)->dcNormal,
							out);
					}
					if (lz + 1 < subDimZ && getCellPtr(lx, ly, lz + 1) && getCellPtr(lx + 1, ly, lz + 1)) {
						addQuad(v0, cx->dcVertex,
							getCellPtr(lx, ly, lz + 1)->dcVertex,
							getCellPtr(lx + 1, ly, lz + 1)->dcVertex,
							n0, cx->dcNormal,
							getCellPtr(lx, ly, lz + 1)->dcNormal,
							getCellPtr(lx + 1, ly, lz + 1)->dcNormal,
							out);
					}
				}

				// --- -X face: between cell (lx,ly,lz) and (lx-1,ly,lz) ---
				if (const DCCell* cl = getCellPtr(lx - 1, ly, lz)) {
					// For -X, we reverse the ordering so that the face orientation is consistent.
					if (ly + 1 < subDimY && getCellPtr(lx - 1, ly + 1, lz) && getCellPtr(lx, ly + 1, lz)) {
						addQuad(cl->dcVertex, v0,
							getCellPtr(lx - 1, ly + 1, lz)->dcVertex,
							getCellPtr(lx, ly + 1, lz)->dcVertex,
							cl->dcNormal, n0,
							getCellPtr(lx - 1, ly + 1, lz)->dcNormal,
							getCellPtr(lx, ly + 1, lz)->dcNormal,
							out);
					}
					if (lz + 1 < subDimZ && getCellPtr(lx - 1, ly, lz + 1) && getCellPtr(lx, ly, lz + 1)) {
						addQuad(cl->dcVertex, v0,
							getCellPtr(lx - 1, ly, lz + 1)->dcVertex,
							getCellPtr(lx, ly, lz + 1)->dcVertex,
							cl->dcNormal, n0,
							getCellPtr(lx - 1, ly, lz + 1)->dcNormal,
							getCellPtr(lx, ly, lz + 1)->dcNormal,
							out);
					}
				}

				// --- +Y face: between cell (lx,ly,lz) and (lx,ly+1,lz) ---
				if (const DCCell* cy = getCellPtr(lx, ly + 1, lz)) {
					if (lz + 1 < subDimZ && getCellPtr(lx, ly, lz + 1) && getCellPtr(lx, ly + 1, lz + 1)) {
						addQuad(v0, cy->dcVertex,
							getCellPtr(lx, ly, lz + 1)->dcVertex,
							getCellPtr(lx, ly + 1, lz + 1)->dcVertex,
							n0, cy->dcNormal,
							getCellPtr(lx, ly, lz + 1)->dcNormal,
							getCellPtr(lx, ly + 1, lz + 1)->dcNormal,
							out);
					}
				}

				// --- -Y face: between cell (lx,ly,lz) and (lx,ly-1,lz) ---
				if (const DCCell* cy = getCellPtr(lx, ly - 1, lz)) {
					if (lz + 1 < subDimZ && getCellPtr(lx, ly - 1, lz + 1) && getCellPtr(lx, ly, lz + 1)) {
						// Reverse the order when generating -Y
						addQuad(cy->dcVertex, v0,
							getCellPtr(lx, ly - 1, lz + 1)->dcVertex,
							getCellPtr(lx, ly, lz + 1)->dcVertex,
							cy->dcNormal, n0,
							getCellPtr(lx, ly - 1, lz + 1)->dcNormal,
							getCellPtr(lx, ly, lz + 1)->dcNormal,
							out);
					}
				}

				// --- +Z face: between cell (lx,ly,lz) and (lx,ly,lz+1) ---
				if (const DCCell* cz = getCellPtr(lx, ly, lz + 1)) {
					if (ly + 1 < subDimY && getCellPtr(lx, ly + 1, lz) && getCellPtr(lx, ly + 1, lz + 1)) {
						addQuad(v0, cz->dcVertex,
							getCellPtr(lx, ly + 1, lz)->dcVertex,
							getCellPtr(lx, ly + 1, lz + 1)->dcVertex,
							n0, cz->dcNormal,
							getCellPtr(lx, ly + 1, lz)->dcNormal,
							getCellPtr(lx, ly + 1, lz + 1)->dcNormal,
							out);
					}
				}

				// --- -Z face: between cell (lx,ly,lz) and (lx,ly,lz-1) ---
				if (const DCCell* cz = getCellPtr(lx, ly, lz - 1)) {
					if (ly + 1 < subDimY && getCellPtr(lx, ly, lz - 1) && getCellPtr(lx, ly + 1, lz - 1) && getCellPtr(lx, ly + 1, lz)) {
						// Reverse the ordering for -Z face:
						addQuad(cz->dcVertex, v0,
							getCellPtr(lx, ly + 1, lz - 1)->dcVertex,
							getCellPtr(lx, ly + 1, lz)->dcVertex,
							cz->dcNormal, n0,
							getCellPtr(lx, ly + 1, lz - 1)->dcNormal,
							getCellPtr(lx, ly + 1, lz)->dcNormal,
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
void AdaptiveDualContouringRenderer::stitchBoundaryFace(
	const VoxelGrid& coarseGrid,
	const OctreeNode* coarseNode,
	int cx0, int cy0, int cz0,
	int cSize, int cLod,
	const VoxelGrid& fineGrid,
	const OctreeNode* fineNode,
	int fx0, int fy0, int fz0,
	int fSize, int fLod,
	std::vector<MCTriangle>& out)
{
	int ratio = cSize / fSize;
	if (ratio < 2) return;

	// --- +X boundary ---
	{
		int boundaryX = cx0 + cSize - 1;
		for (int z = cz0; z < cz0 + cSize; z++) {
			for (int y = cy0; y < cy0 + cSize; y++) {
				subdivideCoarseCell(coarseGrid, boundaryX, y, z, cLod,
					fineGrid, ratio, fLod,
					StitchFace::POS_X, out);
			}
		}
	}

	// --- -X boundary ---
	{
		int boundaryX = cx0;
		for (int z = cz0; z < cz0 + cSize; z++) {
			for (int y = cy0; y < cy0 + cSize; y++) {
				subdivideCoarseCell(coarseGrid, boundaryX, y, z, cLod,
					fineGrid, ratio, fLod,
					StitchFace::NEG_X, out);
			}
		}
	}

	// --- +Y boundary ---
	{
		int boundaryY = cy0 + cSize - 1;
		for (int z = cz0; z < cz0 + cSize; z++) {
			for (int x = cx0; x < cx0 + cSize; x++) {
				subdivideCoarseCell(coarseGrid, x, boundaryY, z, cLod,
					fineGrid, ratio, fLod,
					StitchFace::POS_Y, out);
			}
		}
	}

	// --- -Y boundary ---
	{
		int boundaryY = cy0;
		for (int z = cz0; z < cz0 + cSize; z++) {
			for (int x = cx0; x < cx0 + cSize; x++) {
				subdivideCoarseCell(coarseGrid, x, boundaryY, z, cLod,
					fineGrid, ratio, fLod,
					StitchFace::NEG_Y, out);
			}
		}
	}

	// --- +Z boundary ---
	{
		int boundaryZ = cz0 + cSize - 1;
		for (int y = cy0; y < cy0 + cSize; y++) {
			for (int x = cx0; x < cx0 + cSize; x++) {
				subdivideCoarseCell(coarseGrid, x, y, boundaryZ, cLod,
					fineGrid, ratio, fLod,
					StitchFace::POS_Z, out);
			}
		}
	}

	// --- -Z boundary ---
	{
		int boundaryZ = cz0;
		for (int y = cy0; y < cy0 + cSize; y++) {
			for (int x = cx0; x < cx0 + cSize; x++) {
				subdivideCoarseCell(coarseGrid, x, y, boundaryZ, cLod,
					fineGrid, ratio, fLod,
					StitchFace::NEG_Z, out);
			}
		}
	}
}


void AdaptiveDualContouringRenderer::subdivideCoarseCell(
	const VoxelGrid& coarseGrid,
	int cX, int cY, int cZ,    // coarse cell coordinates (voxel space)
	int cLod,                // coarse LOD level
	const VoxelGrid& fineGrid,
	int ratio,               // how many fine voxels span one coarse voxel along the face
	int fLod,                // fine LOD level
	StitchFace face,         // which face to stitch: one of POS_X, NEG_X, etc.
	std::vector<MCTriangle>& out)
{
	// Retrieve the coarse cell's DC vertex using its key.
	DCCellKey coarseKey{ cLod, cX, cY, cZ };
	auto it = globalCellMap.find(coarseKey);
	if (it == globalCellMap.end()) return;
	const DCCell& coarseCell = it->second;
	if (!coarseCell.isMixed) return;

	glm::vec3 coarseV = coarseCell.dcVertex;
	glm::vec3 coarseN = coarseCell.dcNormal;

	// The face that we are stitching determines which coordinate remains constant,
	// and which two coordinates vary. We will build a 2D grid (of size ratio+1) for the
	// two varying directions.
	int gridSize = ratio + 1;
	std::vector<glm::vec3> fineVerts(gridSize * gridSize);
	std::vector<glm::vec3> fineNorms(gridSize * gridSize);

	auto fineIndex = [=](int a, int b) -> int { return a + gridSize * b; };

	// Depending on the face, compute the appropriate fine coordinates.
	// We assume a simple mapping: the coarse coordinate is multiplied by ratio
	// to give the origin in the fine grid.
	// Adjust these formulas as needed for your coordinate alignment.
	for (int i = 0; i < gridSize; i++) {
		for (int j = 0; j < gridSize; j++) {
			int fineX, fineY, fineZ;
			switch (face) {
			case StitchFace::POS_X:
				// +X face: x is constant; vary y and z.
				// Set constant fineX to the coarse boundary.
				fineX = cX * ratio; // assume cX is the boundary voxel (i.e. cX = coarse cell's max x index)
				fineY = (cY * ratio) + i;
				fineZ = (cZ * ratio) + j;
				break;
			case StitchFace::NEG_X:
				// -X face: x is constant
				// For the negative face, we choose the fine coordinate corresponding to the front side.
				fineX = (cX + 1) * ratio; // adjust as desired (could also be cX*ratio)
				fineY = (cY * ratio) + i;
				fineZ = (cZ * ratio) + j;
				break;
			case StitchFace::POS_Y:
				// +Y face: y is constant; vary x and z.
				fineY = cY * ratio; // constant for +Y face boundary
				fineX = (cX * ratio) + i;
				fineZ = (cZ * ratio) + j;
				break;
			case StitchFace::NEG_Y:
				// -Y face: y is constant
				fineY = (cY + 1) * ratio; // adjust as desired
				fineX = (cX * ratio) + i;
				fineZ = (cZ * ratio) + j;
				break;
			case StitchFace::POS_Z:
				// +Z face: z is constant; vary x and y.
				fineZ = cZ * ratio; // constant for +Z boundary
				fineX = (cX * ratio) + i;
				fineY = (cY * ratio) + j;
				break;
			case StitchFace::NEG_Z:
				// -Z face: z is constant
				fineZ = (cZ + 1) * ratio; // adjust as desired
				fineX = (cX * ratio) + i;
				fineY = (cY * ratio) + j;
				break;
			}

			DCCellKey fineKey{ fLod, fineX, fineY, fineZ };
			int idx = fineIndex(i, j);
			auto itF = globalCellMap.find(fineKey);
			if (itF != globalCellMap.end() && itF->second.isMixed) {
				fineVerts[idx] = itF->second.dcVertex;
				fineNorms[idx] = itF->second.dcNormal;
			}
			else {
				// Fallback: use coarse vertex
				fineVerts[idx] = coarseV;
				fineNorms[idx] = coarseN;
			}
		}
	}

	// Now form bridging geometry.
	// For each subcell (each quad in the 2D grid of fine vertices),
	// create a fan of triangles connecting the coarse cell’s vertex to the four corners.
	for (int j = 0; j < ratio; j++) {
		for (int i = 0; i < ratio; i++) {
			// Get the four corners of the current quad.
			glm::vec3 v00 = fineVerts[fineIndex(i, j)];
			glm::vec3 v10 = fineVerts[fineIndex(i + 1, j)];
			glm::vec3 v11 = fineVerts[fineIndex(i + 1, j + 1)];
			glm::vec3 v01 = fineVerts[fineIndex(i, j + 1)];

			glm::vec3 n00 = fineNorms[fineIndex(i, j)];
			glm::vec3 n10 = fineNorms[fineIndex(i + 1, j)];
			glm::vec3 n11 = fineNorms[fineIndex(i + 1, j + 1)];
			glm::vec3 n01 = fineNorms[fineIndex(i, j + 1)];

			// One simple method is to create a fan around the coarse vertex.
			// We create four triangles:
			addTriangle(coarseV, v00, v10, coarseN, n00, n10, out);
			addTriangle(coarseV, v10, v11, coarseN, n10, n11, out);
			addTriangle(coarseV, v11, v01, coarseN, n11, n01, out);
			addTriangle(coarseV, v01, v00, coarseN, n01, n00, out);
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
glm::vec3 AdaptiveDualContouringRenderer::solveQEF(const std::vector<glm::vec3>& points,
	const std::vector<glm::vec3>& normals) {
	if (points.empty()) return glm::vec3(0.f);

	glm::vec3 centroid(0.f);
	for (auto& p : points) centroid += p;
	centroid /= (float)points.size();

	// Build mass-spring system matrix
	glm::mat3 A(0.f);
	glm::vec3 b(0.f);
	float massWeight = 0.1f; // Tune this parameter

	// Add point constraints with mass
	for (auto& p : points) {
		glm::vec3 d = p - centroid;
		A += massWeight * glm::mat3(1.0f);
		b += massWeight * d;
	}

	// Add normal constraints
	for (size_t i = 0; i < points.size(); i++) {
		glm::vec3 n = glm::normalize(normals[i]);
		A += glm::outerProduct(n, n);
		float d = glm::dot(points[i] - centroid, n);
		b += d * n;
	}

	// Solve using SVD or other stable method
	glm::mat3 U, V;
	glm::vec3 s;
	// Pseudo-inverse with threshold
	const float svdThreshold = 1e-6f;
	for (int i = 0; i < 3; i++) {
		if (s[i] > svdThreshold) s[i] = 1.0f / s[i];
		else s[i] = 0.0f;
	}

	glm::vec3 r = V * glm::mat3(s[0], 0, 0, 0, s[1], 0, 0, 0, s[2]) *
		glm::transpose(U) * b;

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
	std::vector<MCTriangle>& out) {
	// Add degenerate quad check
	float area1 = glm::length(glm::cross(v1 - v0, v2 - v0));
	float area2 = glm::length(glm::cross(v3 - v2, v1 - v2));
	if (area1 < 1e-6f || area2 < 1e-6f) return;

	// Check for quad planarity
	glm::vec3 n = glm::normalize(glm::cross(v1 - v0, v2 - v0));
	if (std::abs(glm::dot(v3 - v0, n)) > 1e-4f) {
		// Non-planar quad - consider alternative triangulation
		if (glm::length2(v2 - v0) < glm::length2(v3 - v1)) {
			addTriangle(v0, v1, v2, n0, n1, n2, out);
			addTriangle(v0, v2, v3, n0, n2, n3, out);
		}
		else {
			addTriangle(v0, v1, v3, n0, n1, n3, out);
			addTriangle(v1, v2, v3, n1, n2, n3, out);
		}
		return;
	}

	// Original triangulation if planar
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
