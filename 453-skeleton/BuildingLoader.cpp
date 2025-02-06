#include "BuildingLoader.h"
#include <gdal_priv.h>
#include <ogrsf_frmts.h>
#include <iostream>
#include <vector>
#include <glm/glm.hpp>
#include <limits>
#include <algorithm>
#include <map>
#include <cmath>


struct BuildingEnvelope {
	float minX, minY, maxX, maxY;
	float minZ, maxZ;
	float height; // derived from maxZ - minZ
};


#define DEBUG_VOXELIZE 0

// Process a single polygon: mark filled voxels.
static void processPolygon(OGRPolygon* polygon,
	VoxelGrid& grid,
	float minX, float minY, float minZ,
	float voxelSize,
	int dimX, int dimY, int dimZ)
{
	OGREnvelope envelope;
	polygon->getEnvelope(&envelope);

#if DEBUG_VOXELIZE
	std::cout << "[processPolygon] Envelope: MinX = " << envelope.MinX
		<< ", MinY = " << envelope.MinY
		<< ", MaxX = " << envelope.MaxX
		<< ", MaxY = " << envelope.MaxY << "\n";
#endif

	int filledCount = 0;
	int startX = std::max(0, (int)((envelope.MinX - minX) / voxelSize));
	int endX = std::min(dimX, (int)((envelope.MaxX - minX) / voxelSize + 1));
	int startY = std::max(0, (int)((envelope.MinY - minY) / voxelSize));
	int endY = std::min(dimY, (int)((envelope.MaxY - minY) / voxelSize + 1));

#if DEBUG_VOXELIZE
	std::cout << "[processPolygon] Voxel X range: " << startX << " to " << endX - 1
		<< ", Y range: " << startY << " to " << endY - 1 << "\n";
#endif

	for (int z = 0; z < grid.dimZ; z++) {
		for (int y = startY; y < endY; y++) {
			for (int x = startX; x < endX; x++) {
				float cx = minX + (x + 0.5f) * voxelSize;
				float cy = minY + (y + 0.5f) * voxelSize;
				float cz = minZ + (z + 0.5f) * voxelSize;

				OGRPoint point(cx, cy);
				if (polygon->Contains(&point)) {
					int idx = x + y * grid.dimX + z * (grid.dimX * grid.dimY);
					if (idx >= 0 && idx < (int)grid.data.size()) {
						grid.data[idx] = VoxelState::FILLED;
						filledCount++;
					}
				}
			}
		}
	}

#if DEBUG_VOXELIZE
	std::cout << "[processPolygon] Filled " << filledCount << " voxels for this polygon.\n";
#endif
}


/**
 * Detect the bounding rectangle (in XY) of the top N tallest buildings in the entire GDB file.
 *
 * @param gdbPath     path to the GDB dataset
 * @param maxFeatures maximum number of features to consider (avoid reading all if huge)
 * @param topCount    how many tallest buildings to consider (e.g., 1 for single tallest,
 *                    or 10 to get a cluster of top-ten tallest)
 * @return            bounding box that encloses these top buildings in the XY plane
 */
BoundingBox2D detectTallestArea(const std::string& gdbPath,
	int maxFeatures = 50000,
	int topCount = 5)
{
	BoundingBox2D result = {
		std::numeric_limits<float>::max(),
		std::numeric_limits<float>::max(),
		-std::numeric_limits<float>::max(),
		-std::numeric_limits<float>::max()
	};

	GDALAllRegister();
	GDALDataset* dataset = (GDALDataset*)GDALOpenEx(gdbPath.c_str(), GDAL_OF_VECTOR, nullptr, nullptr, nullptr);
	if (!dataset) {
		std::cerr << "detectTallestArea: Failed to open GDB: " << gdbPath << "\n";
		std::abort();
	}

	OGRLayer* layer = dataset->GetLayer(0);
	if (!layer) {
		std::cerr << "detectTallestArea: Failed to get layer from GDB.\n";
		GDALClose(dataset);
		std::abort();
	}

	layer->ResetReading();
	OGRFeature* feature = nullptr;

	std::vector<BuildingEnvelope> envelopes;
	envelopes.reserve(1000); // just a guess to avoid repeated reallocation

	int count = 0;
	while ((feature = layer->GetNextFeature()) != nullptr && count < maxFeatures)
	{
		OGRGeometry* geometry = feature->GetGeometryRef();
		if (geometry)
		{
			OGRwkbGeometryType geomType = wkbFlatten(geometry->getGeometryType());
			if (geomType == wkbPolygon || geomType == wkbMultiPolygon ||
				geomType == wkbPolygon25D || geomType == wkbMultiPolygon25D)
			{
				// Envelope3D => so we can get minZ, maxZ
				OGREnvelope3D env3d;
				geometry->getEnvelope(&env3d);

				float h = (float)env3d.MaxZ - (float)env3d.MinZ;
				if (h < 0.f) {
					// pure 2D or nonsense: fallback
					h = 0.f;
				}

				BuildingEnvelope bld;
				bld.minX = (float)env3d.MinX;
				bld.minY = (float)env3d.MinY;
				bld.maxX = (float)env3d.MaxX;
				bld.maxY = (float)env3d.MaxY;
				bld.minZ = (float)env3d.MinZ;
				bld.maxZ = (float)env3d.MaxZ;
				bld.height = h;

				envelopes.push_back(bld);
				count++;
			}
		}
		OGRFeature::DestroyFeature(feature);
	}
	GDALClose(dataset);

	if (envelopes.empty()) {
		std::cout << "detectTallestArea: No building polygons found!\n";
		// Return a default bounding box
		return { 0,0,1000,1000 };
	}

	// Sort envelopes by height descending
	std::sort(envelopes.begin(), envelopes.end(),
		[](const BuildingEnvelope& a, const BuildingEnvelope& b) {
			return a.height > b.height;
		});

	// If topCount > the total number we found, clamp it
	if (topCount > (int)envelopes.size()) {
		topCount = (int)envelopes.size();
	}

	// Union the XY extents of the top N envelopes
	float unionMinX = std::numeric_limits<float>::max();
	float unionMinY = std::numeric_limits<float>::max();
	float unionMaxX = -std::numeric_limits<float>::max();
	float unionMaxY = -std::numeric_limits<float>::max();

	for (int i = 0; i < topCount; i++)
	{
		const auto& b = envelopes[i];
		if (b.minX < unionMinX) unionMinX = b.minX;
		if (b.minY < unionMinY) unionMinY = b.minY;
		if (b.maxX > unionMaxX) unionMaxX = b.maxX;
		if (b.maxY > unionMaxY) unionMaxY = b.maxY;
	}

	// Optionally expand the bounding box by some margin
	float expandDist = 50.f; // expand by 50 meters around
	unionMinX -= expandDist;
	unionMinY -= expandDist;
	unionMaxX += expandDist;
	unionMaxY += expandDist;

	BoundingBox2D topBox = {
		unionMinX, unionMinY,
		unionMaxX, unionMaxY
	};

	std::cout << "Tallest area bounding box (top " << topCount << " bldgs): ["
		<< topBox.minX << ", " << topBox.minY << "] -> ["
		<< topBox.maxX << ", " << topBox.maxY << "]\n";
	return topBox;
}

/**
 * Finds a 2D region whose buildings collectively have the highest average height.
 *  - Loops over buildings (up to maxFeatures).
 *  - For each building, determines its bounding envelope (MinX, MinY, MaxX, MaxY, MinZ, MaxZ).
 *  - Computes the building's height = (MaxZ - MinZ).
 *  - Puts the building's *center* into a bin grid.
 *  - In that bin, accumulates totalHeight += buildingHeight, count++.
 *  - After the pass, picks the bin with the highest averageHeight = totalHeight / count.
 *  - Returns a bounding box around that bin (optionally expanded).
 *
 * @param gdbPath       Path to the GDB file
 * @param maxFeatures   Maximum number of buildings to consider (avoid scanning everything).
 * @param cellSize      Width/height (in meters) of each bin cell in 2D.
 * @return              A bounding box that encloses the bin with the highest *average* building height.
 */
BoundingBox2D findSkyscraperArea(const std::string& gdbPath,
	int maxFeatures = 100000,
	float cellSize = 500.0f)
{
	// We'll accumulate building centers for the bounding region in XY
	BoundingBox2D region = {
		std::numeric_limits<float>::max(),
		std::numeric_limits<float>::max(),
		-std::numeric_limits<float>::max(),
		-std::numeric_limits<float>::max()
	};

	// We'll store each building as {centerX, centerY, height}
	struct BldgInfo { float x, y, h; };
	std::vector<BldgInfo> buildings;

	GDALAllRegister();
	GDALDataset* dataset = (GDALDataset*)GDALOpenEx(gdbPath.c_str(), GDAL_OF_VECTOR, nullptr, nullptr, nullptr);
	if (!dataset) {
		std::cerr << "findSkyscraperArea: Failed to open GDB: " << gdbPath << "\n";
		std::abort();
	}

	OGRLayer* layer = dataset->GetLayer(0);
	if (!layer) {
		std::cerr << "findSkyscraperArea: Failed to get layer from GDB.\n";
		GDALClose(dataset);
		std::abort();
	}

	// (Optional) Print SRS
	{
		OGRSpatialReference* srs = layer->GetSpatialRef();
		if (srs) {
			char* wkt = nullptr;
			srs->exportToWkt(&wkt);
			std::cout << "Layer Spatial Reference (WKT):\n" << wkt << "\n";
			CPLFree(wkt);
		}
	}

	layer->ResetReading();
	OGRFeature* feature = nullptr;
	int count = 0;

	// ========== Pass 1: Collect building center and height ========== 
	while ((feature = layer->GetNextFeature()) != nullptr && count < maxFeatures)
	{
		OGRGeometry* geometry = feature->GetGeometryRef();
		if (geometry)
		{
			OGRwkbGeometryType geomType = wkbFlatten(geometry->getGeometryType());
			if (geomType == wkbPolygon || geomType == wkbMultiPolygon ||
				geomType == wkbPolygon25D || geomType == wkbMultiPolygon25D)
			{
				// Envelope3D to get min/max X, Y, Z
				OGREnvelope3D env3d;
				geometry->getEnvelope(&env3d);

				float minx = (float)env3d.MinX;
				float miny = (float)env3d.MinY;
				float maxx = (float)env3d.MaxX;
				float maxy = (float)env3d.MaxY;
				float minz = (float)env3d.MinZ;
				float maxz = (float)env3d.MaxZ;

				float cx = 0.5f * (minx + maxx);
				float cy = 0.5f * (miny + maxy);
				float height = maxz - minz;

				// We only care about truly "tall" if > 0, but can also skip if height <= 0 
				if (height < 0.f) { // negative or weird
					height = 0.f;
				}

				buildings.push_back({ cx, cy, height });

				// Expand bounding region for the entire data 
				if (cx < region.minX) region.minX = cx;
				if (cy < region.minY) region.minY = cy;
				if (cx > region.maxX) region.maxX = cx;
				if (cy > region.maxY) region.maxY = cy;

				count++;
			}
		}
		OGRFeature::DestroyFeature(feature);
	}
	GDALClose(dataset);

	if (buildings.empty()) {
		std::cout << "findSkyscraperArea: No buildings found!\n";
		return { 0,0,1000,1000 };
	}

	// ========== Pass 2: Bin into a grid, store sumHeights & count for each bin ==========

	// The bounding region for building centers:
	float w = region.maxX - region.minX;
	float h = region.maxY - region.minY;
	if (w <= 0.f || h <= 0.f) {
		std::cout << "findSkyscraperArea: Invalid bounding region for building centers.\n";
		return region; // might be degenerate
	}

	int gridX = (int)std::ceil(w / cellSize);
	int gridY = (int)std::ceil(h / cellSize);

	// Each bin tracks totalHeight & buildingCount
	struct Cell {
		double totalHeight = 0.0;
		int    count = 0;
	};
	std::vector<Cell> bins(gridX * gridY);

	// Populate bins
	for (auto& bldg : buildings)
	{
		float relx = bldg.x - region.minX;  // relative to bounding region
		float rely = bldg.y - region.minY;
		int ix = (int)std::floor(relx / cellSize);
		int iy = (int)std::floor(rely / cellSize);
		if (ix < 0) ix = 0;
		if (ix >= gridX) ix = gridX - 1;
		if (iy < 0) iy = 0;
		if (iy >= gridY) iy = gridY - 1;

		int idx = iy * gridX + ix;
		bins[idx].totalHeight += (double)bldg.h;
		bins[idx].count++;
	}

	// ========== Pass 3: Find bin with the highest average height. ==========

	double bestAverage = 0.0;
	int bestIx = 0, bestIy = 0;

	for (int iy = 0; iy < gridY; iy++)
	{
		for (int ix = 0; ix < gridX; ix++)
		{
			int idx = iy * gridX + ix;
			int c = bins[idx].count;
			if (c > 0) {
				double avg = bins[idx].totalHeight / (double)c;
				if (avg > bestAverage) {
					bestAverage = avg;
					bestIx = ix;
					bestIy = iy;
				}
			}
		}
	}

	// Compute bounding box of that cell in XY
	float cellMinX = region.minX + bestIx * cellSize;
	float cellMinY = region.minY + bestIy * cellSize;
	float cellMaxX = cellMinX + cellSize;
	float cellMaxY = cellMinY + cellSize;

	// Optionally expand this box
	float expand = 0.5f * cellSize; // or a custom approach
	BoundingBox2D skyscraperBox = {
		cellMinX - expand,
		cellMinY - expand,
		cellMaxX + expand,
		cellMaxY + expand
	};

	std::cout << "findSkyscraperArea: highest average building height = "
		<< bestAverage << " in bin ["
		<< bestIx << "," << bestIy << "].\n"
		<< "XY bounds => [(" << skyscraperBox.minX << ", " << skyscraperBox.minY
		<< "), (" << skyscraperBox.maxX << ", " << skyscraperBox.maxY << ")]\n";

	return skyscraperBox;
}

// Detects a “dense area” by counting building center points.
// Parameters:
//   gdbPath: the path to the GDB dataset.
//   maxFeatures: maximum number of features to consider (to avoid processing millions).
//   cellSize: size (in meters) for each grid cell in the density grid.
BoundingBox2D detectDenseArea(const std::string& gdbPath, int maxFeatures = 10000, float cellSize = 500.0f)
{
	BoundingBox2D region = { std::numeric_limits<float>::max(),
							 std::numeric_limits<float>::max(),
							 -std::numeric_limits<float>::max(),
							 -std::numeric_limits<float>::max() };

	std::vector<glm::vec2> centers;

	GDALAllRegister();
	GDALDataset* dataset = (GDALDataset*)GDALOpenEx(gdbPath.c_str(), GDAL_OF_VECTOR, nullptr, nullptr, nullptr);
	if (!dataset) {
		std::cerr << "detectDenseArea: Failed to open GDB: " << gdbPath << "\n";
		std::abort();
	}
	OGRLayer* layer = dataset->GetLayer(0);
	if (!layer) {
		std::cerr << "detectDenseArea: Failed to get layer from GDB.\n";
		GDALClose(dataset);
		std::abort();
	}

	layer->ResetReading();
	OGRFeature* feature = nullptr;
	int count = 0;
	while ((feature = layer->GetNextFeature()) != nullptr && count < maxFeatures)
	{
		OGRGeometry* geometry = feature->GetGeometryRef();
		if (geometry)
		{
			// We consider polygon types.
			OGRwkbGeometryType geomType = wkbFlatten(geometry->getGeometryType());
			if (geomType == wkbPolygon || geomType == wkbMultiPolygon ||
				geomType == wkbPolygon25D || geomType == wkbMultiPolygon25D)
			{
				OGREnvelope envelope;
				geometry->getEnvelope(&envelope);
				float centerX = 0.5f * (envelope.MinX + envelope.MaxX);
				float centerY = 0.5f * (envelope.MinY + envelope.MaxY);
				centers.push_back(glm::vec2(centerX, centerY));

				// Expand overall bounds of centers.
				if (centerX < region.minX) region.minX = centerX;
				if (centerY < region.minY) region.minY = centerY;
				if (centerX > region.maxX) region.maxX = centerX;
				if (centerY > region.maxY) region.maxY = centerY;

				count++;
			}
		}
		OGRFeature::DestroyFeature(feature);
	}
	GDALClose(dataset);

	if (centers.empty())
	{
		std::cout << "detectDenseArea: No building centers found!\n";
		// Return a default bounding box.
		return { 0,0,1000,1000 };
	}

	// Create a grid over the extent of the centers.
	int gridCellsX = (int)std::ceil((region.maxX - region.minX) / cellSize);
	int gridCellsY = (int)std::ceil((region.maxY - region.minY) / cellSize);
	std::vector<int> density(gridCellsX * gridCellsY, 0);

	// Count centers in each grid cell.
	for (const auto& pt : centers)
	{
		int cellX = (int)std::floor((pt.x - region.minX) / cellSize);
		int cellY = (int)std::floor((pt.y - region.minY) / cellSize);
		// Clamp indices
		if (cellX < 0) cellX = 0;
		if (cellX >= gridCellsX) cellX = gridCellsX - 1;
		if (cellY < 0) cellY = 0;
		if (cellY >= gridCellsY) cellY = gridCellsY - 1;
		density[cellY * gridCellsX + cellX]++;
	}

	// Find the grid cell with the highest count.
	int bestCount = 0, bestCellX = 0, bestCellY = 0;
	for (int j = 0; j < gridCellsY; ++j)
	{
		for (int i = 0; i < gridCellsX; ++i)
		{
			int idx = j * gridCellsX + i;
			if (density[idx] > bestCount)
			{
				bestCount = density[idx];
				bestCellX = i;
				bestCellY = j;
			}
		}
	}

	// Compute bounding box of the best cell.
	float cellMinX = region.minX + bestCellX * cellSize;
	float cellMinY = region.minY + bestCellY * cellSize;
	float cellMaxX = cellMinX + cellSize;
	float cellMaxY = cellMinY + cellSize;

	// Optionally, you might expand the cell a bit (e.g., by 50%)
	float expand = 0.5f * cellSize;
	BoundingBox2D downtown = {
		cellMinX - expand,
		cellMinY - expand,
		cellMaxX + expand,
		cellMaxY + expand
	};

	std::cout << "Detected dense area (downtown): ["
		<< downtown.minX << ", " << downtown.minY << "] to ["
		<< downtown.maxX << ", " << downtown.maxY << "], "
		<< "with cell density " << bestCount << "\n";

	return downtown;
}


/**
 * Loads buildings from a GDB within a given bounding rectangle, and limits the number
 * of buildings loaded to maxBuildings.
 *
 * @param gdbPath        Path to the GDB file.
 * @param voxelSize      Size of each voxel (world units).
 * @param minXUser       Left boundary of the region of interest (in meters).
 * @param minYUser       Bottom boundary of the region of interest (in meters).
 * @param maxXUser       Right boundary of the region of interest (in meters).
 * @param maxYUser       Top boundary of the region of interest (in meters).
 * @param maxBuildings   Maximum number of buildings to load.
 *
 * @return               A raw VoxelGrid (unshifted). You may recenter the grid later.
 */
VoxelGrid loadBuildingsFromGDB(const std::string& gdbPath,
	float voxelSize,
	float minXUser,
	float minYUser,
	float maxXUser,
	float maxYUser,
	int maxBuildings)
{
	VoxelGrid grid;

	GDALAllRegister();
	GDALDataset* dataset = (GDALDataset*)GDALOpenEx(gdbPath.c_str(), GDAL_OF_VECTOR, nullptr, nullptr, nullptr);
	if (!dataset) {
		std::cerr << "Failed to open GDB: " << gdbPath << "\n";
		std::abort();
	}

	OGRLayer* layer = dataset->GetLayer(0);
	if (!layer) {
		std::cerr << "Failed to get layer from GDB.\n";
		GDALClose(dataset);
		std::abort();
	}

	// Print the spatial reference for verification.
	OGRSpatialReference* srs = layer->GetSpatialRef();
	if (srs) {
		char* wkt = nullptr;
		srs->exportToWkt(&wkt);
		std::cout << "Layer Spatial Reference (WKT):\n" << wkt << "\n";
		CPLFree(wkt);
		if (srs->IsProjected())
			std::cout << "Data is projected. Measurements are in metres.\n";
		else if (srs->IsGeographic())
			std::cout << "Data is geographic. Measurements are in degrees.\n";
	}
	else {
		std::cout << "No spatial reference found for this layer.\n";
	}

	// Set spatial filter to restrict features to the region of interest.
	layer->SetSpatialFilterRect(minXUser, minYUser, maxXUser, maxYUser);

	std::cout << "Feature count (after spatial filter): " << layer->GetFeatureCount() << std::endl;

	// First pass: determine overall bounding box (in all three dimensions) for the accepted buildings.
	float minX = std::numeric_limits<float>::max();
	float minY = std::numeric_limits<float>::max();
	float minZ = std::numeric_limits<float>::max();
	float maxX = -std::numeric_limits<float>::max();
	float maxY = -std::numeric_limits<float>::max();
	float maxZ = -std::numeric_limits<float>::max();

	layer->ResetReading();
	OGRFeature* feature = nullptr;
	int buildingCount = 0;
	while ((feature = layer->GetNextFeature()) != nullptr) {
		if (buildingCount >= maxBuildings) {
			OGRFeature::DestroyFeature(feature);
			break;
		}

		OGRGeometry* geometry = feature->GetGeometryRef();
		if (!geometry) {
			OGRFeature::DestroyFeature(feature);
			continue;
		}
		OGRwkbGeometryType geomType = wkbFlatten(geometry->getGeometryType());
		if (geomType != wkbPolygon && geomType != wkbMultiPolygon &&
			geomType != wkbPolygon25D && geomType != wkbMultiPolygon25D)
		{
			OGRFeature::DestroyFeature(feature);
			continue;
		}

		// Get full 3D envelope (using Envelope3D)
		OGREnvelope3D env3d;
		geometry->getEnvelope(&env3d);

		if ((float)env3d.MinX < minX) minX = (float)env3d.MinX;
		if ((float)env3d.MinY < minY) minY = (float)env3d.MinY;
		if ((float)env3d.MinZ < minZ) minZ = (float)env3d.MinZ;
		if ((float)env3d.MaxX > maxX) maxX = (float)env3d.MaxX;
		if ((float)env3d.MaxY > maxY) maxY = (float)env3d.MaxY;
		if ((float)env3d.MaxZ > maxZ) maxZ = (float)env3d.MaxZ;

		buildingCount++;
		OGRFeature::DestroyFeature(feature);
	}

	if (buildingCount == 0) {
		std::cout << "No buildings found in the specified bounding region. Returning empty grid.\n";
		GDALClose(dataset);
		return grid;
	}

	// If no proper Z extents are found (pure 2D data), provide a default range.
	if (minZ > maxZ) {
		minZ = 0.f;
		maxZ = 50.f;
	}

	// Pad the bounding box a bit.
	float pad = 2.f * voxelSize;
	minX -= pad;
	minY -= pad;
	minZ -= pad;
	maxX += pad;
	maxY += pad;
	maxZ += pad;

	int dimX = (int)std::ceil((maxX - minX) / voxelSize);
	int dimY = (int)std::ceil((maxY - minY) / voxelSize);
	int dimZ = (int)std::ceil((maxZ - minZ) / voxelSize);

	grid.dimX = dimX;
	grid.dimY = dimY;
	grid.dimZ = dimZ;
	grid.minX = minX;
	grid.minY = minY;
	grid.minZ = minZ;
	grid.voxelSize = voxelSize;
	grid.data.resize(dimX * dimY * dimZ, VoxelState::EMPTY);

	// Second pass: Process each feature and fill the voxel grid.
	layer->ResetReading();
	buildingCount = 0;
	while ((feature = layer->GetNextFeature()) != nullptr) {
		if (buildingCount >= maxBuildings) {
			OGRFeature::DestroyFeature(feature);
			break;
		}

		OGRGeometry* geometry = feature->GetGeometryRef();
		if (!geometry) {
			OGRFeature::DestroyFeature(feature);
			continue;
		}
		OGRwkbGeometryType geomType = wkbFlatten(geometry->getGeometryType());
		if (geomType != wkbPolygon && geomType != wkbMultiPolygon &&
			geomType != wkbPolygon25D && geomType != wkbMultiPolygon25D)
		{
			OGRFeature::DestroyFeature(feature);
			continue;
		}

		// Process single or multi-polygon features.
		if (geomType == wkbPolygon || geomType == wkbPolygon25D) {
			processPolygon((OGRPolygon*)geometry,
				grid, minX, minY, minZ,
				voxelSize, dimX, dimY, dimZ);
		}
		else if (geomType == wkbMultiPolygon || geomType == wkbMultiPolygon25D) {
			OGRMultiPolygon* mp = (OGRMultiPolygon*)geometry;
			for (int i = 0; i < mp->getNumGeometries(); i++) {
				OGRGeometry* subG = mp->getGeometryRef(i);
				if (!subG) continue;
				OGRwkbGeometryType subT = wkbFlatten(subG->getGeometryType());
				if (subT == wkbPolygon || subT == wkbPolygon25D) {
					processPolygon((OGRPolygon*)subG,
						grid, minX, minY, minZ,
						voxelSize, dimX, dimY, dimZ);
				}
			}
		}
		buildingCount++;
		OGRFeature::DestroyFeature(feature);
	}

	GDALClose(dataset);

	std::cout << "Loaded " << buildingCount << " buildings (limit: " << maxBuildings << ").\n";
	return grid;
}

static void processPolygon3D(OGRPolygon* polygon,
	VoxelGrid& grid,
	float minX, float minY, float minZ,
	float voxelSize,
	int dimX, int dimY, int dimZ)
{
	// Retrieve the 3D envelope from the polygon geometry
	OGREnvelope envelope2D;
	polygon->getEnvelope(&envelope2D);  // (2D bounding box in XY)

	// For the Z-range, we rely on the polygon geometry's 3D envelope
	// But since we only have the 2D polygon at this point, we store Z separately (below).
	OGREnvelope3D env3d;
	polygon->getEnvelope(&env3d);  // This is often still the same XY but includes MinZ/MaxZ

	float polyMinZ = static_cast<float>(env3d.MinZ);
	float polyMaxZ = static_cast<float>(env3d.MaxZ);

	if (polyMinZ > polyMaxZ) {
		// If there's no real 3D data, fallback to default.
		polyMinZ = 0.f;
		polyMaxZ = 0.f;
	}

	// Convert the polygon's XY envelope to voxel indices
	int startX = std::max(0, (int)std::floor((envelope2D.MinX - minX) / voxelSize));
	int endX = std::min(dimX, (int)std::ceil((envelope2D.MaxX - minX) / voxelSize));
	int startY = std::max(0, (int)std::floor((envelope2D.MinY - minY) / voxelSize));
	int endY = std::min(dimY, (int)std::ceil((envelope2D.MaxY - minY) / voxelSize));

	// Convert the polygon's Z envelope to voxel indices
	int startZ = std::max(0, (int)std::floor((polyMinZ - minZ) / voxelSize));
	int endZ = std::min(dimZ, (int)std::ceil((polyMaxZ - minZ) / voxelSize));

	std::cout << "[processPolygon3D] XY envelope: (" << envelope2D.MinX << ", " << envelope2D.MinY
		<< ") to (" << envelope2D.MaxX << ", " << envelope2D.MaxY << ")\n"
		<< "Voxel X range = " << startX << ".." << endX - 1
		<< ", Y range = " << startY << ".." << endY - 1 << "\n"
		<< "Z range = " << startZ << ".." << endZ - 1 << std::endl;

	int filledCount = 0;

	// For each voxel in the bounding XY, check polygon containment in XY.
	// Then fill all Z slices from startZ..endZ for that XY cell.
	for (int y = startY; y < endY; y++) {
		for (int x = startX; x < endX; x++) {
			float cx = minX + (x + 0.5f) * voxelSize;
			float cy = minY + (y + 0.5f) * voxelSize;

			OGRPoint point(cx, cy);

			// If the XY center is inside the polygon footprint, fill every Z slice
			if (polygon->Contains(&point)) {
				// Fill Z from startZ to endZ
				for (int z = startZ; z < endZ; z++) {
					int idx = x + y * dimX + z * (dimX * dimY);
					if (idx >= 0 && idx < (int)grid.data.size()) {
						grid.data[idx] = VoxelState::FILLED;
						filledCount++;
					}
				}
			}
		}
	}

	std::cout << "[processPolygon3D] Filled " << filledCount
		<< " voxel cells for this building's extrusion.\n";
}

//-----------------------------------------------------------------
// Helper function: SetZRecursively
//-----------------------------------------------------------------
static void SetZRecursively(OGRGeometry* g, double zVal)
{
	if (!g)
		return;

	// Get the flattened type (ignoring Z/M flags)
	OGRwkbGeometryType type = wkbFlatten(g->getGeometryType());

	if (type == wkbLineString || type == wkbLinearRing)
	{
		OGRLineString* ls = (OGRLineString*)g;
		ls->setCoordinateDimension(3); // ensure we're working in 3D
		int n = ls->getNumPoints();
		for (int i = 0; i < n; i++)
		{
			double x = ls->getX(i);
			double y = ls->getY(i);
			// We ignore the original z value and set it to zVal.
			ls->setPoint(i, x, y, zVal);
		}
	}
	else if (type == wkbPolygon)
	{
		OGRPolygon* poly = (OGRPolygon*)g;
		// Process the exterior ring.
		OGRLinearRing* exterior = poly->getExteriorRing();
		if (exterior)
			SetZRecursively(exterior, zVal);
		// Process interior rings, if any.
		for (int i = 0; i < poly->getNumInteriorRings(); i++)
		{
			OGRLinearRing* interior = poly->getInteriorRing(i);
			SetZRecursively(interior, zVal);
		}
	}
	else if (type == wkbMultiPolygon)
	{
		OGRMultiPolygon* mp = (OGRMultiPolygon*)g;
		for (int i = 0; i < mp->getNumGeometries(); i++)
		{
			OGRGeometry* subG = mp->getGeometryRef(i);
			SetZRecursively(subG, zVal);
		}
	}
	// Add additional geometry types as needed.
}


//-----------------------------------------------------------------
// Function: closeBuildingAtMinZ
//
// Closes a building geometry by constructing a flat base at its minimum Z.
// Returns a new geometry (which you are responsible for destroying)
// that is the union of the original geometry and its flattened 2D base.
//-----------------------------------------------------------------
OGRGeometry* closeBuildingAtMinZ(OGRGeometry* origGeom)
{
	if (!origGeom)
		return nullptr;

	// Clone the original geometry so we do not alter the input.
	OGRGeometry* geom = origGeom->clone();

	// Get the building's 3D envelope.
	OGREnvelope3D env3d;
	geom->getEnvelope(&env3d);
	double minZ = env3d.MinZ;  // This is the lowest z value in the building.

	// Create a clone for the base.
	OGRGeometry* geomBase = geom->clone();
	// Flatten it to 2D (this sets all Z values to 0, typically)
	geomBase->flattenTo2D();
	// Now, force every coordinate in geomBase to have Z = minZ.
	SetZRecursively(geomBase, minZ);

	// Attempt to union the original geometry with the base.
	// (Note: Union() in OGR is primarily 2D; results may vary.)
	OGRGeometry* closedGeom = geom->Union(geomBase);
	if (!closedGeom) {
		std::cerr << "[closeBuildingAtMinZ] Union failed, using original geometry clone instead.\n";
		closedGeom = geom->clone();
	}

	// Clean up temporary geometries.
	OGRGeometryFactory::destroyGeometry(geom);
	OGRGeometryFactory::destroyGeometry(geomBase);

	return closedGeom;
}

//-----------------------------------------
// 1) FULLY 3D VOXELIZATION METHOD
//-----------------------------------------
static void processFully3D(OGRGeometry* geometry,
	VoxelGrid& grid,
	float minX, float minY, float minZ,
	float voxelSize,
	int dimX, int dimY, int dimZ)
{
	// 3D bounding box from the geometry
	OGREnvelope3D env3d;
	geometry->getEnvelope(&env3d);
	float bldMinX = static_cast<float>(env3d.MinX);
	float bldMinY = static_cast<float>(env3d.MinY);
	float bldMinZ = static_cast<float>(env3d.MinZ);
	float bldMaxX = static_cast<float>(env3d.MaxX);
	float bldMaxY = static_cast<float>(env3d.MaxY);
	float bldMaxZ = static_cast<float>(env3d.MaxZ);

	// Convert that bounding region to voxel indices
	int startX = std::max(0, (int)std::floor((bldMinX - minX) / voxelSize));
	int endX = std::min(dimX, (int)std::ceil((bldMaxX - minX) / voxelSize));
	int startY = std::max(0, (int)std::floor((bldMinY - minY) / voxelSize));
	int endY = std::min(dimY, (int)std::ceil((bldMaxY - minY) / voxelSize));
	int startZ = std::max(0, (int)std::floor((bldMinZ - minZ) / voxelSize));
	int endZ = std::min(dimZ, (int)std::ceil((bldMaxZ - minZ) / voxelSize));

	std::cout << "[processFully3D] BLD bounding box => voxel range: X "
		<< startX << ".." << (endX - 1)
		<< ", Y " << startY << ".." << (endY - 1)
		<< ", Z " << startZ << ".." << (endZ - 1) << "\n";

	int filledCount = 0;

	// For each voxel in that region, do a 3D "Contains" test
	for (int z = startZ; z < endZ; z++) {
		for (int y = startY; y < endY; y++) {
			for (int x = startX; x < endX; x++) {
				float cx = minX + (x + 0.5f) * voxelSize;
				float cy = minY + (y + 0.5f) * voxelSize;
				float cz = minZ + (z + 0.5f) * voxelSize;

				// Build a 3D OGRPoint
				OGRPoint pt(cx, cy);
				pt.setZ(cz);  // Force 3D

				// If geometry is a closed volume, Contains() => inside
				if (geometry->Contains(&pt)) {
					int idx = x + y * dimX + z * (dimX * dimY);
					grid.data[idx] = VoxelState::FILLED;
					filledCount++;
				}
			}
		}
	}

	std::cout << "[processFully3D] Marked " << filledCount
		<< " voxels as FILLED for this building.\n";
}

//-----------------------------------------
// 2) MAIN LOADER
//-----------------------------------------
VoxelGrid loadBuildingsFromGDB(const std::string& gdbPath,
	float voxelSize,
	const std::vector<int>& structureIDs)
{
	VoxelGrid grid;

	GDALAllRegister();
	GDALDataset* dataset = (GDALDataset*)GDALOpenEx(gdbPath.c_str(), GDAL_OF_VECTOR, nullptr, nullptr, nullptr);
	if (!dataset) {
		std::cerr << "Failed to open GDB: " << gdbPath << "\n";
		std::abort();
	}

	OGRLayer* layer = dataset->GetLayer(0);
	if (!layer) {
		std::cerr << "Failed to get layer from GDB.\n";
		GDALClose(dataset);
		std::abort();
	}

	// Optional: print the spatial reference
	OGRSpatialReference* srs = layer->GetSpatialRef();
	if (srs) {
		char* wkt = nullptr;
		srs->exportToWkt(&wkt);
		std::cout << "Layer Spatial Reference:\n" << wkt << "\n";
		CPLFree(wkt);
	}

	// If user provided a list of building IDs, set attribute filter
	if (!structureIDs.empty()) {
		std::ostringstream oss;
		oss << "STRUCT_ID IN (";
		for (size_t i = 0; i < structureIDs.size(); i++) {
			oss << structureIDs[i];
			if (i + 1 < structureIDs.size()) oss << ",";
		}
		oss << ")";
		std::string filterStr = oss.str();
		std::cout << "[loadBuildingsFromGDB] Using attribute filter: " << filterStr << "\n";
		layer->SetAttributeFilter(filterStr.c_str());
	}

	// ============ First pass: bounding box =============
	float minX = std::numeric_limits<float>::max();
	float minY = std::numeric_limits<float>::max();
	float minZ = std::numeric_limits<float>::max();
	float maxX = -std::numeric_limits<float>::max();
	float maxY = -std::numeric_limits<float>::max();
	float maxZ = -std::numeric_limits<float>::max();

	layer->ResetReading();
	OGRFeature* feature = nullptr;
	int buildingCount = 0;

	while ((feature = layer->GetNextFeature()) != nullptr) {
		// if structureIDs empty => only first building
		if (structureIDs.empty() && buildingCount > 0) {
			OGRFeature::DestroyFeature(feature);
			break;
		}

		OGRGeometry* geom = feature->GetGeometryRef();
		if (!geom) {
			OGRFeature::DestroyFeature(feature);
			continue;
		}
		OGRwkbGeometryType gType = wkbFlatten(geom->getGeometryType());
		if (gType != wkbPolygon && gType != wkbMultiPolygon &&
			gType != wkbPolygon25D && gType != wkbMultiPolygon25D)
		{
			// skip non-polygon features
			OGRFeature::DestroyFeature(feature);
			continue;
		}

		// get 3D envelope
		OGREnvelope3D env3d;
		geom->getEnvelope(&env3d);
		minX = std::min(minX, (float)env3d.MinX);
		minY = std::min(minY, (float)env3d.MinY);
		minZ = std::min(minZ, (float)env3d.MinZ);
		maxX = std::max(maxX, (float)env3d.MaxX);
		maxY = std::max(maxY, (float)env3d.MaxY);
		maxZ = std::max(maxZ, (float)env3d.MaxZ);

		buildingCount++;
		OGRFeature::DestroyFeature(feature);
	}

	if (buildingCount == 0) {
		std::cout << "[loadBuildingsFromGDB] No matching buildings found. Return empty.\n";
		GDALClose(dataset);
		return grid;
	}
	if (minZ > maxZ) {
		minZ = 0.f;
		maxZ = 50.f;
	}

	// Expand bounding box slightly
	float pad = 2.f * voxelSize;
	minX -= pad;
	minY -= pad;
	minZ -= pad;
	maxX += pad;
	maxY += pad;
	maxZ += pad;

	int dimX = (int)std::ceil((maxX - minX) / voxelSize);
	int dimY = (int)std::ceil((maxY - minY) / voxelSize);
	int dimZ = (int)std::ceil((maxZ - minZ) / voxelSize);

	std::cout << "[loadBuildingsFromGDB] Global bounding box: ["
		<< minX << ", " << minY << ", " << minZ << "] -> ["
		<< maxX << ", " << maxY << ", " << maxZ << "]\n"
		<< "Voxel grid dimension: " << dimX << " x " << dimY << " x " << dimZ << "\n";

	// Allocate VoxelGrid
	grid.dimX = dimX;
	grid.dimY = dimY;
	grid.dimZ = dimZ;
	grid.minX = minX;
	grid.minY = minY;
	grid.minZ = minZ;
	grid.voxelSize = voxelSize;
	grid.data.resize(dimX * dimY * dimZ, VoxelState::EMPTY);

	// ============ Second pass: fill the VoxelGrid with "fully 3D" method =============
	layer->ResetReading();
	buildingCount = 0;

	while ((feature = layer->GetNextFeature()) != nullptr) {
		if (structureIDs.empty() && buildingCount > 0) {
			OGRFeature::DestroyFeature(feature);
			break;
		}

		OGRGeometry* origGeom = feature->GetGeometryRef();
		if (!origGeom) {
			OGRFeature::DestroyFeature(feature);
			continue;
		}

		// Close the building base.
		OGRGeometry* closedGeom = closeBuildingAtMinZ(origGeom);
		if (!closedGeom) {
			// Fallback: use original
			closedGeom = origGeom->clone();
		}

		// Flatten the geometry type
		OGRwkbGeometryType gType = wkbFlatten(closedGeom->getGeometryType());
		if (gType != wkbPolygon && gType != wkbMultiPolygon &&
			gType != wkbPolygon25D && gType != wkbMultiPolygon25D)
		{
			std::cout << "[loadBuildings] Skipping non-polygon geometry.\n";
			OGRFeature::DestroyFeature(feature);
			OGRGeometryFactory::destroyGeometry(closedGeom);
			continue;
		}

		std::cout << "[loadBuildings] Voxelizing feature " << buildingCount << " in 3D...\n";

		// Now, actually do the 3D fill on the CLOSED geometry
		if (gType == wkbMultiPolygon || gType == wkbMultiPolygon25D) {
			// It's a multipolygon => iterate subgeometries
			OGRMultiPolygon* mp = (OGRMultiPolygon*)closedGeom;
			for (int i = 0; i < mp->getNumGeometries(); i++) {
				OGRGeometry* subG = mp->getGeometryRef(i);
				if (!subG) continue;
				processFully3D(subG, grid,
					minX, minY, minZ,
					voxelSize, dimX, dimY, dimZ);
			}
		}
		else {
			// single polygon
			processFully3D(closedGeom, grid,
				minX, minY, minZ,
				voxelSize, dimX, dimY, dimZ);
		}

		// Clean up
		OGRGeometryFactory::destroyGeometry(closedGeom);
		OGRFeature::DestroyFeature(feature);
		buildingCount++;
	}

	GDALClose(dataset);
	std::cout << "[loadBuildingsFromGDB] Voxelized " << buildingCount
		<< " building(s) in fully 3D mode.\n";
	return grid;
}
