#include "BuildingLoader.h"
#include <gdal_priv.h>
#include <ogrsf_frmts.h>
#include <iostream>
#include <vector>
#include <glm/glm.hpp>
#include <limits>

static void processPolygon(OGRPolygon* polygon,
	VoxelGrid& grid,
	float minX, float minY, float minZ,
	float voxelSize,
	int dimX, int dimY, int dimZ)
{
	OGREnvelope envelope;
	// For 2D polygons, getEnvelope is enough. But if 3D is present, see below code
	polygon->getEnvelope(&envelope);

	std::cout << "Processing Polygon: Envelope ("
		<< envelope.MinX << ", " << envelope.MinY << ", "
		<< envelope.MaxX << ", " << envelope.MaxY << ")\n";

	int filledCount = 0;
	int startX = std::max(0, (int)((envelope.MinX - minX) / voxelSize));
	int endX = std::min(dimX, (int)((envelope.MaxX - minX) / voxelSize + 1));
	int startY = std::max(0, (int)((envelope.MinY - minY) / voxelSize));
	int endY = std::min(dimY, (int)((envelope.MaxY - minY) / voxelSize + 1));

	for (int z = 0; z < dimZ; z++) {
		for (int y = startY; y < endY; y++) {
			for (int x = startX; x < endX; x++) {
				float cx = minX + (x + 0.5f) * voxelSize;
				float cy = minY + (y + 0.5f) * voxelSize;
				float cz = minZ + (z + 0.5f) * voxelSize;

				// Only check polygon for x,y => 2D check
				// If you have 3D building footprints, you'd incorporate Z test or
				// building height data here.
				OGRPoint point(cx, cy);
				if (polygon->Contains(&point)) {
					int idx = x + y * dimX + z * (dimX * dimY);
					if (idx >= 0 && idx < (int)grid.data.size()) {
						grid.data[idx] = VoxelState::FILLED;
						filledCount++;
					}
				}
			}
		}
	}

	std::cout << "Filled " << filledCount << " voxels.\n";
}

/**
 * Loads buildings from a GDB.
 * If `structureIDs` is empty, loads only the first building.
 * Otherwise, loads all with matching STRUCT_ID.
 *
 * @param gdbPath       Path to the GDB file.
 * @param voxelSize     Size of each voxel (world units).
 * @param structureIDs  If empty => first building, else by ID list.
 *
 * @return A raw VoxelGrid, unshifted.
 */
VoxelGrid loadBuildingsFromGDB(const std::string& gdbPath,
	float voxelSize,
	const std::vector<int>& structureIDs)
{
	VoxelGrid grid;

	GDALAllRegister();
	GDALDataset* dataset =
		(GDALDataset*)GDALOpenEx(gdbPath.c_str(), GDAL_OF_VECTOR, nullptr, nullptr, nullptr);
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

	std::cout << "Feature count: " << layer->GetFeatureCount() << std::endl;
	layer->SetSpatialFilter(nullptr);

	float minX = std::numeric_limits<float>::max();
	float minY = std::numeric_limits<float>::max();
	float minZ = std::numeric_limits<float>::max();  // We'll override from geometry
	float maxX = -std::numeric_limits<float>::max();
	float maxY = -std::numeric_limits<float>::max();
	float maxZ = -std::numeric_limits<float>::max();

	bool foundAny = false;
	layer->ResetReading();
	OGRFeature* feature = nullptr;

	// ============== 1) FIRST PASS: bounding box in 3D ============== //
	while ((feature = layer->GetNextFeature()) != nullptr) {
		OGRGeometry* geometry = feature->GetGeometryRef();
		if (!geometry) {
			OGRFeature::DestroyFeature(feature);
			continue;
		}
		OGRwkbGeometryType geomType = wkbFlatten(geometry->getGeometryType());
		if (geomType != wkbPolygon && geomType != wkbMultiPolygon &&
			geomType != wkbPolygon25D && geomType != wkbMultiPolygon25D)
		{
			// see note if your data uses 2.5D or 3D polygons
			OGRFeature::DestroyFeature(feature);
			continue;
		}

		bool accept = false;
		if (structureIDs.empty()) {
			accept = !foundAny; // first building only
		}
		else {
			// check STRUCT_ID
			if (feature->IsFieldSet(10)) {
				int fid = feature->GetFieldAsInteger("STRUCT_ID");
				if (std::find(structureIDs.begin(), structureIDs.end(), fid)
					!= structureIDs.end()) {
					accept = true;
				}
			}
		}

		if (accept) {
			foundAny = true;
			// Envelope3D to get minZ,maxZ from geometry
			OGREnvelope3D env3d;
			geometry->getEnvelope(&env3d);

			// update bounding box
			if ((float)env3d.MinX < minX) minX = (float)env3d.MinX;
			if ((float)env3d.MinY < minY) minY = (float)env3d.MinY;
			if ((float)env3d.MinZ < minZ) minZ = (float)env3d.MinZ;

			if ((float)env3d.MaxX > maxX) maxX = (float)env3d.MaxX;
			if ((float)env3d.MaxY > maxY) maxY = (float)env3d.MaxY;
			if ((float)env3d.MaxZ > maxZ) maxZ = (float)env3d.MaxZ;

			if (structureIDs.empty()) {
				OGRFeature::DestroyFeature(feature);
				break;
			}
		}

		OGRFeature::DestroyFeature(feature);
	}

	if (!foundAny) {
		std::cout << "No buildings found. Return empty.\n";
		GDALClose(dataset);
		return grid;
	}

	// If your data is purely 2D, you might see minZ= +∞ or maxZ= -∞ => fallback
	if (minZ > maxZ) {
		// means no Z in data => fallback
		minZ = 0.f;
		maxZ = 50.f; // or some default
	}

	// pad 
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

	layer->ResetReading();
	foundAny = false;

	// ============== 2) SECOND PASS: fill data ============== //
	while ((feature = layer->GetNextFeature()) != nullptr) {
		OGRGeometry* geometry = feature->GetGeometryRef();
		if (!geometry) {
			OGRFeature::DestroyFeature(feature);
			continue;
		}
		OGRwkbGeometryType geomType = wkbFlatten(geometry->getGeometryType());
		if (geomType != wkbPolygon && geomType != wkbMultiPolygon &&
			geomType != wkbPolygon25D && geomType != wkbMultiPolygon25D) {
			OGRFeature::DestroyFeature(feature);
			continue;
		}

		bool accept = false;
		if (structureIDs.empty()) {
			accept = !foundAny;
		}
		else {
			if (feature->IsFieldSet(10)) {
				int fid = feature->GetFieldAsInteger("STRUCT_ID");
				if (std::find(structureIDs.begin(), structureIDs.end(), fid)
					!= structureIDs.end()) {
					accept = true;
				}
			}
		}

		if (accept) {
			foundAny = true;

			// If it's a single polygon
			if (geomType == wkbPolygon || geomType == wkbPolygon25D) {
				processPolygon((OGRPolygon*)geometry,
					grid, minX, minY, minZ,
					voxelSize, dimX, dimY, dimZ);
			}
			else if (geomType == wkbMultiPolygon || geomType == wkbMultiPolygon25D) {
				OGRMultiPolygon* mp = (OGRMultiPolygon*)geometry;
				for (int i = 0; i < mp->getNumGeometries(); i++) {
					OGRGeometry* subG = mp->getGeometryRef(i);
					if (wkbFlatten(subG->getGeometryType()) == wkbPolygon ||
						wkbFlatten(subG->getGeometryType()) == wkbPolygon25D)
					{
						processPolygon((OGRPolygon*)subG,
							grid, minX, minY, minZ,
							voxelSize, dimX, dimY, dimZ);
					}
				}
			}

			if (structureIDs.empty()) {
				OGRFeature::DestroyFeature(feature);
				break;
			}
		}
		OGRFeature::DestroyFeature(feature);
	}

	GDALClose(dataset);

	// Return the raw (unshifted) voxel grid
	return grid;
}
