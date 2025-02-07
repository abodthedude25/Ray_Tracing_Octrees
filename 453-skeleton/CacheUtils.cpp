
#include "CacheUtils.h"
#include <iostream>

bool saveVoxelGrid(const std::string& filename, const VoxelGrid& grid)
{
	std::ofstream out(filename, std::ios::binary);
	if (!out) {
		std::cerr << "Cannot open file for writing: " << filename << "\n";
		return false;
	}

	// Write basic grid parameters
	out.write(reinterpret_cast<const char*>(&grid.dimX), sizeof(grid.dimX));
	out.write(reinterpret_cast<const char*>(&grid.dimY), sizeof(grid.dimY));
	out.write(reinterpret_cast<const char*>(&grid.dimZ), sizeof(grid.dimZ));
	out.write(reinterpret_cast<const char*>(&grid.minX), sizeof(grid.minX));
	out.write(reinterpret_cast<const char*>(&grid.minY), sizeof(grid.minY));
	out.write(reinterpret_cast<const char*>(&grid.minZ), sizeof(grid.minZ));
	out.write(reinterpret_cast<const char*>(&grid.voxelSize), sizeof(grid.voxelSize));

	// Write voxel data size and then data array
	size_t dataSize = grid.data.size();
	out.write(reinterpret_cast<const char*>(&dataSize), sizeof(dataSize));
	out.write(reinterpret_cast<const char*>(grid.data.data()), dataSize * sizeof(grid.data[0]));

	out.close();
	std::cout << "Saved voxel grid to " << filename << "\n";
	return true;
}

// This function loads the entire voxel grid from a binary cache file.
bool loadVoxelGrid(const std::string& filename, VoxelGrid& grid)
{
	std::ifstream in(filename, std::ios::binary);
	if (!in) {
		std::cerr << "Cannot open file for reading: " << filename << "\n";
		return false;
	}

	// Read basic grid parameters.
	in.read(reinterpret_cast<char*>(&grid.dimX), sizeof(grid.dimX));
	in.read(reinterpret_cast<char*>(&grid.dimY), sizeof(grid.dimY));
	in.read(reinterpret_cast<char*>(&grid.dimZ), sizeof(grid.dimZ));
	in.read(reinterpret_cast<char*>(&grid.minX), sizeof(grid.minX));
	in.read(reinterpret_cast<char*>(&grid.minY), sizeof(grid.minY));
	in.read(reinterpret_cast<char*>(&grid.minZ), sizeof(grid.minZ));
	in.read(reinterpret_cast<char*>(&grid.voxelSize), sizeof(grid.voxelSize));

	// Read voxel data size and then the full data array.
	size_t dataSize;
	in.read(reinterpret_cast<char*>(&dataSize), sizeof(dataSize));
	grid.data.resize(dataSize);
	in.read(reinterpret_cast<char*>(grid.data.data()), dataSize * sizeof(grid.data[0]));

	in.close();
	std::cout << "Loaded voxel grid from " << filename << "\n";
	return true;
}

// -----------------------------------------------------------------------------
// NEW FUNCTION: loadVoxelGridPartial
//
// This function loads only a portion (subvolume) of the full voxel grid from file.
// Here we assume that the grid is segmented along the Z–axis. You provide a starting
// layer index (startLayer) and the number of layers (numLayers) you wish to load.
// The function reads the grid parameters (dimX, dimY, full dimZ, etc.), then skips
// ahead in the voxel data to read only the requested portion. It also updates grid.dimZ
// and grid.minZ accordingly.
//
// An interactive UI (for example, a slider) can set startLayer and numLayers (or a fraction)
// so that only a fraction of the scene is loaded at a time.
bool loadVoxelGridPartial(const std::string& filename, VoxelGrid& grid, int startLayer, int numLayers)
{
	std::ifstream in(filename, std::ios::binary);
	if (!in) {
		std::cerr << "Cannot open file for reading: " << filename << "\n";
		return false;
	}

	// Read the full grid parameters.
	in.read(reinterpret_cast<char*>(&grid.dimX), sizeof(grid.dimX));
	in.read(reinterpret_cast<char*>(&grid.dimY), sizeof(grid.dimY));
	in.read(reinterpret_cast<char*>(&grid.dimZ), sizeof(grid.dimZ));
	in.read(reinterpret_cast<char*>(&grid.minX), sizeof(grid.minX));
	in.read(reinterpret_cast<char*>(&grid.minY), sizeof(grid.minY));
	in.read(reinterpret_cast<char*>(&grid.minZ), sizeof(grid.minZ));
	in.read(reinterpret_cast<char*>(&grid.voxelSize), sizeof(grid.voxelSize));

	// Read the full voxel data size.
	size_t fullDataSize;
	in.read(reinterpret_cast<char*>(&fullDataSize), sizeof(fullDataSize));
	// fullDataSize should equal grid.dimX * grid.dimY * (full grid dimZ)

	// Validate the requested subvolume along the Z axis.
	int totalLayers = grid.dimZ;
	if (startLayer < 0 || startLayer >= totalLayers || startLayer + numLayers > totalLayers) {
		std::cerr << "Requested subvolume layers are out of bounds." << std::endl;
		return false;
	}

	// Compute the number of voxels per layer.
	int layerSize = grid.dimX * grid.dimY;
	int partialSize = layerSize * numLayers;
	std::vector<VoxelState> partialData(partialSize);

	// Skip the data for the first 'startLayer' layers.
	std::streamoff offset = static_cast<std::streamoff>(startLayer) * layerSize * sizeof(VoxelState);
	in.seekg(offset, std::ios::cur);

	// Read the partial data.
	in.read(reinterpret_cast<char*>(partialData.data()), partialSize * sizeof(VoxelState));

	in.close();
	// Update the grid parameters to reflect the partial grid.
	grid.dimZ = numLayers;
	grid.minZ += startLayer * grid.voxelSize;
	grid.data = std::move(partialData);

	std::cout << "Loaded partial voxel grid (" << numLayers << " layers starting at layer " << startLayer << ") from " << filename << "\n";
	return true;
}
