
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

bool loadVoxelGrid(const std::string& filename, VoxelGrid& grid)
{
	std::ifstream in(filename, std::ios::binary);
	if (!in) {
		std::cerr << "Cannot open file for reading: " << filename << "\n";
		return false;
	}

	// Read basic grid parameters
	in.read(reinterpret_cast<char*>(&grid.dimX), sizeof(grid.dimX));
	in.read(reinterpret_cast<char*>(&grid.dimY), sizeof(grid.dimY));
	in.read(reinterpret_cast<char*>(&grid.dimZ), sizeof(grid.dimZ));
	in.read(reinterpret_cast<char*>(&grid.minX), sizeof(grid.minX));
	in.read(reinterpret_cast<char*>(&grid.minY), sizeof(grid.minY));
	in.read(reinterpret_cast<char*>(&grid.minZ), sizeof(grid.minZ));
	in.read(reinterpret_cast<char*>(&grid.voxelSize), sizeof(grid.voxelSize));

	// Read voxel data
	size_t dataSize;
	in.read(reinterpret_cast<char*>(&dataSize), sizeof(dataSize));
	grid.data.resize(dataSize);
	in.read(reinterpret_cast<char*>(grid.data.data()), dataSize * sizeof(grid.data[0]));

	in.close();
	std::cout << "Loaded voxel grid from " << filename << "\n";
	return true;
}
