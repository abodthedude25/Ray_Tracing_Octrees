#include "buildingloader.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <glm/glm.hpp>

struct CSVVertex {
	int meshNumber;
	int vertexNumber;
	double easting;
	double northing;
	double elevation;
	double latitude;
	double longitude;
	double elevMin;
};

struct CSVFace {
	int meshNumber;
	int v1;
	int v2;
	int v3;
};

static inline std::string trim(const std::string& s) {
	size_t start = s.find_first_not_of(" \t\n\r");
	size_t end = s.find_last_not_of(" \t\n\r");
	if (start == std::string::npos) return "";
	return s.substr(start, end - start + 1);
}

std::vector<CSVVertex> loadCSVVertices(const std::string& filename) {
	std::vector<CSVVertex> vertices;
	std::ifstream file(filename);
	if (!file) {
		std::cerr << "Error opening vertex file: " << filename << std::endl;
		return vertices;
	}

	std::string line;
	std::getline(file, line); // Skip header

	while (std::getline(file, line)) {
		if (line.empty()) continue;

		std::vector<std::string> tokens;
		std::istringstream ss(line);
		std::string token;

		while (std::getline(ss, token, ',')) {
			tokens.push_back(trim(token));
		}

		if (tokens.size() >= 8) {
			try {
				CSVVertex v;
				v.meshNumber = std::stoi(tokens[0]);
				v.vertexNumber = std::stoi(tokens[1]);
				v.easting = std::stod(tokens[2]);
				v.northing = std::stod(tokens[3]);
				v.elevation = std::stod(tokens[4]);
				v.latitude = std::stod(tokens[5]);
				v.longitude = std::stod(tokens[6]);
				v.elevMin = std::stod(tokens[7]);
				vertices.push_back(v);
			}
			catch (const std::exception& e) {
				std::cerr << "Error parsing vertex line: " << line << std::endl;
				continue;
			}
		}
	}

	if (!vertices.empty()) {
		std::cout << "First vertex: " << vertices[0].easting << ", "
			<< vertices[0].northing << ", " << vertices[0].elevation << std::endl;
		std::cout << "Total vertices: " << vertices.size() << std::endl;
	}

	return vertices;
}

std::vector<CSVFace> loadCSVFaces(const std::string& filename) {
	std::vector<CSVFace> faces;
	std::ifstream file(filename);
	if (!file) {
		std::cerr << "Error opening face file: " << filename << std::endl;
		return faces;
	}

	std::string line;
	std::getline(file, line); // Skip header

	while (std::getline(file, line)) {
		if (line.empty()) continue;

		std::vector<std::string> tokens;
		std::istringstream ss(line);
		std::string token;

		while (std::getline(ss, token, ',')) {
			tokens.push_back(trim(token));
		}

		if (tokens.size() >= 4) {
			try {
				CSVFace f;
				f.meshNumber = std::stoi(tokens[0]);
				f.v1 = std::stoi(tokens[1]);
				f.v2 = std::stoi(tokens[2]);
				f.v3 = std::stoi(tokens[3]);
				faces.push_back(f);
			}
			catch (const std::exception& e) {
				std::cerr << "Error parsing face line: " << line << std::endl;
				continue;
			}
		}
	}

	if (!faces.empty()) {
		std::cout << "Total faces: " << faces.size() << std::endl;
	}

	return faces;
}

bool isPointInTriangle(const glm::vec3& p, const glm::vec3& a, const glm::vec3& b, const glm::vec3& c) {
	glm::vec3 v0 = c - a;
	glm::vec3 v1 = b - a;
	glm::vec3 v2 = p - a;

	float dot00 = glm::dot(v0, v0);
	float dot01 = glm::dot(v0, v1);
	float dot02 = glm::dot(v0, v2);
	float dot11 = glm::dot(v1, v1);
	float dot12 = glm::dot(v1, v2);

	float invDenom = dot00 * dot11 - dot01 * dot01;
	if (std::abs(invDenom) < 1e-7f) return false;
	invDenom = 1.0f / invDenom;

	float u = (dot11 * dot02 - dot01 * dot12) * invDenom;
	float v = (dot00 * dot12 - dot01 * dot02) * invDenom;

	return (u >= 0) && (v >= 0) && (u + v <= 1);
}


VoxelGrid loadCSVDataIntoVoxelGrid(const std::string& vertsFilename, const std::string& facesFilename, float voxelSize = 5.0f) {
	VoxelGrid grid;

	std::vector<CSVVertex> csvVerts = loadCSVVertices(vertsFilename);
	std::vector<CSVFace> csvFaces = loadCSVFaces(facesFilename);

	if (csvVerts.empty() || csvFaces.empty()) return grid;

	// Create vertex lookup map
	std::unordered_map<int, std::unordered_map<int, CSVVertex>> vertexMap;
	for (const auto& v : csvVerts) {
		vertexMap[v.meshNumber][v.vertexNumber] = v;
	}

	// Calculate bounds
	double minX = std::numeric_limits<double>::max();
	double minY = std::numeric_limits<double>::max();
	double minZ = std::numeric_limits<double>::max();
	double maxX = -std::numeric_limits<double>::max();
	double maxY = -std::numeric_limits<double>::max();
	double maxZ = -std::numeric_limits<double>::max();

	for (const auto& v : csvVerts) {
		if (std::isfinite(v.easting) && std::isfinite(v.northing) && std::isfinite(v.elevation)) {
			minX = std::min(minX, v.easting);
			minY = std::min(minY, v.northing);
			minZ = std::min(minZ, v.elevation);
			maxX = std::max(maxX, v.easting);
			maxY = std::max(maxY, v.northing);
			maxZ = std::max(maxZ, v.elevation);
		}
	}

	const double padding = voxelSize;
	minX -= padding;
	minY -= padding;
	minZ -= padding;
	maxX += padding;
	maxY += padding;
	maxZ += padding;

	// Calculate grid dimensions with overflow protection
	size_t dimX = static_cast<size_t>(std::ceil((maxX - minX) / voxelSize));
	size_t dimY = static_cast<size_t>(std::ceil((maxY - minY) / voxelSize));
	size_t dimZ = static_cast<size_t>(std::ceil((maxZ - minZ) / voxelSize));

	// Check for reasonable dimensions
	const size_t MAX_DIM = 1000;
	if (dimX > MAX_DIM || dimY > MAX_DIM || dimZ > MAX_DIM) {
		std::cout << "Adjusting voxel size for reasonable dimensions" << std::endl;
		float scale = std::max({ dimX / MAX_DIM, dimY / MAX_DIM, dimZ / MAX_DIM });
		voxelSize *= scale;

		dimX = static_cast<size_t>(std::ceil((maxX - minX) / voxelSize));
		dimY = static_cast<size_t>(std::ceil((maxY - minY) / voxelSize));
		dimZ = static_cast<size_t>(std::ceil((maxZ - minZ) / voxelSize));
	}

	grid.dimX = static_cast<int>(dimX);
	grid.dimY = static_cast<int>(dimY);
	grid.dimZ = static_cast<int>(dimZ);
	grid.minX = static_cast<float>(minX);
	grid.minY = static_cast<float>(minY);
	grid.minZ = static_cast<float>(minZ);
	grid.voxelSize = voxelSize;

	std::cout << "Final grid dimensions: " << grid.dimX << " x " << grid.dimY << " x " << grid.dimZ << std::endl;

	try {
		grid.data.resize(dimX * dimY * dimZ, VoxelState::EMPTY);
	}
	catch (const std::exception& e) {
		std::cerr << "Failed to allocate grid memory: " << e.what() << std::endl;
		return grid;
	}

	size_t filledVoxels = 0;

#pragma omp parallel for schedule(dynamic) reduction(+:filledVoxels)
	for (size_t i = 0; i < csvFaces.size(); ++i) {
		const auto& face = csvFaces[i];
		auto meshIt = vertexMap.find(face.meshNumber);
		if (meshIt == vertexMap.end()) continue;

		auto& meshVertices = meshIt->second;
		auto v1It = meshVertices.find(face.v1);
		auto v2It = meshVertices.find(face.v2);
		auto v3It = meshVertices.find(face.v3);

		if (v1It == meshVertices.end() || v2It == meshVertices.end() || v3It == meshVertices.end())
			continue;

		glm::vec3 v1(v1It->second.easting, v1It->second.northing, v1It->second.elevation);
		glm::vec3 v2(v2It->second.easting, v2It->second.northing, v2It->second.elevation);
		glm::vec3 v3(v3It->second.easting, v3It->second.northing, v3It->second.elevation);

		float triMinX = std::min({ v1.x, v2.x, v3.x });
		float triMinY = std::min({ v1.y, v2.y, v3.y });
		float triMinZ = std::min({ v1.z, v2.z, v3.z });
		float triMaxX = std::max({ v1.x, v2.x, v3.x });
		float triMaxY = std::max({ v1.y, v2.y, v3.y });
		float triMaxZ = std::max({ v1.z, v2.z, v3.z });

		int startX = std::max(0, static_cast<int>((triMinX - grid.minX) / voxelSize));
		int startY = std::max(0, static_cast<int>((triMinY - grid.minY) / voxelSize));
		int startZ = std::max(0, static_cast<int>((triMinZ - grid.minZ) / voxelSize));
		int endX = std::min(grid.dimX - 1, static_cast<int>((triMaxX - grid.minX) / voxelSize) + 1);
		int endY = std::min(grid.dimY - 1, static_cast<int>((triMaxY - grid.minY) / voxelSize) + 1);
		int endZ = std::min(grid.dimZ - 1, static_cast<int>((triMaxZ - grid.minZ) / voxelSize) + 1);

		if (endX < startX || endY < startY || endZ < startZ) continue;

		for (int z = startZ; z <= endZ; ++z) {
			for (int y = startY; y <= endY; ++y) {
				for (int x = startX; x <= endX; ++x) {
					glm::vec3 center(
						grid.minX + (x + 0.5f) * voxelSize,
						grid.minY + (y + 0.5f) * voxelSize,
						grid.minZ + (z + 0.5f) * voxelSize
					);

					if (isPointInTriangle(center, v1, v2, v3)) {
						size_t idx = static_cast<size_t>(x) +
							static_cast<size_t>(y) * grid.dimX +
							static_cast<size_t>(z) * grid.dimX * grid.dimY;
						if (idx < grid.data.size()) {
#pragma omp atomic write
							grid.data[idx] = VoxelState::FILLED;
							filledVoxels++;
						}
					}
				}
			}
		}
	}

	return grid;
}
