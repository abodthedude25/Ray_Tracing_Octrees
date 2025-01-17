// MarchingCubes.h
#pragma once

#include <vector>
#include "glm/glm.hpp"

/// A simple structure for a single triangle with 3 vertices + normals.
struct MCTriangle {
	glm::vec3 v[3];       // positions
	glm::vec3 normal[3];  // normals for each vertex (optional)
};

/// The main function that runs marching cubes over an entire 3D voxel grid.
/// @param scalarField: a 3D array of float values (size = dimX*dimY*dimZ).
/// @param dimX, dimY, dimZ: dimensions of the grid.
/// @param origin: world-space coordinate of the (0,0,0) corner.
/// @param spacing: distance between adjacent voxel centers in x,y,z.
/// @param isoValue: threshold to decide inside/outside (e.g., 0).
/// @returns a list of triangles forming the isosurface.
std::vector<MCTriangle> marchingCubesVolume(const float* scalarField,
	int dimX, int dimY, int dimZ,
	const glm::vec3& origin,
	float spacing,
	float isoValue);
