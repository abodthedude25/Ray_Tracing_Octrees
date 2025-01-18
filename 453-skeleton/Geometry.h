#pragma once

//------------------------------------------------------------------------------
// This file contains simple classes for storing geometry on the CPU and the GPU.
// Later assignments will require you to expand these classes or create your own
// similar classes with the needed functionality.
//------------------------------------------------------------------------------

#include "VertexArray.h"
#include "VertexBuffer.h"
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <vector>

// CPU-side geometry.
struct CPU_Geometry {
	std::vector<glm::vec3> verts;
	std::vector<glm::vec3> cols;
	std::vector<glm::vec3> normals;
	// Optionally, if you want to store texture coordinates:
	std::vector<glm::vec2> texCoords;
};

// GPU-side geometry. We add a new VertexBuffer for texture coordinates.
class GPU_Geometry {
public:
	GPU_Geometry();

	void bind() { vao.bind(); }

	void setVerts(const std::vector<glm::vec3>& verts);
	void setCols(const std::vector<glm::vec3>& cols);
	void setNormals(const std::vector<glm::vec3>& norms);
	// New: setTexCoords for texture coordinates
	void setTexCoords(const std::vector<glm::vec2>& texCoords);

private:
	VertexArray vao;
	VertexBuffer vertBuffer;
	VertexBuffer colorsBuffer;
	VertexBuffer normalsBuffer;
	// New: A VertexBuffer for texture coordinates.
	VertexBuffer texCoordsBuffer;
};
