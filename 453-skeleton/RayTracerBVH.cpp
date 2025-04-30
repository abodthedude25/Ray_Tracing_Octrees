#include "RayTracerBVH.h"
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <limits>
#include <cmath>
#include <iostream>
#include <queue>
#include <fstream>
#include <sstream>

static const char* g_fsqVertSrc = R"(
#version 430
layout(location = 0) in vec2 inPos;
out vec2 uv;
void main() {
    uv = 0.5*(inPos+vec2(1.0));
    gl_Position = vec4(inPos, 0.0, 1.0);
}
)";

static const char* g_fsqFragSrc = R"(
#version 430
in vec2 uv;
out vec4 fragColor;

uniform sampler2D tex;

void main() {
    fragColor = texture(tex, uv);
}
)";

RayTracerBVH::RayTracerBVH()
	: m_octreeRoot(nullptr),
	m_computeInited(false),
	m_outputTex(0),
	m_fullscreenVAO(0),
	m_fullscreenVBO(0),
	m_computeProg(0),
	m_fsqProg(0),
	m_nodeSSBO(0),
	m_numNodes(0),
	m_frustumCullingEnabled(true),
	m_enableVolumeMeasurement(false),
	m_measuredVolume(0.0f),
	m_volumeSSBO(0),
	m_enableLOD(true),
	m_lodBaseDist(100.0f),
	m_lodFactor(1.5f),
	m_minVoxelSize(1.0f)
{
}

RayTracerBVH::~RayTracerBVH()
{
	// Clean up GL resources
	if (m_outputTex) {
		glDeleteTextures(1, &m_outputTex);
	}
	if (m_fullscreenVBO) {
		glDeleteBuffers(1, &m_fullscreenVBO);
	}
	if (m_fullscreenVAO) {
		glDeleteVertexArrays(1, &m_fullscreenVAO);
	}
	if (m_computeProg) {
		glDeleteProgram(m_computeProg);
	}
	if (m_fsqProg) {
		glDeleteProgram(m_fsqProg);
	}
	if (m_nodeSSBO) {
		glDeleteBuffers(1, &m_nodeSSBO);
	}
}

void RayTracerBVH::setOctree(OctreeNode* root, const VoxelGrid& grid)
{
	m_octreeRoot = root;
	m_grid = grid;

	// Flatten the octree into a single array for the SSBO
	// BFS or DFS approach:
	m_flatNodes.clear();

	if (!root) return;

	// We'll store each node in a queue
	// We'll keep the index in the array so children can reference it
	std::queue<OctreeNode*> q;
	q.push(root);

	// We also need a map from OctreeNode* -> index
	std::unordered_map<OctreeNode*, int> indexMap;
	indexMap[root] = 0;

	// Pre-insert the root
	m_flatNodes.push_back(GPUNodes{ 0, 0, 0, 0, 0, 0, 0, { -1,-1,-1,-1,-1,-1,-1,-1 } });

	while (!q.empty()) {
		OctreeNode* nd = q.front();
		q.pop();
		int idx = indexMap[nd];

		// Fill in data
		m_flatNodes[idx].x = nd->x;
		m_flatNodes[idx].y = nd->y;
		m_flatNodes[idx].z = nd->z;
		m_flatNodes[idx].size = nd->size;
		m_flatNodes[idx].isLeaf = (nd->isLeaf ? 1 : 0);
		m_flatNodes[idx].isSolid = (nd->isSolid ? 1 : 0);
		m_flatNodes[idx].isUniform = (nd->isUniform ? 1 : 0);

		for (int i = 0; i < 8; i++) {
			m_flatNodes[idx].child[i] = -1; // default
		}

		if (!nd->isLeaf) {
			// push children
			for (int i = 0; i < 8; i++) {
				OctreeNode* c = nd->children[i];
				if (c) {
					if (indexMap.find(c) == indexMap.end()) {
						int newIdx = (int)m_flatNodes.size();
						indexMap[c] = newIdx;
						m_flatNodes.push_back(GPUNodes{});
						for (int j = 0; j < 8; j++) {
							m_flatNodes.back().child[j] = -1;
						}
					}
					int cIndex = indexMap[c];
					m_flatNodes[idx].child[i] = cIndex;
					q.push(c);
				}
			}
		}
	}

	m_numNodes = (int)m_flatNodes.size();

	// Create or update SSBO
	if (!m_nodeSSBO) {
		glGenBuffers(1, &m_nodeSSBO);
	}
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_nodeSSBO);
	glBufferData(GL_SHADER_STORAGE_BUFFER,
		m_numNodes * sizeof(GPUNodes),
		m_flatNodes.data(),
		GL_STATIC_DRAW);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, m_nodeSSBO);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

// Function to load a shader from an external file
static inline std::string loadShaderFromFile(const std::string& filename) {
	std::ifstream file(filename);
	if (!file.is_open()) {
		std::cerr << "Failed to open shader file: " << filename << std::endl;
		return "";
	}

	std::stringstream buffer;
	buffer << file.rdbuf();
	return buffer.str();
}

// Ensure we have created/compiled the compute pipeline, the fullscreen pass, etc.
void RayTracerBVH::ensureComputeInitialized()
{
	if (m_computeInited) return;
	m_computeInited = true;

	{
		// Load the compute shader from file
		std::string shaderSource = loadShaderFromFile("453-skeleton/shaders/raytraceCompute.glsl");
		if (shaderSource.empty()) {
			std::cerr << "Failed to load compute shader source" << std::endl;
			return;
		}

		GLuint cs = glCreateShader(GL_COMPUTE_SHADER);
		const char* src = shaderSource.c_str();
		glShaderSource(cs, 1, &src, nullptr);
		glCompileShader(cs);

		GLint status;
		glGetShaderiv(cs, GL_COMPILE_STATUS, &status);
		if (!status) {
			char log[512];
			glGetShaderInfoLog(cs, 512, nullptr, log);
			std::cerr << "[Compute Shader] Compile Error:\n" << log << std::endl;
			glDeleteShader(cs);
			m_computeProg = 0;
			return;
		}

		m_computeProg = glCreateProgram();
		glAttachShader(m_computeProg, cs);
		glLinkProgram(m_computeProg);

		glGetProgramiv(m_computeProg, GL_LINK_STATUS, &status);
		if (!status) {
			char log[512];
			glGetProgramInfoLog(m_computeProg, 512, nullptr, log);
			std::cerr << "[Compute Program] Link Error:\n" << log << std::endl;
			glDeleteProgram(m_computeProg);
			m_computeProg = 0;
			return;
		}
		glDeleteShader(cs);
	}

	{
		// Vertex shader
		GLuint vs = glCreateShader(GL_VERTEX_SHADER);
		glShaderSource(vs, 1, &g_fsqVertSrc, nullptr);
		glCompileShader(vs);

		GLint status;
		glGetShaderiv(vs, GL_COMPILE_STATUS, &status);
		if (!status) {
			char log[512];
			glGetShaderInfoLog(vs, 512, nullptr, log);
			std::cerr << "[FSQ Vertex Shader] Compile Error:\n" << log << std::endl;
			glDeleteShader(vs);
			return;
		}

		// Fragment shader
		GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);
		glShaderSource(fs, 1, &g_fsqFragSrc, nullptr);
		glCompileShader(fs);

		glGetShaderiv(fs, GL_COMPILE_STATUS, &status);
		if (!status) {
			char log[512];
			glGetShaderInfoLog(fs, 512, nullptr, log);
			std::cerr << "[FSQ Fragment Shader] Compile Error:\n" << log << std::endl;
			glDeleteShader(vs);
			glDeleteShader(fs);
			return;
		}

		m_fsqProg = glCreateProgram();
		glAttachShader(m_fsqProg, vs);
		glAttachShader(m_fsqProg, fs);
		glLinkProgram(m_fsqProg);

		glGetProgramiv(m_fsqProg, GL_LINK_STATUS, &status);
		if (!status) {
			char log[512];
			glGetProgramInfoLog(m_fsqProg, 512, nullptr, log);
			std::cerr << "[FSQ Program] Link Error:\n" << log << std::endl;
			glDeleteProgram(m_fsqProg);
			m_fsqProg = 0;
		}
		glDeleteShader(vs);
		glDeleteShader(fs);
	}

	{
		glGenVertexArrays(1, &m_fullscreenVAO);
		glBindVertexArray(m_fullscreenVAO);

		glGenBuffers(1, &m_fullscreenVBO);
		glBindBuffer(GL_ARRAY_BUFFER, m_fullscreenVBO);
		// 2D positions covering the entire screen
		float fsqVerts[] = {
			-1.f, -1.f,
			+1.f, -1.f,
			-1.f, +1.f,
			+1.f, +1.f
		};
		glBufferData(GL_ARRAY_BUFFER, sizeof(fsqVerts), fsqVerts, GL_STATIC_DRAW);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);

		glBindVertexArray(0);
	}
}

void RayTracerBVH::renderSceneCompute(const Camera& camera,
	int width, int height,
	float aspect,
	float fovDeg)
{
	if (!m_computeInited || m_computeProg == 0 || m_fsqProg == 0) {
		std::cerr << "[RayTracerBVH] Compute pipeline not initialized or failed.\n";
		return;
	}

	// If we have no data, skip
	if (m_numNodes <= 0) {
		return;
	}

	// Resize or create output image
	if (!m_outputTex) {
		glGenTextures(1, &m_outputTex);
	}
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, m_outputTex);

	// Allocate or re-allocate as RGBA32F
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height,
		0, GL_RGBA, GL_FLOAT, nullptr);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	// Bind SSBO
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, m_nodeSSBO);

	// Use compute shader
	glUseProgram(m_computeProg);

	// Bind image for output
	glBindImageTexture(0, m_outputTex, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);

	// set uniforms
	GLint locNumNodes = glGetUniformLocation(m_computeProg, "numNodes");
	GLint locGridMin = glGetUniformLocation(m_computeProg, "gridMin");
	GLint locVoxelSize = glGetUniformLocation(m_computeProg, "voxelSize");
	GLint locInvVP = glGetUniformLocation(m_computeProg, "invVP");
	GLint locViewMat = glGetUniformLocation(m_computeProg, "viewMat");
	GLint locCamPos = glGetUniformLocation(m_computeProg, "cameraPos");
	GLint locAspect = glGetUniformLocation(m_computeProg, "aspect");
	GLint locFov = glGetUniformLocation(m_computeProg, "fov");
	GLint locWidth = glGetUniformLocation(m_computeProg, "imageWidth");
	GLint locHeight = glGetUniformLocation(m_computeProg, "imageHeight");

	glUniform1i(locNumNodes, m_numNodes);
	glUniform3f(locGridMin, m_grid.minX, m_grid.minY, m_grid.minZ);
	glUniform1f(locVoxelSize, m_grid.voxelSize);

	// For reference, we can pass invVP if you prefer unproject approach:
	glm::mat4 view = camera.getView();
	glm::mat4 invVP = glm::inverse(glm::perspective(glm::radians(fovDeg), aspect, 0.01f, 5000.f) * view);
	glUniformMatrix4fv(locInvVP, 1, GL_FALSE, &invVP[0][0]);
	glUniformMatrix4fv(locViewMat, 1, GL_FALSE, &view[0][0]);

	glm::vec3 camPos = camera.getPos();
	glUniform3f(locCamPos, camPos.x, camPos.y, camPos.z);

	glUniform1f(locAspect, aspect);
	glUniform1f(locFov, fovDeg);

	glUniform1i(locWidth, width);
	glUniform1i(locHeight, height);

	// Dispatch
	int gx = (width + 7) / 8;   // match local_size_x=8
	int gy = (height + 7) / 8;   // match local_size_y=8
	glDispatchCompute(gx, gy, 1);

	// Wait for compute
	glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

	// Now draw a fullscreen quad to show the result
	glUseProgram(m_fsqProg);

	// Our texture is bound to GL_TEXTURE0, so set the sampler uniform
	GLint locTex = glGetUniformLocation(m_fsqProg, "tex");
	glUniform1i(locTex, 0); // texture unit 0

	glBindVertexArray(m_fullscreenVAO);
	glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
	glBindVertexArray(0);

	// Unbind
	glBindImageTexture(0, 0, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);
	glUseProgram(0);
}

void RayTracerBVH::renderSceneComputeWithCulling(
	const Camera& camera,
	int width,
	int height,
	float aspect,
	float fovDeg,
	bool updateFrustum)
{
	// 1) Bail out if compute pipeline is not ready
	if (!m_computeInited || m_computeProg == 0 || m_fsqProg == 0) {
		std::cerr << "[RayTracerBVH] Compute pipeline not initialized or failed.\n";
		return;
	}

	// 2) If no nodes, skip
	if (m_numNodes <= 0) {
		std::cout << "[RayTracerBVH] No octree nodes to render.\n";
		return;
	}

	// 3) Optionally perform frustum culling
	if (updateFrustum) {
		std::cout << "Applying frustum update with camera at position: ("
			<< camera.getPos().x << ", "
			<< camera.getPos().y << ", "
			<< camera.getPos().z << ")\n";

		// Create the frustum from the camera's view-projection
		glm::mat4 view = camera.getView();
		glm::mat4 proj = glm::perspective(glm::radians(fovDeg), aspect, 0.01f, 5000.f);
		Frustum frustum(proj * view);

		// Clear the visible‐nodes list
		m_visibleNodes.clear();

		// 3.1) Mark which nodes pass frustum test
		std::vector<bool> isVisible(m_flatNodes.size(), false);
		int visibleCount = 0;

		for (size_t i = 0; i < m_flatNodes.size(); i++) {
			const auto& node = m_flatNodes[i];

			// Build world‐space bounding box
			glm::vec3 nodeMin(m_grid.minX + node.x * m_grid.voxelSize,
				m_grid.minY + node.y * m_grid.voxelSize,
				m_grid.minZ + node.z * m_grid.voxelSize);
			glm::vec3 nodeMax = nodeMin + glm::vec3(node.size * m_grid.voxelSize);

			int frustumTest = frustum.testAABB(nodeMin, nodeMax, 150.0f);
			if (frustumTest != -1) {
				isVisible[i] = true;
				visibleCount++;
			}
		}

		// 3.2) Remap old indices -> new indices
		std::vector<int> oldToNew(m_flatNodes.size(), -1);
		int newIndex = 0;
		for (size_t i = 0; i < m_flatNodes.size(); i++) {
			if (isVisible[i]) {
				oldToNew[i] = newIndex++;
			}
		}

		// 3.3) Build m_visibleNodes
		m_visibleNodes.resize(visibleCount);
		newIndex = 0;
		for (size_t i = 0; i < m_flatNodes.size(); i++) {
			if (!isVisible[i]) continue;
			m_visibleNodes[newIndex] = m_flatNodes[i];

			// Re-map children
			if (!m_flatNodes[i].isLeaf) {
				for (int c = 0; c < 8; c++) {
					int childIdx = m_flatNodes[i].child[c];
					if (childIdx >= 0 && isVisible[childIdx]) {
						m_visibleNodes[newIndex].child[c] = oldToNew[childIdx];
					}
					else {
						m_visibleNodes[newIndex].child[c] = -1;
					}
				}
			}
			newIndex++;
		}

		std::cout << "[RayTracerBVH] Frustum culling: " << m_numNodes << " -> "
			<< visibleCount << " nodes\n";

		// 3.4) Update GPU buffer with just the visible nodes
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_nodeSSBO);
		glBufferData(GL_SHADER_STORAGE_BUFFER,
			m_visibleNodes.size() * sizeof(GPUNodes),
			m_visibleNodes.data(),
			GL_DYNAMIC_DRAW);
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
	}

	// 4) Prepare output image
	if (!m_outputTex) {
		glGenTextures(1, &m_outputTex);
	}
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, m_outputTex);

	// Re‐allocate as RGBA32F
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height,
		0, GL_RGBA, GL_FLOAT, nullptr);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	// Bind node SSBO at binding=1
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, m_nodeSSBO);

	if (m_enableVolumeMeasurement && m_volumeSSBO) {
		// 1) Zero out the SSBO each frame
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_volumeSSBO);

		// Write int zero (since 'accumulatedVolume' is an int in the shader)
		int zero = 0;
		glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(int), &zero);

		// Optionally re‐bind at binding=2 if not still bound
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, m_volumeSSBO);

		glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
	}

	// 6) Use the compute shader
	glUseProgram(m_computeProg);

	// Bind image for output
	glBindImageTexture(0, m_outputTex, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);

	// 7) Set uniforms
	//    — NB: set "enableVolumeMeasurement" uniform if your shader uses it
	GLint locNumNodes = glGetUniformLocation(m_computeProg, "numNodes");
	GLint locGridMin = glGetUniformLocation(m_computeProg, "gridMin");
	GLint locVoxel = glGetUniformLocation(m_computeProg, "voxelSize");
	GLint locInvVP = glGetUniformLocation(m_computeProg, "invVP");
	GLint locViewMat = glGetUniformLocation(m_computeProg, "viewMat");
	GLint locCamPos = glGetUniformLocation(m_computeProg, "cameraPos");
	GLint locAspect = glGetUniformLocation(m_computeProg, "aspect");
	GLint locFov = glGetUniformLocation(m_computeProg, "fov");
	GLint locWidth = glGetUniformLocation(m_computeProg, "imageWidth");
	GLint locHeight = glGetUniformLocation(m_computeProg, "imageHeight");
	GLint locEnable = glGetUniformLocation(m_computeProg, "enableVolumeMeasurement");
	GLint locRenderMode = glGetUniformLocation(m_computeProg, "renderMode");
	GLint locEnableLOD = glGetUniformLocation(m_computeProg, "enableLOD");
	GLint locLODBaseDist = glGetUniformLocation(m_computeProg, "lodBaseDist");
	GLint locLODFactor = glGetUniformLocation(m_computeProg, "lodFactor");
	GLint locMinVoxelSize = glGetUniformLocation(m_computeProg, "minVoxelSize");

	if (locEnableLOD >= 0) {
		glUniform1i(locEnableLOD, m_enableLOD ? 1 : 0);
	}
	if (locLODBaseDist >= 0) {
		glUniform1f(locLODBaseDist, m_lodBaseDist);
	}
	if (locLODFactor >= 0) {
		glUniform1f(locLODFactor, m_lodFactor);
	}
	if (locMinVoxelSize >= 0) {
		glUniform1f(locMinVoxelSize, m_minVoxelSize);
	}

	if (locRenderMode >= 0) {
		glUniform1i(locRenderMode, static_cast<int>(m_renderMode));
	}

	int nodeCount = updateFrustum ? int(m_visibleNodes.size()) : m_numNodes;
	glUniform1i(locNumNodes, nodeCount);
	glUniform3f(locGridMin, m_grid.minX, m_grid.minY, m_grid.minZ);
	glUniform1f(locVoxel, m_grid.voxelSize);

	glm::mat4 viewMatrix = camera.getView();
	glm::mat4 vp = glm::perspective(glm::radians(fovDeg), aspect, 0.01f, 5000.f) * viewMatrix;
	glm::mat4 invVP = glm::inverse(vp);

	glUniformMatrix4fv(locInvVP, 1, GL_FALSE, &invVP[0][0]);
	glUniformMatrix4fv(locViewMat, 1, GL_FALSE, &viewMatrix[0][0]);

	glm::vec3 cpos = camera.getPos();
	glUniform3f(locCamPos, cpos.x, cpos.y, cpos.z);

	glUniform1f(locAspect, aspect);
	glUniform1f(locFov, fovDeg);
	glUniform1i(locWidth, width);
	glUniform1i(locHeight, height);

	if (locEnable >= 0) {
		glUniform1i(locEnable, m_enableVolumeMeasurement ? 1 : 0);
	}

	// 8) Dispatch compute
	int gx = (width + 7) / 8;
	int gy = (height + 7) / 8;
	glDispatchCompute(gx, gy, 1);

	// 9) Wait for completion
	glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);

	// 10) If measuring, read the volume result back from the SSBO
	if (m_enableVolumeMeasurement && m_volumeSSBO) {
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_volumeSSBO);

		// Map as an int*
		int* ptr = (int*)glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);
		if (ptr) {
			int accumInt = *ptr; // total scaled volume
			glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);

			// Convert back:
			const float scale = 1e8f; // same scale as in the shader
			float totalVolume = float(accumInt) / scale;

			m_measuredVolume = totalVolume;
			std::cout << "[RayTracerBVH] Measured volume: " << totalVolume << " units\n";
		}
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
	}

	// 11) Draw a fullscreen quad to display the result
	glUseProgram(m_fsqProg);

	GLint locTex = glGetUniformLocation(m_fsqProg, "tex");
	glUniform1i(locTex, 0); // texture unit 0

	glBindVertexArray(m_fullscreenVAO);
	glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
	glBindVertexArray(0);

	// 12) Cleanup
	glBindImageTexture(0, 0, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);
	glUseProgram(0);
}

void RayTracerBVH::updateNodesWithFrustumCulling(const Frustum& frustum, float extraMargin) {
	m_visibleNodes.clear();

	// Reserve space for efficiency
	m_visibleNodes.reserve(m_flatNodes.size());

	// Go through all nodes and check visibility
	for (const auto& node : m_flatNodes) {
		// Calculate node bounds in world space
		glm::vec3 minPoint(
			m_grid.minX + node.x * m_grid.voxelSize,
			m_grid.minY + node.y * m_grid.voxelSize,
			m_grid.minZ + node.z * m_grid.voxelSize
		);
		glm::vec3 maxPoint = minPoint + glm::vec3(node.size * m_grid.voxelSize);

		// Test against frustum with margin
		int frustumTest = frustum.testAABB(minPoint, maxPoint, extraMargin);

		// If not completely outside, include it
		if (frustumTest != -1) {
			m_visibleNodes.push_back(node);
		}
	}
}


// On the CPU, do the same:
void RayTracerBVH::enableVolumeMeasurement(bool enable)
{
	m_enableVolumeMeasurement = enable;

	if (enable) {
		// If we haven't created the SSBO yet, do so
		if (!m_volumeSSBO) {
			glGenBuffers(1, &m_volumeSSBO);
		}

		// Bind it
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_volumeSSBO);

		// We only need space for one integer
		int zero = 0;
		// Allocate and initialize to zero
		glBufferData(GL_SHADER_STORAGE_BUFFER,
			sizeof(int),
			&zero,
			GL_DYNAMIC_DRAW);

		// Bind it at binding index=2 (must match your compute shader)
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, m_volumeSSBO);

		// Unbind
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

		// Also reset our CPU‐side value
		m_measuredVolume = 0.0f;
	}
	else {
		// If disabling measurement, optionally free the buffer
		if (m_volumeSSBO) {
			glDeleteBuffers(1, &m_volumeSSBO);
			m_volumeSSBO = 0;
		}
	}
}

void RayTracerBVH::resetVolumeMeasurement() {
	m_measuredVolume = 0.0f;

	if (m_volumeSSBO) {
		// Clear the SSBO to zero as an integer
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_volumeSSBO);
		int zero = 0;
		glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(int), &zero);
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
	}
}

