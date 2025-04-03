#include "RayTracerBVH.h"
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <limits>
#include <cmath>
#include <iostream>
#include <queue>

/*
static const char* g_computeShaderSrc = R"(
#version 430 core

// For local workgroup sizes
layout(local_size_x = 8, local_size_y = 8) in;

// Output image: final rendered image
layout(rgba32f, binding = 0) uniform image2D outputImage;

// GPU octree node structure – note the added 'isUniform' field.
struct OctreeNodeGPUStruct {
	int x;
	int y;
	int z;
	int size;
	int isLeaf;
	int isSolid;
	int isUniform;  // 1 if uniform (all voxels are the same), else 0.
	int child[8];
};

layout(std430, binding = 1) buffer OctreeNodes {
	OctreeNodeGPUStruct nodes[];
};

uniform int numNodes;
uniform vec3 gridMin;
uniform float voxelSize;
uniform mat4 invVP;
uniform mat4 viewMat;
uniform vec3 cameraPos;
uniform float aspect;
uniform float fov;  // in degrees
uniform int imageWidth;
uniform int imageHeight;

struct Ray {
	vec3 origin;
	vec3 direction;
};

bool intersectAABB(vec3 rayOrigin, vec3 rayDir, vec3 bmin, vec3 bmax, out float tNear, out float tFar)
{
	vec3 invDir = 1.0 / rayDir;
	vec3 t1 = (bmin - rayOrigin) * invDir;
	vec3 t2 = (bmax - rayOrigin) * invDir;
	vec3 tMin = min(t1, t2);
	vec3 tMax = max(t1, t2);
	tNear = max(max(tMin.x, tMin.y), tMin.z);
	tFar = min(min(tMax.x, tMax.y), tMax.z);
	return (tNear <= tFar && tFar > 0.0);
}

bool intersectOctreeIterative(vec3 rayOrigin, vec3 rayDir,
							  out vec3 hitPoint, out vec3 hitNormal)
{
	float closestT = 1e30;
	bool hitFound = false;
	vec3 bestNormal = vec3(0.0);

	// Stack for node indices
	int stack[128];
	int sp = 0;
	stack[sp++] = 0;  // Push root node (assumed index 0)

	while (sp > 0) {
		sp--;
		int nodeIdx = stack[sp];
		if (nodeIdx < 0) continue;

		OctreeNodeGPUStruct node = nodes[nodeIdx];

		// Compute world-space AABB for the node.
		vec3 nodeMin = gridMin + vec3(node.x, node.y, node.z) * voxelSize;
		vec3 nodeMax = nodeMin + vec3(node.size) * voxelSize;

		float tNear, tFar;
		if (!intersectAABB(rayOrigin, rayDir, nodeMin, nodeMax, tNear, tFar))
			continue;

		// Early exit if tNear is not better than current hit.
		if (tNear >= closestT)
			continue;

		if (node.isUniform == 1) {
			if (node.isSolid == 1) {  // Uniform and solid: update hit directly.
				float tHit = max(0.0, tNear);
				if (tHit < closestT && tHit <= tFar) {
					closestT = tHit;
					hitFound = true;
					vec3 center = 0.5 * (nodeMin + nodeMax);
					vec3 p = rayOrigin + rayDir * tHit;
					bestNormal = normalize(p - center);
				}
			}
			// If uniform and empty, nothing to do.
			continue;
		}

		// If this is a leaf (non–uniform leaves are possible, though unlikely)
		if (node.isLeaf == 1) {
			if (node.isSolid == 1) {
				float tHit = max(0.0, tNear);
				if (tHit < closestT && tHit <= tFar) {
					closestT = tHit;
					hitFound = true;
					vec3 center = 0.5 * (nodeMin + nodeMax);
					vec3 p = rayOrigin + rayDir * tHit;
					bestNormal = normalize(p - center);
				}
			}
			continue;
		}
		else {
			// For non-uniform internal nodes, push all children.
			for (int i = 0; i < 8; i++) {
				int childIdx = node.child[i];
				if (childIdx >= 0)
					stack[sp++] = childIdx;
			}
		}
	}

	if (hitFound) {
		hitPoint = rayOrigin + rayDir * closestT;
		hitNormal = bestNormal;
	}
	return hitFound;
}


// Simple Lambert shading function.
vec3 shade(vec3 hitPoint, vec3 normal)
{
	vec3 lightDir = normalize(vec3(-1.0, -1.0, -1.0));
	float ndotl = max(0.0, dot(normal, -lightDir));
	return vec3(1.0, 0.8, 0.6) * ndotl + vec3(0.1, 0.1, 0.1);
}

Ray generateRay(int px, int py, int w, int h, vec3 camPos, mat4 view, float fovDeg, float aspect)
{
	float fovRad = radians(fovDeg);
	float nx = (float(px) + 0.5) / float(w) * 2.0 - 1.0;
	float ny = 1.0 - (float(py) + 0.5) / float(h) * 2.0;
	nx *= aspect;
	float tanHalfFov = tan(fovRad * 0.5);
	nx *= tanHalfFov;
	ny *= tanHalfFov;

	mat4 invView = inverse(view);
	vec4 rayDirView = normalize(vec4(nx, ny, -1.0, 0.0));
	vec4 rayDirWorld = invView * rayDirView;
	Ray r;
	r.origin = camPos;
	r.direction = normalize(vec3(rayDirWorld));
	return r;
}

void main()
{
	ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
	if (gid.x >= imageWidth || gid.y >= imageHeight)
		return;

	Ray ray = generateRay(gid.x, gid.y, imageWidth, imageHeight, cameraPos, viewMat, fov, aspect);
	vec3 hitPoint, hitNormal;
	bool hit = intersectOctreeIterative(ray.origin, ray.direction, hitPoint, hitNormal);
	vec3 color = hit ? shade(hitPoint, hitNormal) : vec3(0.0);
	imageStore(outputImage, gid, vec4(color, 1.0));
}
)";*/

static const char* g_computeShaderSrc = R"(
#version 430 core

// For local workgroup sizes
layout(local_size_x = 8, local_size_y = 8) in;

// Output image: final rendered image
layout(rgba32f, binding = 0) uniform image2D outputImage;

// Define maximum traversal steps per ray
#define MAX_TRAVERSAL_STEPS 512

// GPU octree node structure – note the added 'isUniform' field.
struct OctreeNodeGPUStruct {
    int x;
    int y;
    int z;
    int size;
    int isLeaf;
    int isSolid;
    int isUniform;  // 1 if uniform (all voxels are the same), else 0.
    int child[8];
};

layout(std430, binding = 1) buffer OctreeNodes {
    OctreeNodeGPUStruct nodes[];
};

uniform int numNodes;
uniform vec3 gridMin;
uniform float voxelSize;
uniform mat4 invVP;
uniform mat4 viewMat;
uniform vec3 cameraPos;
uniform float aspect;
uniform float fov;  // in degrees
uniform int imageWidth;
uniform int imageHeight;

struct Ray {
    vec3 origin;
    vec3 direction;
};

bool intersectAABB(vec3 rayOrigin, vec3 rayDir, vec3 bmin, vec3 bmax, out float tNear, out float tFar)
{
    vec3 invDir = 1.0 / rayDir;
    vec3 t1 = (bmin - rayOrigin) * invDir;
    vec3 t2 = (bmax - rayOrigin) * invDir;
    vec3 tMin = min(t1, t2);
    vec3 tMax = max(t1, t2);
    tNear = max(max(tMin.x, tMin.y), tMin.z);
    tFar = min(min(tMax.x, tMax.y), tMax.z);
    return (tNear <= tFar && tFar > 0.0);
}

// Optimized Octree Traversal with Workload Limiting 
bool intersectOctreeIterative(vec3 rayOrigin, vec3 rayDir,
                              out vec3 hitPoint, out vec3 hitNormal)
{
    float closestT = 1e30;
    bool hitFound = false;
    vec3 bestNormal = vec3(0.0);

    // Stack for node indices
    int stack[128];
    int sp = 0;
    stack[sp++] = 0;  // Push root node (assumed index 0)
    
    // Track traversal steps to limit workload
    int traversalSteps = 0;

    while (sp > 0 && traversalSteps < MAX_TRAVERSAL_STEPS) {
        sp--;
        int nodeIdx = stack[sp];
        if (nodeIdx < 0) continue;
        
        // Increment step counter
        traversalSteps++;

        OctreeNodeGPUStruct node = nodes[nodeIdx];

        // Compute world-space AABB for the node.
        vec3 nodeMin = gridMin + vec3(node.x, node.y, node.z) * voxelSize;
        vec3 nodeMax = nodeMin + vec3(node.size) * voxelSize;

        float tNear, tFar;
        if (!intersectAABB(rayOrigin, rayDir, nodeMin, nodeMax, tNear, tFar))
            continue;

        // Early exit if tNear is not better than current hit.
        if (tNear >= closestT)
            continue;
        
        // If this node is uniform (all voxels are the same), treat it as a leaf.
        if (node.isUniform == 1) {
            if (node.isSolid == 1) {  // Uniform and solid: update hit directly.
                float tHit = max(0.0, tNear);
                if (tHit < closestT && tHit <= tFar) {
                    closestT = tHit;
                    hitFound = true;
                    vec3 center = 0.5 * (nodeMin + nodeMax);
                    vec3 p = rayOrigin + rayDir * tHit;
                    bestNormal = normalize(p - center);
                    
                    // Early exit once we found a hit - reduces warp divergence
                    break;
                }
            }
            // If uniform and empty, nothing to do.
            continue;
        }
        
        // If this is a leaf (non–uniform leaves are possible, though unlikely)
        if (node.isLeaf == 1) {
            if (node.isSolid == 1) {
                float tHit = max(0.0, tNear);
                if (tHit < closestT && tHit <= tFar) {
                    closestT = tHit;
                    hitFound = true;
                    vec3 center = 0.5 * (nodeMin + nodeMax);
                    vec3 p = rayOrigin + rayDir * tHit;
                    bestNormal = normalize(p - center);
                    
                    // Early exit once we found a hit - reduces warp divergence
                    break;
                }
            }
            continue;
        }
        else {
            // For non-uniform internal nodes, push all children.
            for (int i = 0; i < 8; i++) {
                int childIdx = node.child[i];
                if (childIdx >= 0)
                    stack[sp++] = childIdx;
            }
        }
    }

    if (hitFound) {
        hitPoint = rayOrigin + rayDir * closestT;
        hitNormal = bestNormal;
    }
    return hitFound;
}


// Simple Lambert shading function.
vec3 shade(vec3 hitPoint, vec3 normal)
{
    vec3 lightDir = normalize(vec3(-1.0, -1.0, -1.0));
    float ndotl = max(0.0, dot(normal, -lightDir));
    return vec3(1.0, 0.8, 0.6) * ndotl + vec3(0.1, 0.1, 0.1);
}

Ray generateRay(int px, int py, int w, int h, vec3 camPos, mat4 view, float fovDeg, float aspect)
{
    float fovRad = radians(fovDeg);
    float nx = (float(px) + 0.5) / float(w) * 2.0 - 1.0;
    float ny = 1.0 - (float(py) + 0.5) / float(h) * 2.0;
    nx *= aspect;
    float tanHalfFov = tan(fovRad * 0.5);
    nx *= tanHalfFov;
    ny *= tanHalfFov;

    mat4 invView = inverse(view);
    vec4 rayDirView = normalize(vec4(nx, ny, -1.0, 0.0));
    vec4 rayDirWorld = invView * rayDirView;
    Ray r;
    r.origin = camPos;
    r.direction = normalize(vec3(rayDirWorld));
    return r;
}

void main()
{
    ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
    if (gid.x >= imageWidth || gid.y >= imageHeight)
        return;

    Ray ray = generateRay(gid.x, gid.y, imageWidth, imageHeight, cameraPos, viewMat, fov, aspect);
    vec3 hitPoint, hitNormal;
    bool hit = intersectOctreeIterative(ray.origin, ray.direction, hitPoint, hitNormal);
    vec3 color = hit ? shade(hitPoint, hitNormal) : vec3(0.0);
    imageStore(outputImage, gid, vec4(color, 1.0));
}
)";

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
	m_frustumCullingEnabled(true)
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

// Ensure we have created/compiled the compute pipeline, the fullscreen pass, etc.
void RayTracerBVH::ensureComputeInitialized()
{
	if (m_computeInited) return;
	m_computeInited = true;

	{
		GLuint cs = glCreateShader(GL_COMPUTE_SHADER);
		glShaderSource(cs, 1, &g_computeShaderSrc, nullptr);
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
	int width, int height,
	float aspect,
	float fovDeg,
	bool updateFrustum)
{
	if (!m_computeInited || m_computeProg == 0 || m_fsqProg == 0) {
		std::cerr << "[RayTracerBVH] Compute pipeline not initialized or failed.\n";
		return;
	}

	// If we have no data, skip
	if (m_numNodes <= 0) {
		printf("No nodes to render.\n");
		return;
	}

	// Update the frustum culling if requested
	if (updateFrustum) {
		std::cout << "Applying frustum update with camera at position: ("
			<< camera.getPos().x << ", "
			<< camera.getPos().y << ", "
			<< camera.getPos().z << ")" << std::endl;

		// Create the frustum from the camera's view-projection matrix
		glm::mat4 view = camera.getView();
		glm::mat4 proj = glm::perspective(glm::radians(fovDeg), aspect, 0.01f, 5000.f);
		Frustum frustum(proj * view);

		// Clear the visible nodes list
		m_visibleNodes.clear();

		// First pass: Determine which nodes are visible
		std::vector<bool> isVisible(m_flatNodes.size(), false);
		int visibleCount = 0;

		for (size_t i = 0; i < m_flatNodes.size(); i++) {
			const auto& node = m_flatNodes[i];

			// Calculate node bounds in world space
			glm::vec3 minPoint(
				m_grid.minX + node.x * m_grid.voxelSize,
				m_grid.minY + node.y * m_grid.voxelSize,
				m_grid.minZ + node.z * m_grid.voxelSize
			);
			glm::vec3 maxPoint = minPoint + glm::vec3(node.size * m_grid.voxelSize);

			// Test against frustum with margin
			int frustumTest = frustum.testAABB(minPoint, maxPoint, 150.0f);

			// If not completely outside, include it
			if (frustumTest != -1) {
				isVisible[i] = true;
				visibleCount++;
			}
		}

		// Create mapping from old indices to new indices
		std::vector<int> oldToNewIndex(m_flatNodes.size(), -1);
		int newIndex = 0;

		for (size_t i = 0; i < m_flatNodes.size(); i++) {
			if (isVisible[i]) {
				oldToNewIndex[i] = newIndex++;
			}
		}

		// Second pass: Build visible nodes list with remapped child indices
		m_visibleNodes.resize(visibleCount);
		newIndex = 0;

		for (size_t i = 0; i < m_flatNodes.size(); i++) {
			if (isVisible[i]) {
				// Copy the node
				m_visibleNodes[newIndex] = m_flatNodes[i];

				// Remap child indices
				if (!m_flatNodes[i].isLeaf) {
					for (int c = 0; c < 8; c++) {
						int oldChildIndex = m_flatNodes[i].child[c];

						// If child is visible, update its index
						if (oldChildIndex >= 0 && oldChildIndex < static_cast<int>(m_flatNodes.size()) &&
							isVisible[oldChildIndex]) {
							m_visibleNodes[newIndex].child[c] = oldToNewIndex[oldChildIndex];
						}
						else {
							// Child not visible, mark as invalid
							m_visibleNodes[newIndex].child[c] = -1;
						}
					}
				}

				newIndex++;
			}
		}

		// Debug output
		std::cout << "Frustum culling: " << m_numNodes << " -> " << visibleCount
			<< " nodes (" << (visibleCount * 100 / m_numNodes) << "%)\n";

		// Update the GPU buffer with visible nodes
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_nodeSSBO);
		glBufferData(GL_SHADER_STORAGE_BUFFER,
			m_visibleNodes.size() * sizeof(GPUNodes),
			m_visibleNodes.data(), GL_DYNAMIC_DRAW);
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

	// Set uniforms
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

	// Pass the number of visible nodes
	int nodeCount = updateFrustum ? m_visibleNodes.size() : m_numNodes;

	glUniform1i(locNumNodes, nodeCount);
	glUniform3f(locGridMin, m_grid.minX, m_grid.minY, m_grid.minZ);
	glUniform1f(locVoxelSize, m_grid.voxelSize);

	// Calculate view and projection matrices
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
