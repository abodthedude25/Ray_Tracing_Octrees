#version 430 core

#ifdef GL_ARB_shader_atomic_float
#extension GL_ARB_shader_atomic_float : enable
#endif

layout(local_size_x = 8, local_size_y = 8) in;

// OUTPUT IMAGE
layout(rgba32f, binding = 0) uniform image2D outputImage;

// VOLUME MEASUREMENT BUFFER
layout(std430, binding = 2) buffer VolumeBuffer {
    int accumulatedVolume;
};

// UNIFORMS
uniform int   numNodes;
uniform vec3  gridMin;
uniform float voxelSize;
uniform mat4  invVP;
uniform mat4  viewMat;
uniform vec3  cameraPos;
uniform float aspect;
uniform float fov;
uniform int   imageWidth;
uniform int   imageHeight;
uniform bool  enableVolumeMeasurement;
uniform int   renderMode = 0;  // 0: normal, 1: x-ray, 2: see-through
uniform bool  enableLOD = true;          // Whether to enable LOD rendering
uniform float lodBaseDist = 100.0;       // Base distance for LOD calculations
uniform float lodFactor = 1.2;           // LOD scaling factor
uniform float minVoxelSize = 1.0;        // Minimum voxel size (won't go smaller)

// Add this function to calculate LOD level based on distance
float calculateLODLevel(float distance) {
    if (!enableLOD) return 1.0;
    
    // Calculate LOD level based on distance
    // Start with level 1 (original size) at close distances
    // and increase with distance
    float lodLevel = max(1.0, distance / lodBaseDist);
    
    // Apply scaling factor to control how quickly LOD increases with distance
    lodLevel = pow(lodLevel, lodFactor);
    
    // Round to nearest power of 2 for cleaner transitions
    // This makes voxels combine in powers of 2 (2x, 4x, 8x, etc.)
    lodLevel = pow(2.0, floor(log2(lodLevel) + 0.5));
    
    return lodLevel;
}

const float VOLUME_SCALE = 1e8;

struct OctreeNodeGPUStruct {
    int x;
    int y;
    int z;
    int size;
    int isLeaf;
    int isSolid;
    int isUniform;
    int child[8];
};

layout(std430, binding = 1) buffer OctreeNodes {
    OctreeNodeGPUStruct nodes[];
};

struct Ray {
    vec3 origin;
    vec3 direction;
};

struct VolumeInterval {
    float tStart;
    float tEnd;
};

struct HitInfo {
    float t;
    vec3 position;
    vec3 normal;
    int nodeIndex;
    int nodeSize;
    bool isEdge;
};

// Store multiple hits for x-ray and see-through modes
const int MAX_HITS = 64;
struct HitArray {
    HitInfo hits[MAX_HITS];
    int count;
};

bool intersectAABB(vec3 rayOrigin, vec3 rayDir, vec3 bmin, vec3 bmax,
                    out float tNear, out float tFar)
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

// Check if a point is at a voxel boundary
bool isAtVoxelBoundary(vec3 pos) {
    vec3 normalizedPos = (pos - gridMin) / voxelSize;
    vec3 fractionalPart = fract(normalizedPos);
    
    // Check if any component is close to 0 or 1
    float threshold = 0.05;
    return any(lessThan(fractionalPart, vec3(threshold))) || 
           any(greaterThan(fractionalPart, vec3(1.0 - threshold)));
}

// Determine if a position is on a building edge
bool isOnBuildingEdge(vec3 pos, vec3 normal) {
    // Use voxel boundaries as building edges
    vec3 normalizedPos = (pos - gridMin) / voxelSize;
    vec3 fractionalPart = fract(normalizedPos);
    
    // Thicker, more visible edges
    float threshold = 0.1; // Increased from 0.05
    
    return any(lessThan(fractionalPart, vec3(threshold))) || 
           any(greaterThan(fractionalPart, vec3(1.0 - threshold)));
}

// Get color based on depth (distance from camera)
vec3 getDepthColor(float depth) {
    // Normalize depth to a reasonable range
    float normalizedDepth = clamp(depth / 500.0, 0.0, 1.0);
    
    // Define vibrant color gradient: blue (close) -> cyan -> green -> yellow -> red (far)
    vec3 closeColor = vec3(0.1, 0.4, 1.0);    // Bright blue
    vec3 midColor1 = vec3(0.0, 0.8, 0.8);     // Bright cyan
    vec3 midColor2 = vec3(0.2, 0.9, 0.2);     // Bright green
    vec3 midColor3 = vec3(0.9, 0.9, 0.1);     // Bright yellow
    vec3 farColor = vec3(1.0, 0.2, 0.1);      // Bright red
    
    // Multi-step interpolation
    vec3 color;
    if (normalizedDepth < 0.25) {
        color = mix(closeColor, midColor1, normalizedDepth * 4.0);
    } else if (normalizedDepth < 0.5) {
        color = mix(midColor1, midColor2, (normalizedDepth - 0.25) * 4.0);
    } else if (normalizedDepth < 0.75) {
        color = mix(midColor2, midColor3, (normalizedDepth - 0.5) * 4.0);
    } else {
        color = mix(midColor3, farColor, (normalizedDepth - 0.75) * 4.0);
    }
    
    return color;
}

// Get building color based on size and position
vec3 getBuildingColor(vec3 pos, int nodeSize) {
    // Create a color hash based on position to differentiate buildings
    vec3 blockPos = floor(pos / (voxelSize * 8.0));
    float hash = fract(sin(dot(blockPos.xyz, vec3(12.9898, 78.233, 45.543))) * 43758.5453);
    
    // Different base colors for different building sizes
    vec3 smallBuildingColor = vec3(0.85, 0.7, 0.55);  // Tan
    vec3 mediumBuildingColor = vec3(0.75, 0.75, 0.75); // Gray
    vec3 largeBuildingColor = vec3(0.8, 0.6, 0.5);    // Brick red
    
    // Vary color slightly based on hash
    float variance = 0.2;
    vec3 colorOffset = vec3(hash, fract(hash * 2.31), fract(hash * 3.79)) * variance - (variance/2.0);
    
    // Select base color based on node size (larger nodes = larger buildings)
    vec3 baseColor;
    if (nodeSize <= 2) {
        baseColor = smallBuildingColor;
    } else if (nodeSize <= 8) {
        baseColor = mediumBuildingColor;
    } else {
        baseColor = largeBuildingColor;
    }
    
    // Apply variance
    return clamp(baseColor + colorOffset, 0.2, 0.95);
}

// Standard shading function for normal mode
vec3 shadeNormal(vec3 hitPoint, vec3 normal, int nodeSize) {
    vec3 lightDir = normalize(vec3(-1, -1, -1));
    float ndotl = max(0.0, dot(normal, -lightDir));
    
    // Get base color from building characteristics
    vec3 baseColor = getBuildingColor(hitPoint, nodeSize);
    
    // Add some ambient, diffuse and specular lighting
    float ambient = 0.4;
    float diffuse = ndotl * 0.7;
    
    // Simple specular highlight
    vec3 viewDir = normalize(cameraPos - hitPoint);
    vec3 reflectDir = reflect(lightDir, normal);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 16);
    float specular = spec * 0.2;
    
    // Edge highlighting
    bool isEdge = isOnBuildingEdge(hitPoint, normal);
    float edgeFactor = isEdge ? 0.7 : 1.0;
    
    vec3 finalColor = baseColor * (ambient + diffuse) * edgeFactor + vec3(specular);
    
    return finalColor;
}

// X-ray effect shading
vec4 shadeXRay(vec3 hitPoint, vec3 normal, float depth, bool isEdge, int nodeSize) {
    // Get base color from depth
    vec3 depthColor = getDepthColor(depth);
    
    // Enhance edges - they are more opaque and whiter
    if (isEdge) {
        // Bright white-ish edge with high opacity
        return vec4(vec3(0.6, 0.6, 0.7), 0.9);
    }
    
    // Interior points are more transparent and colored by depth
    // Smaller buildings should be more visible
    float sizeOpacity = 1.0 / (0.5 * float(nodeSize));
    sizeOpacity = clamp(sizeOpacity, 0.05, 0.7);
    
    return vec4(depthColor, sizeOpacity * 0.4);
}

// See-through shading with opacity based on position and size
vec4 shadeSeeThrough(vec3 hitPoint, vec3 normal, float depth, int nodeSize) {
    // Get building color - make it brighter
    vec3 baseColor = getBuildingColor(hitPoint, nodeSize) * 1.3;
    
    // Calculate lighting
    vec3 lightDir = normalize(vec3(-1, -1, -1));
    float ndotl = max(0.0, dot(normal, -lightDir));
    
    // Enhanced lighting
    float ambient = 0.5; // Higher ambient
    vec3 finalColor = baseColor * (ndotl * 0.8 + ambient);
    
    // Opacity depends on:
    // 1. Depth (further = more transparent)
    // 2. Building size (smaller = more opaque)
    // 3. View angle (grazing angles = more opaque)
    
    float depthFactor = 1.0 - clamp(depth / 1000.0, 0.0, 0.8);
    float sizeFactor = 1.0 / (0.15 * float(nodeSize));
    sizeFactor = clamp(sizeFactor, 0.3, 1.0);
    
    float viewAngle = abs(dot(normal, normalize(cameraPos - hitPoint)));
    float angleFactor = pow(1.0 - viewAngle, 2.0) * 0.5 + 0.5;
    
    // Combine factors for final opacity - increased base opacity
    float opacity = depthFactor * sizeFactor * angleFactor;
    opacity = clamp(opacity, 0.3, 0.9); // Ensure minimum visibility
    
    return vec4(finalColor, opacity);
}

// Collect multiple intersections along the ray for x-ray and see-through rendering
void intersectOctreeMultiHit(vec3 rayOrigin, vec3 rayDir, float maxDistance, out HitArray hits) {
    // Initialize hits
    hits.count = 0;
    
    // Stack for BFS
    int stack[128];
    int sp = 0;
    stack[sp++] = 0; // root node index = 0

    while (sp > 0 && hits.count < MAX_HITS) {
        sp--;
        int nodeIdx = stack[sp];
        if (nodeIdx < 0) continue;

        // Read the node
        OctreeNodeGPUStruct node = nodes[nodeIdx];

        // Build the node's AABB
        vec3 nodeMin = gridMin + vec3(node.x, node.y, node.z) * voxelSize;
        vec3 nodeMax = nodeMin + vec3(node.size) * voxelSize;

        float tNear, tFar;
        if (!intersectAABB(rayOrigin, rayDir, nodeMin, nodeMax, tNear, tFar))
            continue;

        // Skip if beyond our max distance
        if (tNear > maxDistance)
            continue;
        
        // Apply LOD based on distance
        float distToNode = tNear;
        float lodLevel = calculateLODLevel(distToNode);
        
        // Determine if we should stop traversal based on LOD level
        bool isTerminalForLOD = node.isLeaf == 1 || 
                               node.isUniform == 1 || 
                               float(node.size * voxelSize) < (minVoxelSize * lodLevel);

        // For solid leaf, uniform solid, or LOD-terminal nodes
        if ((node.isLeaf == 1 || node.isUniform == 1 || isTerminalForLOD) && node.isSolid == 1) {
            float tHit = max(0.0, tNear);
            if (tHit <= tFar && tHit <= maxDistance) {
                // Calculate hit point and normal
                vec3 hitPoint = rayOrigin + rayDir * tHit;
                vec3 center = 0.5 * (nodeMin + nodeMax);
                vec3 normal = normalize(hitPoint - center);
                
                // Check if on edge
                bool edge = isOnBuildingEdge(hitPoint, normal);
                
                // Store hit information
                if (hits.count < MAX_HITS) {
                    hits.hits[hits.count].t = tHit;
                    hits.hits[hits.count].position = hitPoint;
                    hits.hits[hits.count].normal = normal;
                    hits.hits[hits.count].nodeIndex = nodeIdx;
                    hits.hits[hits.count].nodeSize = node.size;
                    hits.hits[hits.count].isEdge = edge;
                    hits.count++;
                }
            }
        }
        else if (node.isLeaf == 0 && node.isUniform == 0 && !isTerminalForLOD) {
            // For non-leaf non-uniform, push children
            for (int c = 0; c < 8; c++) {
                int childIdx = node.child[c];
                if (childIdx >= 0 && sp < 127) {
                    stack[sp++] = childIdx;
                }
            }
        }
    }
    
    // Sort hits by distance (closest first)
    // Simple bubble sort since array is small
    for (int i = 0; i < hits.count - 1; i++) {
        for (int j = 0; j < hits.count - i - 1; j++) {
            if (hits.hits[j].t > hits.hits[j+1].t) {
                // Swap
                HitInfo temp = hits.hits[j];
                hits.hits[j] = hits.hits[j+1];
                hits.hits[j+1] = temp;
            }
        }
    }
}

vec3 calculateFaceNormal(vec3 hitPoint, vec3 nodeMin, vec3 nodeMax) {
    vec3 center = 0.5 * (nodeMin + nodeMax);
    vec3 toHit = hitPoint - center;
    
    // Find which component (x, y, or z) has the largest absolute value
    // This tells us which face we're closest to
    vec3 absToHit = abs(toHit);
    float maxComponent = max(max(absToHit.x, absToHit.y), absToHit.z);
    
    // Create a face normal based on which component is largest
    vec3 normal = vec3(0.0);
    
    if (absToHit.x == maxComponent) {
        normal.x = sign(toHit.x);
    } else if (absToHit.y == maxComponent) {
        normal.y = sign(toHit.y);
    } else {
        normal.z = sign(toHit.z);
    }
    
    return normal;
}


// Original single hit intersection function for normal mode and volume computation
void intersectOctreeForVolume(vec3 rayOrigin, vec3 rayDir, 
                            out float closestT, 
                            out vec3 outNormal,
                            out VolumeInterval intervals[16],
                            out int intervalCount)
{
    bool hitFound = false;
    vec3 bestNormal = vec3(0);
    float bestT = 1e30;
    
    // Start with zero intervals
    intervalCount = 0;
    
    // Stack for BFS
    int stack[128];
    int sp = 0;
    stack[sp++] = 0; // root node index = 0

    while (sp > 0) {
        sp--;
        int nodeIdx = stack[sp];
        if (nodeIdx < 0) continue;

        // Read the node
        OctreeNodeGPUStruct node = nodes[nodeIdx];

        // Build the node's AABB
        vec3 nodeMin = gridMin + vec3(node.x, node.y, node.z) * voxelSize;
        vec3 nodeMax = nodeMin + vec3(node.size) * voxelSize;

        float tNear, tFar;
        if (!intersectAABB(rayOrigin, rayDir, nodeMin, nodeMax, tNear, tFar))
            continue;

        // If we're looking for the closest hit for shading
        if (tNear >= bestT) continue;
        
        // Apply LOD based on distance
        float distToNode = tNear;
        float lodLevel = calculateLODLevel(distToNode);
        
        // Determine if we should stop traversal based on LOD level
        // If node size is large enough for current LOD level or is a leaf,
        // treat it as a terminal node
        bool isTerminalForLOD = node.isLeaf == 1 || 
                               node.isUniform == 1 || 
                               float(node.size * voxelSize) < (minVoxelSize * lodLevel);
        
        // For solid leaf, uniform solid, or LOD-terminal nodes
        bool isSolid = false;
        
        // Uniform node => treat as leaf
        if (node.isUniform == 1) {
            if (node.isSolid == 1) {
                // This node is fully solid - for shading
                float tHit = max(0.0, tNear);
                if (tHit < bestT && tHit <= tFar) {
                    bestT = tHit;
                    hitFound = true;
                    vec3 center = 0.5 * (nodeMin + nodeMax);
                    vec3 p = rayOrigin + rayDir * tHit;
                    bestNormal = calculateFaceNormal(p, nodeMin, nodeMax);
                }
                
                // And for volume calculation
                isSolid = true;
            }
        }
        else if (node.isLeaf == 1) {
            // Leaf node
            if (node.isSolid == 1) {
                // Solid leaf - for shading
                float tHit = max(0.0, tNear);
                if (tHit < bestT && tHit <= tFar) {
                    bestT = tHit;
                    hitFound = true;
                    vec3 center = 0.5 * (nodeMin + nodeMax);
                    vec3 p = rayOrigin + rayDir * tHit;
                    bestNormal = calculateFaceNormal(p, nodeMin, nodeMax);
                }
                
                // And for volume calculation
                isSolid = true;
            }
        }
        else if (isTerminalForLOD && node.isSolid == 1) {
            // Node is terminal due to LOD and is solid
            float tHit = max(0.0, tNear);
            if (tHit < bestT && tHit <= tFar) {
                bestT = tHit;
                hitFound = true;
                vec3 center = 0.5 * (nodeMin + nodeMax);
                vec3 p = rayOrigin + rayDir * tHit;
                bestNormal = calculateFaceNormal(p, nodeMin, nodeMax);
            }
            
            // And for volume calculation
            isSolid = true;
        }
        else if (!isTerminalForLOD) {
            // For non-leaf, non-terminal nodes, push children
            for (int c = 0; c < 8; c++) {
                int childIdx = node.child[c];
                if (childIdx >= 0) {
                    stack[sp++] = childIdx;
                }
            }
        }
        
        // Add volume interval if solid
        if (isSolid && enableVolumeMeasurement) {
            // Make sure we're within ray limits
            tNear = max(0.0, tNear);
            
            // Don't add zero-length intervals
            if (tFar > tNear && intervalCount < 16) {
                intervals[intervalCount].tStart = tNear;
                intervals[intervalCount].tEnd = tFar;
                intervalCount++;
            }
        }
    }

    if (hitFound) {
        closestT = bestT;
        outNormal = bestNormal;
    } else {
        closestT = -1.0;
    }
}

// Merge overlapping intervals for accurate volume calculation
void mergeIntervals(inout VolumeInterval intervals[16], inout int intervalCount) {
    if (intervalCount <= 1) return;
    
    // Sort intervals by start time (simple bubble sort)
    for (int i = 0; i < intervalCount - 1; i++) {
        for (int j = 0; j < intervalCount - i - 1; j++) {
            if (intervals[j].tStart > intervals[j+1].tStart) {
                VolumeInterval temp = intervals[j];
                intervals[j] = intervals[j+1];
                intervals[j+1] = temp;
            }
        }
    }
    
    // Merge overlapping intervals
    int indexToKeep = 0;
    for (int i = 1; i < intervalCount; i++) {
        // If current interval overlaps with previous
        if (intervals[indexToKeep].tEnd >= intervals[i].tStart) {
            // Update end of previous interval if current end is greater
            intervals[indexToKeep].tEnd = max(intervals[indexToKeep].tEnd, intervals[i].tEnd);
        } else {
            // No overlap, keep this interval
            indexToKeep++;
            intervals[indexToKeep] = intervals[i];
        }
    }
    
    // Update interval count
    intervalCount = indexToKeep + 1;
}

vec4 renderXRayMode(Ray ray, float maxDistance) {
    // Get all intersections along the ray
    HitArray hits;
    intersectOctreeMultiHit(ray.origin, ray.direction, maxDistance, hits);
    
    // If no hits, return black background
    if (hits.count == 0) {
        return vec4(0.0, 0.0, 0.0, 1.0);
    }
    
    // Render hits from back to front (for correct alpha blending)
    vec4 finalColor = vec4(0.0, 0.0, 0.0, 0.0);
    
    // Process hits in reverse order (back to front)
    for (int i = hits.count - 1; i >= 0; i--) {
        // Get hit information
        HitInfo hit = hits.hits[i];
        
        // X-ray shading with depth-based color and opacity
        vec4 hitColor = shadeXRay(hit.position, hit.normal, hit.t, hit.isEdge, hit.nodeSize);
        
        // Alpha blending (back to front)
        finalColor.rgb = hitColor.rgb * hitColor.a + finalColor.rgb * (1.0 - hitColor.a);
        finalColor.a = hitColor.a + finalColor.a * (1.0 - hitColor.a);
        
        // Early termination when opaque enough
        if (finalColor.a > 0.98) break;
    }
    
    // Ensure fully opaque output
    finalColor.a = 1.0;
    
    return finalColor;
}

vec4 renderSeeThroughMode(Ray ray, float maxDistance) {
    // Get all intersections along the ray
    HitArray hits;
    intersectOctreeMultiHit(ray.origin, ray.direction, maxDistance, hits);
    
    // If no hits, return black background
    if (hits.count == 0) {
        return vec4(0.0, 0.0, 0.0, 1.0);
    }
    
    // Render hits from back to front (for correct alpha blending)
    vec4 finalColor = vec4(0.0, 0.0, 0.0, 0.0);
    
    // Process hits in reverse order (back to front)
    for (int i = hits.count - 1; i >= 0; i--) {
        // Get hit information
        HitInfo hit = hits.hits[i];
        
        // See-through shading with partial transparency
        vec4 hitColor = shadeSeeThrough(hit.position, hit.normal, hit.t, hit.nodeSize);
        
        // Alpha blending (front to back)
        finalColor.rgb = finalColor.rgb + (1.0 - finalColor.a) * hitColor.a * hitColor.rgb;
        finalColor.a = finalColor.a + (1.0 - finalColor.a) * hitColor.a;
        
        // Early termination when opaque enough
        if (finalColor.a > 0.99) break;
    }
    
    // Ensure fully opaque output
    finalColor.a = 1.0;
    
    return finalColor;
}

vec4 renderNormalMode(Ray ray) {
    // For volume measurement, collect all intervals
    VolumeInterval intervals[16];
    int intervalCount = 0;
    
    // Intersect the octree for rendering and volume
    float tClosest;
    vec3 normal;
    intersectOctreeForVolume(ray.origin, ray.direction, tClosest, normal, intervals, intervalCount);
    
    // Render the scene
    vec4 color = vec4(0.0, 0.0, 0.0, 1.0);
    if (tClosest > 0.0) {
        vec3 hitPoint = ray.origin + ray.direction * tClosest;
        
        // Find node size by traversing to hit point
        int nodeSize = 1; // Default if we can't find the node
        
        // Compute world-space AABB for the scene
        vec3 sceneMin = gridMin;
        vec3 sceneMax = gridMin + vec3(voxelSize) * vec3(numNodes);
        
        // Apply normal shading
        color = vec4(shadeNormal(hitPoint, normal, nodeSize), 1.0);
    }
    
    return color;
}

Ray generateRay(int px, int py, int w, int h,
                vec3 camPos, mat4 view, float fovDeg, float aspect)
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
    if (gid.x >= imageWidth || gid.y >= imageHeight) {
        return;
    }

    // Generate ray for this pixel
    Ray ray = generateRay(gid.x, gid.y,
                          imageWidth, imageHeight,
                          cameraPos, viewMat, fov, aspect);

    // Maximum ray distance (for x-ray and see-through modes)
    float maxDistance = 5000.0;
    
    // Choose rendering mode
    vec4 finalColor;
    
    if (renderMode == 1) {
        // X-ray mode
        finalColor = renderXRayMode(ray, maxDistance);
    } 
    else if (renderMode == 2) {
        // See-through mode
        finalColor = renderSeeThroughMode(ray, maxDistance);
    }
    else {
        // Normal mode
        finalColor = renderNormalMode(ray);
        
        // Compute volume if enabled (only in normal mode)
        if (enableVolumeMeasurement) {
            // For volume measurement, collect all intervals
            VolumeInterval intervals[16];
            int intervalCount = 0;
            
            // Intersect the octree for volume calculation
            float tClosest;
            vec3 normal;
            intersectOctreeForVolume(ray.origin, ray.direction, tClosest, normal, intervals, intervalCount);
            
            if (intervalCount > 0) {
                // Merge intervals to handle overlaps
                mergeIntervals(intervals, intervalCount);
                
                // Compute variable cone size for each interval
                float totalVolume = 0.0;
                
                // Get base pixel size at distance 1.0
                float fovRad = radians(fov);
                float pixelSize = 2.0 * tan(fovRad * 0.5) / float(imageHeight);
                
                // For each interval, calculate volume with proper perspective scaling
                for (int i = 0; i < intervalCount; i++) {
                    float tStart = intervals[i].tStart;
                    float tEnd = intervals[i].tEnd;
                    float intervalLength = tEnd - tStart;
                    
                    // Skip tiny intervals (numerical precision issues)
                    if (intervalLength < 0.0001) continue;
                    
                    // Calculate middle distance for this interval
                    float midDist = (tStart + tEnd) * 0.5;
                    
                    // For perspective correction, scale pixel area by square of distance
                    float areaAtMidDist = pixelSize * pixelSize * midDist * midDist;
                    
                    // Volume = area × length
                    float intervalVolume = areaAtMidDist * intervalLength;
                    
                    // Accumulate volume
                    totalVolume += intervalVolume;
                }
                
                // Scale by a factor to account for entire image
                float scalingFactor = 1.0 / float(imageWidth * imageHeight);
                totalVolume *= scalingFactor;
                
                // Convert to integer for atomic addition
                int scaledVolume = int(totalVolume * VOLUME_SCALE + 0.5);
                
                // Add to the accumulated volume
                atomicAdd(accumulatedVolume, scaledVolume);
            }
        }
    }
    
    // Store the output color
    imageStore(outputImage, gid, finalColor);
}
