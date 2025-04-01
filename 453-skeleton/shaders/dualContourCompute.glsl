#version 430 core
layout(local_size_x = 8, local_size_y = 8, local_size_z = 4) in;

// Reduced workgroup size for better reliability

// Grid information buffer - header only now
layout(std430, binding = 0) readonly buffer VoxelGridBuffer {
    int dimX;
    int dimY;
    int dimZ;
    float minX;
    float minY;
    float minZ;
    float voxelSize;
} grid;

// Grid data is now in a separate buffer
layout(std430, binding = 4) readonly buffer GridDataBuffer {
    int data[];
} gridData;

struct HermitePoint {
    vec3 position;
    vec3 normal;
};

layout(std430, binding = 1) buffer HermitePointsBuffer {
    HermitePoint points[];
} hermiteData;

layout(std430, binding = 2) buffer VertexBuffer {
    vec4 vertices[];
} vertexBuffer;

struct Triangle {
    vec3 v0;  vec3 n0;
    vec3 v1;  vec3 n1;
    vec3 v2;  vec3 n2;
};

layout(std430, binding = 3) buffer TriangleBuffer {
    int triangleCount;
    Triangle triangles[];
} triangleBuffer;

uniform ivec3 chunkStart;
uniform int chunkSize;
uniform float cellSizeWorld;

// Check if a voxel is solid (filled)
bool isSolid(ivec3 pos) {
    // Treat out-of-bounds as empty
    if (pos.x < 0 || pos.y < 0 || pos.z < 0 ||
        pos.x >= grid.dimX || pos.y >= grid.dimY || pos.z >= grid.dimZ) {
        return false;
    }
    
    // Get index into the grid data array
    int index = pos.x + pos.y * grid.dimX + pos.z * (grid.dimX * grid.dimY);
    
    // Safety check for index bounds
    if (index < 0 || index >= gridData.data.length()) {
        return false;
    }
    
    return (gridData.data[index] == 1);
}

// Convert grid coordinates to world space
vec3 gridToWorld(ivec3 pos) {
    return vec3(
        grid.minX + pos.x * grid.voxelSize,
        grid.minY + pos.y * grid.voxelSize,
        grid.minZ + pos.z * grid.voxelSize
    );
}

// Estimate the normal at a given position
vec3 estimateNormal(ivec3 pos) {
    vec3 n = vec3(0);
    
    // Safety check for position bounds
    if (pos.x <= 0 || pos.x >= grid.dimX - 1 ||
        pos.y <= 0 || pos.y >= grid.dimY - 1 ||
        pos.z <= 0 || pos.z >= grid.dimZ - 1) {
        return vec3(1, 0, 0); // Default direction for edge cases
    }
    
    // Use central differences for normal estimation
    n.x = float(isSolid(pos + ivec3(-1,0,0))) - float(isSolid(pos + ivec3(1,0,0)));
    n.y = float(isSolid(pos + ivec3(0,-1,0))) - float(isSolid(pos + ivec3(0,1,0)));
    n.z = float(isSolid(pos + ivec3(0,0,-1))) - float(isSolid(pos + ivec3(0,0,1)));
    
    // Normalize or provide a default direction
    if (length(n) < 0.01) {
        return vec3(1, 0, 0);
    }
    
    return normalize(n);
}

// Simplified QEF solver for dual vertex position
vec3 solveQEF(vec3 cellCenter, vec3 minB, vec3 maxB, int startIdx, int endIdx) {
    if (startIdx >= endIdx) {
        return cellCenter;
    }
    
    // Simple mass point calculation (average of intersections)
    vec3 sumPos = vec3(0);
    vec3 sumN = vec3(0);
    int count = endIdx - startIdx;
    
    for (int i = startIdx; i < endIdx; i++) {
        // Bounds check
        if (i >= 0 && i < hermiteData.points.length()) {
            sumPos += hermiteData.points[i].position;
            sumN += hermiteData.points[i].normal;
        }
    }
    
    if (count <= 0) return cellCenter;
    
    vec3 avgPos = sumPos / float(count);
    
    // Project onto normal plane if possible
    if (length(sumN) > 0.01) {
        vec3 n = normalize(sumN);
        float d = -dot(n, avgPos);
        float t = -(dot(n, cellCenter) + d);
        vec3 proj = cellCenter + t*n;
        
        // Ensure result is within cell bounds
        return clamp(proj, minB, maxB);
    }
    
    // Fallback to average position within bounds
    return clamp(avgPos, minB, maxB);
}

// Process a cell to find surface intersections and create a dual vertex
void processCell(ivec3 cellPos, int size) {
    // Bounds check for cell position
    if (cellPos.x < 0 || cellPos.y < 0 || cellPos.z < 0 ||
        cellPos.x >= grid.dimX || cellPos.y >= grid.dimY || cellPos.z >= grid.dimZ) {
        return;
    }

    // Check for sign change in this cell (surface crossing)
    bool hasInside = false;
    bool hasOutside = false;
    
    // Sample multiple points to detect surface
    for (int z = 0; z < size && z < 2; z++) {
        for (int y = 0; y < size && y < 2; y++) {
            for (int x = 0; x < size && x < 2; x++) {
                ivec3 vpos = cellPos + ivec3(x,y,z);
                bool fill = isSolid(vpos);
                if (fill) hasInside = true;
                else hasOutside = true;
                
                // Early exit if surface found
                if (hasInside && hasOutside) break;
            }
            if (hasInside && hasOutside) break;
        }
        if (hasInside && hasOutside) break;
    }
    
    // Skip if no surface crosses this cell
    if (!hasInside || !hasOutside) {
        return;
    }

    // Store starting index for hermite points
    int hermiteStart = atomicAdd(triangleBuffer.triangleCount, 0);
    
    // Sample grid within the cell to find edge intersections
    ivec3 minIdx = cellPos;
    ivec3 maxIdx = cellPos + ivec3(min(size, 2));  // Limit size for efficiency
    
    // Clamp to grid boundaries
    maxIdx = min(maxIdx, ivec3(grid.dimX, grid.dimY, grid.dimZ));
    
    // Check edges for surface crossings
    for (int z = minIdx.z; z < maxIdx.z; z++) {
        for (int y = minIdx.y; y < maxIdx.y; y++) {
            for (int x = minIdx.x; x < maxIdx.x; x++) {
                ivec3 pos = ivec3(x,y,z);
                bool cFill = isSolid(pos);
                
                // Check +X edge
                if (x+1 < grid.dimX) {
                    ivec3 nxt = pos + ivec3(1,0,0);
                    bool nFill = isSolid(nxt);
                    
                    if (cFill != nFill) {
                        // Create hermite point at intersection
                        float t = 0.5;  // Simplified interpolation
                        vec3 wPos = gridToWorld(pos) + vec3(t*grid.voxelSize, 0, 0);
                        vec3 norm = estimateNormal(cFill ? pos : nxt);
                        
                        // Store the hermite point
                        int idx = atomicAdd(triangleBuffer.triangleCount, 1);
                        if (idx < hermiteData.points.length()) {
                            hermiteData.points[idx].position = wPos;
                            hermiteData.points[idx].normal = norm;
                        }
                    }
                }
                
                // Check +Y edge
                if (y+1 < grid.dimY) {
                    ivec3 nxt = pos + ivec3(0,1,0);
                    bool nFill = isSolid(nxt);
                    
                    if (cFill != nFill) {
                        float t = 0.5;
                        vec3 wPos = gridToWorld(pos) + vec3(0, t*grid.voxelSize, 0);
                        vec3 norm = estimateNormal(cFill ? pos : nxt);
                        
                        int idx = atomicAdd(triangleBuffer.triangleCount, 1);
                        if (idx < hermiteData.points.length()) {
                            hermiteData.points[idx].position = wPos;
                            hermiteData.points[idx].normal = norm;
                        }
                    }
                }
                
                // Check +Z edge
                if (z+1 < grid.dimZ) {
                    ivec3 nxt = pos + ivec3(0,0,1);
                    bool nFill = isSolid(nxt);
                    
                    if (cFill != nFill) {
                        float t = 0.5;
                        vec3 wPos = gridToWorld(pos) + vec3(0, 0, t*grid.voxelSize);
                        vec3 norm = estimateNormal(cFill ? pos : nxt);
                        
                        int idx = atomicAdd(triangleBuffer.triangleCount, 1);
                        if (idx < hermiteData.points.length()) {
                            hermiteData.points[idx].position = wPos;
                            hermiteData.points[idx].normal = norm;
                        }
                    }
                }
            }
        }
    }
    
    // Get ending index for hermite points
    int hermiteEnd = atomicAdd(triangleBuffer.triangleCount, 0);
    
    // Generate the dual vertex for this cell
    if (hermiteEnd > hermiteStart) {
        // Calculate cell bounds and center
        vec3 cCenter = gridToWorld(cellPos) + vec3(0.5*grid.voxelSize);
        vec3 minB = gridToWorld(cellPos);
        vec3 maxB = gridToWorld(cellPos + ivec3(size));
        
        // Solve for optimal vertex position
        vec3 vertex = solveQEF(cCenter, minB, maxB, hermiteStart, hermiteEnd);
        
        // Store vertex in buffer
        int cellID = cellPos.x + cellPos.y*1024 + cellPos.z*1048576;
        if (cellID < vertexBuffer.vertices.length()) {
            vertexBuffer.vertices[cellID] = vec4(vertex, 1.0);
        }
    }
    else {
        // No intersections found, check for nearby surface
        bool nearSurface = false;
        
        for (int dz=-1; dz<=1 && !nearSurface; dz++) {
            for (int dy=-1; dy<=1 && !nearSurface; dy++) {
                for (int dx=-1; dx<=1 && !nearSurface; dx++) {
                    if (dx==0 && dy==0 && dz==0) continue;
                    
                    ivec3 nbr = cellPos + ivec3(dx,dy,dz);
                    bool cFill = isSolid(cellPos);
                    bool nFill = isSolid(nbr);
                    
                    if (cFill != nFill) {
                        nearSurface = true;
                        break;
                    }
                }
            }
        }
        
        if (nearSurface) {
            // Create a fallback vertex at cell center
            vec3 fallback = gridToWorld(cellPos) + vec3(0.5*grid.voxelSize);
            int cellID = cellPos.x + cellPos.y*1024 + cellPos.z*1048576;
            
            if (cellID < vertexBuffer.vertices.length()) {
                vertexBuffer.vertices[cellID] = vec4(fallback, 1.0);
            }
        }
    }
}

// Forward declaration
void emitQuad(vec3 V00, vec3 V01, vec3 V11, vec3 V10);

// Get cell ID for vertex lookup
int cellID(ivec3 c) {
    return c.x + c.y*1024 + c.z*1048576;
}

// Check if a cell is solid
bool isSolidCell(ivec3 c) {
    return isSolid(c);
}

// Ensure a vertex exists at the given position
bool ensureVertex(ivec3 c) {
    // Bounds check
    if (c.x < 0 || c.x >= grid.dimX ||
        c.y < 0 || c.y >= grid.dimY ||
        c.z < 0 || c.z >= grid.dimZ) {
        return false;
    }
    
    // Get vertex ID
    int id = cellID(c);
    
    // Make sure buffer index is valid
    if (id >= vertexBuffer.vertices.length()) {
        return false;
    }
    
    // Create a fallback vertex if none exists
    if (vertexBuffer.vertices[id].w <= 0.0) {
        vec3 fallbackPos = gridToWorld(c) + vec3(0.5*grid.voxelSize);
        vertexBuffer.vertices[id] = vec4(fallbackPos, 0.5);
    }
    
    return true;
}

// Generate triangles for a cell
void generateTriangles(ivec3 cellPos) {
    // Bounds check
    if (cellPos.x < 0 || cellPos.y < 0 || cellPos.z < 0 ||
        cellPos.x >= grid.dimX || cellPos.y >= grid.dimY || cellPos.z >= grid.dimZ) {
        return;
    }
    
    // Check cell state
    bool s0 = isSolidCell(cellPos);
    
    // Check +X face
    if (cellPos.x+1 < grid.dimX) {
        bool s1 = isSolidCell(cellPos + ivec3(1,0,0));
        
        // Only create face if there's a sign change
        if (s0 != s1) {
            // Ensure vertices exist
            for (int dy=0; dy<=1; dy++) {
                for (int dz=0; dz<=1; dz++) {
                    ensureVertex(cellPos + ivec3(0,dy,dz));
                    ensureVertex(cellPos + ivec3(1,dy,dz));
                }
            }
            
            // Create quad faces
            for (int dy=0; dy<1; dy++) {
                for (int dz=0; dz<1; dz++) {
                    // Get vertex positions
                    ivec3 c00 = cellPos + ivec3(0, dy, dz);
                    ivec3 c01 = cellPos + ivec3(0, dy+1, dz);
                    ivec3 c10 = cellPos + ivec3(1, dy, dz);
                    ivec3 c11 = cellPos + ivec3(1, dy+1, dz);
                    
                    // Check if all vertices are valid
                    if (cellID(c00) < vertexBuffer.vertices.length() &&
                        cellID(c01) < vertexBuffer.vertices.length() &&
                        cellID(c10) < vertexBuffer.vertices.length() &&
                        cellID(c11) < vertexBuffer.vertices.length()) {
                        
                        vec3 V00 = vertexBuffer.vertices[cellID(c00)].xyz;
                        vec3 V01 = vertexBuffer.vertices[cellID(c01)].xyz;
                        vec3 V10 = vertexBuffer.vertices[cellID(c10)].xyz;
                        vec3 V11 = vertexBuffer.vertices[cellID(c11)].xyz;
                        
                        // Create triangles for this quad
                        emitQuad(V00, V01, V11, V10);
                    }
                }
            }
        }
    }

// Check +Y face
    if (cellPos.y+1 < grid.dimY) {
        bool s2 = isSolidCell(cellPos + ivec3(0,1,0));
        
        if (s0 != s2) {
            for (int dx=0; dx<=1; dx++) {
                for (int dz=0; dz<=1; dz++) {
                    ensureVertex(cellPos + ivec3(dx,0,dz));
                    ensureVertex(cellPos + ivec3(dx,1,dz));
                }
            }
            
            for (int dx=0; dx<1; dx++) {
                for (int dz=0; dz<1; dz++) {
                    ivec3 c00 = cellPos + ivec3(dx, 0, dz);
                    ivec3 c01 = cellPos + ivec3(dx+1, 0, dz);
                    ivec3 c10 = cellPos + ivec3(dx, 1, dz);
                    ivec3 c11 = cellPos + ivec3(dx+1, 1, dz);
                    
                    if (cellID(c00) < vertexBuffer.vertices.length() &&
                        cellID(c01) < vertexBuffer.vertices.length() &&
                        cellID(c10) < vertexBuffer.vertices.length() &&
                        cellID(c11) < vertexBuffer.vertices.length()) {
                            
                        vec3 V00 = vertexBuffer.vertices[cellID(c00)].xyz;
                        vec3 V01 = vertexBuffer.vertices[cellID(c01)].xyz;
                        vec3 V10 = vertexBuffer.vertices[cellID(c10)].xyz;
                        vec3 V11 = vertexBuffer.vertices[cellID(c11)].xyz;
                        
                        emitQuad(V00, V01, V11, V10);
                    }
                }
            }
        }
    }
    
    // Check +Z face
    if (cellPos.z+1 < grid.dimZ) {
        bool s3 = isSolidCell(cellPos + ivec3(0,0,1));
        
        if (s0 != s3) {
            for (int dx=0; dx<=1; dx++) {
                for (int dy=0; dy<=1; dy++) {
                    ensureVertex(cellPos + ivec3(dx,dy,0));
                    ensureVertex(cellPos + ivec3(dx,dy,1));
                }
            }
            
            for (int dx=0; dx<1; dx++) {
                for (int dy=0; dy<1; dy++) {
                    ivec3 c00 = cellPos + ivec3(dx, dy, 0);
                    ivec3 c01 = cellPos + ivec3(dx, dy+1, 0);
                    ivec3 c10 = cellPos + ivec3(dx, dy, 1);
                    ivec3 c11 = cellPos + ivec3(dx, dy+1, 1);
                    
                    if (cellID(c00) < vertexBuffer.vertices.length() &&
                        cellID(c01) < vertexBuffer.vertices.length() &&
                        cellID(c10) < vertexBuffer.vertices.length() &&
                        cellID(c11) < vertexBuffer.vertices.length()) {
                            
                        vec3 V00 = vertexBuffer.vertices[cellID(c00)].xyz;
                        vec3 V01 = vertexBuffer.vertices[cellID(c01)].xyz;
                        vec3 V10 = vertexBuffer.vertices[cellID(c10)].xyz;
                        vec3 V11 = vertexBuffer.vertices[cellID(c11)].xyz;
                        
                        emitQuad(V00, V01, V11, V10);
                    }
                }
            }
        }
    }
}

// Create two triangles from a quad
void emitQuad(in vec3 V00, in vec3 V01, in vec3 V11, in vec3 V10) {
    // Safety check to avoid out-of-bounds writes
    int count = triangleBuffer.triangleCount;
    if (count + 2 >= triangleBuffer.triangles.length()) {
        return;
    }
    
    // First triangle
    int t0 = atomicAdd(triangleBuffer.triangleCount, 1);
    triangleBuffer.triangles[t0].v0 = V00; 
    triangleBuffer.triangles[t0].v1 = V01; 
    triangleBuffer.triangles[t0].v2 = V11;

    vec3 e1 = V01 - V00;
    vec3 e2 = V11 - V00;
    vec3 n = normalize(cross(e1, e2));
    
    triangleBuffer.triangles[t0].n0 = n;
    triangleBuffer.triangles[t0].n1 = n;
    triangleBuffer.triangles[t0].n2 = n;

    // Second triangle
    int t1 = atomicAdd(triangleBuffer.triangleCount, 1);
    triangleBuffer.triangles[t1].v0 = V00; 
    triangleBuffer.triangles[t1].v1 = V11; 
    triangleBuffer.triangles[t1].v2 = V10;

    e1 = V11 - V00;
    e2 = V10 - V00;
    n = normalize(cross(e1, e2));
    
    triangleBuffer.triangles[t1].n0 = n;
    triangleBuffer.triangles[t1].n1 = n;
    triangleBuffer.triangles[t1].n2 = n;
}

void main() {
    // Get cell position from global invocation ID
    ivec3 cellPos = ivec3(gl_GlobalInvocationID.xyz) + chunkStart;
    
    // Safety check for bounds
    if (cellPos.x < 0 || cellPos.y < 0 || cellPos.z < 0 ||
        cellPos.x >= grid.dimX || cellPos.y >= grid.dimY || cellPos.z >= grid.dimZ) {
        return;
    }
    
    // Process cell to find surface intersections and create vertex
    processCell(cellPos, chunkSize);
    
    // Wait for all threads to complete
    barrier();
    memoryBarrier();
    
    // Generate triangles for this cell
    generateTriangles(cellPos);
}
