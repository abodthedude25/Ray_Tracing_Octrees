#version 430 core
in vec2 vTexCoord;
out vec4 FragColor;

uniform float octreeSkipT;
uniform mat4 invView;
uniform mat4 invProj;
uniform vec3 camPos;
uniform vec3 boxMin;
uniform vec3 boxMax;
uniform sampler3D volumeTex;
uniform sampler3D radiationTex;
uniform sampler3D gradientMagTex;
uniform sampler3D gradientDirTex;
uniform sampler3D edgeFactorTex;
uniform sampler3D ambientOcclusionTex;
uniform sampler3D indirectLightTex;
uniform float timeValue;
uniform bool useFrustumCulling;
uniform vec3 previousCamPos;
uniform vec3 previousLookDir;
uniform bool enableOctreeSkip;
uniform sampler3D octreeSkipTex;
uniform int maxMipLevel;
uniform bool useMipMappedSkipping;
uniform sampler3D workingVolumeTex; 

// Lighting parameters
const vec3 mainLightDir = normalize(vec3(0.5, 0.9, 0.4));
const vec3 mainLightColor = vec3(1.0, 0.98, 0.9) * 4.0;
const vec3 skyLightColor = vec3(0.6, 0.7, 0.9) * 0.4;
const vec3 groundLightColor = vec3(0.3, 0.25, 0.2) * 0.2;

// Material parameters
const float edgeThreshold = 0.5;
const float edgeWidth = 0.03;
const float alphaCutoff = 0.95;

// Window parameters
const float windowDensity = 0.4;
const float windowWidth = 0.3;
const float windowHeight = 0.7;

// TAA jitter patterns (8 samples is sufficient for good results while maintaining performance)
const vec2 haltonJitter[8] = vec2[8](
    vec2(0.5000, 0.3333), vec2(0.2500, 0.6667), vec2(0.7500, 0.1111), vec2(0.1250, 0.4444),
    vec2(0.6250, 0.7778), vec2(0.3750, 0.2222), vec2(0.8750, 0.5556), vec2(0.0625, 0.8889)
);

// Hash function for material variation and noise
float hash(vec3 p) {
    p = fract(p * vec3(0.1031, 0.1030, 0.0973));
    p += dot(p, p.yxz + 33.33);
    return fract((p.x + p.y) * p.z);
}

// Improved hash function with better distribution
float hash2(vec3 p) {
    p = fract(p * vec3(5.3983, 5.4427, 6.9371));
    p += dot(p.yzx, p.xyz + vec3(21.5351, 14.3137, 15.3219));
    return fract(dot(p, vec3(23.62431, 34.81539, 18.36229)));
}

// Compute box intersection for ray
vec2 intersectBox(vec3 ro, vec3 rd, vec3 bmin, vec3 bmax) {
    // Inverse ray direction optimization to avoid division
    vec3 invRd = 1.0 / rd;
    
    // Handle near-zero ray direction components to avoid NaN
    const float eps = 1e-10;
    if (abs(rd.x) < eps) invRd.x = rd.x >= 0.0 ? 1e10 : -1e10;
    if (abs(rd.y) < eps) invRd.y = rd.y >= 0.0 ? 1e10 : -1e10;
    if (abs(rd.z) < eps) invRd.z = rd.z >= 0.0 ? 1e10 : -1e10;
    
    // Calculate entry and exit points
    vec3 t1 = (bmin - ro) * invRd;
    vec3 t2 = (bmax - ro) * invRd;
    
    vec3 tmin = min(t1, t2);
    vec3 tmax = max(t1, t2);
    
    float tN = max(max(tmin.x, tmin.y), tmin.z);
    float tF = min(min(tmax.x, tmax.y), tmax.z);
    
    return vec2(tN, tF);
}

// Calculate ambient occlusion in corners
float calculateCornersAO(vec3 uvw, vec3 normal) {
    // Skip AO calculation for distant areas (optimization)
    float distToCamera = length((uvw * (boxMax - boxMin) + boxMin) - camPos);
    if (distToCamera > 150.0) return 0.85; // Default medium AO for distant areas
    
    // Check for corners by sampling in perpendicular directions
    vec3 perp1 = normalize(cross(normal, vec3(0,1,0)));
    if (length(perp1) < 0.01) perp1 = normalize(cross(normal, vec3(1,0,0)));
    vec3 perp2 = normalize(cross(normal, perp1));
    
    float cornerRadius = 0.05;
    
    // Sample nearby points in perpendicular directions
    float s1 = texture(volumeTex, uvw + perp1 * cornerRadius).r;
    float s2 = texture(volumeTex, uvw + perp2 * cornerRadius).r;
    float s3 = texture(volumeTex, uvw - perp1 * cornerRadius).r;
    float s4 = texture(volumeTex, uvw - perp2 * cornerRadius).r;
    
    // If we're at an inside corner, darken
    int solidCount = 0;
    if (s1 > edgeThreshold) solidCount++;
    if (s2 > edgeThreshold) solidCount++;
    if (s3 > edgeThreshold) solidCount++;
    if (s4 > edgeThreshold) solidCount++;
    
    if (solidCount >= 3) return 0.6; // Inside corner - darker
    else if (solidCount <= 1) return 0.9; // Outside corner or edge
    
    return 0.75; // Regular surface
}

// Material and texture functions
bool isWindowPosition(vec3 pos, vec3 normal) {
    // Add slight randomness to window positions for more natural appearance
    float randomOffset = hash(floor(pos / 5.0)) * 0.2;
    
    // Use different window pattern based on normal direction
    if (abs(normal.x) > 0.7) {
        // X-facing facades
        float windowGridY = mod(pos.y * windowDensity + randomOffset, 1.0);
        float windowGridZ = mod(pos.z * (windowDensity * 0.7), 1.0);
        return (windowGridY > (1.0 - windowWidth) * 0.5 && windowGridY < (1.0 + windowWidth) * 0.5) && 
               (windowGridZ > (1.0 - windowHeight) * 0.5 && windowGridZ < (1.0 + windowHeight) * 0.5);
    } 
    else if (abs(normal.z) > 0.7) {
        // Z-facing facades
        float windowGridX = mod(pos.x * windowDensity + randomOffset, 1.0);
        float windowGridY = mod(pos.y * (windowDensity * 0.7), 1.0);
        return (windowGridX > (1.0 - windowWidth) * 0.5 && windowGridX < (1.0 + windowWidth) * 0.5) && 
               (windowGridY > (1.0 - windowHeight) * 0.5 && windowGridY < (1.0 + windowHeight) * 0.5);
    }
    else if (normal.y > 0.7) {
        // Roof - fewer/no windows
        return false;
    }
    else if (normal.y < -0.7) {
        // Bottom - no windows
        return false;
    }
    
    // Default for other orientations
    float windowGridA = mod(dot(pos, vec3(1.0, 0, 0)) * windowDensity + randomOffset, 1.0);
    float windowGridB = mod(dot(pos, vec3(0, 1.0, 0)) * (windowDensity * 0.7), 1.0);
    return (windowGridA > (1.0 - windowWidth) * 0.5 && windowGridA < (1.0 + windowWidth) * 0.5) && 
           (windowGridB > (1.0 - windowHeight) * 0.5 && windowGridB < (1.0 + windowHeight) * 0.5);
}


// Improved building color function with better boundary definition
vec3 getBuildingColor(vec3 pos, vec3 normal) {
    // Create a hash based on position
    // Use a smaller divisor to create larger, more distinct building blocks
    vec3 buildingPos = floor(pos / 8.0); // Larger buildings for better distinction
    float buildingID = hash(buildingPos);
    
    // Create more distinct building colors with greater variation
    vec3 building1 = vec3(0.90, 0.85, 0.75); // Light cream
    vec3 building2 = vec3(0.73, 0.68, 0.62); // Medium gray
    vec3 building3 = vec3(0.86, 0.75, 0.65); // Tan
    vec3 building4 = vec3(0.65, 0.62, 0.58); // Dark gray
    vec3 building5 = vec3(0.81, 0.71, 0.65); // Light brown
    vec3 building6 = vec3(0.78, 0.82, 0.76); // Light green-gray
    
    // Pick a color based on building ID with more contrast between adjacent buildings
    vec3 baseColor;
    if (buildingID < 0.17) baseColor = building1;
    else if (buildingID < 0.34) baseColor = building2;
    else if (buildingID < 0.50) baseColor = building3;
    else if (buildingID < 0.67) baseColor = building4;
    else if (buildingID < 0.84) baseColor = building5;
    else baseColor = building6;
    
    // Add subtle height-based gradient for better vertical distinction
    float heightFactor = (pos.y - boxMin.y) / (boxMax.y - boxMin.y);
    baseColor = mix(baseColor, baseColor * 1.15, heightFactor * 0.3);
    
    // Check if we're near a building boundary
    vec3 buildingPosNext = buildingPos + sign(pos - (buildingPos * 8.0 + 4.0));
    float distToEdge = min(
        min(abs(mod(pos.x, 8.0) - 0.1), abs(mod(pos.x, 8.0) - 7.9)),
        min(abs(mod(pos.z, 8.0) - 0.1), abs(mod(pos.z, 8.0) - 7.9))
    );
    
    // Darken color near building boundaries
    if (distToEdge < 0.2) {
        // Calculate a smooth transition at boundaries
        float edgeFactor = smoothstep(0.0, 0.2, distToEdge);
        baseColor *= mix(0.7, 1.0, edgeFactor);
    }
    
    // Add more building-specific variation
    baseColor *= 0.9 + 0.2 * hash(buildingPos * 42.1);
    
    return baseColor;
}

// Enhance edge detection between buildings
float detectBuildingBoundaries(vec3 pos, vec3 normal) {
    // Check for building boundaries by looking at grid cells
    vec3 buildingGrid = pos / 8.0;
    
    // Calculate distance to nearest building boundary
    vec3 cellPosition = fract(buildingGrid);
    vec3 distanceToEdge = min(cellPosition, 1.0 - cellPosition);
    float minDist = min(min(distanceToEdge.x, distanceToEdge.z), 0.5); // Only consider x and z (not y/height)
    
    // Convert to a 0-1 factor (0 = on edge, 1 = far from edge)
    float boundaryFactor = smoothstep(0.0, 0.1, minDist);
    
    return boundaryFactor;
}

// Modified shadow calculation to enhance building separation
float calculateShadow(vec3 pos, vec3 lightDir) {
    // Start slightly offset to avoid self-shadowing artifacts
    vec3 ro = pos + lightDir * 0.05;
    vec3 rd = lightDir;
    
    // Intersect with volume bounds
    vec2 tHit = intersectBox(ro, rd, boxMin, boxMax);
    if (tHit.x > tHit.y) return 1.0; // No intersection, fully lit
    
    float T = max(tHit.x, 0.0);
    float Tfar = min(tHit.y, 5.0);
    
    float stepSize = (Tfar - T) / 10.0;
    float shadow = 1.0;
    
    // Check for building boundaries near the sampling point
    float boundaryFactor = detectBuildingBoundaries(pos, normalize(-lightDir));
    
    // Enhance shadows at building boundaries
    if (boundaryFactor < 0.6) {
        shadow *= 0.8 + 0.2 * boundaryFactor; // Darker shadows near building edges
    }
    
    for (int i = 0; i < 8; i++) {
        if (T > Tfar) break;
        
        // Sample position
        vec3 p = ro + rd * T;
        vec3 uvw = (p - boxMin) / (boxMax - boxMin);
        
        // Skip if outside volume
        if (any(lessThan(uvw, vec3(0.0))) || any(greaterThan(uvw, vec3(1.0)))) {
            T += stepSize;
            continue;
        }
        
        // Check density and radiation
        float den = texture(volumeTex, uvw).r;
        float radVal = texture(radiationTex, uvw).r;
        
        // If it's solid and not carved, it casts shadow
        if (den > edgeThreshold && radVal < 0.5) {
            shadow -= 0.2;
        }
        
        T += stepSize;
    }
    
    return max(shadow, 0.2);
}

// Modified shading function with enhanced building separation
vec3 calculateShading(vec3 pos, vec3 normal, vec3 rayDir, float edgeFactor) {
    // Get building color with boundary enhancement
    vec3 baseColor = getBuildingColor(pos, normal);
    
    // Check building boundaries
    float boundaryFactor = detectBuildingBoundaries(pos, normal);
    
    // Enhance edges at building boundaries
    if (boundaryFactor < 0.4) {
        // We're near a building boundary - enhance the edge
        edgeFactor = max(edgeFactor, 1.0 - boundaryFactor * 2.0);
    }
    
    // Check if this is a window
    bool isWindow = isWindowPosition(pos, normal);
    if (isWindow) {
        // Windows are darker with slight blue tint
        return vec3(0.2, 0.25, 0.3);
    }
    
    // Get normalized volume coordinates
    vec3 uvw = (pos - boxMin) / (boxMax - boxMin);
    
    // Use precomputed ambient occlusion
    float precomputedAO = texture(ambientOcclusionTex, uvw).r;
    
    // Enhance AO near building boundaries
    float ao = 1.0 - precomputedAO * 0.7;
    if (boundaryFactor < 0.5) {
        // Darker AO near building boundaries
        ao *= mix(0.7, 1.0, boundaryFactor);
    }
    
    // Main directional light contribution
    float NdotL = max(0.0, dot(normal, mainLightDir));
    float shadowFactor = calculateShadow(pos, mainLightDir);
    vec3 directLight = mainLightColor * NdotL * shadowFactor;
    
    // Ambient light from sky (hemispherical)
    float skyFactor = 0.5 + 0.5 * normal.y;
    vec3 skyLight = skyLightColor * skyFactor;
    
    // Ground bounce light
    float groundFactor = 0.5 - 0.5 * normal.y;
    vec3 groundLight = groundLightColor * groundFactor;
    
    // Indirect bounce lighting
    vec3 indirectLight = texture(indirectLightTex, uvw).rgb;
    
    // Enhanced rim lighting for building edges
    float rimFactor = pow(1.0 - max(0.0, dot(normal, -rayDir)), 3.0);
    vec3 rimLight = vec3(1.0) * rimFactor * 0.3;
    
    // Stronger rim light at building boundaries
    if (boundaryFactor < 0.6) {
        rimLight *= 2.0 * (1.0 - boundaryFactor);
    }
    
    // Edge darkening for better definition
    float edgeDarkening = 1.0;
    if (edgeFactor > 0.7) {
        // Strong edge (higher value = stronger edge)
        edgeDarkening = 0.7;
    } else if (edgeFactor > 0.3) {
        // Gradient from edge to normal
        edgeDarkening = mix(0.85, 1.0, (0.7 - edgeFactor) / 0.4);
    }
    
    // Additional darkening at building boundaries
    if (boundaryFactor < 0.4) {
        edgeDarkening *= mix(0.6, 1.0, boundaryFactor);
    }
    
    // Combine all lighting with color and effects
    vec3 finalColor = baseColor * (directLight + skyLight + groundLight + indirectLight * 3.0) * ao * edgeDarkening + rimLight;
    
    return finalColor;
}

// Smooth the density samples for better anti-aliasing
float smoothDensitySample(vec3 uvw) {
    // Get base sample
    float den = texture(volumeTex, uvw).r;

	if (useFrustumCulling) {
        // Check if this point is in the culled working volume
        float visibilityCheck = texture(workingVolumeTex, uvw).r;
        
        // If this point was culled (visibility near zero), force density to zero
        if (visibilityCheck < 0.001) {
            return 0.0;
        }
    }

    // Only smooth near edges (optimization)
    if (abs(den - edgeThreshold) < edgeWidth) {
        vec3 texelSize = 1.0 / vec3(textureSize(volumeTex, 0));
        float sum = den;
        float weight = 1.0;
        
        // Sample nearby points along coordinate axes (6 samples is sufficient and efficient)
        for (int i = 0; i < 3; i++) {
            vec3 offset = vec3(0.0);
            offset[i] = texelSize[i] * 0.8;
            
           float s1 = texture(volumeTex, uvw + offset).r;
            float s2 = texture(volumeTex, uvw - offset).r;
            
            if (useFrustumCulling) {
                float v1 = texture(workingVolumeTex, uvw + offset).r;
                float v2 = texture(workingVolumeTex, uvw - offset).r;
                
                // Only include samples that are visible in working volume
                if (v1 > 0.001) {
                    sum += s1;
                    weight += 1.0;
                }
                if (v2 > 0.001) {
                    sum += s2;
                    weight += 1.0;
                }
            } else {
                sum += s1 + s2;
                weight += 2.0;
            }
        }
        
        return sum / weight;
    }
    
    return den;
}

// Blue noise texture coordinates for dithering (if you don't have a texture, we'll use procedural noise)
vec3 blueNoiseOffset(vec2 screenCoord, float time) {
    // Generate high-quality pseudo-random offset
    vec3 offset;
    // Hash function based on screen position and time
    vec2 n = screenCoord + vec2(time * 0.11, time * 0.17);
    float h1 = fract(sin(dot(n, vec2(127.1, 311.7))) * 43758.5453123);
    float h2 = fract(sin(dot(n, vec2(269.5, 183.3))) * 41749.6721234);
    float h3 = fract(sin(dot(n, vec2(419.2, 371.9))) * 81749.8975432);
    offset = vec3(h1, h2, h3) * 2.0 - 1.0;
    return offset * 0.001; // Scale to a very small value
}

// Optimized skip distance function with separated MIP mapping and octree skipping
float getOptimizedSkipDistance(vec3 rayOrigin, vec3 rayDir, vec3 currentPos) {
    // Only check skip optimizations if either is enabled
    if (!enableOctreeSkip && !useMipMappedSkipping) {
        return 0.0;
    }
    
    // Convert current position to normalized texture coordinates
    vec3 normalizedPos = (currentPos - boxMin) / (boxMax - boxMin);
    
    // Check if we're outside the volume
    if (any(lessThan(normalizedPos, vec3(0.0))) || any(greaterThan(normalizedPos, vec3(1.0)))) {
        return 0.0;
    }


    // Skip distance to return
    float skipDistance = 0.0;
    
    // MIP mapping based empty space skipping
	if (useMipMappedSkipping) {
		// Start with a more conservative MIP level
		for (int mipLevel = min(maxMipLevel, 4); mipLevel >= 0; mipLevel--) {
			// Calculate the scale factor for this MIP level
			float mipScale = pow(2.0, float(mipLevel));
        
			// Sample with trilinear filtering
			float density = textureLod(volumeTex, normalizedPos, float(mipLevel)).r;
        
			// Use a lower threshold for higher mip levels to reduce artifacts
			float mipThreshold = mipLevel > 2 ? 0.001 : 0.01;
        
			// If empty at this level, we can potentially skip
			if (density < mipThreshold) {
				// Calculate cell size at this MIP level in normalized coordinates
				vec3 texSize = vec3(textureSize(volumeTex, mipLevel));
				vec3 cellSize = vec3(1.0) / texSize;
            
				// Calculate cell boundaries in normalized coordinates
				vec3 cellMin = floor(normalizedPos * texSize) / texSize;
				vec3 cellMax = cellMin + cellSize;
            
				// Use larger inset for higher mip levels to avoid artifacts
				float insetFactor = 0.001 + float(mipLevel) * 0.001;
				cellMin += cellSize * insetFactor;
				cellMax -= cellSize * insetFactor;
            
				// Convert to world space
				vec3 worldCellMin = mix(boxMin, boxMax, cellMin);
				vec3 worldCellMax = mix(boxMin, boxMax, cellMax);
            
				// Find distance to exit this cell
				vec2 tBox = intersectBox(currentPos, rayDir, worldCellMin, worldCellMax);
				float exitDist = max(0.0, tBox.y - 0.01 * (float(mipLevel) + 1.0));
            
				// Only consider valid distances
				if (exitDist > 0.0) {
					// Apply more conservative factor for higher mip levels
					skipDistance = max(skipDistance, exitDist * (1.0 - float(mipLevel) * 0.05));
					break;
				}
			}
		}
	}
    // Pre-computed octree skipping (only if MIP mapping didn't find a good distance)
    if (enableOctreeSkip && skipDistance == 0.0) {
        // Use pre-computed skip texture
        float precomputedSkip = texture(octreeSkipTex, normalizedPos).r;
        skipDistance = precomputedSkip * length(boxMax - boxMin);
    }
    
    // Apply a safety factor to avoid skipping too far
    // Use a more conservative factor for MIP mapping to avoid artifacts
    float safetyFactor = useMipMappedSkipping ? 0.9 : 0.95;
	skipDistance *= safetyFactor;
	if (!enableOctreeSkip && useMipMappedSkipping){
		return skipDistance;
	}
    
    // Calculate distance from camera to current position
    float distanceFromCamera = length(currentPos - camPos);
    
    // Normalize to a 0-1 range for typical scene distances
    float normalizedDistance = clamp(distanceFromCamera / 50.0, 0.0, 1.0);
    
    // Apply distance-based scaling to skip distance before returning
    float distanceScaleFactor = mix(0.001, 12.0,  pow(normalizedDistance, 3.5));
    return skipDistance * distanceScaleFactor;
}

// Modified traceRay function with optimized octree skipping and frustum culling integration
vec4 traceRay(vec2 coord) {
    // Get current frame number for temporal effects
    int frameNumber = int(mod(timeValue * 60.0, 16.0));
    
    // Set up the ray from camera
    vec2 ndc = vec2(2.0 * coord.x - 1.0, 1.0 - 2.0 * coord.y);

    vec4 clipPos = vec4(ndc, 1.0, 1.0);
    vec4 viewPos = invProj * clipPos;
    viewPos /= viewPos.w;
    vec4 worldPos4 = invView * viewPos;
    vec3 rayDir = normalize(worldPos4.xyz - camPos);

    // Only apply jitter when the camera is actually moving
    bool cameraIsMoving = length(camPos - previousCamPos) > 0.001;
    if (!cameraIsMoving) {
        // Also check if view direction changed
        if (dot(normalize(previousLookDir), rayDir) < 0.9999) {
            cameraIsMoving = true;
        }
    }

    // Enhanced temporal jittering - use higher quality pattern
    vec2 jitter;
    if (cameraIsMoving) {
        // Use halton sequence for better distribution
        jitter = haltonJitter[frameNumber % 8] * 1.5 / vec2(textureSize(volumeTex, 0).xy);
    } else {
        jitter = vec2(0.0);
    }

    // Apply jitter to coordinates
    vec2 jitteredCoord = coord + jitter;

    // Set up the ray with jittered coordinates
    ndc = vec2(2.0 * jitteredCoord.x - 1.0, 1.0 - 2.0 * jitteredCoord.y);
    clipPos = vec4(ndc, 1.0, 1.0);
    viewPos = invProj * clipPos;
    viewPos /= viewPos.w;
    worldPos4 = invView * viewPos;
    rayDir = normalize(worldPos4.xyz - camPos);
    
    // Intersect ray with volume bounds
    vec2 tHit = intersectBox(camPos, rayDir, boxMin, boxMax);
    float tNear = max(tHit.x, 0.0);
    float tFar = tHit.y;
    
    // Use the pre-computed octree skip distance if available
    if (octreeSkipT > 0.0) {
        tNear = max(tNear, octreeSkipT);
    }
    
    if (tNear > tFar) {
        // No intersection with volume
        return vec4(0.0, 0.0, 0.0, 1.0); // Black background
    }
    
    // Calculate view distance for adaptive quality
    float viewDistance = length(worldPos4.xyz - camPos);
    float distanceFactor = clamp(viewDistance / 500.0, 0.0, 1.0);
    
    // Base step size - adaptive based on view distance
    // Use smaller step size for better quality, especially at boundaries
    float baseStep = mix(
        min(length(boxMax - boxMin)/2048.0, (tFar - tNear)/2048.0),  // Higher quality when close
        min(length(boxMax - boxMin)/1024.0, (tFar - tNear)/1024.0),  // Higher quality when far
        distanceFactor
    );
    
    // Enhanced blue noise dithering to break up edge artifacts
    vec3 noiseOffset = blueNoiseOffset(gl_FragCoord.xy, timeValue);
    
    // Add temporal jitter and per-pixel variance to completely break up horizontal patterns
    float pixelNoise = hash(vec3(gl_FragCoord.xy, timeValue * 1111.0));
    float T = tNear + baseStep * pixelNoise * 0.5; // Reduced initial jitter
    
    // Create varying step sizes based on ray angle to avoid horizontal banding
    float rayAngleVariance = abs(dot(rayDir, vec3(0.0, 1.0, 0.0)));
    float angleBasedJitter = rayAngleVariance * 0.1 * baseStep; // Reduced jitter
    T += angleBasedJitter;
    
    // Initialize ray marching variables
    float accumAlpha = 0.0;
    vec3 accumColor = vec3(0.0);
    int maxSteps = 800 - int(distanceFactor * 350.0); // More steps for quality
    
    // Edge detection state
    bool wasInside = false;
    vec3 edgeNormal = vec3(0.0);
    vec3 edgePos = vec3(0.0);
    float edgeDepth = 0.0;
    
    // Adaptive sampling variables
    float emptySpaceCounter = 0.0;
    float detailRegionCounter = 0.0;
    bool wasEmpty = false;
    
    // These variables will help create varying step sizes
    float currentStepSize = baseStep;
    float prevDensity = 0.0;
    
    // Enhanced boundary detection
    bool nearBoundary = false;
    float boundaryCrossingCounter = 0.0;
    
    // Frustum culling optimization counter
    int emptyFrustumSkips = 0;
    
    // March along the ray
    for (int i = 0; i < maxSteps; i++) {
        if (T > tFar) break;
        if (accumAlpha > alphaCutoff) break; // Early termination when mostly opaque
        
        // Current sample position
        vec3 posWorld = camPos + rayDir * T;
        
        // Check if we can skip a section using our optimized function
        // Only check for skipping periodically to avoid the overhead of the skip calculation
        if (i % 5 == 0 && (enableOctreeSkip || useMipMappedSkipping)) {
            float skipDistance = getOptimizedSkipDistance(camPos, rayDir, posWorld);
            if (skipDistance > 0.0) {
                // Skip ahead by the calculated distance
                T += skipDistance;
                emptyFrustumSkips++;
                continue;
            }
        }
        
        // Apply subtle noise offset to position (breaks up patterns)
        posWorld += noiseOffset * mix(0.5, 2.0, distanceFactor); // Reduced offset magnitude
        
        vec3 uvw = (posWorld - boxMin) / (boxMax - boxMin);
        // Enhanced boundary check for mipmapped mode
		if (useFrustumCulling && useMipMappedSkipping) {
			// Get visibility from working volume with no mipmapping to avoid bleeding
			float visibilityFactor = textureLod(workingVolumeTex, uvw, 0.0).r;
    
			// Create a hard boundary at the frustum edge
			if (visibilityFactor < 0.001) {
				// Take a larger step past this invisible region
				T += baseStep * 4.0;
				emptyFrustumSkips++;
				continue;
			}
    
			// Add a small boundary region with reduced opacity
			if (visibilityFactor < 0.1) {
				// We're near the boundary, create a gradient falloff
				float falloff = smoothstep(0.0, 0.1, visibilityFactor);
        
				// Apply smooth boundary transition
				if (falloff < 0.2) {
					T += baseStep * (1.0 + (1.0 - falloff) * 3.0);
					emptyFrustumSkips++;
					continue;
				}
			}
		}

        // Enhanced boundary handling - use clamped coordinates for edge samples
        bool isOutside = any(lessThan(uvw, vec3(0.0))) || any(greaterThan(uvw, vec3(1.0)));
        vec3 uvwClamped = clamp(uvw, vec3(0.001), vec3(0.999));
        
        // If outside but very close to boundary, still sample with clamped coordinates
        if (isOutside) {
            // Check how far outside we are
            vec3 outsideDist = max(vec3(0.0) - uvw, uvw - vec3(1.0));
            float maxOutsideDist = max(max(outsideDist.x, outsideDist.y), outsideDist.z);
            
            if (maxOutsideDist < 0.05) {
                // Close to boundary, use clamped coordinates
                uvw = uvwClamped;
                nearBoundary = true;
                boundaryCrossingCounter += 1.0;
            } else {
                // Far outside, take larger steps
                T += baseStep * (3.0 + maxOutsideDist * 10.0);
                continue;
            }
        } else {
            nearBoundary = false;
            boundaryCrossingCounter = max(0.0, boundaryCrossingCounter - 0.5);
        }
        
        // Check if this region was culled by frustum culling
        // Use the working volume texture which has frustum culling applied
        if (useFrustumCulling) {
            float visibilityFactor = texture(workingVolumeTex, uvw).r;
            // If this region was culled (value is 0 in the working texture), skip it
            if (visibilityFactor < 0.001) {
                // Take a larger step in culled regions
                T += baseStep * mix(1.0, 4.0, 1.0 - visibilityFactor * 10.0);;
                emptyFrustumSkips++;
                continue;
            }
        }
        
        // Check if carved by radiation
        float radVal = texture(radiationTex, uvw).r;
        
		// Sample density with enhanced filtering for boundary regions or mipmapped regions
		float den;
		if (nearBoundary || boundaryCrossingCounter > 0.0 || distanceFactor < 0.5 || useMipMappedSkipping) {
			// High quality sampling at boundaries, when close, or when using mipmapping
			vec3 uvwJittered = uvw + (noiseOffset * 0.001);
    
			// For mipmapping mode, sample multiple LODs and blend between them
			if (useMipMappedSkipping) {
				// Calculate adaptive LOD based on distance and whether we're in a detail region
				float baseLOD = clamp(distanceFactor * 2.0, 0.0, 3.0);
    
				// Use multiple jittered samples at the same LOD level
				const int sampleCount = 4;
				const vec3 offsets[4] = vec3[4](
					vec3(0.0, 0.0, 0.0),
					vec3(0.001, 0.001, 0.001),
					vec3(-0.001, 0.001, -0.001),
					vec3(0.001, -0.001, 0.001)
				);
    
				// Sample with temporal jitter to reduce flickering
				float jitterOffset = fract(timeValue * 0.1 + pixelNoise * 0.5) * 0.5;
				float lod = baseLOD + jitterOffset;
    
				den = 0.0;
				for (int s = 0; s < sampleCount; s++) {
					// Apply position jitter scaled by LOD to break up patterns
					vec3 sampleOffset = offsets[s] * (lod + 0.5);
					den += textureLod(volumeTex, uvw + sampleOffset, lod).r;
				}
				den /= float(sampleCount);
			} else {
				// Standard high-quality sampling
				const int sampleCount = 2;
				const vec3 offsets[2] = vec3[2](
					vec3(0.001, 0.001, 0.001),
					vec3(-0.001, 0.001, -0.001)
				);
        
				den = texture(volumeTex, uvwJittered).r;
				for (int s = 0; s < sampleCount; s++) {
					den += texture(volumeTex, uvw + offsets[s] * (1.0 + pixelNoise * 0.5)).r;
				}
				den /= float(sampleCount + 1);
			}
		} else {
			// Standard sampling for interior regions
			float lodLevel = mix(0.0, 2.0, distanceFactor);
			den = textureLod(volumeTex, uvw, lodLevel).r;
		}

        // Apply subtle dithering to density value to break up banding
        den += (pixelNoise - 0.5) * 0.01; // Reduced dithering magnitude
        
        // Adaptive step for radiation
        if (radVal > 0.05) {
            emptySpaceCounter += 1.0;
            
            // Use varying step size based on position and ray direction
            float variableStep = baseStep * mix(1.0, 4.0, min(1.0, emptySpaceCounter / 15.0));
            
            // Add ray-angle dependent jitter to break horizontal patterns
            variableStep *= 1.0 + 0.1 * sin(dot(rayDir, vec3(1.0, 3.0, 2.0)) * 10.0 + timeValue);
            
            T += variableStep * mix(1.0, 1.5, radVal);
            wasEmpty = true;
            detailRegionCounter = 0.0;
            continue;
        }
        
        // If the voxel is empty, take a larger step
        if (den < 0.01) {
            emptySpaceCounter += 1.0;
            
            // Use variable step size that changes with ray direction
            float variableStep = baseStep * mix(1.5, 6.0, min(1.0, emptySpaceCounter / 25.0));

			// Smoother steps for mipmapped mode to reduce artifacts
			if (useMipMappedSkipping) {
				// Smoother steps for mipmapped mode to reduce artifacts
				variableStep *= 0.7 + 0.3 * smoothstep(0.0, 30.0, emptySpaceCounter);
    
				// Further reduce step size if we're using higher LODs
				float distanceStepFactor = distanceFactor < 0.3 ? 0.6 : 0.9;
				variableStep *= distanceStepFactor;
    
				// Add temporal stability to prevent jittering
				float timeJitter = sin(timeValue * 3.0 + hash(vec3(gl_FragCoord.xy, 0.0)) * 6.28) * 0.5 + 0.5;
				timeJitter = mix(0.9, 1.0, timeJitter); // Subtle temporal variation
				variableStep *= timeJitter;
			}

            // Add directional variance to avoid uniform stepping
            variableStep *= 1.0 + 0.1 * sin(rayDir.y * 20.0 + timeValue);
            
            // Smaller steps near boundaries
            if (nearBoundary || boundaryCrossingCounter > 0.0) {
                variableStep *= 0.3;
            }
            
            T += variableStep;
            wasEmpty = true;
            detailRegionCounter = 0.0;
            continue;
        }
        
        // If we just transitioned from empty to non-empty, reset counters
        if (wasEmpty) {
            emptySpaceCounter = 0.0;
            wasEmpty = false;
        }
        
        // Edge detection with enhanced refinement
        bool isInside = den > edgeThreshold;
        bool crossingEdge = (isInside && !wasInside) || (!isInside && wasInside) || 
                           (abs(den - prevDensity) > 0.15);
        prevDensity = den;
        
        if (crossingEdge) {
            // Refine edge position with binary search for better precision
            float refinedT = T;
            vec3 refinedPos = posWorld;
            
            // Binary search for exact edge (prevents stair-stepping)
            if (i > 0 && !nearBoundary) {
                float prevT = T - currentStepSize;
                float curT = T;
                
                // 6 iterations of binary search for more precise edges
                for (int r = 0; r < 6; r++) {
                    float midT = 0.5 * (prevT + curT);
                    vec3 midPos = camPos + rayDir * midT;
                    vec3 midUVW = (midPos - boxMin) / (boxMax - boxMin);
                    
                    // Skip if outside volume
                    if (any(lessThan(midUVW, vec3(0.0))) || any(greaterThan(midUVW, vec3(1.0)))) {
                        break;
                    }
                    
                    float midDen = texture(volumeTex, midUVW).r;
                    
                    if ((midDen > edgeThreshold) == isInside) {
                        curT = midT;
                    } else {
                        prevT = midT;
                    }
                }
                
                refinedT = curT;
                refinedPos = camPos + rayDir * refinedT;
            }
            
            // Store refined edge information
            edgeNormal = texture(gradientDirTex, (refinedPos - boxMin) / (boxMax - boxMin)).xyz;
            edgePos = refinedPos;
            edgeDepth = refinedT;
            
            // Set boundary crossing counter for enhanced edge sampling
            boundaryCrossingCounter = 5.0;
        }
        
        wasInside = isInside;
        
        // Get gradient information for shading
        vec3 uvwFiltered = uvw;
        float gradMag = texture(gradientMagTex, uvwFiltered).r;
        vec3 normal = texture(gradientDirTex, uvwFiltered).xyz;
        float edgeDist = texture(edgeFactorTex, uvwFiltered).r;
        
        // Process material near edges or inside buildings
        if (edgeDist > 0.1 || isInside) {
            detailRegionCounter += 1.0;
            
            // Calculate opacity - vary slightly based on position to break patterns
            float alpha = min(0.9999, 0.95 + pixelNoise * 0.02);
            if (edgeDist > 0.5) {
                alpha = 0.9999;
            }
            
            // Enhanced opacity at volume boundaries
            if (nearBoundary) {
                alpha = min(0.95, alpha * 0.9);
            }
    
            // Calculate building color and shading
            vec3 litColor = calculateShading(posWorld, normal, rayDir, edgeDist);
    
            // Front-to-back compositing
            float oldAlpha = accumAlpha;
            accumAlpha = oldAlpha + (1.0 - oldAlpha) * alpha;
            accumColor += (1.0 - oldAlpha) * alpha * litColor;
    
            // Variable step size based on multiple factors
            float baseStepScale;
            if (edgeDist > 0.5 || nearBoundary) {
                // Very small steps at sharp edges and volume boundaries
                baseStepScale = mix(0.02, 0.1, distanceFactor);
            } else if (edgeDist > 0.2 || gradMag > 0.8) {
                // Small steps near edges or high gradient magnitude
                baseStepScale = mix(0.05, 0.2, distanceFactor);
            } else if (gradMag > 0.6) {
                // Medium steps in transitional regions
                baseStepScale = mix(0.1, 0.3, distanceFactor);
            } else {
                // Larger steps in uniform regions
                float detailFactor = min(1.0, detailRegionCounter / 20.0);
                baseStepScale = mix(0.2, 0.1, detailFactor);
                baseStepScale = mix(baseStepScale, 0.5, distanceFactor);
            }
            
            // Add variance based on ray direction and pixel position to break banding
            float dirJitter = 0.1 * sin(rayDir.y * 15.0 + timeValue + pixelNoise * 6.28);
            float stepScale = baseStepScale * (1.0 + dirJitter);
            
            // Save current step size for edge refinement
            currentStepSize = baseStep * stepScale;
            T += currentStepSize;
        } else {
            // Empty space with variable step size
            float stepScale = 1.5 * (1.0 + 0.1 * sin(rayDir.y * 10.0 + gl_FragCoord.x * 0.01));
            
            // Smaller steps near boundaries
            if (nearBoundary || boundaryCrossingCounter > 0.0) {
                stepScale *= 0.3;
            }
            
            currentStepSize = baseStep * stepScale;
            T += currentStepSize;
        }
    }
    
    // Keep black background if nothing substantial was hit
    if (accumAlpha < 0.1) {
        return vec4(0.0, 0.0, 0.0, 1.0);
    }
    
    // Enhanced post-processing for edges
    if (boundaryCrossingCounter > 0.0) {
        // Smooth edges by slightly reducing opacity
        accumAlpha = min(accumAlpha, 0.98);
    }
    
    // Final color adjustments
    vec3 finalColor = pow(accumColor, vec3(1.0/2.2)); // Gamma correction
    
    // Add faint noise dithering to break up any remaining banding
    finalColor += (vec3(hash(vec3(gl_FragCoord.xy, timeValue * 591.3))) - 0.5) * 0.01;
    
    // Enhance contrast
    finalColor = finalColor / (finalColor + vec3(0.15));
    
    // Add subtle atmospheric depth for distant objects
    float fogFactor = 1.0 - exp(-viewDistance * 0.0001);
    vec3 fogColor = vec3(0.15, 0.17, 0.2); // Dark fog for black background
    finalColor = mix(finalColor, fogColor, fogFactor * 0.15);

	// Before returning the final color in traceRay function, add:
	if (useMipMappedSkipping) {
		// Add subtle temporal stabilization for mipmapped mode
		// This slightly blends with previous results to reduce flickering
		float stabilityFactor = 0.85;
    
		// Use noise based on screen position and time for temporal jitter
		vec3 timeNoise = blueNoiseOffset(gl_FragCoord.xy, timeValue) * 0.5 + 0.5;
    
		// Apply subtle variance to color to break up patterns
		finalColor = mix(
			finalColor,
			finalColor * (0.97 + timeNoise.x * 0.06), // Subtle color variation
			0.4  // Blend amount
		);
	}

    // Output fully opaque result
    return vec4(finalColor, 1.0);
}

void main() {
    // Call ray tracer with texture coordinates
    FragColor = traceRay(vTexCoord);
}
