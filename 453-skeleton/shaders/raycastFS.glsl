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
uniform sampler3D ambientOcclusionTex;  // New ambient occlusion texture
uniform sampler3D indirectLightTex;     // New indirect lighting texture
uniform float timeValue; // For temporal jittering

// Lighting parameters
const vec3 mainLightDir = normalize(vec3(0.5, 0.9, 0.4)); // Primary light direction
const vec3 mainLightColor = vec3(1.0, 0.98, 0.9) * 4.0; // Slightly warm sunlight
const vec3 skyLightColor = vec3(0.6, 0.7, 0.9) * 0.4; // Blue-ish ambient from sky
const vec3 groundLightColor = vec3(0.3, 0.25, 0.2) * 0.2; // Brownish bounce light from ground

// Material parameters
const float edgeThreshold = 0.5;
const float edgeWidth = 0.03;
const float alphaCutoff = 0.97; // Early ray termination

// Window parameters
const float windowDensity = 0.4;    // How many windows per unit
const float windowWidth = 0.3;      // Width of windows relative to spacing
const float windowHeight = 0.7;     // Height of windows relative to spacing

// Random hash function for material variation
float hash(vec3 p) {
    p = fract(p * vec3(0.1031, 0.1030, 0.0973));
    p += dot(p, p.yxz + 33.33);
    return fract((p.x + p.y) * p.z);
}

// Compute box intersection for ray
vec2 intersectBox(vec3 ro, vec3 rd, vec3 bmin, vec3 bmax) {
    vec3 t1 = (bmin - ro) / rd;
    vec3 t2 = (bmax - ro) / rd;
    vec3 tmin = min(t1, t2);
    vec3 tmax = max(t1, t2);
    float tN = max(max(tmin.x, tmin.y), tmin.z);
    float tF = min(min(tmax.x, tmax.y), tmax.z);
    return vec2(tN, tF);
}

// Calculate ambient occlusion in corners
float calculateCornersAO(vec3 uvw, vec3 normal) {
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

// Calculate shadow by sampling along light direction
float calculateShadow(vec3 pos, vec3 lightDir) {
    // Start slightly offset to avoid self-shadowing artifacts
    vec3 ro = pos + lightDir * 0.02;
    vec3 rd = lightDir;
    
    // Intersect with volume bounds
    vec2 tHit = intersectBox(ro, rd, boxMin, boxMax);
    if (tHit.x > tHit.y) return 1.0; // No intersection, fully lit
    
    float T = max(tHit.x, 0.0);
    float Tfar = min(tHit.y, 3.0); // Limit shadow ray length for performance
    
    // Take fewer samples for performance
    float stepSize = (Tfar - T) / 8.0;
    float shadow = 1.0;
    
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
            shadow -= 0.2; // Softer shadows (0.12 instead of 0.125 = 1/8)
        }
        
        T += stepSize;
    }
    
    return max(shadow, 0.15); // Never completely dark
}

// Material and texture functions
bool isWindowPosition(vec3 pos, vec3 normal) {
    // Use different window pattern based on normal direction
    if (abs(normal.x) > 0.7) {
        // X-facing facades
        float windowGridY = mod(pos.y * windowDensity, 1.0);
        float windowGridZ = mod(pos.z * (windowDensity * 0.7), 1.0);
        return (windowGridY > (1.0 - windowWidth) * 0.5 && windowGridY < (1.0 + windowWidth) * 0.5) && 
               (windowGridZ > (1.0 - windowHeight) * 0.5 && windowGridZ < (1.0 + windowHeight) * 0.5);
    } 
    else if (abs(normal.z) > 0.7) {
        // Z-facing facades
        float windowGridX = mod(pos.x * windowDensity, 1.0);
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
    float windowGridA = mod(dot(pos, vec3(1.0, 0, 0)) * windowDensity, 1.0);
    float windowGridB = mod(dot(pos, vec3(0, 1.0, 0)) * (windowDensity * 0.7), 1.0);
    return (windowGridA > (1.0 - windowWidth) * 0.5 && windowGridA < (1.0 + windowWidth) * 0.5) && 
           (windowGridB > (1.0 - windowHeight) * 0.5 && windowGridB < (1.0 + windowHeight) * 0.5);
}

// Get building color based on position (to differentiate buildings)
vec3 getBuildingColor(vec3 pos) {
    // Create a hash based on building "block" position
    // Smaller divisor = larger buildings with same color
    vec3 buildingPos = floor(pos / 2.0);
    float buildingID = hash(buildingPos);
    
    // Building material variations
    vec3 building1 = vec3(0.85, 0.82, 0.78); // Light tan
    vec3 building2 = vec3(0.75, 0.72, 0.68); // Medium gray
    vec3 building3 = vec3(0.82, 0.78, 0.72); // Beige
    vec3 building4 = vec3(0.70, 0.68, 0.65); // Darker gray
    
    // Pick a color based on building ID
    vec3 baseColor;
    if (buildingID < 0.25) baseColor = building1;
    else if (buildingID < 0.5) baseColor = building2;
    else if (buildingID < 0.75) baseColor = building3;
    else baseColor = building4;
    
    // Add subtle height-based gradient
    float heightFactor = (pos.y - boxMin.y) / (boxMax.y - boxMin.y);
    baseColor = mix(baseColor, baseColor * 1.15, heightFactor * 0.3);
    
    // Add subtle noise to break flat surfaces
    float noise = fract(sin(dot(pos * 10.0, vec3(12.9898, 78.233, 45.164))) * 43758.5453);
    baseColor *= 0.9 + noise * 0.2;
    
    return baseColor;
}

// Calculate complete shading with AO, shadows, lighting
vec3 calculateShading(vec3 pos, vec3 normal, vec3 rayDir, float edgeDist) {
    // Get base building color
    vec3 baseColor = getBuildingColor(pos);
    
    // Check if this is a window
    bool isWindow = isWindowPosition(pos, normal);
    if (isWindow) {
        // Windows are darker with slight blue tint
        return vec3(0.2, 0.25, 0.3);
    }
    
    // Calculate corner ambient occlusion
    float customAO = calculateCornersAO(normalize(pos - boxMin) / normalize(boxMax - boxMin), normal);
    
    // Get precomputed ambient occlusion
    vec3 uvw = (pos - boxMin) / (boxMax - boxMin);
    float precomputedAO = texture(ambientOcclusionTex, uvw).r;
    
    // Combine both AO effects
    float ao = customAO * (1.0 - precomputedAO * 0.9);
    
    // Main directional light contribution
    float NdotL = max(0.0, dot(normal, mainLightDir));
    float shadowFactor = calculateShadow(pos, mainLightDir);
    vec3 directLight = mainLightColor * NdotL * shadowFactor;
    
    // Ambient light from sky (hemispherical)
    float skyFactor = 0.5 + 0.5 * normal.y; // More sky contribution on upward facing surfaces
    vec3 skyLight = skyLightColor * skyFactor;
    
    // Ground bounce light
    float groundFactor = 0.5 - 0.5 * normal.y; // More ground bounce on downward facing surfaces
    vec3 groundLight = groundLightColor * groundFactor;
    
    // Indirect bounce lighting
    vec3 indirectLight = texture(indirectLightTex, uvw).rgb;
    
    // Rim lighting for edges (view dependent)
    float rimFactor = pow(1.0 - max(0.0, dot(normal, -rayDir)), 4.0);
    vec3 rimLight = vec3(1.0) * rimFactor * 0.2;
    
    // Edge darkening for better definition
    float edgeDarkening = 1.0;
    if (edgeDist < 0.1 * edgeWidth) {
        // Strong edge
        edgeDarkening = 0.3;
    } else if (edgeDist < edgeWidth) {
        // Gradient from edge to normal
        edgeDarkening = mix(0.7, 1.0, (edgeDist - 0.1 * edgeWidth) / (0.9 * edgeWidth));
    }
    
    // Combine all lighting with color and effects
	vec3 finalColor = baseColor * (directLight + skyLight + groundLight + indirectLight * 3.0) * ao * edgeDarkening + rimLight;
    
    return finalColor;
}

// Smooth the density samples for better anti-aliasing
float smoothDensitySample(vec3 uvw) {
    // Get base sample
    float den = texture(volumeTex, uvw).r;
    
    // Only smooth near edges
    if (abs(den - edgeThreshold) < edgeWidth) {
        vec3 texelSize = 1.0 / vec3(textureSize(volumeTex, 0));
        float sum = den;
        float weight = 1.0;
        
        // Sample nearby points
        for (int i = 0; i < 3; i++) {
            vec3 offset = vec3(0.0);
            offset[i] = texelSize[i] * 0.8;
            
            float s1 = texture(volumeTex, uvw + offset).r;
            float s2 = texture(volumeTex, uvw - offset).r;
            
            sum += s1 + s2;
            weight += 2.0;
        }
        
        return sum / weight;
    }
    
    return den;
}

// Main ray tracing function
vec4 traceRay(vec2 coord) {
    // Set up the ray from camera
    vec2 ndc = vec2(2.0 * coord.x - 1.0, 1.0 - 2.0 * coord.y);
    vec4 clipPos = vec4(ndc, 1.0, 1.0);
    vec4 viewPos = invProj * clipPos;
    viewPos /= viewPos.w;
    vec4 worldPos4 = invView * viewPos;
    vec3 rayDir = normalize(worldPos4.xyz - camPos);
    
    // Intersect ray with volume bounds
    vec2 tHit = intersectBox(camPos, rayDir, boxMin, boxMax);
    float tNear = max(tHit.x, 0.0);
    float tFar = tHit.y;
    tNear = max(tNear, octreeSkipT);
    
    if (tNear > tFar) {
        // No intersection with volume
        return vec4(0.0, 0.0, 0.0, 1.0); // Black background
    }
    
    // Calculate view distance for adaptive quality
    float viewDistance = length(worldPos4.xyz - camPos);
    float distanceFactor = clamp(viewDistance / 500.0, 0.0, 1.0);
    
    // Adaptive step size - smaller when close, larger when far
    float baseStep = mix(
        min(length(boxMax - boxMin)/768.0, (tFar - tNear)/1024.0),  // Higher quality when close
        min(length(boxMax - boxMin)/384.0, (tFar - tNear)/512.0),   // Lower quality when far
        distanceFactor
    );
    
    // Add temporal jitter for better anti-aliasing
    float jitter = fract(sin(dot(floor(gl_FragCoord.xy / 4.0) * 4.0, vec2(12.9898, 78.233))) * 43758.5453);
    float T = tNear + baseStep * 0.25 * jitter;
    
    // Initialize ray marching variables
    float accumAlpha = 0.0;
    vec3 accumColor = vec3(0.0);
    int maxSteps = 512 - int(distanceFactor * 128.0); // Fewer steps when far away
    
    // Edge detection state
    bool wasInside = false;
    vec3 edgeNormal = vec3(0.0);
    vec3 edgePos = vec3(0.0);
    float edgeDepth = 0.0;
    
    // March along the ray
    for (int i = 0; i < maxSteps; i++) {
        if (T > tFar) break;
        if (accumAlpha > alphaCutoff) break; // Early termination when mostly opaque
        
        // Current sample position
        vec3 posWorld = camPos + rayDir * T;
        vec3 uvw = (posWorld - boxMin) / (boxMax - boxMin);
        
        // Skip if outside volume
        if (any(lessThan(uvw, vec3(0.0))) || any(greaterThan(uvw, vec3(1.0)))) {
            T += baseStep * 2.0;
            continue;
        }
        
        // Check if carved by radiation
        float radVal = texture(radiationTex, uvw).r;
        if (radVal > 0.05) {
            T += baseStep * mix(1.0, 2.0, radVal);
            continue;
        }
        
        // Sample density with smoothing when close
        float den;
        if (distanceFactor < 0.7) {
            den = smoothDensitySample(uvw);
        } else {
            den = texture(volumeTex, uvw).r;
        }
        
        // Edge detection
        bool isInside = den > edgeThreshold;
        bool crossingEdge = (isInside && !wasInside);
        
        if (crossingEdge) {
            // Store information about the edge we just crossed
            edgeNormal = texture(gradientDirTex, uvw).xyz;
            edgePos = posWorld;
            edgeDepth = T;
        }
        
        wasInside = isInside;
        
        // Get gradient information for normal
        float gradMag = texture(gradientMagTex, uvw).r;
        vec3 normal = texture(gradientDirTex, uvw).xyz;
        
        // Process material near edges or inside buildings
        float edgeDist = abs(den - edgeThreshold);
        if (edgeDist < edgeWidth || isInside) {
            // Higher opacity near edges and inside
            float alpha = min(0.9999, 0.95);
            if (edgeDist < edgeWidth * 0.5) {
                alpha = 0.9999; // Maximum opacity at exact edges
            }
            
            // Apply edge enhancement
            float edgeEnhancement = 1.0;
            if (abs(T - edgeDepth) < baseStep * 5.0 && crossingEdge) {
                edgeEnhancement = 0.8; // Darken newly crossed edges
            }
            
            // Calculate building color and shading
            vec3 litColor = calculateShading(posWorld, normal, rayDir, edgeDist) * edgeEnhancement;
            
            // Front-to-back compositing
            float oldAlpha = accumAlpha;
            accumAlpha = oldAlpha + (1.0 - oldAlpha) * alpha;
            accumColor += (1.0 - oldAlpha) * alpha * litColor;
            
            // Adaptive step size - smaller near edges
            float stepScale;
            if (edgeDist < edgeWidth * 0.5) {
                stepScale = mix(0.05, 0.2, distanceFactor); // Smaller steps at edges when close
            } else if (edgeDist < edgeWidth) {
                stepScale = mix(0.1, 0.3, distanceFactor);
            } else if (gradMag > 0.6) {
                stepScale = mix(0.2, 0.5, distanceFactor);
            } else {
                stepScale = mix(0.4, 0.8, distanceFactor);
            }
            T += baseStep * stepScale;
        } else {
            // Empty space - use larger steps
            T += baseStep * 2.0;
        }
    }
    
    // Keep black background if nothing substantial was hit
    if (accumAlpha < 0.1) {
        return vec4(0.0, 0.0, 0.0, 1.0);
    }
    
    // Final color adjustments
    vec3 finalColor = pow(accumColor, vec3(1.0/2.2)); // Gamma correction
    
    // Enhance contrast
    finalColor = finalColor / (finalColor + vec3(0.15));
    
    // Add subtle atmospheric depth for distant objects
    float fogFactor = 1.0 - exp(-viewDistance * 0.0001);
    vec3 fogColor = vec3(0.15, 0.17, 0.2); // Dark fog for black background
    finalColor = mix(finalColor, fogColor, fogFactor * 0.15);
    
    // Output fully opaque result
    return vec4(finalColor, 1.0);
}

vec4 debugView(vec2 coord) {
    vec3 uvw = vec3(coord, 0.5); // Show the middle slice
    
    // Choose what to visualize
    int visualizationMode = 2; // 0=normal, 1=AO, 2=indirect light
    
    if (visualizationMode == 1) {
        float ao = texture(ambientOcclusionTex, uvw).r;
        return vec4(vec3(1.0 - ao), 1.0); // Visualize AO (white=no occlusion, black=occluded)
    }
    else if (visualizationMode == 2) {
        vec3 indirect = texture(indirectLightTex, uvw).rgb * 5.0; // Amplify for visibility
        return vec4(indirect, 1.0); // Visualize indirect lighting
    }
    
    // Default: normal render
    return traceRay(coord);
}

void main() {
    // Call the debug function instead
    FragColor = traceRay(vTexCoord);
}
