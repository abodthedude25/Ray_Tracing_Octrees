#version 430 core
layout (location = 0) in vec3 pos;
layout (location = 1) in vec3 color;
layout (location = 2) in vec3 normal;

uniform mat4 M;
uniform mat4 V;
uniform mat4 P;

out vec3 FragPos;
out vec3 Normal;
out vec3 Color;

void main() {
    FragPos = vec3(M * vec4(pos, 1.0));
    Normal = mat3(transpose(inverse(M))) * normal;
    Color = color;
    gl_Position = P * V * M * vec4(pos, 1.0);
}
