#version 330 core

in vec3 fragPos;
in vec3 fragColor;
in vec3 n;

//uniform vec3 lightPosition;

out vec4 color;

void main() {
	color = vec4(fragColor, 1.0);
}
