#version 430 core
out vec4 FragColor;

in vec2 TexCoords;

uniform sampler2D gScalar;
uniform float tMin;
uniform float tMax;

void main(){
	float scalar = texture(gScalar, TexCoords).r;
	scalar = (scalar - tMin) / (tMax - tMin);
	FragColor = vec4(vec3(scalar), 1.0);
}
