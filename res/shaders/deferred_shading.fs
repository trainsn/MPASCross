#version 430 core
out vec4 FragColor;

in vec2 TexCoords;

uniform sampler2D gScalar;
uniform float tMin;
uniform float tMax;

const float scalars[8] = {
0, 
0.142857143, 
0.285714286, 
0.428571429, 
0.571428571, 
0.714285714, 
0.857142857, 
1
};
const float RGB_r[8] = {
0, 
0.139557065, 
0.028425819, 
0.021455557, 
0.03062584, 
0.439530034, 
0.979548868, 
1
};
const float RGB_g[8] = {
0, 
0.021982792, 
0.242674525, 
0.449208208, 
0.625455867, 
0.767449458, 
0.815288711, 
1
};
const float RGB_b[8] = {
0, 
0.459840275, 
0.587281065, 
0.381696896, 
0.082912936, 
0.037212909, 
0.571652069, 
1
};

void main(){
	float scalar = texture(gScalar, TexCoords).r;
	scalar = (scalar - tMin) / (tMax - tMin);
	float r, g, b;
	for (int i = 0; i < 7; i++){
		if (scalar > scalars[i] && scalar < scalars[i + 1]){
			r = RGB_r[i] + (scalar - scalars[i]) / (scalars[i + 1] - scalars[i]) * (RGB_r[i + 1] -  RGB_r[i]);
			g = RGB_g[i] + (scalar - scalars[i]) / (scalars[i + 1] - scalars[i]) * (RGB_g[i + 1] -  RGB_g[i]);
			b = RGB_b[i] + (scalar - scalars[i]) / (scalars[i + 1] - scalars[i]) * (RGB_b[i + 1] -  RGB_b[i]);
			break;
		}
	}
	FragColor = vec4(vec3(r, g, b), 1.0);
}
