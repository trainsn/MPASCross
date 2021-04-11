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
0.001461996,
0.158780545,
0.396786519,
0.623447508,
0.830892564,
0.961593201,
0.98163156,
0.98836208
};
const float RGB_g[8] = {
0.000465991,
0.044145885,
0.082921034,
0.164863286,
0.282655486,
0.48967993,
0.75583726,
0.998361647
};
const float RGB_b[8] = {
0.013866006,
0.328737055,
0.433172687,
0.388066347,
0.258636436,
0.083564484,
0.152913003,
0.644924098
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
	FragColor = vec4(vec3(r, g ,b), 1.0f);
}
