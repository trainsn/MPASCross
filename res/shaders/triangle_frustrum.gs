#version 430 core
layout (points) in;
layout (triangle_strip, max_vertices = 24) out;

in V2G{
	flat int vertex_id;
	flat int layer_id;
}v2g[];

out G2F{
	flat int triangle_id;
	flat int layer_id;
	flat int hitFaceid;
	smooth vec3 o_pos;
}g2f;

struct MPASPrism {
	uint m_prismId;
	int m_iLayer;
	vec3 vtxCoordTop[3]; // 3 top vertex coordinates
	vec3 vtxCoordBottom[3]; // 3 botton vertex coordinates 
	int m_idxEdge[3];
	int idxVtx[3]; // triangle vertex index is equvalent to hexagon cell index 
};

//3D transformation matrices
uniform mat4 uMVMatrix;
//connectivity
uniform samplerBuffer latCell;
uniform samplerBuffer lonCell;
uniform samplerBuffer depths;
uniform isamplerBuffer cellsOnVertex;
uniform isamplerBuffer edgesOnVertex;
uniform isamplerBuffer cellsOnEdge;
uniform isamplerBuffer verticesOnEdge;

#define TRIANGLE_TO_EDGES_VAR edgesOnVertex
#define EDGE_CORNERS_VAR cellsOnEdge
#define CORNERS_LAT_VAR latCell
#define CORNERS_LON_VAR lonCell
#define EDGE_TO_TRIANGLES_VAR verticesOnEdge

#define EPS 1e-4
const float M_PI = 3.14159265358;
uniform int TOTAL_LAYERS;
uniform float threshold;

int	d_mpas_faceCorners[24] = {
    0, 1, 2,  3, 4, 5,//top 0 and bottom 1
    4, 2, 1,  4, 5, 2,//front 2,3
    5, 0, 2,  5, 3, 0,//right 4,5
    0, 3, 1,  3, 4, 1//left 6,7
};

bool ReloadVtxInfo(in int triangle_id, in int iLayer, inout MPASPrism prism) {
	prism.m_prismId = triangle_id;
	prism.m_iLayer = iLayer;

	prism.idxVtx[0] = -1;
	prism.idxVtx[1] = -1;
	prism.idxVtx[2] = -1;

	// load first edge 
	// index of first edge of triangle
	ivec3 idxEdges = texelFetch(TRIANGLE_TO_EDGES_VAR, triangle_id - 1).xyz;
	int idxEdge = idxEdges.x; // TRIANGLE_TO_EDGES_VAR[m_prismId * 3 + 0];
	prism.m_idxEdge[0] = idxEdge;
	// index of start corner of this edge
	ivec2 cornerIdxs = texelFetch(EDGE_CORNERS_VAR, idxEdge - 1).xy;
	int iS = cornerIdxs.x; //EDGE_CORNERS_VAR[idxEdge * 2];	
	// index of end corner of this edge 
	int iE = cornerIdxs.y; //EDGE_CORNERS_VAR[idxEdge * 2 + 1];
	prism.idxVtx[0] = iS;
	prism.idxVtx[1] = iE;
	int edge1E = iE;

	// load second edge 
	idxEdge = idxEdges.y; // TRIANGLE_TO_EDGES_VAR[m_prismId * 3 + 1];
	prism.m_idxEdge[1] = idxEdge;
	// index of start corner of second edge
	cornerIdxs = texelFetch(EDGE_CORNERS_VAR, idxEdge - 1).xy;
	iS = cornerIdxs.x; //EDGE_CORNERS_VAR[idxEdge * 2];	
	// index of end corner of this edge 
	iE = cornerIdxs.y; //EDGE_CORNERS_VAR[idxEdge * 2 + 1];

	bool normalCase = (edge1E != iS) && (edge1E != iE); // the second edge connects corner 0 and corner 2
	if (iS != prism.idxVtx[0] && iS != prism.idxVtx[1]) {	// find the index of the third corner.
		prism.idxVtx[2] = iS;
	}
	else {
		prism.idxVtx[2] = iE;
	}

	// index of third edge.
	idxEdge = idxEdges.z; // TRIANGLE_TO_EDGES_VAR[m_prismId * 3 + 1];
	prism.m_idxEdge[2] = idxEdge;
	if (!normalCase) {
		// swap m_idxEdge[1] and m_idxEdge[2]
		prism.m_idxEdge[1] ^= prism.m_idxEdge[2];
		prism.m_idxEdge[2] ^= prism.m_idxEdge[1];
		prism.m_idxEdge[1] ^= prism.m_idxEdge[2];
	}

	// load vertex info based on edge's info
	float lon[3], lat[3]; // longtitude and latitude of three corners

	for (int i = 0; i < 3; i++) {	// for each corner index of the triangle (specified by prismId)
		int idxCorner = prism.idxVtx[i];	//TRIANGLE_TO_CORNERS_VAR[m_prismId*3+i];
		lat[i] = texelFetch(CORNERS_LAT_VAR, idxCorner - 1).r;	// CORNERS_LAT_VAR[idxCorner]
		lon[i] = texelFetch(CORNERS_LON_VAR, idxCorner - 1).r;	// CORNERS_LON_VAR[idxCorner]
		prism.vtxCoordTop[i] = vec3(lon[i], lat[i], texelFetch(depths, (idxCorner - 1) * TOTAL_LAYERS + iLayer).r);
		prism.vtxCoordBottom[i] = vec3(lon[i], lat[i], texelFetch(depths, (idxCorner - 1) * TOTAL_LAYERS + iLayer + 1).r);
	}

	return true; 
}

void main(void){
	int vertexId = v2g[0].vertex_id;
	//query three neighbor cell id 
	ivec3 cellId3 = texelFetch(cellsOnVertex, vertexId - 1).xyz;

	if (cellId3.x == 0 || cellId3.y == 0 || cellId3.z == 0){ // on boundary
	}
	else {
		int cellIds[3];
		cellIds[0] = cellId3.x;
		cellIds[1] = cellId3.y;
		cellIds[2] = cellId3.z;

		float lat[3];
		float lon[3];
		bool showTriangle = true; 
		for (int i = 0; i < 3; i++){
			int cell_id = cellIds[i];
			lat[i] = texelFetch(latCell, cell_id - 1).x;
			lon[i] = texelFetch(lonCell, cell_id - 1).x; 
			if (abs(lat[i]) > M_PI * 80.0 / 90.0 / 2.0){		// only show tropical areas 
				showTriangle = false;
				break;
			}
		}
		if (showTriangle){	// filter out triangle cross Prime meridian
			showTriangle = abs(lon[0] - lon[1]) < M_PI / 2 && abs(lon[0] - lon[2]) < M_PI / 2 && abs(lon[1] - lon[2]) < M_PI / 2;
		}

		if (showTriangle){
			int layerId = v2g[0].layer_id;
			MPASPrism curPrismHitted;
			curPrismHitted.idxVtx[0] = -1;
			curPrismHitted.idxVtx[1] = -1;
			curPrismHitted.idxVtx[2] = -1;
			ReloadVtxInfo(vertexId, layerId, curPrismHitted);

			vec3 vtxCoord[6];
			vtxCoord[0] = curPrismHitted.vtxCoordTop[0];
			vtxCoord[1] = curPrismHitted.vtxCoordTop[1];
			vtxCoord[2] = curPrismHitted.vtxCoordTop[2];
			vtxCoord[3] = curPrismHitted.vtxCoordBottom[0];
			vtxCoord[4] = curPrismHitted.vtxCoordBottom[1];
			vtxCoord[5] = curPrismHitted.vtxCoordBottom[2];

			showTriangle = false;
			vec4 xyzw = vec4(vtxCoord[0], 1.0);	
			vec4 v_pos = uMVMatrix * xyzw;
			double diff0 = v_pos.z - threshold;
			for (int i = 1; i < 6; i++){
				xyzw = vec4(vtxCoord[i], 1.0);	
				v_pos = uMVMatrix * xyzw;
				if (diff0 * (v_pos.z - threshold) < EPS){
					showTriangle = true;
					break;
				}
			}

			int nFaces = 8;
			const int faceId_to_edgeId[8] = { -1, -1, 2, 2, 1, 1, 0, 0 };
			for (int idxFace = 0; idxFace < nFaces; idxFace++) {	// 6 faces except the top bottom faces
				if (showTriangle){
					for (int i = 0; i < 3; i++){
						int idxCell = d_mpas_faceCorners[idxFace * 3 + i];	
						vec4 xyzw = vec4(vtxCoord[idxCell], 1.0);	

						gl_Position = uMVMatrix * xyzw;
						g2f.o_pos = xyzw.xyz;
						g2f.triangle_id = vertexId;
						g2f.layer_id = layerId;
						g2f.hitFaceid = idxFace;
						EmitVertex();
					}
					EndPrimitive();				
				}
			}
		}
	}
}