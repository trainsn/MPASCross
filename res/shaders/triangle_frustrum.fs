#version 430 core 

in float CLIMATE_VALS_VAR;
layout (location = 0) out float gScalar;

uniform int maxEdges;
uniform float threshold;

in G2F{
	flat int triangle_id;
	flat int layer_id;
	flat int hitFaceid;
	smooth vec3 o_pos;
}g2f;

uniform mat4 uInvMVMatrix;
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
uniform isamplerBuffer verticesOnCell;
uniform isamplerBuffer nEdgesOnCell;
uniform isamplerBuffer maxLevelCell;
uniform samplerBuffer temperature;
uniform samplerBuffer salinity;
//symbolic definition of dual triangle mesh
#define TRIANGLE_TO_EDGES_VAR edgesOnVertex
#define EDGE_CORNERS_VAR cellsOnEdge
#define CORNERS_LAT_VAR latCell
#define CORNERS_LON_VAR lonCell
#define FACEID_TO_EDGEID faceId_to_edgeId
#define EDGE_TO_TRIANGLES_VAR verticesOnEdge
#define CORNER_TO_TRIANGLES_VAR verticesOnCell
#define CORNER_TO_TRIANGLES_DIMSIZES nEdgesOnCell

#define FLOAT_MAX  3.402823466e+38F
#define FLOAT_MIN -2.402823466e+36F
#define EPS 1e-4
const float M_PI = 3.14159265358;
uniform int TOTAL_LAYERS;

int	d_mpas_faceCorners[24] = {
    0, 1, 2,  3, 4, 5,//top 0 and bottom 1
    4, 2, 1,  4, 5, 2,//front 2,3
    5, 0, 2,  5, 3, 0,//right 4,5
    0, 3, 1,  3, 4, 1//left 6,7
};

struct Ray{
	dvec3 o;
	dvec3 d;
};

struct HitRec{
	double t;	//  t value along the hitted face 
	int hitFaceid;	// hitted face id 
	int nextlayerId;
};

struct MPASPrism {
	uint m_prismId;
	int m_iLayer;
	dvec3 vtxCoordTop[3]; // 3 top vertex coordinates
	dvec3 vtxCoordBottom[3]; // 3 botton vertex coordinates 
	int m_idxEdge[3];
	int idxVtx[3]; // triangle vertex index is equvalent to hexagon cell index 
};

#define DOUBLE_ERROR 1.0e-8
bool rayIntersectsTriangleDouble(dvec3 p, dvec3 d,
    dvec3 v0, dvec3 v1, dvec3 v2, inout double u, inout double v, inout double t)
{
    dvec3 e1, e2, h, s, q;
    double a, f;
    //float error = 1.0e-4;//0.005f;
    e1 = v1 - v0;
    e2 = v2 - v0;
    //crossProduct(h, d, e2);
    h = cross(d, e2);
    a = dot(e1, h);//innerProduct(e1, h);

    if (a > -DOUBLE_ERROR && a < DOUBLE_ERROR)
        return(false);

    f = 1.0 / a;
    s = p - v0;//_vector3d(s, p, v0);
    u = f * dot(s, h);//(innerProduct(s, h));

    if (u < -DOUBLE_ERROR || u >(1.0 + DOUBLE_ERROR))
        return(false);

    q = cross(s, e1);//crossProduct(q, s, e1);
    v = f * dot(d, q);//innerProduct(d, q);

    if (v < -DOUBLE_ERROR || u + v >(1.0 + DOUBLE_ERROR))
        return(false);

    // at this stage we can compute t to find out where
    // the intersection point is on the line
    t = f * dot(e2, q);//innerProduct(e2, q);

    if (t > DOUBLE_ERROR)//ray intersection
        return(true);
    else // this means that there is a line intersection
        // but not a ray intersection
        return (false);
}

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

// Return adjacent mesh cell (which is triangle in the remeshed MPAS mesh) id 
// which shares with current triangle (specified by curTriangleId) the edge belongs to the face (denoted by faceId)
int getAdjacentCellId(inout MPASPrism prism, int faceId) {
	if (prism.m_iLayer == TOTAL_LAYERS - 2 && faceId == 1) {
		return -1; // we reached the deepest layer, no more layers beyond current layer
	} 
	if (prism.m_iLayer == 0 && faceId == 0){
		return -1;	// we reached the most top layer
	}
	if (faceId == 0 || faceId == 1) {
		return int(prism.m_prismId);	// currentTriangleId
	}

	const int faceId_to_edgeId[8] = { -1, -1, 2, 2, 1, 1, 0, 0 };
	int edgeId = FACEID_TO_EDGEID[faceId];
	int idxEdge = prism.m_idxEdge[edgeId];
	ivec2 nextTriangleIds = texelFetch(EDGE_TO_TRIANGLES_VAR, idxEdge - 1).xy;	// EDGE_TO_TRIANGLES_VAR[idxEdge * 2];
	if (nextTriangleIds.x == prism.m_prismId){ // curTriangleId
		ivec3 cellId3 = texelFetch(cellsOnVertex, nextTriangleIds.y - 1).xyz;
		if (cellId3.x == 0 || cellId3.y == 0 || cellId3.z == 0){ // on boundary
			return -1;
		}
		return nextTriangleIds.y;
	}
	ivec3 cellId3 = texelFetch(cellsOnVertex, nextTriangleIds.x - 1).xyz;
	if (cellId3.x == 0 || cellId3.y == 0 || cellId3.z == 0){ // on boundary
		return -1;
	}
	return nextTriangleIds.x;
}

int rayPrismIntersection(inout MPASPrism prism, in Ray r, inout HitRec tInRec,
	inout HitRec tOutRec, inout int nextCellId) {
	nextCellId = -1;	// assume no next prism to shot into
	int nHit = 0;
	int nFaces = 8;
	tOutRec.hitFaceid = -1; // initialize to tOutRec
	tOutRec.t = -1.0f;
	double min_t = FLOAT_MAX, max_t = -1.0f;
	dvec3 vtxCoord[6];
	vtxCoord[0] = prism.vtxCoordTop[0];
	vtxCoord[1] = prism.vtxCoordTop[1];
	vtxCoord[2] = prism.vtxCoordTop[2];
	vtxCoord[3] = prism.vtxCoordBottom[0];
	vtxCoord[4] = prism.vtxCoordBottom[1];
	vtxCoord[5] = prism.vtxCoordBottom[2];

	for (int idxFace = 0; idxFace < nFaces; idxFace++) {	// 8 faces
		dvec3 v0 = vtxCoord[d_mpas_faceCorners[idxFace * 3]];
		dvec3 v1 = vtxCoord[d_mpas_faceCorners[idxFace * 3 + 1]];
		dvec3 v2 = vtxCoord[d_mpas_faceCorners[idxFace * 3 + 2]];

		double t = 0.0;
		dvec3 rayO = dvec3(r.o);
		dvec3 rayD = dvec3(r.d);
		dvec3 vtxTB0 = dvec3(v0);
        dvec3 vtxTB1 = dvec3(v1);
        dvec3 vtxTB2 = dvec3(v2);
		double u,v;
		bool bhit = rayIntersectsTriangleDouble(rayO, rayD,
                    vtxTB0, vtxTB1, vtxTB2, u, v, t);
		if (bhit) {
			nHit++;
			
			if (min_t > t) {
				min_t = t;
				tInRec.t = t;
				tInRec.hitFaceid = idxFace; 
			}
			if (max_t < t) {
				max_t = t;
				tOutRec.t = t;
				tOutRec.hitFaceid = idxFace;
				if (idxFace == 1) {
					tOutRec.nextlayerId = prism.m_iLayer + 1;	// the next prism to be traversed is in the lower layer 
				}
				else if (idxFace == 0) {
					tOutRec.nextlayerId = prism.m_iLayer - 1;	// the next prism to be traversed is in the upper layer
				} 
				else {
					tOutRec.nextlayerId = prism.m_iLayer;
				}
			}
		}
	}

	if (nHit == 2) {
		nextCellId = getAdjacentCellId(prism, tOutRec.hitFaceid);
	}
	else {	// specical case when ray hit on the edge 
		nextCellId = -1;
	}

	return nHit;
}

void GetScalarValue(inout MPASPrism prism, samplerBuffer CLIMATE_VALS_VAR, inout dvec3 scalars[2]) {
	for (int iFace = 0; iFace < 2; iFace++) {
		int layerId = prism.m_iLayer + iFace;
		scalars[iFace].x = texelFetch(CLIMATE_VALS_VAR, (prism.idxVtx[0]-1) * TOTAL_LAYERS + layerId).r; //CLIMATE_VALS_VAR[idxVtx[0] * TOTAL_LAYERS + (m_iLayer + iFace)];
		scalars[iFace].y = texelFetch(CLIMATE_VALS_VAR, (prism.idxVtx[1]-1) * TOTAL_LAYERS + layerId).r; //CLIMATE_VALS_VAR[idxVtx[0] * TOTAL_LAYERS + (m_iLayer + iFace)];
		scalars[iFace].z = texelFetch(CLIMATE_VALS_VAR, (prism.idxVtx[2]-1) * TOTAL_LAYERS + layerId).r; //CLIMATE_VALS_VAR[idxVtx[0] * TOTAL_LAYERS + (m_iLayer + iFace)];
	}
}

void GetMaxLevelCell(inout MPASPrism prism, inout ivec3 maxLevel){
	maxLevel.x = texelFetch(maxLevelCell, prism.idxVtx[0]-1).r;
	maxLevel.y = texelFetch(maxLevelCell, prism.idxVtx[1]-1).r;
	maxLevel.z = texelFetch(maxLevelCell, prism.idxVtx[2]-1).r;
}

void GetUV(in dvec3 O, in dvec3 Q, inout double A[12],
	inout double u, inout double v) {
	dvec3 QO = (Q - O);//*Factor;
	double denominator = (A[9] * QO.x - A[8] * QO.y + A[7] * QO.z);
	u = (A[3] * QO.x - A[2] * QO.y + A[1] * QO.z) / denominator;
	v = (A[6] * QO.x - A[5] * QO.y + A[4] * QO.z) / denominator;
}

double GetInterpolateValue2(in MPASPrism prism, in const double u, in const double v,
	in const dvec3 Q, in dvec3 fT, in dvec3 fB) {
	dvec3 baryCoord = vec3(1.0 - u - v, u, v);
	dvec3 m1 = baryCoord.x * prism.vtxCoordTop[0] + baryCoord.y * prism.vtxCoordTop[1] + baryCoord.z * prism.vtxCoordTop[2];
	dvec3 m2 = baryCoord.x * prism.vtxCoordBottom[0] + baryCoord.y * prism.vtxCoordBottom[1] + baryCoord.z * prism.vtxCoordBottom[2];

	double scalar_m1 = dot(baryCoord, fT);
	double scalar_m2 = dot(baryCoord, fB);
	double t3 = length(Q - m2) / length(m1 - m2);
	double lerpedVal = mix(scalar_m2, scalar_m1, t3);	//lerp()
	return lerpedVal;
}

void main(){
	int triangle_id = g2f.triangle_id;
	dvec3 v_pos =  dvec3((uMVMatrix * vec4(g2f.o_pos, 1.0)).xyz);
	dvec3 o_eye =  dvec3((uInvMVMatrix * vec4(v_pos.xy, -1.01, 1.0)).xyz);
	Ray ray;
	ray.o = o_eye;
	ray.d = normalize(g2f.o_pos - o_eye);
	dvec3 sample_pos = dvec3((uInvMVMatrix * vec4(v_pos.xy, threshold, 1.0)).xyz);
	double sample_t = length(sample_pos - o_eye);

	HitRec tInHitRecord, tOutHitRecord, tmpInRec, tmpOutRec;
	tInHitRecord.hitFaceid = -1;
	tInHitRecord.t = FLOAT_MAX;
	tInHitRecord.nextlayerId = -1;

	tOutHitRecord.hitFaceid = -1;
	tOutHitRecord.t = FLOAT_MIN;
	tOutHitRecord.nextlayerId = -1;
	
	tmpInRec.t = FLOAT_MAX;
	tmpOutRec.t = FLOAT_MIN;
	tmpInRec.nextlayerId = g2f.layer_id;
	tmpOutRec.nextlayerId = -1;

	int nHit = 0;
	int nextCellId = -1;
	int tmpNextCellId = -1;
	uint curPrismHittedId = triangle_id;
	
	MPASPrism curPrismHitted;
	curPrismHitted.idxVtx[0] = -1;
	curPrismHitted.idxVtx[1] = -1;
	curPrismHitted.idxVtx[2] = -1;
	ReloadVtxInfo(int(curPrismHittedId), tmpInRec.nextlayerId, curPrismHitted);
	
	int tmpNHit = rayPrismIntersection(curPrismHitted, ray, tmpInRec, tmpOutRec, tmpNextCellId);

	if (tmpNHit > 0) {
		nHit = tmpNHit;
		nextCellId = tmpNextCellId;
		curPrismHittedId = triangle_id;
		tInHitRecord = (tInHitRecord.t > tmpInRec.t) ? tmpInRec : tInHitRecord; 
		tOutHitRecord = (tOutHitRecord.t < tmpOutRec.t) ? tmpOutRec : tOutHitRecord;

		dvec3 position = dvec3(0.0);
		bool hasValue = false;

		ivec3 maxLevel;
		GetMaxLevelCell(curPrismHitted, maxLevel);
		if (curPrismHitted.m_iLayer < maxLevel.x - 1 && 
		curPrismHitted.m_iLayer < maxLevel.y - 1 && 
		curPrismHitted.m_iLayer < maxLevel.z - 1) 
		{
			dvec3 fTB[2];
			GetScalarValue(curPrismHitted, temperature, fTB);
			
			if (tInHitRecord.t < 0.0f)
				tInHitRecord.t = 0.0f;
			double t = tInHitRecord.t;	
			position = ray.o + ray.d * t;
			double u, v;
			dvec3 rayO = dvec3(position.xy, -1.0);
			dvec3 rayD = dvec3(0.0, 0.0, 1.0);
			double dummmy;
			rayIntersectsTriangleDouble(rayO, rayD, curPrismHitted.vtxCoordTop[0], curPrismHitted.vtxCoordTop[1], curPrismHitted.vtxCoordTop[2], u, v, dummmy);
			double scalar_st = GetInterpolateValue2(curPrismHitted, u, v, position, fTB[0], fTB[1]);

			t = tOutHitRecord.t; 
			position = ray.o + ray.d * t;
			rayO = dvec3(position.xy, -1.0);
			rayD = dvec3(0.0, 0.0, 1.0);
			rayIntersectsTriangleDouble(rayO, rayD, curPrismHitted.vtxCoordTop[0], curPrismHitted.vtxCoordTop[1], curPrismHitted.vtxCoordTop[2], u, v, dummmy);
			double scalar_en = GetInterpolateValue2(curPrismHitted, u, v, position, fTB[0], fTB[1]);

			if ((tInHitRecord.t - sample_t) * (tOutHitRecord.t - sample_t) < EPS) {	// sample point inside the line segment 
				double scalar = scalar_st + (scalar_en - scalar_st) * (sample_t - tInHitRecord.t) / (tOutHitRecord.t - tInHitRecord.t);
				gScalar = float(scalar);
				hasValue = true;
			}
		}
		if (!hasValue)
			discard;
	}	
	else {
		discard;
	}

}