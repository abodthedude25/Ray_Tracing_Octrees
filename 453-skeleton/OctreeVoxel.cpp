#include "OctreeVoxel.h"
#include <cmath>
#include <algorithm>
#include <array>
#include <unordered_map>
#include "glm/gtc/type_ptr.hpp"

// -------------------- MARCHING CUBES LOOKUP TABLES --------------------

static const int edgeTable[256] = {
	0x0,
	0x109,
	0x203,
	0x30a,
	0x406,
	0x50f,
	0x605,
	0x70c,
	0x80c,
	0x905,
	0xa0f,
	0xb06,
	0xc0a,
	0xd03,
	0xe09,
	0xf00,
	0x190,
	0x99,
	0x393,
	0x29a,
	0x596,
	0x49f,
	0x795,
	0x69c,
	0x99c,
	0x895,
	0xb9f,
	0xa96,
	0xd9a,
	0xc93,
	0xf99,
	0xe90,
	0x230,
	0x339,
	0x33,
	0x13a,
	0x636,
	0x73f,
	0x435,
	0x53c,
	0xa3c,
	0xb35,
	0x83f,
	0x936,
	0xe3a,
	0xf33,
	0xc39,
	0xd30,
	0x3a0,
	0x2a9,
	0x1a3,
	0xaa,
	0x7a6,
	0x6af,
	0x5a5,
	0x4ac,
	0xbac,
	0xaa5,
	0x9af,
	0x8a6,
	0xfaa,
	0xea3,
	0xda9,
	0xca0,
	0x460,
	0x569,
	0x663,
	0x76a,
	0x66,
	0x16f,
	0x265,
	0x36c,
	0xc6c,
	0xd65,
	0xe6f,
	0xf66,
	0x86a,
	0x963,
	0xa69,
	0xb60,
	0x5f0,
	0x4f9,
	0x7f3,
	0x6fa,
	0x1f6,
	0xff,
	0x3f5,
	0x2fc,
	0xdfc,
	0xcf5,
	0xfff,
	0xef6,
	0x9fa,
	0x8f3,
	0xbf9,
	0xaf0,
	0x650,
	0x759,
	0x453,
	0x55a,
	0x256,
	0x35f,
	0x55,
	0x15c,
	0xe5c,
	0xf55,
	0xc5f,
	0xd56,
	0xa5a,
	0xb53,
	0x859,
	0x950,
	0x7c0,
	0x6c9,
	0x5c3,
	0x4ca,
	0x3c6,
	0x2cf,
	0x1c5,
	0xcc,
	0xfcc,
	0xec5,
	0xdcf,
	0xcc6,
	0xbca,
	0xac3,
	0x9c9,
	0x8c0,
	0x8c0,
	0x9c9,
	0xac3,
	0xbca,
	0xcc6,
	0xdcf,
	0xec5,
	0xfcc,
	0xcc,
	0x1c5,
	0x2cf,
	0x3c6,
	0x4ca,
	0x5c3,
	0x6c9,
	0x7c0,
	0x950,
	0x859,
	0xb53,
	0xa5a,
	0xd56,
	0xc5f,
	0xf55,
	0xe5c,
	0x15c,
	0x55,
	0x35f,
	0x256,
	0x55a,
	0x453,
	0x759,
	0x650,
	0xaf0,
	0xbf9,
	0x8f3,
	0x9fa,
	0xef6,
	0xfff,
	0xcf5,
	0xdfc,
	0x2fc,
	0x3f5,
	0xff,
	0x1f6,
	0x6fa,
	0x7f3,
	0x4f9,
	0x5f0,
	0xb60,
	0xa69,
	0x963,
	0x86a,
	0xf66,
	0xe6f,
	0xd65,
	0xc6c,
	0x36c,
	0x265,
	0x16f,
	0x66,
	0x76a,
	0x663,
	0x569,
	0x460,
	0xca0,
	0xda9,
	0xea3,
	0xfaa,
	0x8a6,
	0x9af,
	0xaa5,
	0xbac,
	0x4ac,
	0x5a5,
	0x6af,
	0x7a6,
	0xaa,
	0x1a3,
	0x2a9,
	0x3a0,
	0xd30,
	0xc39,
	0xf33,
	0xe3a,
	0x936,
	0x83f,
	0xb35,
	0xa3c,
	0x53c,
	0x435,
	0x73f,
	0x636,
	0x13a,
	0x33,
	0x339,
	0x230,
	0xe90,
	0xf99,
	0xc93,
	0xd9a,
	0xa96,
	0xb9f,
	0x895,
	0x99c,
	0x69c,
	0x795,
	0x49f,
	0x596,
	0x29a,
	0x393,
	0x99,
	0x190,
	0xf00,
	0xe09,
	0xd03,
	0xc0a,
	0xb06,
	0xa0f,
	0x905,
	0x80c,
	0x70c,
	0x605,
	0x50f,
	0x406,
	0x30a,
	0x203,
	0x109,
	0x0
};


static const int triTable[256][16] = {
	{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 8, 3, 9, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 8, 3, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 9, 2, 10, 0, 2, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 2, 8, 3, 2, 10, 8, 10, 9, 8, -1, -1, -1, -1, -1, -1, -1 },
	{ 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 11, 2, 8, 11, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 9, 0, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 11, 2, 1, 9, 11, 9, 8, 11, -1, -1, -1, -1, -1, -1, -1 },
	{ 3, 10, 1, 11, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 10, 1, 0, 8, 10, 8, 11, 10, -1, -1, -1, -1, -1, -1, -1 },
	{ 3, 9, 0, 3, 11, 9, 11, 10, 9, -1, -1, -1, -1, -1, -1, -1 },
	{ 9, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 4, 3, 0, 7, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 1, 9, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 4, 1, 9, 4, 7, 1, 7, 3, 1, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 2, 10, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 3, 4, 7, 3, 0, 4, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1 },
	{ 9, 2, 10, 9, 0, 2, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1 },
	{ 2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, -1, -1, -1, -1 },
	{ 8, 4, 7, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 11, 4, 7, 11, 2, 4, 2, 0, 4, -1, -1, -1, -1, -1, -1, -1 },
	{ 9, 0, 1, 8, 4, 7, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1 },
	{ 4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1, -1, -1, -1, -1 },
	{ 3, 10, 1, 3, 11, 10, 7, 8, 4, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4, -1, -1, -1, -1 },
	{ 4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3, -1, -1, -1, -1 },
	{ 4, 7, 11, 4, 11, 9, 9, 11, 10, -1, -1, -1, -1, -1, -1, -1 },
	{ 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 9, 5, 4, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 5, 4, 1, 5, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 8, 5, 4, 8, 3, 5, 3, 1, 5, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 2, 10, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 3, 0, 8, 1, 2, 10, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1 },
	{ 5, 2, 10, 5, 4, 2, 4, 0, 2, -1, -1, -1, -1, -1, -1, -1 },
	{ 2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, -1, -1, -1, -1 },
	{ 9, 5, 4, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 11, 2, 0, 8, 11, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 5, 4, 0, 1, 5, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1 },
	{ 2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, -1, -1, -1, -1 },
	{ 10, 3, 11, 10, 1, 3, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1 },
	{ 4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10, -1, -1, -1, -1 },
	{ 5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, -1, -1, -1, -1 },
	{ 5, 4, 8, 5, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1 },
	{ 9, 7, 8, 5, 7, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 9, 3, 0, 9, 5, 3, 5, 7, 3, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 7, 8, 0, 1, 7, 1, 5, 7, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 9, 7, 8, 9, 5, 7, 10, 1, 2, -1, -1, -1, -1, -1, -1, -1 },
	{ 10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, -1, -1, -1, -1 },
	{ 8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2, -1, -1, -1, -1 },
	{ 2, 10, 5, 2, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1 },
	{ 7, 9, 5, 7, 8, 9, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1 },
	{ 9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11, -1, -1, -1, -1 },
	{ 2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7, -1, -1, -1, -1 },
	{ 11, 2, 1, 11, 1, 7, 7, 1, 5, -1, -1, -1, -1, -1, -1, -1 },
	{ 9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, -1, -1, -1, -1 },
	{ 5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0, -1 },
	{ 11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0, -1 },
	{ 11, 10, 5, 7, 11, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 8, 3, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 9, 0, 1, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 8, 3, 1, 9, 8, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 6, 5, 2, 6, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 6, 5, 1, 2, 6, 3, 0, 8, -1, -1, -1, -1, -1, -1, -1 },
	{ 9, 6, 5, 9, 0, 6, 0, 2, 6, -1, -1, -1, -1, -1, -1, -1 },
	{ 5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, -1, -1, -1, -1 },
	{ 2, 3, 11, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 11, 0, 8, 11, 2, 0, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 1, 9, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1 },
	{ 5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, -1, -1, -1, -1 },
	{ 6, 3, 11, 6, 5, 3, 5, 1, 3, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, -1, -1, -1, -1 },
	{ 3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1, -1, -1, -1 },
	{ 6, 5, 9, 6, 9, 11, 11, 9, 8, -1, -1, -1, -1, -1, -1, -1 },
	{ 5, 10, 6, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 4, 3, 0, 4, 7, 3, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 9, 0, 5, 10, 6, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1 },
	{ 10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4, -1, -1, -1, -1 },
	{ 6, 1, 2, 6, 5, 1, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7, -1, -1, -1, -1 },
	{ 8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6, -1, -1, -1, -1 },
	{ 7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9, -1 },
	{ 3, 11, 2, 7, 8, 4, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1 },
	{ 5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11, -1, -1, -1, -1 },
	{ 0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1 },
	{ 9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6, -1 },
	{ 8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, -1, -1, -1, -1 },
	{ 5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11, -1 },
	{ 0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7, -1 },
	{ 6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9, -1, -1, -1, -1 },
	{ 10, 4, 9, 6, 4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 4, 10, 6, 4, 9, 10, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1 },
	{ 10, 0, 1, 10, 6, 0, 6, 4, 0, -1, -1, -1, -1, -1, -1, -1 },
	{ 8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10, -1, -1, -1, -1 },
	{ 1, 4, 9, 1, 2, 4, 2, 6, 4, -1, -1, -1, -1, -1, -1, -1 },
	{ 3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4, -1, -1, -1, -1 },
	{ 0, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 8, 3, 2, 8, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1 },
	{ 10, 4, 9, 10, 6, 4, 11, 2, 3, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6, -1, -1, -1, -1 },
	{ 3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10, -1, -1, -1, -1 },
	{ 6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1, -1 },
	{ 9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3, -1, -1, -1, -1 },
	{ 8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1, -1 },
	{ 3, 11, 6, 3, 6, 0, 0, 6, 4, -1, -1, -1, -1, -1, -1, -1 },
	{ 6, 4, 8, 11, 6, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 7, 10, 6, 7, 8, 10, 8, 9, 10, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10, -1, -1, -1, -1 },
	{ 10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0, -1, -1, -1, -1 },
	{ 10, 6, 7, 10, 7, 1, 1, 7, 3, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, -1, -1, -1, -1 },
	{ 2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9, -1 },
	{ 7, 8, 0, 7, 0, 6, 6, 0, 2, -1, -1, -1, -1, -1, -1, -1 },
	{ 7, 3, 2, 6, 7, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7, -1, -1, -1, -1 },
	{ 2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7, -1 },
	{ 1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11, -1 },
	{ 11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1, -1, -1, -1, -1 },
	{ 8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6, -1 },
	{ 0, 9, 1, 11, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0, -1, -1, -1, -1 },
	{ 7, 11, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 3, 0, 8, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 1, 9, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 8, 1, 9, 8, 3, 1, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1 },
	{ 10, 1, 2, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 2, 10, 3, 0, 8, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1 },
	{ 2, 9, 0, 2, 10, 9, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1 },
	{ 6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8, -1, -1, -1, -1 },
	{ 7, 2, 3, 6, 2, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 7, 0, 8, 7, 6, 0, 6, 2, 0, -1, -1, -1, -1, -1, -1, -1 },
	{ 2, 7, 6, 2, 3, 7, 0, 1, 9, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6, -1, -1, -1, -1 },
	{ 10, 7, 6, 10, 1, 7, 1, 3, 7, -1, -1, -1, -1, -1, -1, -1 },
	{ 10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8, -1, -1, -1, -1 },
	{ 0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7, -1, -1, -1, -1 },
	{ 7, 6, 10, 7, 10, 8, 8, 10, 9, -1, -1, -1, -1, -1, -1, -1 },
	{ 6, 8, 4, 11, 8, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 3, 6, 11, 3, 0, 6, 0, 4, 6, -1, -1, -1, -1, -1, -1, -1 },
	{ 8, 6, 11, 8, 4, 6, 9, 0, 1, -1, -1, -1, -1, -1, -1, -1 },
	{ 9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6, -1, -1, -1, -1 },
	{ 6, 8, 4, 6, 11, 8, 2, 10, 1, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6, -1, -1, -1, -1 },
	{ 4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9, -1, -1, -1, -1 },
	{ 10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3, -1 },
	{ 8, 2, 3, 8, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8, -1, -1, -1, -1 },
	{ 1, 9, 4, 1, 4, 2, 2, 4, 6, -1, -1, -1, -1, -1, -1, -1 },
	{ 8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1, -1, -1, -1, -1 },
	{ 10, 1, 0, 10, 0, 6, 6, 0, 4, -1, -1, -1, -1, -1, -1, -1 },
	{ 4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3, -1 },
	{ 10, 9, 4, 6, 10, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 4, 9, 5, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 8, 3, 4, 9, 5, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1 },
	{ 5, 0, 1, 5, 4, 0, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1 },
	{ 11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5, -1, -1, -1, -1 },
	{ 9, 5, 4, 10, 1, 2, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1 },
	{ 6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5, -1, -1, -1, -1 },
	{ 7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2, -1, -1, -1, -1 },
	{ 3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6, -1 },
	{ 7, 2, 3, 7, 6, 2, 5, 4, 9, -1, -1, -1, -1, -1, -1, -1 },
	{ 9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7, -1, -1, -1, -1 },
	{ 3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0, -1, -1, -1, -1 },
	{ 6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8, -1 },
	{ 9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7, -1, -1, -1, -1 },
	{ 1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4, -1 },
	{ 4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10, -1 },
	{ 7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10, -1, -1, -1, -1 },
	{ 6, 9, 5, 6, 11, 9, 11, 8, 9, -1, -1, -1, -1, -1, -1, -1 },
	{ 3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5, -1, -1, -1, -1 },
	{ 0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11, -1, -1, -1, -1 },
	{ 6, 11, 3, 6, 3, 5, 5, 3, 1, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6, -1, -1, -1, -1 },
	{ 0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10, -1 },
	{ 11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5, -1 },
	{ 6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3, -1, -1, -1, -1 },
	{ 5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, -1, -1, -1, -1 },
	{ 9, 5, 6, 9, 6, 0, 0, 6, 2, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8, -1 },
	{ 1, 5, 6, 2, 1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6, -1 },
	{ 10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0, -1, -1, -1, -1 },
	{ 0, 3, 8, 5, 6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 10, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 11, 5, 10, 7, 5, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 11, 5, 10, 11, 7, 5, 8, 3, 0, -1, -1, -1, -1, -1, -1, -1 },
	{ 5, 11, 7, 5, 10, 11, 1, 9, 0, -1, -1, -1, -1, -1, -1, -1 },
	{ 10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1, -1, -1, -1, -1 },
	{ 11, 1, 2, 11, 7, 1, 7, 5, 1, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11, -1, -1, -1, -1 },
	{ 9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7, -1, -1, -1, -1 },
	{ 7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2, -1 },
	{ 2, 5, 10, 2, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1 },
	{ 8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5, -1, -1, -1, -1 },
	{ 9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2, -1, -1, -1, -1 },
	{ 9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2, -1 },
	{ 1, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 8, 7, 0, 7, 1, 1, 7, 5, -1, -1, -1, -1, -1, -1, -1 },
	{ 9, 0, 3, 9, 3, 5, 5, 3, 7, -1, -1, -1, -1, -1, -1, -1 },
	{ 9, 8, 7, 5, 9, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 5, 8, 4, 5, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1 },
	{ 5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0, -1, -1, -1, -1 },
	{ 0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5, -1, -1, -1, -1 },
	{ 10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4, -1 },
	{ 2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8, -1, -1, -1, -1 },
	{ 0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11, -1 },
	{ 0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5, -1 },
	{ 9, 4, 5, 2, 11, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4, -1, -1, -1, -1 },
	{ 5, 10, 2, 5, 2, 4, 4, 2, 0, -1, -1, -1, -1, -1, -1, -1 },
	{ 3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9, -1 },
	{ 5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2, -1, -1, -1, -1 },
	{ 8, 4, 5, 8, 5, 3, 3, 5, 1, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 4, 5, 1, 0, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5, -1, -1, -1, -1 },
	{ 9, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 4, 11, 7, 4, 9, 11, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11, -1, -1, -1, -1 },
	{ 1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11, -1, -1, -1, -1 },
	{ 3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4, -1 },
	{ 4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2, -1, -1, -1, -1 },
	{ 9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3, -1 },
	{ 11, 7, 4, 11, 4, 2, 2, 4, 0, -1, -1, -1, -1, -1, -1, -1 },
	{ 11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4, -1, -1, -1, -1 },
	{ 2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9, -1, -1, -1, -1 },
	{ 9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7, -1 },
	{ 3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10, -1 },
	{ 1, 10, 2, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 4, 9, 1, 4, 1, 7, 7, 1, 3, -1, -1, -1, -1, -1, -1, -1 },
	{ 4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1, -1, -1, -1, -1 },
	{ 4, 0, 3, 7, 4, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 4, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 9, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 3, 0, 9, 3, 9, 11, 11, 9, 10, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 1, 10, 0, 10, 8, 8, 10, 11, -1, -1, -1, -1, -1, -1, -1 },
	{ 3, 1, 10, 11, 3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 2, 11, 1, 11, 9, 9, 11, 8, -1, -1, -1, -1, -1, -1, -1 },
	{ 3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9, -1, -1, -1, -1 },
	{ 0, 2, 11, 8, 0, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 3, 2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 2, 3, 8, 2, 8, 10, 10, 8, 9, -1, -1, -1, -1, -1, -1, -1 },
	{ 9, 10, 2, 0, 9, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8, -1, -1, -1, -1 },
	{ 1, 10, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 3, 8, 9, 1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 9, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 3, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }
};

/// A small corner structure
struct Corner {
	glm::vec3 pos;
	float val;
};

OctreeNode* getParentCube(OctreeNode* node) {
	return node ? node->parent : nullptr;
}

int getSubcubeIndex(int x, int y, int z,
	int halfSize,
	int x0, int y0, int z0)
{
	// Return 0..7 depending on whether x>=x0+half, y>=..., z>=...
	// bit 0 => x, bit 1 => y, bit 2 => z
	int idx = 0;
	if (x >= x0 + halfSize) idx |= 1;
	if (y >= y0 + halfSize) idx |= 2;
	if (z >= z0 + halfSize) idx |= 4;
	return idx;
}

// A simple function to retrieve 6 face neighbors
// (We skip edge/corner neighbors for simplicity)
std::vector<OctreeNode*> getNeighbors(OctreeNode* node,
	const std::unordered_map<long long, OctreeNode*>& nodeMap)
{
	std::vector<OctreeNode*> neighbors;
	if (!node) {
		return neighbors;
	}

	// A 64-bit key builder
	auto buildKey = [&](int xx, int yy, int zz) {
		long long k = ((long long)xx << 20) |
			((long long)yy << 10) |
			(long long)zz;
		return k;
		};

	// For face neighbors, we offset by ±(node->size) in each dimension:
	// +X
	{
		int nx = node->x + node->size;
		int ny = node->y;
		int nz = node->z;
		long long key = buildKey(nx, ny, nz);
		if (nodeMap.count(key)) {
			neighbors.push_back(nodeMap.at(key));
		}
	}
	// -X
	{
		int nx = node->x - node->size;
		int ny = node->y;
		int nz = node->z;
		long long key = buildKey(nx, ny, nz);
		if (nodeMap.count(key)) {
			neighbors.push_back(nodeMap.at(key));
		}
	}
	// +Y
	{
		int nx = node->x;
		int ny = node->y + node->size;
		int nz = node->z;
		long long key = buildKey(nx, ny, nz);
		if (nodeMap.count(key)) {
			neighbors.push_back(nodeMap.at(key));
		}
	}
	// -Y
	{
		int nx = node->x;
		int ny = node->y - node->size;
		int nz = node->z;
		long long key = buildKey(nx, ny, nz);
		if (nodeMap.count(key)) {
			neighbors.push_back(nodeMap.at(key));
		}
	}
	// +Z
	{
		int nx = node->x;
		int ny = node->y;
		int nz = node->z + node->size;
		long long key = buildKey(nx, ny, nz);
		if (nodeMap.count(key)) {
			neighbors.push_back(nodeMap.at(key));
		}
	}
	// -Z
	{
		int nx = node->x;
		int ny = node->y;
		int nz = node->z - node->size;
		long long key = buildKey(nx, ny, nz);
		if (nodeMap.count(key)) {
			neighbors.push_back(nodeMap.at(key));
		}
	}

	return neighbors;
}

// standard MC single-cell
inline glm::vec3 vertexInterp(float isoLevel, const glm::vec3& p1, const glm::vec3& p2, float valp1, float valp2) {
	if (std::abs(isoLevel - valp1) < 0.00001f) return p1;
	if (std::abs(isoLevel - valp2) < 0.00001f) return p2;
	if (std::abs(valp1 - valp2) < 0.00001f) return p1;

	float mu = (isoLevel - valp1) / (valp2 - valp1);
	return p1 + mu * (p2 - p1);
}


// The actual MC cell polygonization
std::vector<MCTriangle> marchingCubesCell(const std::array<Corner, 8>& corners, float isoValue)
{
	std::vector<MCTriangle> cellTris;
	int cubeIndex = 0;
	for (int i = 0; i < 8; i++) {
		if (corners[i].val < isoValue) {
			cubeIndex |= (1 << i);
		}
	}
	int edges = edgeTable[cubeIndex];
	if (edges == 0) {
		return cellTris;
	}

	glm::vec3 vertList[12];
	// The edge-to-corner mapping depends on how you define them. Make sure it matches.
	static const int e2c[12][2] = {
		{0,1},{1,2},{2,3},{3,0},
		{4,5},{5,6},{6,7},{7,4},
		{0,4},{1,5},{2,6},{3,7}
	};

	for (int e = 0; e < 12; e++) {
		if (edges & (1 << e)) {
			int c1 = e2c[e][0];
			int c2 = e2c[e][1];
			vertList[e] = vertexInterp(
				isoValue,
				corners[c1].pos, corners[c2].pos,
				corners[c1].val, corners[c2].val
			);
		}
	}

	const int* triEdges = triTable[cubeIndex];
	for (int t = 0; t < 5; t++) {
		if (triEdges[3 * t] == -1) break;
		MCTriangle tri;
		for (int v = 0; v < 3; v++) {
			int edgeId = triEdges[3 * t + v];
			tri.v[v] = vertList[edgeId];
			tri.normal[v] = glm::vec3(0, 1, 0); // or compute gradient if you wish
		}
		cellTris.push_back(tri);
	}
	return cellTris;
}

VoxelState getVoxelSafe(const VoxelGrid& grid, int x, int y, int z) {
	if (x < 0 || y < 0 || z < 0 ||
		x >= grid.dimX ||
		y >= grid.dimY ||
		z >= grid.dimZ)
	{
		return VoxelState::EMPTY; // treat out-of-range as empty
	}
	return grid.data[grid.index(x, y, z)];
}

// Build Octree
OctreeNode* buildOctreeRec(const VoxelGrid& grid,
	int x0, int y0, int z0,
	int size,
	std::unordered_map<long long, OctreeNode*>& nodeMap)
{
	OctreeNode* node = new OctreeNode(x0, y0, z0, size);

	// Build the key and add to the map (assuming you already fixed the key issues).
	long long key = ((long long)x0 << 40) | ((long long)y0 << 20) | (long long)z0;
	nodeMap[key] = node;

	// If the cell is of size 1, it is a leaf.
	if (size == 1) {
		node->isLeaf = true;
		VoxelState state = getVoxelSafe(grid, x0, y0, z0);
		node->isSolid = (state == VoxelState::FILLED);
		node->isUniform = true;  // When size==1 it’s trivially uniform.
		return node;
	}

	// Check if all voxels in this region are the same.
	bool allSame = true;
	VoxelState firstVal = getVoxelSafe(grid, x0, y0, z0);
	for (int zz = z0; zz < z0 + size; zz++) {
		for (int yy = y0; yy < y0 + size; yy++) {
			for (int xx = x0; xx < x0 + size; xx++) {
				if (getVoxelSafe(grid, xx, yy, zz) != firstVal) {
					allSame = false;
					break;
				}
			}
			if (!allSame) break;
		}
		if (!allSame) break;
	}

	if (allSame) {
		node->isLeaf = true;
		node->isUniform = true;
		node->isSolid = (firstVal == VoxelState::FILLED);
		return node;
	}

	// Not uniform => subdivide.
	node->isLeaf = false;
	node->isUniform = false;
	int half = size / 2;
	for (int i = 0; i < 8; i++) {
		int ox = x0 + ((i & 1) ? half : 0);
		int oy = y0 + ((i & 2) ? half : 0);
		int oz = z0 + ((i & 4) ? half : 0);
		OctreeNode* child = buildOctreeRec(grid, ox, oy, oz, half, nodeMap);
		node->children[i] = child;
		if (child) {
			child->parent = node;
		}
	}
	return node;
}


OctreeNode* createOctreeFromVoxelGrid(const VoxelGrid& grid) {
	if (grid.dimX == 0 || grid.dimY == 0 || grid.dimZ == 0) return nullptr;

	int maxDim = std::max({ grid.dimX, grid.dimY, grid.dimZ });
	int sizePow2 = 1;
	while (sizePow2 < maxDim) sizePow2 <<= 1;

	// We'll fill this map so we can retrieve neighbors
	extern std::unordered_map<long long, OctreeNode*> g_octreeMap;
	g_octreeMap.clear(); // ensure empty before building

	OctreeNode* root = buildOctreeRec(grid, 0, 0, 0, sizePow2, g_octreeMap);
	return root;
}

std::vector<MCTriangle> localMC(const VoxelGrid& grid, int x0, int y0, int z0, int size) {
	std::vector<MCTriangle> results;
	results.reserve(size * size * size / 2);  // Pre-allocate space

	const float EPSILON = 1e-6f;
	float vx = grid.voxelSize;

	auto getScalar = [&](int x, int y, int z) -> float {
		if (x < 0 || y < 0 || z < 0 || x >= grid.dimX || y >= grid.dimY || z >= grid.dimZ) {
			return 1.0f;
		}
		return (grid.data[grid.index(x, y, z)] == VoxelState::FILLED) ? -1.0f : 1.0f;
		};

	// Process one cell at a time
	for (int z = z0; z < (z0 + size) && z < grid.dimZ - 1; z++) {
		for (int y = y0; y < (y0 + size) && y < grid.dimY - 1; y++) {
			for (int x = x0; x < (x0 + size) && x < grid.dimX - 1; x++) {
				// Build corners
				std::array<Corner, 8> corners;

				// Corner positions in world space
				corners[0] = { glm::vec3(grid.minX + x * vx, grid.minY + y * vx, grid.minZ + z * vx),
							 getScalar(x, y, z) };
				corners[1] = { glm::vec3(grid.minX + (x + 1) * vx, grid.minY + y * vx, grid.minZ + z * vx),
							 getScalar(x + 1, y, z) };
				corners[2] = { glm::vec3(grid.minX + (x + 1) * vx, grid.minY + (y + 1) * vx, grid.minZ + z * vx),
							 getScalar(x + 1, y + 1, z) };
				corners[3] = { glm::vec3(grid.minX + x * vx, grid.minY + (y + 1) * vx, grid.minZ + z * vx),
							 getScalar(x, y + 1, z) };
				corners[4] = { glm::vec3(grid.minX + x * vx, grid.minY + y * vx, grid.minZ + (z + 1) * vx),
							 getScalar(x, y, z + 1) };
				corners[5] = { glm::vec3(grid.minX + (x + 1) * vx, grid.minY + y * vx, grid.minZ + (z + 1) * vx),
							 getScalar(x + 1, y, z + 1) };
				corners[6] = { glm::vec3(grid.minX + (x + 1) * vx, grid.minY + (y + 1) * vx, grid.minZ + (z + 1) * vx),
							 getScalar(x + 1, y + 1, z + 1) };
				corners[7] = { glm::vec3(grid.minX + x * vx, grid.minY + (y + 1) * vx, grid.minZ + (z + 1) * vx),
							 getScalar(x, y + 1, z + 1) };

				// Check if cell intersects surface
				bool hasPos = false, hasNeg = false;
				for (const auto& c : corners) {
					if (c.val < 0) hasNeg = true;
					if (c.val > 0) hasPos = true;
					if (hasPos && hasNeg) break;
				}

				if (!hasPos || !hasNeg) continue;

				// Process cell
				int cubeIndex = 0;
				for (int i = 0; i < 8; i++) {
					if (corners[i].val < 0) {
						cubeIndex |= (1 << i);
					}
				}

				// Get edge flags
				int edgeFlags = edgeTable[cubeIndex];
				if (edgeFlags == 0) continue;

				// Calculate vertices
				glm::vec3 vertList[12];
				if (edgeFlags & 1)    vertList[0] = vertexInterp(0.0f, corners[0].pos, corners[1].pos, corners[0].val, corners[1].val);
				if (edgeFlags & 2)    vertList[1] = vertexInterp(0.0f, corners[1].pos, corners[2].pos, corners[1].val, corners[2].val);
				if (edgeFlags & 4)    vertList[2] = vertexInterp(0.0f, corners[2].pos, corners[3].pos, corners[2].val, corners[3].val);
				if (edgeFlags & 8)    vertList[3] = vertexInterp(0.0f, corners[3].pos, corners[0].pos, corners[3].val, corners[0].val);
				if (edgeFlags & 16)   vertList[4] = vertexInterp(0.0f, corners[4].pos, corners[5].pos, corners[4].val, corners[5].val);
				if (edgeFlags & 32)   vertList[5] = vertexInterp(0.0f, corners[5].pos, corners[6].pos, corners[5].val, corners[6].val);
				if (edgeFlags & 64)   vertList[6] = vertexInterp(0.0f, corners[6].pos, corners[7].pos, corners[6].val, corners[7].val);
				if (edgeFlags & 128)  vertList[7] = vertexInterp(0.0f, corners[7].pos, corners[4].pos, corners[7].val, corners[4].val);
				if (edgeFlags & 256)  vertList[8] = vertexInterp(0.0f, corners[0].pos, corners[4].pos, corners[0].val, corners[4].val);
				if (edgeFlags & 512)  vertList[9] = vertexInterp(0.0f, corners[1].pos, corners[5].pos, corners[1].val, corners[5].val);
				if (edgeFlags & 1024) vertList[10] = vertexInterp(0.0f, corners[2].pos, corners[6].pos, corners[2].val, corners[6].val);
				if (edgeFlags & 2048) vertList[11] = vertexInterp(0.0f, corners[3].pos, corners[7].pos, corners[3].val, corners[7].val);

				// Create triangles
				for (int i = 0; triTable[cubeIndex][i] != -1; i += 3) {
					MCTriangle tri;
					tri.v[0] = vertList[triTable[cubeIndex][i]];
					tri.v[1] = vertList[triTable[cubeIndex][i + 1]];
					tri.v[2] = vertList[triTable[cubeIndex][i + 2]];

					// Calculate normal
					glm::vec3 edge1 = tri.v[1] - tri.v[0];
					glm::vec3 edge2 = tri.v[2] - tri.v[0];
					glm::vec3 normal = glm::normalize(glm::cross(edge1, edge2));

					tri.normal[0] = normal;
					tri.normal[1] = normal;
					tri.normal[2] = normal;

					results.push_back(tri);
				}
			}
		}
	}

	return results;
}
void freeOctree(OctreeNode* node)
{
	if (!node) return;
	for (int i = 0; i < 8; i++) {
		freeOctree(node->children[i]);
	}
	delete node;
}
