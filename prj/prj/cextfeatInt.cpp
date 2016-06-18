
#include <memory>
#include "cextfeatInt.h"

CExtfeatInt::CExtfeatInt()
{
	// table of the uniform mode
	const int tmp[58] = 
	{	
	//  1   2	  3	   4	5	 6	  7			number of zero
		127, 63,  31,  15,  7,   3,   1,
		191, 159, 143, 135, 131, 129, 128,
		223, 207, 199, 195, 193, 192, 64,
		239, 231, 227, 225, 224, 96,  32,
		247, 243, 241, 240, 112, 48,  16,
		251, 249, 248, 120, 56,  24,  8,
		253, 252, 124, 60,  28,  12,  4,
		254, 126, 62,  30,  14,  6,   2,
	//  0   8
		255, 0
	};
	utable = new int[256];
	memset(utable, 0, 256 * sizeof(int));
	for (int i = 0; i < 58; ++i)
	{
		utable[tmp[i]] = 1;
	}
}
