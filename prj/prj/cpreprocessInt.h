#ifndef __CPREPROCESS_Ci_H
#define __CPREPROCESS_H

class ImgWrap;

class CPreprocessInt
{
	/* interface */
public:
	virtual void doit( ImgWrap *imgWrapSrc )=0;
};

#endif