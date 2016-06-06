#ifndef __CPREPROCESS_INT_H
#define __CPREPROCESS_INT_H

class ImgWrap;

class CPreprocessInt
{
	/* interface */
public:
	virtual void doit( ImgWrap *imgWrapSrc )=0;
};

#endif