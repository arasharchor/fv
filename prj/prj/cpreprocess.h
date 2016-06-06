#ifndef __CPRE_PROCESS_H
#define __CPRE_PROCESS_H

class ImgWrap;

class CPreprocess
{
	/* interface */
public:
	virtual void doit( ImgWrap *imgWrapSrc )=0;
};

#endif