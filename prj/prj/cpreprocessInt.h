#ifndef __CPREPROCESS_INT_H
#define __CPREPROCESS_INT_H

class ImgWrap;

class CPreprocessInt
{
	/* ctor and de-ctor */
public:
	virtual ~CPreprocessInt(){}

	/* interface */
public:
	virtual bool doit( cv::Mat *imgSrc )=0;
};

#endif