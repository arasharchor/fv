#ifndef __CFEATURE_H
#define __CFEATURE_H

class ImgWrap;

class CFeatureModel
{
public:	CFeatureModel(void *ptr=NULL):context(ptr){}
public:	void *context;
};

class CFeature
{
	/* ctor and de-ctor */
public:
	CFeature(){}
	CFeature(ImgWrap *imgWrapSrc1, ImgWrap *imgWrapSrc2);
	~CFeature(){}

	/* interface */
public:

	/* member fun */
private:

	/* member var */
private:
	CFeatureModel mFeatureModel;
};

#endif 