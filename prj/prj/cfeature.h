#ifndef __CFEATURE_H
#define __CFEATURE_H

#include <vector>
#include <cv.h>

class ImgWrap;

//特征组织类
class CFeatureModel
{
public:
	CFeatureModel(){}
public:
	std::vector<double> mixfeat;
};

//特征组织类
class CFeatureImg
{
public:
	CFeatureImg(){}
public:
	std::vector<int> lbpfeat;
	cv::Mat siftfeat;
	std::vector<cv::Mat> gaborfeat;
	std::vector<std::vector<int> > catgabor;
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
	void _mixfeature(CFeatureImg *featImg1, CFeatureImg *featImg2);

	void _mixlbpfeat(CFeatureImg *featImg1, CFeatureImg *featImg2);
	void _mixsiftfeat(CFeatureImg *featImg1, CFeatureImg *featImg2);
	void _mixgaborfeat(CFeatureImg *featImg1, CFeatureImg *featImg2);

	void _mixcatgaborfeat(CFeatureImg *featImg1, CFeatureImg *featImg2);
	/* member var */
public:
	CFeatureModel mFeatureMode;
	//CFeatureImg mFeatureImgA;
	//CFeatureImg mFeatureImgB;
};

#endif 