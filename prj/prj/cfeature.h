#ifndef __CFEATURE_H
#define __CFEATURE_H

#include <vector>
#include <cv.h>

//#include "iofile.h"

class ImgWrap;
class iofile;

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
	CFeature(cv::Mat *imgWrapSrc1, cv::Mat *imgWrapSrc2);
	CFeature(iofile imgCoupleDataSet, int nth);
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
	int label;
	CFeatureModel mFeatureMode;
};

#endif 