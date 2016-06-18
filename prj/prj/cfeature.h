#ifndef __CFEATURE_H
#define __CFEATURE_H

#include <vector>
#include <cv.h>

class ImgWrap;

//特征组织类
class CFeatureModel
{
public:
	CFeatureModel(){ mixlbp.clear();mixsift.release();}
public:
	std::vector<double> mixfeat;
	std::vector<int> mixlbp;
	cv::Mat mixsift;
	std::vector<cv::Mat> gaborfeat;
};

//特征组织类
class CFeatureImg
{
public:
	CFeatureImg(){ lbpfeat.clear();siftfeat.release();}
public:
	std::vector<int> lbpfeat;
	cv::Mat siftfeat;
	std::vector<cv::Mat> gaborfeat;
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
	void _mixedfeature(const CFeatureImg *featImg1, const CFeatureImg *featImg2, CFeatureModel *featMode);
	/* member var */
public:
	CFeatureModel mFeatureMode;
	CFeatureImg mFeatureImgA;
	CFeatureImg mFeatureImgB;
};

#endif 