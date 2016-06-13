#ifndef __CFEATURE_H
#define __CFEATURE_H

#include <vector>
#include <cv.h>

using namespace std;
using namespace cv;

class ImgWrap;

//特征组织类
class CFeatureModel
{
public:
	CFeatureModel(){ mixlbp.clear();mixsift.release();}
public:
	vector<int> mixlbp;
	Mat mixsift;
	vector<Mat> gaborfeat;
};

//特征组织类
class CFeatureImg
{
public:
	CFeatureImg(){ lbpfeat.clear();siftfeat.release();}
public:
	vector<int> lbpfeat;
	Mat siftfeat;
	vector<Mat> gaborfeat;
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
	void _mixfeature(const CFeatureImg *featImg1, const CFeatureImg *featImg2, CFeatureModel *featMode);
	/* member var */
public:
	CFeatureModel mFeatureMode;
	CFeatureImg mFeatureImgA;
	CFeatureImg mFeatureImgB;
};

#endif 