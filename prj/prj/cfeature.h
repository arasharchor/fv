#ifndef __CFEATURE_H
#define __CFEATURE_H

#include <vector>
#include <cv.h>

using namespace std;
using namespace cv;

class ImgWrap;

//������֯��
class CFeatureModel
{
public:
	CFeatureModel(){ mixlbp.clear();mixsift.release();}
public:
	vector<int> mixlbp;
	Mat mixsift;
};

//������֯��
class CFeatureImg
{
public:
	CFeatureImg(){ lbpfeat.clear();siftfeat.release();}
public:
	vector<int> lbpfeat;
	Mat siftfeat;
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