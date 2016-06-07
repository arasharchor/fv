#ifndef __CFEATURE_H
#define __CFEATURE_H

#include <vector>
#include <cv.h>

using namespace std;
using namespace cv;

class ImgWrap;

//特征组织类
class CFeatureStore
{
public:
	CFeatureStore(){ lbpfeat.clear();siftfeat.release();}
public:
	vector<int> lbpfeat;
	Mat siftfeat;
};

class CFeature
{
	/* ctor and de-ctor */
public:
	CFeature(){}
	CFeature(ImgWrap *imgWrapSrc);
	~CFeature(){}

	/* interface */
public:

	/* member fun */
private:

	/* member var */
public:
	CFeatureStore mFeatureStore;
};

#endif 