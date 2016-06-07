#ifndef __CFEATURE_H
#define __CFEATURE_H

#include <vector>

class ImgWrap;

//特征组织类
class CFeatureStore
{
public:
	CFeatureStore(){ featStore.clear();numberPerFeatType.clear(); }
public:
	std::vector<std::vector<double>> featStore;		//整个特征的存储由数个vector组织起来
	std::vector<int>	numberPerFeatType;			//每种特征的数量
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