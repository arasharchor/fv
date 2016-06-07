#ifndef __CFEATURE_H
#define __CFEATURE_H

#include <vector>

class ImgWrap;

//������֯��
class CFeatureStore
{
public:
	CFeatureStore(){ featStore.clear();numberPerFeatType.clear(); }
public:
	std::vector<std::vector<double>> featStore;		//���������Ĵ洢������vector��֯����
	std::vector<int>	numberPerFeatType;			//ÿ������������
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