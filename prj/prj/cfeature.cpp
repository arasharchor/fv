
#include <iostream>
#include <vector>
#include <opencv2/legacy/legacy.hpp>

#include "cfeature.h"
#include "cextfeature.h"
#include "cpreprocess.h"


CFeature::CFeature(ImgWrap *imgWrapSrc1, ImgWrap *imgWrapSrc2)
{
	CPreprocessInt *preprocess = new CPreprocess();
	CExtfeatInt *extfeat = new CExtfeature();

	//预处理
	if(preprocess)
	{
		preprocess->doit(imgWrapSrc1);
		preprocess->doit(imgWrapSrc2);
	}

	//提取特征
	if(extfeat)
	{
		extfeat->doit(imgWrapSrc1, &this->mFeatureImgA);
		extfeat->doit(imgWrapSrc2, &this->mFeatureImgB);
		this->_mixedfeature(&mFeatureImgA, &mFeatureImgB, &mFeatureMode);
	}
}

void CFeature::_mixedfeature( const CFeatureImg *featImg1, const CFeatureImg *featImg2, CFeatureModel *featMode )
{
	// for lbp feature
	vector<int> feat1(featImg1->lbpfeat);
	vector<int> feat2(featImg2->lbpfeat);
	float lbpDistance = 0;
	for (unsigned int i = 0; i < feat1.size(); ++i)
	{
		lbpDistance += (feat1[i] - feat2[i]) * (feat1[i] - feat2[i]);
	}
	lbpDistance = sqrt(lbpDistance) / 57;
	featMode->mixfeat.push_back(lbpDistance);

	// for sift feature
	BruteForceMatcher<L2<float> > matcher;					// brute force matcher
	vector<DMatch> matches1to2;								// result of matches 1-->2
	vector<DMatch> matches2to1;								// result of matches 2-->1

	matcher.match(featImg1->siftfeat, featImg2->siftfeat, matches1to2);
	matcher.match(featImg2->siftfeat, featImg1->siftfeat, matches2to1);

	float siftDistance = 0;
	for (unsigned int i = 0; i < matches1to2.size(); ++i)
	{
		siftDistance += matches1to2[i].distance;
	}
	for (unsigned int i = 0; i < matches2to1.size(); ++i)
	{
		siftDistance += matches2to1[i].distance;
	}
	siftDistance = siftDistance / (matches1to2.size() + matches2to1.size());
	featMode->mixfeat.push_back(siftDistance);

	// for gabor feature

	// ...

}