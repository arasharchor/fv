
#include <iostream>
#include <vector>
#include <opencv2/legacy/legacy.hpp>

#include "cfeature.h"
#include "cextfeature.h"
#include "cpreprocess.h"
#include "iofile.h"

using namespace std;
using namespace cv;


CFeature::CFeature(Mat *imgSrc1, Mat *imgSrc2)
{
	CPreprocessInt *preprocess = new CPreprocess();
	CExtfeatInt *extfeat = new CExtfeature();

	//预处理
	if(preprocess)
	{
		if (!preprocess->doit(imgSrc1))
		{
			return;
		}
		if (!preprocess->doit(imgSrc2))
		{
			return;
		}
	}

	//提取特征
	if(extfeat)
	{
		CFeatureImg mFeatureImgA, mFeatureImgB;
		extfeat->doit(imgSrc1, &mFeatureImgA);
		extfeat->doit(imgSrc2, &mFeatureImgB);
		_mixfeature(&mFeatureImgA, &mFeatureImgB, vector<int>());
	}
}

CFeature::CFeature(iofile imgCoupleDataSet, int nth, bool type)
{
    if (!type)
    {
        nth += imgCoupleDataSet.posSamplesNums();      // 负样本偏移量
    }

	// 载入第n个样本特征
    if ( imgCoupleDataSet.load(this->mFeatureMode.mixfeat, nth) )
    {
        return;
    }

    // 提取第n个样本信息
    coupleImageInf imgInf;

    imgInf.label = type;
    imgCoupleDataSet.load(imgInf, nth);

    // 人脸检测
	Mat imgSrc1 = imread(imgInf.imgPath1, IMREAD_GRAYSCALE);
	Mat imgSrc2 = imread(imgInf.imgPath2, IMREAD_GRAYSCALE);

    CPreprocessInt *preprocess = new CPreprocess();
    if (preprocess)
    {
        errLogInf logInf = {nth, logInf.imgNoErr, imgInf};

        if ( !preprocess->doit(&imgSrc1) )          // 第一个图片没检测到人脸
		{
			logInf.errImg = logInf.img1Err;
		}

		if ( !preprocess->doit(&imgSrc2) )          // 第二个图片没检测到人脸
		{
            if (logInf.errImg == logInf.img1Err)    // 都没检测到
            {
                logInf.errImg = logInf.imgAllErr;
            }
            else
            {
			    logInf.errImg = logInf.img2Err;
            }
		}
		// 输出到日志
		if (logInf.errImg != logInf.imgNoErr)
		{
			imgCoupleDataSet.save(logInf);
		}
	}
	
    resize(imgSrc1, imgSrc1, Size(80, 80), 0, 0, CV_INTER_LINEAR);
    resize(imgSrc2, imgSrc2, Size(80, 80), 0, 0, CV_INTER_LINEAR);

	// 提取特征
	CExtfeatInt *extfeat = new CExtfeature();
    vector<int> table;
    extfeat->getTable(table);
	if(extfeat)
	{
		CFeatureImg mFeatureImgA, mFeatureImgB;
		extfeat->doit(&imgSrc1, &mFeatureImgA);
		extfeat->doit(&imgSrc2, &mFeatureImgB);
		_mixfeature(&mFeatureImgA, &mFeatureImgB, table);
	}
    // 写入到特征集
    imgCoupleDataSet.save(this->mFeatureMode.mixfeat, nth);
}

void CFeature::_mixfeature(CFeatureImg *featImg1, CFeatureImg *featImg2, vector<int> &table)
{
	_mixlbpfeat(featImg1, featImg2, table);
	//_mixsiftfeat(featImg1, featImg2);
	//_mixgaborfeat(featImg1, featImg2);
	//_mixcatgaborfeat(featImg1, featImg2);
}

//void CFeature::_mixlbpfeat(CFeatureImg *featImg1, CFeatureImg *featImg2)
//{
//	// for lbp feature
//	vector<int> feat1(featImg1->lbpfeat);
//	vector<int> feat2(featImg2->lbpfeat);
//	float lbpDistance = 0;
//	for (size_t i = 0; i < feat1.size(); ++i)
//	{
//		lbpDistance += (feat1[i] - feat2[i]) * (feat1[i] - feat2[i]);
//	}
//	lbpDistance = sqrt(lbpDistance) / 58;
//
//	mFeatureMode.mixfeat.push_back(lbpDistance);
//}

void CFeature::_mixlbpfeat(CFeatureImg *featImg1, CFeatureImg *featImg2, vector<int> &table)
{
    // for lbp feature
    for (size_t i = 0; i < featImg1->lbpfeat.size(); ++i)
    {
        if (table[i] == 1)
        {
            mFeatureMode.mixfeat.push_back(abs(featImg1->lbpfeat[i] - featImg2->lbpfeat[i]));
        }
    }
}
void CFeature::_mixsiftfeat(CFeatureImg *featImg1, CFeatureImg *featImg2)
{
	// for sift feature
	BruteForceMatcher<L2<float> > matcher;					// brute force matcher
	vector<DMatch> matches1to2;								// result of matches 1-->2
	vector<DMatch> matches2to1;								// result of matches 2-->1
	if (featImg1->siftfeat.empty() || featImg2->siftfeat.empty())
	{
		printf("----------楼下没有SIFT特征------------\n");
		return;
	}
	matcher.match(featImg1->siftfeat, featImg2->siftfeat, matches1to2);
	matcher.match(featImg2->siftfeat, featImg1->siftfeat, matches2to1);

	float siftDistance = 0;
	for (size_t i = 0; i < matches1to2.size(); ++i)
	{
		siftDistance += matches1to2[i].distance;
	}
	for (size_t i = 0; i < matches2to1.size(); ++i)
	{
		siftDistance += matches2to1[i].distance;
	}
	siftDistance = siftDistance / (matches1to2.size() + matches2to1.size());

	mFeatureMode.mixfeat.push_back(siftDistance);
}

void CFeature::_mixgaborfeat(CFeatureImg *featImg1, CFeatureImg *featImg2)
{
	// for gabor feature
	CExtfeature *extfeat = new CExtfeature;
	for (size_t i = 0; i < featImg1->gaborfeat.size(); ++i)
	{
		CFeatureImg mFeatImgA, mFeatImgB;

		Mat *f1(&featImg1->gaborfeat[i]);
		Mat *f2(&featImg2->gaborfeat[i]);

		extfeat->_cextlbp(f1, &mFeatImgA);
		extfeat->_cextlbp(f2, &mFeatImgB);

		_mixlbpfeat(&mFeatImgA, &mFeatImgB, vector<int>());
	}
}

void CFeature::_mixcatgaborfeat(CFeatureImg *featImg1, CFeatureImg *featImg2)
{
	// for cat gabor
	for (size_t i = 0; i < featImg1->catgabor.size(); ++i)
	{
		float sum = 0;
		for (size_t j = 0; j < featImg1->catgabor[i].size(); ++j)
		{
			sum += (featImg1->catgabor[i][j] - featImg2->catgabor[i][j]) * (featImg1->catgabor[i][j] - featImg2->catgabor[i][j]);
		}
		sum /= featImg1->catgabor.size();
		mFeatureMode.mixfeat.push_back(sum);
	}
}
