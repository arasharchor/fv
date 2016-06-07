
#include <iostream>
#include "cfeature.h"
#include "cextfeatDemo.h"
#include "wrap.h"

#include <cv.h>
#include <opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

void CExtfeatDemo::doit( const ImgWrap *imgWrapSrc, CFeatureStore *featStore )
{
	_do(imgWrapSrc, featStore);
}

void CExtfeatDemo::_do( const ImgWrap *imgWrapSrc, CFeatureStore *featStore )
{
	_cextlbp(imgWrapSrc, featStore);
	_cextsift(imgWrapSrc, featStore);
}

void CExtfeatDemo::_cextlbp(const ImgWrap *imgWrapSrc, CFeatureStore *featStore)
{
	Mat *img = (Mat *)imgWrapSrc->context;
	assert(img->channels() == 1);//single channel

	int row = img->rows;
	int col = img->cols;
	// p0	p1	p2
	// p7	x	p3
	// p6	p5	p4
	vector<double> lbpfeat(256, 0);			// lbp feature (histogram of lbp image)
	for (int i = 1; i < row - 1; ++i)
	{
		for (int j = 1; j < col - 1; ++j)
		{
			int tmp[8] = {0};
			
			if (img->at<uchar>(i - 1, j - 1) > img->at<uchar>(i, j))	{ tmp[0] = 1;  }	// p0
			if (img->at<uchar>(i - 1, j + 0) > img->at<uchar>(i, j))	{ tmp[1] = 2;  }	// p1
			if (img->at<uchar>(i - 1, j + 1) > img->at<uchar>(i, j))	{ tmp[2] = 4;  }	// p2
			if (img->at<uchar>(i + 0, j + 1) > img->at<uchar>(i, j))	{ tmp[3] = 8;  }	// p3
			if (img->at<uchar>(i + 1, j + 1) > img->at<uchar>(i, j))	{ tmp[4] = 16; }	// p4
			if (img->at<uchar>(i + 1, j + 0) > img->at<uchar>(i, j))	{ tmp[5] = 32; }	// p5
			if (img->at<uchar>(i + 1, j - 1) > img->at<uchar>(i, j))	{ tmp[6] = 64; }	// p6
			if (img->at<uchar>(i + 0, j - 1) > img->at<uchar>(i, j))	{ tmp[7] = 128;}	// p7

			int lbpvalue = tmp[0] + tmp[1] + tmp[2] + tmp[3]+ tmp[4] + tmp[5] + tmp[6] + tmp[7];
			lbpfeat[lbpvalue] += 1;		// cumulative
		}
	}
	featStore->featStore.push_back(lbpfeat);		// lbp feature
	featStore->numberPerFeatType.push_back(256);	// lbp dimension
}

void CExtfeatDemo::_cextlbp(const ImgWrap *imgWrapSrc, CFeatureStore *featStore, int scale)
{
	Mat *img = (Mat *)imgWrapSrc->context;
	Mat imgIntegral;
	integral(*img, imgIntegral);

	// ...
}

void CExtfeatDemo::_cextsift(const ImgWrap *imgWrapSrc, CFeatureStore *featStore)
{
	// ...
}