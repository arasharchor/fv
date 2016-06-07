
#include <iostream>
#include "cfeature.h"
#include "cextfeature.h"
#include "wrap.h"

#include <opencv2/nonfree/nonfree.hpp>

using namespace std;
using namespace cv;

void CExtfeature::doit( const ImgWrap *imgWrapSrc, CFeatureStore *featStore )
{
	_do(imgWrapSrc, featStore);
}

void CExtfeature::_do( const ImgWrap *imgWrapSrc, CFeatureStore *featStore )
{
	_cextlbp(imgWrapSrc, featStore);
	//_cextlbp(imgWrapSrc, featStore, 5);
	_cextsift(imgWrapSrc, featStore);
}

void CExtfeature::_cextlbp(const ImgWrap *imgWrapSrc, CFeatureStore *featStore)
{
	Mat *img = (Mat *)imgWrapSrc->context;
	assert(img->channels() == 1);	//single channel

	int row = img->rows;
	int col = img->cols;
	// p0	p1	p2
	// p7	x	p3
	// p6	p5	p4
	featStore->lbpfeat.resize(256);	// lbp feature (histogram of lbp image)
	for (int i = 1; i < row - 1; ++i)
	{
		for (int j = 1; j < col - 1; ++j)
		{
			int lbpvalue = 0;
			
			if (img->at<uchar>(i - 1, j - 1) > img->at<uchar>(i, j))	{ lbpvalue += 1;  }	// p0
			if (img->at<uchar>(i - 1, j + 0) > img->at<uchar>(i, j))	{ lbpvalue += 2;  }	// p1
			if (img->at<uchar>(i - 1, j + 1) > img->at<uchar>(i, j))	{ lbpvalue += 4;  }	// p2
			if (img->at<uchar>(i + 0, j + 1) > img->at<uchar>(i, j))	{ lbpvalue += 8;  }	// p3
			if (img->at<uchar>(i + 1, j + 1) > img->at<uchar>(i, j))	{ lbpvalue += 16; }	// p4
			if (img->at<uchar>(i + 1, j + 0) > img->at<uchar>(i, j))	{ lbpvalue += 32; }	// p5
			if (img->at<uchar>(i + 1, j - 1) > img->at<uchar>(i, j))	{ lbpvalue += 64; }	// p6
			if (img->at<uchar>(i + 0, j - 1) > img->at<uchar>(i, j))	{ lbpvalue += 128;}	// p7

			featStore->lbpfeat[lbpvalue] += 1;		// cumulative	
		}
	}
}

void CExtfeature::_cextlbp(const ImgWrap *imgWrapSrc, CFeatureStore *featStore, int scale)
{
	Mat *img = (Mat *)imgWrapSrc->context;
	assert(img->channels() == 1);	//single channel

	Mat imgIntegral;
	integral(*img, imgIntegral);
	
	featStore->lbpfeat.resize(256);	// mb-lbp feature (histogram of mb-lbp image)
	// mb0		mb1		mb2
	// mb7	  center	mb3
	// mb6		mb5		mb4
	for (int i = 0; i < img->rows - 3 * scale; ++i)		//row
	{
		for (int j = 0; j < img->cols - 3 * scale; ++j)	//col
		{
			// nA	nB	nC	nD
			// nE	nF	nG	nH
			// nI	nJ	nK	nL
			// nM	nN	nO	nP
			int nA = imgIntegral.at<int>(i, j);
			int nB = imgIntegral.at<int>(i, j + scale);
			int nC = imgIntegral.at<int>(i, j + 2 * scale);
			int nD = imgIntegral.at<int>(i, j + 3 * scale);

			int nE = imgIntegral.at<int>(i + scale, j);
			int nF = imgIntegral.at<int>(i + scale, j + scale);
			int nG = imgIntegral.at<int>(i + scale, j + 2 * scale);
			int nH = imgIntegral.at<int>(i + scale, j + 3 * scale);

			int nI = imgIntegral.at<int>(i + 2 * scale, j);
			int nJ = imgIntegral.at<int>(i + 2 * scale, j + scale);
			int nK = imgIntegral.at<int>(i + 2 * scale, j + 2 * scale);
			int nL = imgIntegral.at<int>(i + 2 * scale, j + 3 * scale);

			int nM = imgIntegral.at<int>(i + 3 * scale, j);
			int nN = imgIntegral.at<int>(i + 3 * scale, j + scale);
			int nO = imgIntegral.at<int>(i + 3 * scale, j + 2 * scale);
			int nP = imgIntegral.at<int>(i + 3 * scale, j + 3 * scale);

			int mblock[8] = {0}, centblock = 0;
			mblock[0] = nF + nA - nB - nE;	// mb0 ( sum of all pixels in block0 )
			mblock[1] = nG + nB - nC - nF;	// mb1
			mblock[2] = nH + nC - nD - nG;	// mb2
			mblock[3] = nL + nG - nH - nK;	// mb3
			mblock[4] = nP + nK - nL - nO;	// mb4
			mblock[5] = nO + nJ - nK - nN;	// mb5
			mblock[6] = nN + nI - nJ - nM;	// mb6
			mblock[7] = nJ + nE - nF - nI;	// mb7
			centblock = nK + nF - nG - nJ;	// center

			int lbpvalue = 0;
			if (mblock[0] >= centblock) { lbpvalue += 1;  }
			if (mblock[1] >= centblock) { lbpvalue += 2;  }
			if (mblock[2] >= centblock) { lbpvalue += 4;  }
			if (mblock[3] >= centblock) { lbpvalue += 8;  }
			if (mblock[4] >= centblock) { lbpvalue += 16; }
			if (mblock[5] >= centblock) { lbpvalue += 32; }
			if (mblock[6] >= centblock) { lbpvalue += 64; }
			if (mblock[7] >= centblock) { lbpvalue += 128;}

			featStore->lbpfeat[lbpvalue] += 1;	// cumulative	
		}
	}
}

void CExtfeature::_cextsift(const ImgWrap *imgWrapSrc, CFeatureStore *featStore)
{
	Mat *img = (Mat *)imgWrapSrc->context;
	assert(img->channels() == 1);//single channel
	vector<KeyPoint> keypoint;
	Mat des, mask;

	SIFT sift;
	sift(*img, mask, keypoint, featStore->siftfeat);
}