
#include <iostream>
#include "cfeature.h"
#include "cextfeature.h"
#include "wrap.h"

#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

#include <memory>
#include "cextfeatInt.h"

CExtfeature::CExtfeature()
{
	// table of the uniform mode
	const int tmp[58] = 
	{	
		//  1   2	  3	   4	5	 6	  7			number of zero
		127, 63,  31,  15,  7,   3,   1,
		191, 159, 143, 135, 131, 129, 128,
		223, 207, 199, 195, 193, 192, 64,
		239, 231, 227, 225, 224, 96,  32,
		247, 243, 241, 240, 112, 48,  16,
		251, 249, 248, 120, 56,  24,  8,
		253, 252, 124, 60,  28,  12,  4,
		254, 126, 62,  30,  14,  6,   2,
		//  0   8
		255, 0
	};
	utable = new int[256];
	memset(utable, 0, 256 * sizeof(int));
	for (int i = 0; i < 58; ++i)
	{
		utable[tmp[i]] = 1;
	}
}

void CExtfeature::doit(const ImgWrap *imgWrapSrc, CFeatureImg *featImg )
{
	_do(imgWrapSrc, featImg);
}

void CExtfeature::_do(const ImgWrap *imgWrapSrc, CFeatureImg *featImg )
{
	_cextlbp(imgWrapSrc, featImg);
	//_cextlbp(imgWrapSrc, featStore, 5);
	_cextsift(imgWrapSrc, featImg);
	_cextgabor(imgWrapSrc, featImg);
}

void CExtfeature::_cextlbp(const ImgWrap *imgWrapSrc, CFeatureImg *featImg )
{
	Mat *img = (Mat *)imgWrapSrc->context;
	assert(img->channels() == 1);	//single channel

	int row = img->rows;
	int col = img->cols;
	// p0	p1	p2
	// p7	x	p3
	// p6	p5	p4
	featImg->lbpfeat.resize(256);	// lbp feature (histogram of lbp image)
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

			if (this->utable[lbpvalue] == 1)
			{
				featImg->lbpfeat[lbpvalue] += 1;		// cumulative
			}
		} // end of for
	} // end of for
} // end of function

void CExtfeature::_cextlbp(const ImgWrap *imgWrapSrc, CFeatureImg *featImg, int scale )
{
	Mat *img = (Mat *)imgWrapSrc->context;
	assert(img->channels() == 1);	//single channel

	Mat imgIntegral;
	integral(*img, imgIntegral);
	
	featImg->lbpfeat.resize(256);	// mb-lbp feature (histogram of mb-lbp image)
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

			if (this->utable)
			{
				featImg->lbpfeat[lbpvalue] += 1;	// cumulative
			}

		} // end of for
	} // end of for
} // end of function

void CExtfeature::_cextsift(const ImgWrap *imgWrapSrc, CFeatureImg *featImg )
{
	Mat *img = (Mat *)imgWrapSrc->context;
	assert(img->channels() == 1);	//single channel
	vector<KeyPoint> keypoint;
	Mat des, mask;

	SIFT sift;
	sift(*img, mask, keypoint, featImg->siftfeat);
}

// ¦Õ_uv(z) = (|K_uv| ^ 2 / (sigma ^ 2)) * exp(-(|K_uv|^2 * |z|^2) / (2 * sigma^2)) * (exp(i * K_uv) - exp(-sigma^2 / 2))
// K_uv = K_v * exp(i * Phi_u)		//Phi_u ---> ¦Õ_u
// K_v = K_max / (f ^ v)
// Phi_u = u * pi / 8
// K_max = pi / 2
// f = sqrt(2)
// sigma = 2 * pi
void CExtfeature::_cextgabor(const ImgWrap *imgWrapSrc, CFeatureImg *featImg )
{
	Mat *imgSrc = (Mat *)imgWrapSrc->context;
	assert(imgSrc->channels() == 1);				//single channel

	double sigma = 2 * CV_PI;
	double f = sqrt(2.0);
	double K_max = CV_PI / 2;

	for (int v = 0; v < 5; ++v)					// scale
	{
		for (int u = 0; u < 8; ++u)				// orientation
		{
			int mask_width = cvRound(6 * sigma * pow(f, v) / K_max) + 1;	//width of gabor filter
			double Phi_u = CV_PI * u / 8;
			double K_v = K_max / pow(f, v);	

			Mat imgMaskReal = Mat::zeros(mask_width, mask_width, CV_64FC1);
			Mat imgMaskIm = Mat::zeros(mask_width, mask_width, CV_64FC1);

			Mat imgReal, imgIm, imgMac;

			for (int i = 0; i < mask_width; ++i)
			{
				for (int j = 0; j < mask_width; ++j)
				{
					int x = i - (mask_width - 1) / 2;	// offset
					int y = j - (mask_width - 1) / 2;

					double K_uv = K_v * (cos(Phi_u) * x + sin(Phi_u) * y);
					double part1 = ((K_v * K_v) / (sigma * sigma)) * exp(-(K_v * K_v) * (x * x + y * y) / (2 * sigma * sigma));
					double part2 = cos(K_uv) - exp(-sigma * sigma / 2);
					double part3 = sin(K_uv);

					double realPart = part1 * part2;	// real part
					double imPart = part1 * part3;		// imaginary part

					imgMaskReal.at<double>(i, j) = realPart;
					imgMaskIm.at<double>(i, j) = imPart;
				}
			}

#define GABOR_MODE 0

#if GABOR_MODE == 1
			filter2D(*imgSrc, imgReal, CV_32F, imgMaskReal, Point((mask_width - 1) / 2, (mask_width - 1) / 2));		// real filter
			Mat imgDst(imgReal);
#elif GABOR_MODE == 2
			filter2D(*imgSrc, imgIm, CV_32F, imgMaskIm, Point((mask_width - 1) / 2, (mask_width - 1) / 2));			// imaginary filter
			Mat imgDst(imgIm);
#else
			// magnitude filter
			filter2D(*imgSrc, imgReal, CV_32F, imgMaskReal, Point((mask_width - 1) / 2, (mask_width - 1) / 2));
			filter2D(*imgSrc, imgIm, CV_32F, imgMaskIm, Point((mask_width - 1) / 2, (mask_width - 1) / 2));

			cv::pow(imgReal, 2, imgReal);
			cv::pow(imgIm, 2, imgIm);
			cv::add(imgReal, imgIm, imgMac);
			cv::pow(imgMac, 0.5, imgMac);
			Mat imgDst(imgMac);
#endif
			Mat imgGabor;
			normalize(imgDst, imgDst, 255, 0, NORM_MINMAX);	// normalize value of pixel from 0 to 255
			convertScaleAbs(imgDst, imgGabor, 1, 0 );		// 8-bit unsigned integers

			featImg->gaborfeat.push_back(imgGabor);			// push back the gabor feature

			//imshow("gabor response", imgGabor); waitKey(0);	// show the image

		} // end of for
	} // end of for
} // end of function
