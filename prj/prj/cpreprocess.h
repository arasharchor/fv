#ifndef __CPREPROCESS_H
#define __CPREPROCESS_H
#pragma once

#include "cpreprocessInt.h"
#include <stdio.h>
#include <iostream>
#include <vector>
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

class CPreprocess : public CPreprocessInt
{
	/* ctor and de-ctor */
public:

	/* interface */
public:
	CPreprocess();
	void doit( ImgWrap *imgWrapSrc ) override;

	/* member fun */
private:
	void _do( ImgWrap *imgWrapSrc);

	void _detectObjectsCustom(Mat &img, CascadeClassifier &cascade, vector<Rect> &objects, int scaledWidth, int flags, Size minFeatureSize, float searchScaleFactor, int minNeighbors);
	void _detectLargestObject(Mat &img, CascadeClassifier &cascade, Rect &largestObject, int scaledWidth = 320);//寻找一个图像的特征
	//void _detectManyObjects(const ImgWrap *imgWrapSrc, CascadeClassifier &cascade, vector<Rect> &objects, int scaledWidth = 320);//寻找多个图像的特征
	void _drawFaceImage(const Mat img, vector<Rect> objects);
	void _drawFaceImage(const Mat img, Rect largestObject);
	//添加函数；
	/* member var */
private:
	CascadeClassifier *classifier;
	vector<Rect> objects;
	Rect largestObject;
	int scaledWidth;
	int flags;
	Size minFeatureSize;
	float searchScaleFactor;
	int minNeighbors;
};

#endif