#ifndef __CPREPROCESS_H
#define __CPREPROCESS_H
#pragma once

#include <iostream>
#include <vector>

#include <opencv2/opencv.hpp>

#include "cpreprocessInt.h"


class CPreprocess : public CPreprocessInt
{
	/* ctor and de-ctor */
public:

	/* interface */
public:
	CPreprocess();
	bool doit( ImgWrap *imgWrapSrc ) override;

	/* member fun */
private:
	bool _do( ImgWrap *imgWrapSrc);

	bool _detectObjectsCustom(cv::Mat &img);
	bool _detectObjectsCustom(cv::Mat &img, cv::CascadeClassifier &cascade, std::vector<cv::Rect> &objects, int scaledWidth, int flags, cv::Size minFeatureSize, float searchScaleFactor, int minNeighbors);
	void _detectLargestObject(cv::Mat &img, cv::CascadeClassifier &cascade, cv::Rect &largestObject, int scaledWidth = 320);//寻找一个图像的特征
	//void _detectManyObjects(const ImgWrap *imgWrapSrc, CascadeClassifier &cascade, vector<Rect> &objects, int scaledWidth = 320);//寻找多个图像的特征
	void _drawFaceImage(const cv::Mat img, std::vector<cv::Rect> objects);
	void _drawFaceImage(const cv::Mat img, cv::Rect largestObject);

	//添加函数；
	/* member var */
private:
	cv::CascadeClassifier *classifier;
	std::vector<cv::Rect> objects;
	cv::Rect largestObject;
	int scaledWidth;
	int flags;
	cv::Size minFeatureSize;
	float searchScaleFactor;
	int minNeighbors;
};

#endif