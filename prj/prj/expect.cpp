#pragma warning(disable:4996)

#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <opencv.hpp>


#include "common.h"
#include "cmodelSVM.h"
#include "cfeature.h"
#include "wrap.h"

#include "iofile.h"

using namespace std;
using namespace cv;

void expect(void)
{
	iofile coupleImgDataSet("datalist.txt");
	int predNum=20;

	vector<CFeature>			featureSet(predNum);	//特征集
	vector<float>				labelSet(predNum);		//标签集
	vector<float>				similSet(predNum);		//相似度集
	vector<pair<float,float>>	rocSet(100);			//fpr tpr集

	//1).提取测试样本特征，并得到对应标签
	for(int i=0; i<predNum/2; i++)
	{
		tuple<string, string> path;

		coupleImgDataSet.extCoupleImg_path(path, i, true);						// 第i对正样本

		Mat img1 = imread(get<0>(path), IMREAD_GRAYSCALE);
		Mat img2 = imread(get<1>(path), IMREAD_GRAYSCALE);

		featureSet[i] =  CFeature(&ImgWrap(&img1), &ImgWrap(&img2));
		labelSet[i] = 1.0;
	}

	for(int i=0; i<predNum/2; i++)
	{
		tuple<string, string> path;

		coupleImgDataSet.extCoupleImg_path(path, i, false);						// 第i对正样本

		Mat img1 = imread(get<0>(path), IMREAD_GRAYSCALE);
		Mat img2 = imread(get<1>(path), IMREAD_GRAYSCALE);

		featureSet[i+predNum/2] =  CFeature(&ImgWrap(&img1), &ImgWrap(&img2));
		labelSet[i+predNum/2] = -1.0;
	}

	//2).加载识别模型
	CModelInt *model = new CModelSVM();
	model->loadModel("svm_model");

	//3).计算所有特征的相似度
	for(int i=0; i<predNum; i++)
	{
		similSet[i] = model->similarity(featureSet[i]);
	}

	//4).计算所有hold对应的FPR TPR
	for(int i=0; i<100; i+=1)
	{
		CalFPR_TPR(rocSet[int(i)].first, rocSet[int(i)].second, labelSet, similSet, i*0.01);
	}

	//5).生成输出文件
	/*
	数据说明:
	  1).similSet数据用以构造相似性文件(因为不管阈值是多少，相似性都是那么大)
	     最佳阈值可以用0.5或者是低FPR且高TPR的阈值
	  2).rocSet数据用以构造ROC文本
	     rocSet[i]指阈值为i*0.01时的FPR与TPR，rocSet[i]是个pair<float, float>数据，first是FPR在ROC中是横轴，second是TPR在ROC中是纵轴
	     FPR可以理解为“负样本识别错误率”，TPR可以理解为“正样本识别正确率”
	*/
}