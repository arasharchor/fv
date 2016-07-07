#pragma warning(disable:4996)

#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <opencv.hpp>


#include "common.h"
#include "cmodelSVM.h"
#include "cmodelANN.h"
#include "cfeature.h"

#include "iofile.h"

using namespace std;
using namespace cv;

//const int predNum = 6000;
//const int jumpNum = 1000;

const int predNum = 1200;
const int jumpNum = 3600;

void expect(void)
{
	iofile imgCoupleDataSet("datalist.txt");

	vector<CFeature>				featureSet(predNum);    //特征集
	vector<float>					labelSet(predNum);	    //标签集
	vector<float>					similSet(predNum);	    //相似度集
	vector<pair<float,float> >		rocSet(100);            //fpr tpr集
	vector<float>					rightRate(100);

	//1).提取测试样本特征，并得到对应标签

    for(int i = 0; i < predNum / 2; i++)
    {
        featureSet[i] = CFeature(imgCoupleDataSet, jumpNum + i, true);
        labelSet[i] = 1;
        printf("finish %d\n", i + jumpNum);
    }
    for(int i = 0; i < predNum / 2; i++)
    {
        featureSet[i + predNum / 2] = CFeature(imgCoupleDataSet, jumpNum + i, false);
        labelSet[i + predNum / 2] = -1.0;
        printf("finish %d\n", i + jumpNum);
    }
	//2).加载识别模型
	CModelInt *model = new CModelSVM();
	model->loadModel("svm_model");
	//CModelInt *model = new CModelANN();
	//model->loadModel("ann_model");

	//3).计算所有特征的相似度
	for(int i=0; i<predNum; i++)
	{
		printf("%d\n", i);
		similSet[i] = model->similarity(featureSet[i]);
	}

	//4).计算所有hold对应的FPR TPR
	for(int i=0; i<100; i+=1)
	{
		CalFPR_TPR(rocSet[int(i)].first, rocSet[int(i)].second, rightRate[i], labelSet, similSet, i*0.01);
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
    imgCoupleDataSet.outputSimilarFile(similSet);
    imgCoupleDataSet.outputRocFile(rocSet);

	delete model;
}
