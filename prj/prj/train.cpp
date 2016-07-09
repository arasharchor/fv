
#pragma warning(disable:4996)

#include <iostream>
#include <vector>
#include <string>
#include <opencv.hpp>


#include "common.h"
#include "cmodelSVM.h"
#include "cmodelANN.h"
#include "cfeature.h"
#include "iofile.h"

static int trainNums = 7200;			// 训练样本数
static int jumpNums = 0;               // 跳过

using namespace std;
using namespace cv;

void train(void)
{
    iofile obj("datalist.FERET",            // 数据集
                "Dataset.FERET",             // 特征集
                "逍遥_Distance.txt",         // 相似度
                "逍遥_ROC.txt",              // ROC
                "errInf.log"                // 日志
                );

    Mat labelSet;                           // 标签集

    // 1).提取所有图像的特征
    for(int i = 0; i < trainNums / 2; i++)
    {
        labelSet.push_back(1.0);
    }
    for(int i = 0; i < trainNums / 2; i++)
    {
        labelSet.push_back(-1.0);
    }
    //labelSet = labelSet.t();              // 列向量—>行向量

    Mat featureSet;                                 //特征集
    obj.load(featureSet, trainNums, jumpNums);      // 载入

    //2).训练模型
    //CModelInt *model = new CModelSVM();
    //model->train(featureSet, labelSet);
    //model->saveModel("svm_model");
    
	CModelInt *model = new CModelANN(0.001, 10, 2);
	model->train(featureSet, labelSet);
	model->saveModel("ann_model");
	delete model;
}