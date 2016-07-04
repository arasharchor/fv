
#pragma warning(disable:4996)

#include <iostream>
#include <vector>
#include <string>
#include <opencv.hpp>


#include "common.h"
#include "cmodelSVM.h"
#include "cfeature.h"
#include "iofile.h"

static int TRAIN_NUM = 2;			//训练图像对个数

using namespace std;
using namespace cv;

void train(void)
{
    iofile imgCoupleDataSet("datalist.txt");

    vector<CFeature> featureSet(TRAIN_NUM);	//特征集
    vector<float> labelSet(TRAIN_NUM);
    // 1).提取所有图像的特征
    for(int i = 0; i < TRAIN_NUM / 2; i++)
    {
        featureSet[i] = CFeature(imgCoupleDataSet, i, true);
        labelSet[i] = 1;
        printf("finish %d\n", i);
        //showMemoryInfo();
    }
    for(int i = 0; i < TRAIN_NUM / 2; i++)
    {
        featureSet[i + TRAIN_NUM / 2] = CFeature(imgCoupleDataSet, i,false);
        labelSet[i + TRAIN_NUM / 2] = -1.0;
        printf("finish %d\n", i + TRAIN_NUM / 2);
        //showMemoryInfo();
    }
    //2).训练模型
    
    CModelInt *model = new CModelSVM();
    model->train(featureSet, labelSet);
    model->saveModel("svm_model");
}