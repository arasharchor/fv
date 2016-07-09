#include "iofile.h"
#include "cfeature.h"

using namespace std;
using namespace cv;


// 提取数据集中所有样本的特征，归一化并保存到文件

void prefeat(void)
{
    iofile obj("datalist.FERET",            // 数据集
                "Dataset.FERET",             // 特征集
                "逍遥_Distance.txt",         // 相似度
                "逍遥_ROC.txt",              // ROC
                "errInf.log"                 // 日志
                );

    int totalNums = obj.posSamplesNums() + obj.negSamplesNums();

    // 1).提取所有样本的特征
    for(int i = 0; i < totalNums / 2; i++)
    {
        CFeature(obj, i, true);
        printf("finish %d\n", i);
    }
    for(int i = 0; i < totalNums / 2; i++)
    {
        CFeature(obj, i, false);
        printf("finish %d\n", i + totalNums / 2);
    }

    Mat featureSet;

    obj.load(featureSet);
    obj.dataNormalize(featureSet);
    obj.save(featureSet);
}