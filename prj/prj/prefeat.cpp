#include "iofile.h"
#include "cfeature.h"

using namespace std;
using namespace cv;


// ��ȡ���ݼ���������������������һ�������浽�ļ�

void prefeat(void)
{
    iofile obj("datalist.txt",            // ���ݼ�
                "Dataset.ZET",             // ������
                "��ң_Distance.txt",         // ���ƶ�
                "��ң_ROC.txt",              // ROC
                "errInf.log"                 // ��־
                );

    int totalNums = obj.posSamplesNums() + obj.negSamplesNums();

    // 1).��ȡ��������������
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