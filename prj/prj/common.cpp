
#include "common.h"

using namespace std;

// 将int转换成string 
string itos(int i)
{
	stringstream s;
	s << i;
	return s.str();
}

void CalFPR_TPR(float &FPR, float &TPR, const std::vector<float> &labelSet, const std::vector<float> &similSet, const float hold)
{
	int pos_number = 0, neg_number = 0;

	FPR = 0;
	TPR = 0;

	//计算正负样本个数
	for(int i=0; i<labelSet.size(); i++)
	{
		if(labelSet[i] == 1.0)
		{
			pos_number++;
		}
		else
		{
			neg_number++;
		}
	}

	// FPR and TPR
	// 负样本的错误率 以及 正样本的正确率
	for(int i=0; i<labelSet.size(); i++)
	{
		float yi = similSet[i]>=hold ? 1.0 : -1.0;
		if ((yi == 1.0) && (labelSet[i] == -1.0))	//实际不是同一个人，却认为是同一个人 的样本数
		{
			FPR += 1;
		}

		if ((yi == 1.0) && (labelSet[i] == 1.0))	//实际是同一个人，也认为是同一个人 的样本数
		{
			TPR += 1;
		}
	}

	FPR = FPR / neg_number;
	TPR = TPR / pos_number;
}