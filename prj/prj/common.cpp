
#include "common.h"

using namespace std;

// ��intת����string 
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

	//����������������
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
	// �������Ĵ����� �Լ� ����������ȷ��
	for(int i=0; i<labelSet.size(); i++)
	{
		float yi = similSet[i]>=hold ? 1.0 : -1.0;
		if ((yi == 1.0) && (labelSet[i] == -1.0))	//ʵ�ʲ���ͬһ���ˣ�ȴ��Ϊ��ͬһ���� ��������
		{
			FPR += 1;
		}

		if ((yi == 1.0) && (labelSet[i] == 1.0))	//ʵ����ͬһ���ˣ�Ҳ��Ϊ��ͬһ���� ��������
		{
			TPR += 1;
		}
	}

	FPR = FPR / neg_number;
	TPR = TPR / pos_number;
}