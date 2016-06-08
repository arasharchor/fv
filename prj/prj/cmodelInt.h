#ifndef __CMODEL_H
#define __CMODEL_H

#include <vector>
#include <string>

class CFeature;

class CModelInt
{
	/* interface */
public:
	virtual void train( const std::vector<CFeature> &feaSet )=0;					//������������ѵ��ģ��
	virtual double similarity( const CFeature &fea )=0;								//�����Լ���
	virtual void saveModel(std::string model_file)=0;
	virtual void loadModel(std::string model_file)=0;
};

#endif