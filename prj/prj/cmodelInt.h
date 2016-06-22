#ifndef __CMODEL_H
#define __CMODEL_H

#include <vector>
#include <string>
#include <assert.h>
#include <opencv2\ml\ml.hpp>
#include <opencv2\core\core.hpp>

class CFeature;

class CModelInt
{
	/* ctor and de-ctor */
public:
	virtual ~CModelInt(){}

	/* interface */
public:
	virtual void train( const std::vector<CFeature> &feaSet, const std::vector<float> &labSet )=0;			//�����������ͱ�ǩ����ѵ��ģ��
	virtual void validation_model( const std::vector<CFeature> &feaSet , const std::vector<float> &labSet )=0;
	virtual double similarity( const CFeature &fea )=0;								//�����Լ���
	virtual void saveModel(std::string model_file)=0;
	virtual void loadModel(std::string model_file)=0;
};

#endif