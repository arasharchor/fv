#ifndef __CMODEL_H
#define __CMODEL_H

#include <vector>
#include <string>

class CFeature;

class CModelInt
{
	/* interface */
public:
	virtual void train( const std::vector<CFeature> &feaSet )=0;					//输入特征集，训练模型
	virtual double similarity( const CFeature &fea1, const CFeature &fea2 )=0;		//相似性计算
	virtual void storeModel(std::string model_file)=0;
	virtual void readModel(std::string model_file)=0;
};

#endif