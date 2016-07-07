#ifndef __CMODEL_H
#define __CMODEL_H

#include <vector>
#include <string>
#include <assert.h>
#include <opencv2\ml\ml.hpp>
#include <opencv2\core\core.hpp>
#include "cfeature.h"

class CFeature;

class CModelInt
{
	/* ctor and de-ctor */
public:
	virtual ~CModelInt(){}

	/* interface */
public:
	virtual void train( const std::vector<CFeature> &feaSet, const std::vector<float> &labSet )=0;			//输入特征集和标签集，训练模型
	virtual void validation_model( const std::vector<CFeature> &feaSet , const std::vector<float> &labSet )=0;
	virtual double similarity( const CFeature &fea )=0;								//相似性计算
	virtual void saveModel(std::string model_file)=0;
	virtual void loadModel(std::string model_file)=0;

protected:
	void _loadTrain( cv::Mat &trainData, const std::vector<CFeature> &feaSet )
	{
		for(int i=0; i<feaSet.size(); i++)
		{
			for(int j=0; j<feaSet[i].mFeatureMode.mixfeat.size(); j++)
			{
				trainData.at<float>(i, j) = feaSet[i].mFeatureMode.mixfeat[j];
			}
		}
	}
	void _loadLabel( cv::Mat &labelData, const std::vector<float> &labSet )
	{
		for(int i=0; i<labSet.size(); i++)
		{
			labelData.at<float>(i) = labSet.at(i);
		}
	}
};

#endif