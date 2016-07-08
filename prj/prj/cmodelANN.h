#ifndef __CMODEL_ANN_H
#define __CMODEL_ANN_H

#include "cmodelInt.h"

class CvANN_MLP;

class CModelANN : public CModelInt
{
	/* ctor and de-ctor */
public:
	CModelANN(double _scale=0.001, int _hiddenSize=10, int type=CvANN_MLP::GAUSSIAN, int max_iter=1e3, double epsilon=1e-20);
	~CModelANN();
	/* interface */
public:
	void train( const std::vector<CFeature> &feaSet , const std::vector<float> &labSet ) override;
	void train( const cv::Mat &trains, const cv::Mat &labels) override;			//输入特征集和标签集，训练模型
	double similarity( const CFeature &fea ) override;
	void saveModel(std::string model_file) override;
	void loadModel(std::string model_file) override;

	/* member var */
private:
	CvANN_MLP *ANN;
	double scale;
	int hiddenSize;
	int type;
	int max_iter;
	double epsilon;
};

#endif