#ifndef _IOFILE_H
#define _IOFILE_H

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <iomanip>
#include <cv.h>

//样本图像
struct coupleImageInf
{
	std::string imgPath1;		// 样本的第1幅图片
	std::string imgPath2;		// -------2-----
	int label;					// 样本标签
};
// 日志打包
struct errLogInf
{
    enum{imgNoErr = 0, img1Err, img2Err, imgAllErr};
	int errOrder;				// 样本序号
	int errImg;					// 出错的图像
	coupleImageInf errInf;		// 其他样本信息
};

class iofile
{
public:

    iofile::iofile(std::string datalist, 
        std::string featlist,
        std::string distfile,
        std::string rocfile,
        std::string errfile
        );

	// 正、负样本数
	int posSamplesNums(void);
	int negSamplesNums(void);

    int rowNums(void);
    int cloNums(void);
    // 数据归一化
    void dataNormalize(cv::Mat &featdata);

	// 提取第n个样本路径信息
	void load(coupleImageInf &inf, int nth);

	// 读、写第n个样本特征	[true->存在	false->不存在]
	bool load(std::vector<double> &feat, int nth);
	void save(std::vector<double> &feat, int nth);

    // 载入样本特征
    void load(cv::Mat &featdata);
    void load(cv::Mat &featdata, int trainNums, int jumpNums);

	// 输出错误到日志
	void save(const errLogInf &errLog);
    void load(errLogInf &errLog, int nth);

    // 文件输出
    void save(const std::vector<float> similSet);
    void save(const std::vector<std::pair<float,float> > rocSet);
    void save(const cv::Mat &featureSet);

private:

	int posNums;					    // 正样本数
	int negNums;					    // 负样本数

	int lineLength;						// 列表中一行的长度
	std::vector<int> colonPos;		    // ':'的位置

	std::string dataList;				// 数据集文件
    std::string featList;               // 特征文件
    std::string errorLog;               // 日志文件
    std::string distFile;               // 距离、相似度
    std::string rocFile;                // ROC文件
};

#endif

