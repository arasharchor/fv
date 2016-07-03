#ifndef _IOFILE_H
#define _IOFILE_H

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <io.h>
#include <sstream>
#include <iomanip>
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

	iofile(std::string dataListFile);

	//返回正、负样本数
	int posCoupleNums();
	int negCoupleNums();

	// 提取第n个样本信息
	void extCoupleImageInf(coupleImageInf &inf, int nth);

	// 读、写第n个样本特征	[true->存在	false->不存在]
	bool readFeature(std::vector<double> &feat, int nth);
	void writeFeature(std::vector<double> &feat, int nth);

	// 输出错误到日志
	void writeErrorLog(const errLogInf &errLog);
    void readErrorLog(errLogInf &errLog, int nth);

    // 文件输出
    void outputSimilarFile(const std::vector<float> similSet);
    void outputRocFile(const std::vector<std::pair<float,float> > rocSet);

private:

	int posCoupleSize;					//正样本数
	int negCoupleSize;					//负样本数

	int lineLength;						//列表中一行的长度

	std::string dataList;				//列表文件名

	std::vector<int> subStringPos;		//截取位置

};

#endif

