#ifndef _IOFILE_H
#define _IOFILE_H

#include <iostream>
#include <fstream>
#include <vector>
#include <io.h>

typedef struct{
	std::string path1;
	std::string path2;
}imgPath;

class iofile
{
public:
	iofile(int n, int m);
	void coupleImg_org();		// 组织图像
	void posCoupleImg_org();	// 组织正样本对
	void negCoupleImg_org();	// 组织负样本对
	void extCoupleImg_path();	// 提取正、负样本对路径

	void extCoupleImg_path(std::string &path1, std::string &path2, int nth, bool type);

private:
	int N;						// N个人
	int M;						// 每个人M张人脸图像
	int posCoupleSize;
	int negCoupleSize;
	std::vector<imgPath> posImg;			// 正样本图像对路径
	std::vector<imgPath> negImg;			// 负样本图像对路径
	std::vector<std::string> allPath;		// 所有图像的路径
};

#endif

