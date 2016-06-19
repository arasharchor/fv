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
	void coupleImg_org();		// ��֯ͼ��
	void posCoupleImg_org();	// ��֯��������
	void negCoupleImg_org();	// ��֯��������
	void extCoupleImg_path();	// ��ȡ������������·��

	void extCoupleImg_path(std::string &path1, std::string &path2, int nth, bool type);

private:
	int N;						// N����
	int M;						// ÿ����M������ͼ��
	int posCoupleSize;
	int negCoupleSize;
	std::vector<imgPath> posImg;			// ������ͼ���·��
	std::vector<imgPath> negImg;			// ������ͼ���·��
	std::vector<std::string> allPath;		// ����ͼ���·��
};

#endif

