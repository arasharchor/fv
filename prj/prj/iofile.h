#ifndef _IOFILE_H
#define _IOFILE_H

#include <iostream>
#include <fstream>
#include <string>
#include <tuple>
#include <vector>
#include <io.h>

class iofile
{
public:
	iofile(std::string datalist);

	int posCoupleNums();
	int negCoupleNums();
	void extCoupleImg_path(std::tuple<std::string, std::string> &path, int nth, bool type);
	void writeFeatFile(std::string fileName, int nth, const std::vector<double> &data);

private:
	int lineLength;
	std::string pathList;
	std::vector<int> subStringPos;
	int posCoupleSize;
	int negCoupleSize;
};

#endif

