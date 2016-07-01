#include "iofile.h"

using namespace std;


iofile::iofile(string datalist)
{
	pathList = datalist;
	posCoupleSize = 0;
	negCoupleSize = 0;

	string str;
	ifstream dataList(pathList);

	getline(dataList, str);
	lineLength = str.length();

	for (size_t i = 0; i < str.length(); ++i)
	{
		if (str[i] == ':')
		{
			subStringPos.push_back(i);
		}
	}

	if (str[subStringPos[1] + 1] == '1')	// label
	{
		posCoupleSize++;
	}
	else
	{
		negCoupleSize++;
	}

	while(!dataList.eof())
	{
		getline(dataList, str);
		if (str.length() == lineLength)
		{
			if (str[subStringPos[1] + 1] == '1')	// label
			{
				posCoupleSize++;
			}
			else
			{
				negCoupleSize++;
			}	
		}
	}

	dataList.close();
}

void iofile::extCoupleImg_path(tuple<string, string> &path, int nth, bool type)
{
	string coupleImg_path;

	ifstream pnFile(pathList);

	if (type)
	{
		pnFile.seekg(nth * (lineLength + 2));
	}
	else
	{
		pnFile.seekg((nth + posCoupleSize) * (lineLength + 2));
	}

	string str;

	pnFile >> str;
	path = make_tuple(str.substr(0, subStringPos[0]), str.substr(subStringPos[0] + 1, subStringPos[0]));
}

void iofile::writeFeatFile(string fileName, int nth, const std::vector<double> &data)
{
	// ÌØÕ÷¼¯
	ofstream dataFile(fileName, ios::app);	// 'w+'
	dataFile << nth << endl;
	for (size_t i = 0; i < data.size() - 1; ++i)
	{
		dataFile << data[i] << '\t';
	}
	dataFile << data.back() << endl;
	dataFile.close();
}

int iofile::posCoupleNums()
{
	return posCoupleSize;
}

int iofile::negCoupleNums()
{
	return negCoupleSize;
}
