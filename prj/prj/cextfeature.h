#ifndef __CEXTFEAT_H
#define __CEXTFEAT_H

#include "cextfeatInt.h"

class CExtfeature : public CExtfeatInt
{
	/* ctor and de-ctor */
public:

	/* interface */
public:
	void doit( const ImgWrap *imgWrapSrc, CFeatureStore *featStore ) override;
	/* member fun */
private:
	void _do( const ImgWrap *imgWrapSrc, CFeatureStore *featStore );
	void _cextlbp(const ImgWrap *imgWrapSrc, CFeatureStore *featStore);				// lbp
	void _cextlbp(const ImgWrap *imgWrapSrc, CFeatureStore *featStore, int scale);		// mb-lbp

	void _cextsift(const ImgWrap *imgWrapSrc, CFeatureStore *featStore);				// sift

	/* member var */
private:
};

#endif
