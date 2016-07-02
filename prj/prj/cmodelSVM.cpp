#include "cmodelSVM.h"
#include "cfeature.h"
#include <assert.h>

using namespace std;
using namespace cv;

typedef struct CvSparseVecElem32f
{
    int idx;
    float val;
}
CvSparseVecElem32f;



static int icvCmpSparseVecElems( const void* a, const void* b )
{
    return ((CvSparseVecElem32f*)a)->idx - ((CvSparseVecElem32f*)b)->idx;
}

void
cvPreparePredictData( const CvArr* _sample, int dims_all,
                      const CvMat* comp_idx, int class_count,
                      const CvMat* prob, float** _row_sample,
                      int as_sparse CV_DEFAULT(0))
{
    float* row_sample = 0;
    int* inverse_comp_idx = 0;

    CV_FUNCNAME( "cvPreparePredictData" );

    __BEGIN__;

    const CvMat* sample = (const CvMat*)_sample;
    float* sample_data;
    int sample_step;
    int is_sparse = CV_IS_SPARSE_MAT(sample);
    int d, sizes[CV_MAX_DIM];
    int i, dims_selected;
    int vec_size;

    if( !is_sparse && !CV_IS_MAT(sample) )
        CV_ERROR( !sample ? CV_StsNullPtr : CV_StsBadArg, "The sample is not a valid vector" );

    if( cvGetElemType( sample ) != CV_32FC1 )
        CV_ERROR( CV_StsUnsupportedFormat, "Input sample must have 32fC1 type" );

    CV_CALL( d = cvGetDims( sample, sizes ));

    if( !((is_sparse && d == 1) || (!is_sparse && d == 2 && (sample->rows == 1 || sample->cols == 1))) )
        CV_ERROR( CV_StsBadSize, "Input sample must be 1-dimensional vector" );

    if( d == 1 )
        sizes[1] = 1;

    if( sizes[0] + sizes[1] - 1 != dims_all )
        CV_ERROR( CV_StsUnmatchedSizes,
        "The sample size is different from what has been used for training" );

    if( !_row_sample )
        CV_ERROR( CV_StsNullPtr, "INTERNAL ERROR: The row_sample pointer is NULL" );

    if( comp_idx && (!CV_IS_MAT(comp_idx) || comp_idx->rows != 1 ||
        CV_MAT_TYPE(comp_idx->type) != CV_32SC1) )
        CV_ERROR( CV_StsBadArg, "INTERNAL ERROR: invalid comp_idx" );

    dims_selected = comp_idx ? comp_idx->cols : dims_all;

    if( prob )
    {
        if( !CV_IS_MAT(prob) )
            CV_ERROR( CV_StsBadArg, "The output matrix of probabilities is invalid" );

        if( (prob->rows != 1 && prob->cols != 1) ||
            (CV_MAT_TYPE(prob->type) != CV_32FC1 &&
            CV_MAT_TYPE(prob->type) != CV_64FC1) )
            CV_ERROR( CV_StsBadSize,
            "The matrix of probabilities must be 1-dimensional vector of 32fC1 type" );

        if( prob->rows + prob->cols - 1 != class_count )
            CV_ERROR( CV_StsUnmatchedSizes,
            "The vector of probabilities must contain as many elements as "
            "the number of classes in the training set" );
    }

    vec_size = !as_sparse ? dims_selected*sizeof(row_sample[0]) :
                (dims_selected + 1)*sizeof(CvSparseVecElem32f);

    if( CV_IS_MAT(sample) )
    {
        sample_data = sample->data.fl;
        sample_step = CV_IS_MAT_CONT(sample->type) ? 1 : sample->step/sizeof(row_sample[0]);

        if( !comp_idx && CV_IS_MAT_CONT(sample->type) && !as_sparse )
            *_row_sample = sample_data;
        else
        {
            CV_CALL( row_sample = (float*)cvAlloc( vec_size ));

            if( !comp_idx )
                for( i = 0; i < dims_selected; i++ )
                    row_sample[i] = sample_data[sample_step*i];
            else
            {
                int* comp = comp_idx->data.i;
                for( i = 0; i < dims_selected; i++ )
                    row_sample[i] = sample_data[sample_step*comp[i]];
            }

            *_row_sample = row_sample;
        }

        if( as_sparse )
        {
            const float* src = (const float*)row_sample;
            CvSparseVecElem32f* dst = (CvSparseVecElem32f*)row_sample;

            dst[dims_selected].idx = -1;
            for( i = dims_selected - 1; i >= 0; i-- )
            {
                dst[i].idx = i;
                dst[i].val = src[i];
            }
        }
    }
    else
    {
        CvSparseNode* node;
        CvSparseMatIterator mat_iterator;
        const CvSparseMat* sparse = (const CvSparseMat*)sample;
        assert( is_sparse );

        node = cvInitSparseMatIterator( sparse, &mat_iterator );
        CV_CALL( row_sample = (float*)cvAlloc( vec_size ));

        if( comp_idx )
        {
            CV_CALL( inverse_comp_idx = (int*)cvAlloc( dims_all*sizeof(int) ));
            memset( inverse_comp_idx, -1, dims_all*sizeof(int) );
            for( i = 0; i < dims_selected; i++ )
                inverse_comp_idx[comp_idx->data.i[i]] = i;
        }

        if( !as_sparse )
        {
            memset( row_sample, 0, vec_size );

            for( ; node != 0; node = cvGetNextSparseNode(&mat_iterator) )
            {
                int idx = *CV_NODE_IDX( sparse, node );
                if( inverse_comp_idx )
                {
                    idx = inverse_comp_idx[idx];
                    if( idx < 0 )
                        continue;
                }
                row_sample[idx] = *(float*)CV_NODE_VAL( sparse, node );
            }
        }
        else
        {
            CvSparseVecElem32f* ptr = (CvSparseVecElem32f*)row_sample;

            for( ; node != 0; node = cvGetNextSparseNode(&mat_iterator) )
            {
                int idx = *CV_NODE_IDX( sparse, node );
                if( inverse_comp_idx )
                {
                    idx = inverse_comp_idx[idx];
                    if( idx < 0 )
                        continue;
                }
                ptr->idx = idx;
                ptr->val = *(float*)CV_NODE_VAL( sparse, node );
                ptr++;
            }

            qsort( row_sample, ptr - (CvSparseVecElem32f*)row_sample,
                   sizeof(ptr[0]), icvCmpSparseVecElems );
            ptr->idx = -1;
        }

        *_row_sample = row_sample;
    }

    __END__;

    if( inverse_comp_idx )
        cvFree( &inverse_comp_idx );

    if( cvGetErrStatus() < 0 && _row_sample )
    {
        cvFree( &row_sample );
        *_row_sample = 0;
    }
}


float predict_( const CvSVM *svm, const CvMat* sample, bool returnDFVal )
{
    float result = 0;
    float* row_sample = 0;

    CV_FUNCNAME( "CvSVM::predict" );

    __BEGIN__;

    int class_count;

    if( !svm->kernel )
        CV_ERROR( CV_StsBadArg, "The SVM should be trained first" );

    class_count = svm->class_labels ? svm->class_labels->cols :
                  svm->params.svm_type == CvSVM::ONE_CLASS ? 1 : 0;

    CV_CALL( cvPreparePredictData( sample, svm->var_all, svm->var_idx,
                                   class_count, 0, &row_sample ));
    result = predict__( svm, row_sample, svm->get_var_count(), returnDFVal );

    __END__;

    if( sample && (!CV_IS_MAT(sample) || sample->data.fl != row_sample) )
        cvFree( &row_sample );

    return result;
}

float predict__( const CvSVM *svm, const float* row_sample, int row_len, bool returnDFVal )
{
    assert( svm->kernel );
    assert( row_sample );

    int var_count = svm->get_var_count();
    assert( row_len == var_count );
    (void)row_len;

    int class_count = svm->class_labels ? svm->class_labels->cols :
                  svm->params.svm_type == CvSVM::ONE_CLASS ? 1 : 0;

    float result = 0;
    cv::AutoBuffer<float> _buffer(svm->sv_total + (class_count+1)*2);
    float* buffer = _buffer;

    if( svm->params.svm_type == CvSVM::EPS_SVR ||
        svm->params.svm_type == CvSVM::NU_SVR ||
        svm->params.svm_type == CvSVM::ONE_CLASS )
    {
        CvSVMDecisionFunc* df = (CvSVMDecisionFunc*)svm->decision_func;
        int i, sv_count = df->sv_count;
        double sum = -df->rho;

        svm->kernel->calc( sv_count, var_count, (const float**)svm->sv, row_sample, buffer );
        for( i = 0; i < sv_count; i++ )
            sum += buffer[i]*df->alpha[i];

        result = svm->params.svm_type == CvSVM::ONE_CLASS ? (float)(sum > 0) : (float)sum;
    }
    else if( svm->params.svm_type == CvSVM::C_SVC ||
             svm->params.svm_type == CvSVM::NU_SVC )
    {
//		svm->decision_func->rho;
        CvSVMDecisionFunc* df = (CvSVMDecisionFunc*)svm->decision_func;
        int* vote = (int*)(buffer + svm->sv_total);
        int i, j, k;

        memset( vote, 0, class_count*sizeof(vote[0]));
        svm->kernel->calc( svm->sv_total, var_count, (const float**)svm->sv, row_sample, buffer );
        double sum = 0.;

        for( i = 0; i < class_count; i++ )
        {
            for( j = i+1; j < class_count; j++, df++ )
            {
                sum = -df->rho;
                int sv_count = df->sv_count;
                for( k = 0; k < sv_count; k++ )
                    sum += df->alpha[k]*buffer[df->sv_index[k]];

                vote[sum > 0 ? i : j]++;
            }
        }

        for( i = 1, k = 0; i < class_count; i++ )
        {
            if( vote[i] > vote[k] )
                k = i;
        }
        result = returnDFVal && class_count == 2 ? (float)sum : (float)(svm->class_labels->data.i[k]);
    }
    else
        CV_Error( CV_StsBadArg, "INTERNAL ERROR: Unknown SVM type, "
                                "the SVM structure is probably corrupted" );

    return result;
}

float _predict( CvSVM *svm, const Mat& _sample, bool returnDFVal )
{
    CvMat sample = _sample;
    return predict_(svm, &sample, returnDFVal);
}

CModelSVMStore::~CModelSVMStore()
{
	if(SVM)
	{
		delete SVM;
		SVM = NULL;
	}
}

void CModelSVM::train(const std::vector<CFeature> &feaSet, const std::vector<float> &labSet)
{
	//---------------------------------0. set train data and labels------------------------------
	assert( feaSet.size()==labSet.size() && feaSet.size() );

	int trainSize = feaSet.size();
	int featureSize = feaSet[0].mFeatureMode.mixfeat.size();

	Mat trains(trainSize, featureSize, CV_32FC1);
	Mat labels(trainSize, 1, CV_32FC1);


	_loadTrain(trains, feaSet);
	_loadLabel(labels, labSet);

	//---------------------------------1. set SVM parameters-------------------------------------
	CvSVMParams params;

	params.svm_type = CvSVM::C_SVC;
	params.C = 0.1;
	params.kernel_type = CvSVM::LINEAR;
	params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 1e4, 1e-6);

	//---------------------------------2. train--------------------------------------------------
	mModelStore.SVM->train(trains, labels, Mat(), Mat(), params);
}

void CModelSVM::validation_model( const std::vector<CFeature> &feaSet , const std::vector<float> &labSet )
{
	assert( feaSet.size()==labSet.size() && feaSet.size() );

	float accuracy=0;

	cout<<"respone\tlabel"<<endl;
	for(int i=0; i<feaSet.size(); i++)
	{
		Mat sampleMat(1, feaSet[i].mFeatureMode.mixfeat.size(), CV_32FC1);
		
		for(int j=0; j<feaSet[i].mFeatureMode.mixfeat.size(); j++)
			sampleMat.at<float>(j) = feaSet[i].mFeatureMode.mixfeat.at(j);

//		float response = mModelStore.SVM->predict(sampleMat);
		float response = _predict(mModelStore.SVM, sampleMat, false);
		cout<<i<<" . "<<response<<"\t"<<labSet[i]<<endl;
		accuracy += ((response==labSet[i])?1:0);
	}
	cout<<"accuracy : "<<accuracy/labSet.size()*100<<"%"<<endl;
}

double CModelSVM::similarity(const CFeature &feat)
{
	return 0.0;
}


//=======================Model IO==========================
void CModelSVM::loadModel(std::string model_file)
{
	this->mModelStore.SVM->load(model_file.c_str());
}

void CModelSVM::saveModel(std::string model_file)
{
	this->mModelStore.SVM->save(model_file.c_str());
}

//=======================load sample========================
void CModelSVM::_loadTrain( cv::Mat &trainData, const std::vector<CFeature> &feaSet )
{
	for(int i=0; i<feaSet.size(); i++)
	{
		for(int j=0; j<feaSet[i].mFeatureMode.mixfeat.size(); j++)
		{
			trainData.at<float>(i, j) = feaSet[i].mFeatureMode.mixfeat[j];
		}
	}
}

void CModelSVM::_loadLabel( cv::Mat &labelData, const std::vector<float> &labSet )
{
	for(int i=0; i<labSet.size(); i++)
	{
		labelData.at<float>(i) = labSet.at(i);
	}
}

CModelSVMStore::CModelSVMStore():SVM(NULL)
{
	SVM = new CvSVM();
}