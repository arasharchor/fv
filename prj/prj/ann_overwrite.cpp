#include <vector>
#include <string>
#include <assert.h>
#include <opencv2\ml\ml.hpp>
#include <opencv2\core\core.hpp>
#include "cfeature.h"
#include <assert.h>

using namespace std;
using namespace cv;

int train_backprop( CvANN_MLP *ann, CvVectors x0, CvVectors u, const double* sw )
{
    CvMat* dw = 0;
    CvMat* buf = 0;
    double **x = 0, **df = 0;
    CvMat* _idx = 0;
    int iter = -1, count = x0.count;

    CV_FUNCNAME( "CvANN_MLP::train_backprop" );

    __BEGIN__;

    int i, j, k, ivcount, ovcount, l_count, total = 0, max_iter;
    double *buf_ptr;
    double prev_E = DBL_MAX*0.5, E = 0, epsilon;

    max_iter = ann->params.term_crit.max_iter*count;
    epsilon = ann->params.term_crit.epsilon*count;

    l_count = ann->layer_sizes->cols;
    ivcount = ann->layer_sizes->data.i[0];
    ovcount = ann->layer_sizes->data.i[l_count-1];

    // allocate buffers
    for( i = 0; i < l_count; i++ )
        total += ann->layer_sizes->data.i[i] + 1;

    CV_CALL( dw = cvCreateMat( ann->wbuf->rows, ann->wbuf->cols, ann->wbuf->type ));
    cvZero( dw );
    CV_CALL( buf = cvCreateMat( 1, (total + ann->max_count)*2, CV_64F ));
    CV_CALL( _idx = cvCreateMat( 1, count, CV_32SC1 ));
    for( i = 0; i < count; i++ )
        _idx->data.i[i] = i;

    CV_CALL( x = (double**)cvAlloc( total*2*sizeof(x[0]) ));
    df = x + total;
    buf_ptr = buf->data.db;

    for( j = 0; j < l_count; j++ )
    {
        x[j] = buf_ptr;
        df[j] = x[j] + ann->layer_sizes->data.i[j];
        buf_ptr += (df[j] - x[j])*2;
    }

    // run back-propagation loop
    /*
        y_i = w_i*x_{i-1}
        x_i = f(y_i)
        E = 1/2*||u - x_N||^2
        grad_N = (x_N - u)*f'(y_i)
        dw_i(t) = momentum*dw_i(t-1) + dw_scale*x_{i-1}*grad_i
        w_i(t+1) = w_i(t) + dw_i(t)
        grad_{i-1} = w_i^t*grad_i
    */
    for( iter = 0; iter < max_iter; iter++ )
    {
        int idx = iter % count;
        double* w = ann->weights[0];
        double sweight = sw ? count*sw[idx] : 1.;
        CvMat _w, _dw, hdr1, hdr2, ghdr1, ghdr2, _df;
        CvMat *x1 = &hdr1, *x2 = &hdr2, *grad1 = &ghdr1, *grad2 = &ghdr2, *temp;

        if( idx == 0 )
        {
            printf("%d. E = %g\n", iter/count, E);
            if( fabs(prev_E - E) < epsilon )
                break;
            prev_E = E;
            E = 0;

            // shuffle indices
            for( i = 0; i < count; i++ )
            {
                int tt;
                j = (*ann->rng)(count);
                k = (*ann->rng)(count);
                CV_SWAP( _idx->data.i[j], _idx->data.i[k], tt );
            }
        }

        idx = _idx->data.i[idx];

        if( x0.type == CV_32F )
        {
            const float* x0data = x0.data.fl[idx];
            for( j = 0; j < ivcount; j++ )
                x[0][j] = x0data[j]*w[j*2] + w[j*2 + 1];
        }
        else
        {
            const double* x0data = x0.data.db[idx];
            for( j = 0; j < ivcount; j++ )
                x[0][j] = x0data[j]*w[j*2] + w[j*2 + 1];
        }

        cvInitMatHeader( x1, 1, ivcount, CV_64F, x[0] );

        // forward pass, compute y[i]=w*x[i-1], x[i]=f(y[i]), df[i]=f'(y[i])
        for( i = 1; i < l_count; i++ )
        {
            cvInitMatHeader( x2, 1, ann->layer_sizes->data.i[i], CV_64F, x[i] );
            cvInitMatHeader( &_w, x1->cols, x2->cols, CV_64F, ann->weights[i] );
            cvGEMM( x1, &_w, 1, 0, 0, x2 );
            _df = *x2;
            _df.data.db = df[i];
            ann->calc_activ_func_deriv( x2, &_df, _w.data.db + _w.rows*_w.cols );
            CV_SWAP( x1, x2, temp );
        }

        cvInitMatHeader( grad1, 1, ovcount, CV_64F, buf_ptr );
        *grad2 = *grad1;
        grad2->data.db = buf_ptr + ann->max_count;

        w = ann->weights[l_count+1];

        // calculate error
        if( u.type == CV_32F )
        {
            const float* udata = u.data.fl[idx];
            for( k = 0; k < ovcount; k++ )
            {
                double t = udata[k]*w[k*2] + w[k*2+1] - x[l_count-1][k];
                grad1->data.db[k] = t*sweight;
                E += t*t;
            }
        }
        else
        {
            const double* udata = u.data.db[idx];
            for( k = 0; k < ovcount; k++ )
            {
                double t = udata[k]*w[k*2] + w[k*2+1] - x[l_count-1][k];
                grad1->data.db[k] = t*sweight;
                E += t*t;
            }
        }
        E *= sweight;

        // backward pass, update weights
        for( i = l_count-1; i > 0; i-- )
        {
            int n1 = ann->layer_sizes->data.i[i-1], n2 = ann->layer_sizes->data.i[i];
            cvInitMatHeader( &_df, 1, n2, CV_64F, df[i] );
            cvMul( grad1, &_df, grad1 );
            cvInitMatHeader( &_w, n1+1, n2, CV_64F, ann->weights[i] );
            cvInitMatHeader( &_dw, n1+1, n2, CV_64F, dw->data.db + (ann->weights[i] - ann->weights[0]) );
            cvInitMatHeader( x1, n1+1, 1, CV_64F, x[i-1] );
            x[i-1][n1] = 1.;
            cvGEMM( x1, grad1, ann->params.bp_dw_scale, &_dw, ann->params.bp_moment_scale, &_dw );
            cvAdd( &_w, &_dw, &_w );
            if( i > 1 )
            {
                grad2->cols = n1;
                _w.rows = n1;
                cvGEMM( grad1, &_w, 1, 0, 0, grad2, CV_GEMM_B_T );
            }
            CV_SWAP( grad1, grad2, temp );
        }
    }

    iter /= count;

    __END__;

    cvReleaseMat( &dw );
    cvReleaseMat( &buf );
    cvReleaseMat( &_idx );
    cvFree( &x );

    return iter;
}


int train_( CvANN_MLP *ann, const CvMat* _inputs, const CvMat* _outputs,
                      const CvMat* _sample_weights, const CvMat* _sample_idx,
                      CvANN_MLP_TrainParams _params, int flags )
{
    const int MAX_ITER = 1000;
    const double DEFAULT_EPSILON = FLT_EPSILON;

    double* sw = 0;
    CvVectors x0, u;
    int iter = -1;

    x0.data.ptr = u.data.ptr = 0;

    CV_FUNCNAME( "CvANN_MLP::train" );

    __BEGIN__;

    int max_iter;
    double epsilon;

    ann->params = _params;

    // initialize training data
    CV_CALL( ann->prepare_to_train( _inputs, _outputs, _sample_weights,
                               _sample_idx, &x0, &u, &sw, flags ));

    // ... and link weights
    if( !(flags & CvANN_MLP::UPDATE_WEIGHTS) )
        ann->init_weights();

    max_iter = ann->params.term_crit.type & CV_TERMCRIT_ITER ? ann->params.term_crit.max_iter : MAX_ITER;
    max_iter = MAX( max_iter, 1 );

    epsilon = ann->params.term_crit.type & CV_TERMCRIT_EPS ? ann->params.term_crit.epsilon : DEFAULT_EPSILON;
    epsilon = MAX(epsilon, DBL_EPSILON);

    ann->params.term_crit.type = CV_TERMCRIT_ITER + CV_TERMCRIT_EPS;
    ann->params.term_crit.max_iter = max_iter;
    ann->params.term_crit.epsilon = epsilon;

    if( ann->params.train_method == CvANN_MLP_TrainParams::BACKPROP )
    {
        CV_CALL( iter = train_backprop( ann, x0, u, sw ));
    }
    else
    {
        CV_CALL( iter = ann->train_rprop( x0, u, sw ));
    }

    __END__;

    cvFree( &x0.data.ptr );
    cvFree( &u.data.ptr );
    cvFree( &sw );

    return iter;
}


int ann_train( CvANN_MLP *ann, const Mat& _inputs, const Mat& _outputs,
                     const Mat& _sample_weights, const Mat& _sample_idx,
                     CvANN_MLP_TrainParams _params, int flags )
{
    CvMat inputs = _inputs, outputs = _outputs, sweights = _sample_weights, sidx = _sample_idx;
    return train_(ann, &inputs, &outputs, sweights.data.ptr ? &sweights : 0,
                 sidx.data.ptr ? &sidx : 0, _params, flags);
}