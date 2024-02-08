#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <mex.h>
#include <math.h>
#include "matrix.h"

#define tol 1e-8

/*
-------------------------- Function prf_en -----------------------------

 Euclidean Projection onto Elastic Net (prf_en)
 
        min  1/2 ||x- c||_2^2
        s.t. ||x||_1 + gamma/2*||x||_2^2 <= t

 Usage (in matlab):
 [x, theta, iter]=prf_en(c, n, t, gamma); n is the dimension of the vector c

-------------------------- Function prf_en -----------------------------
 */

void prf_en(double * x, double *root, double * steps, double * c,int n, double t, double gamma)
{
	int i, gnum, rho = 0;
	double theta = 0;
	double aa, bb, cc, s = 0;
	double iter_step=0; 

	if (t < 0)
	{
		printf("\n t should be nonnegative!");
		return;
	}
	gnum = 0;
	for(i=0;i<n;i++)
	{
		//if (c[i]!=0)
		//{
			x[gnum] = fabs(c[i]);
			s += x[gnum] + 0.5*gamma*x[gnum]*x[gnum];
			gnum++;
		//}
	}

	/* ||c||_1 + gamma/2*||c||_2^2 <= t, then c is the solution  */
	if (s <= t)
	{
        theta=0;
        for(i=0;i<n;i++)
		{
            x[i]=c[i];
        }
        *root=theta;
        *steps=iter_step;
        return;
    }
	
	bb = 2*t*gamma + gnum;
	aa = 0.5*bb*gamma;
	cc = t - s;
	/*while loops*/
	while (abs(aa*theta*theta + bb*theta + cc) > tol)
	{
		iter_step++;
		theta = (-bb + sqrt(bb*bb-4*aa*cc))/(2.0*aa);
		s=0; rho = 0;
		for (i=0;i<gnum;i++)
		{
			if (x[i]>=theta)
			{
				x[rho] = x[i];
				s+=x[i] + 0.5*gamma*x[i]*x[i];
				rho++;
			}
		}
		gnum = rho;
		/*i=V_i_b; j=V_i_e; s=0;
        while (i <= j)
		{            
            while( (i <= V_i_e) && (x[i] < theta) )
			{
                i++;
            }
            while( (j>=V_i_b) && (x[j] >= theta) )
			{
                s+=x[j] + 0.5*gamma*x[j]*x[j];                
                j--;
            }
            if (i<j)
			{
                s+=x[i] + 0.5*gamma*x[i]*x[i];
                
                temp=x[i];  x[i]=x[j];  x[j]=temp;
                i++;  j--;
            }
		}
		V_i_b = i; rho = V_i_e - V_i_b + 1;*/
		
		bb = 2*t*gamma + gnum;
		aa = 0.5*bb*gamma;
		cc = t - s;
	}/*end of while*/

	/*projection result*/
	for(i=0;i<n;i++)
	{        
		x[i] = (c[i] > theta)?(c[i]-theta)/(1.0 + gamma*theta):((c[i]< -theta)?(c[i]+theta)/(1.0 + gamma*theta):0);
    }
    *root=theta;
    *steps=iter_step;
	return;
}


void mexFunction (int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    /*set up input arguments */
    double* c=            mxGetPr(prhs[0]);
    int     n=       (int)mxGetScalar(prhs[1]);
    double  t=            mxGetScalar(prhs[2]);
	double  gamma=        mxGetScalar(prhs[3]);
    
    double *x, *theta;
	double *iter_step;


    /* set up output arguments */
    plhs[0] = mxCreateDoubleMatrix(n,1,mxREAL);
    plhs[1] = mxCreateDoubleMatrix(1,1,mxREAL);
    plhs[2] = mxCreateDoubleMatrix(1,1,mxREAL);
    
    x=mxGetPr(plhs[0]);
    theta=mxGetPr(plhs[1]);
	iter_step= mxGetPr(plhs[2]);
    
    prf_en(x, theta, iter_step, c, n, t, gamma);
}


