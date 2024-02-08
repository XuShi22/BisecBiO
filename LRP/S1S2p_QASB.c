#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <mex.h>
#include <math.h>
#include "matrix.h" 
#define delta 1e-9

#define innerIter 1000
#define outerIter 1000

/*
-------------------------- Function S1S2p_QASB -----------------------------

Projection onto the intersection of l1 ball and l2 ball using QASB method

In paper "Projections onto the Intersection of an l1 Ball or Sphere and an l2 Ball or Sphere"
Hongying Liu ¡¤ Hao Wang ¡¤ Mengmeng Song
 
        min  1/2 ||x- v||_2^2
        s.t. ||x||_1 = z
             ||x||_2 = 1

 Usage (in matlab):
[U, Area, x, lambda, iter_step, flag, x_eplb, lambda_eplb, iter_step_eplb]=l12p_QASB(w, N, t, Area_Known);
 Area_Known=0,1,2,3(If Area of v is known, Area=1,2,3. Else, Area=0.)
 U: The number of elements in [lambda1, lambda2] during iteration
 Area: If Area_Known=0, the area where x is located. Else, Area=Area_Known.
 x: projected point
 lambda: zero point of phi
 iter_step:  iter step to find zero point of phi
 
 x_eplb, lambda_eplb, iter_step_eplb: results from eplb, if Area_Known=0. if Area_Known!=0, they have no meaning.
--------------------- Function eplb as a subfunction------------------------
 */

/*--------------------------------------------------------------------------
                    function  S1S2p_QASB
--------------------------------------------------------------------------*/


void S1S2p_QASB(double * U, double * x, double *root, int * steps, int * fla, double * v,int n, double z)
{
    
    int i, j, flag=0;
    int rho_1, rho_2, rho, rho_Q, rho_S;
    int V_i_b, V_i_e, V_i;
    /*double l=0, r;*/
    double lambda_1, lambda_2, lambda_Q, lambda_S, lambda=0;
    double s_1, s_2, s, s_Q, s_S, q_1, q_2, q, q_Q, q_S, temp, norm2x, v_max=0, r;
    double f_lambda_1, f_lambda_2, f_lambda, f1_lambda_1, f_lambda_Q, f_lambda_S, psi0;
    int iter_step=0;
    double z2=z*z; 
    double temp1=0;
	if (z<0){
		printf("\n z should be nonnegative!");
		return;
	}
    
    s=0;q=0;
    for (i=0;i<n; i++){
        s+=fabs(v[i]);
        q+=v[i]*v[i];
        if(fabs(v[i])>v_max)
        v_max=fabs(v[i]);
    }
    r=v_max;
    
    
   i=n-1;
   /*printf("%f\n",s*s-z2*q);*/
   if(s*s-z2*q<0){
                lambda=(s-z*sqrt(((i+1)*q-s*s)/(i+1-z*z)))/(i+1);
                /*printf("%f\n",lambda);*/
                flag=0;
    }
   else
   {
    V_i=0;  rho_1=0; rho_2=0; rho_Q=0; rho_S=0; 
    s_1=0; s_2=0; s_Q=0; s_S=0; q_1=0; q_2=0; q_Q=0; q_S=0; s=0; q=0;
    lambda_1=0; lambda_2=r;
    for (i=0;i<n; i++){
        if(fabs(v[i]) >= lambda_2){
        rho_2++; s_2+= fabs(v[i]);
        q_2+= v[i]*v[i];
        rho_1++;
        s_1+= fabs(v[i]);
        q_1+= v[i]*v[i];
        s+= fabs(v[i]);
        q+= v[i]*v[i];
    }
        if (fabs(v[i]) >= lambda_1&& fabs(v[i]) < lambda_2){
            x[V_i]=fabs(v[i]); s_1+= x[V_i]; q_1+= x[V_i]*x[V_i]; rho_1++;
            V_i++;
        }
    }
    z2=z*z;
    f_lambda_1=(rho_1-z2)*(rho_1*lambda_1-2*s_1)*lambda_1+s_1*s_1-z2*q_1;
    f_lambda_2=(rho_2-z2)*(rho_2*lambda_2-2*s_2)*lambda_2+s_2*s_2-z2*q_2;
    
    /*f_lambda_1=s_1-rho_1* lambda_1 -z;*/

    V_i_b=0; V_i_e=V_i-1;
    /*-------------------------------------------------------------------
                  Initialization with the root
     *-------------------------------------------------------------------
     */
           
        
    while (!flag){
        iter_step++;
         /* compute lambda_T  */
        /*Netown strategy */
        /*f1_lambda_1=2*(rho_1-z2)*(rho_1*lambda_1-s_1);
        lambda_T=lambda_1-f_lambda_1/f1_lambda_1;*/    
        
        /* compute lambda_Q  */
        /*Q strategy */
        temp1=sqrt((rho_1*q_1-s_1*s_1)/(rho_1-z2));
        lambda_Q=(s_1-z*temp1)/rho_1; 
        
        /* compute lambda_S */
        lambda_S=lambda_1-f_lambda_1*(lambda_2-lambda_1)/(f_lambda_2-f_lambda_1);
        
        if (fabs(lambda_Q-lambda_S) <= delta){
            lambda=lambda_Q; flag=1;
            break;
        }
        
        /* set lambda as the middle point of lambda_Q and lambda_S */
        lambda=(lambda_Q+lambda_S)/2;
        s_Q=s_S=s=0;q_Q=q_S=q=0;rho_Q=rho_S=rho=0;
        i=V_i_b; j=V_i_e;
        while (i <= j){            
            while( (i <= V_i_e) && (x[i] <= lambda) ){
                if (x[i]> lambda_Q){
                    s_Q+=x[i]; q_Q+=x[i]*x[i]; rho_Q++;
                }
                i++;
            }
            while( (j>=V_i_b) && (x[j] > lambda) ){
                if (x[j] > lambda_S){
                    s_S+=x[j]; q_S+=x[j]*x[j]; rho_S++;
                }
                else{
                    s+=x[j]; q+=x[j]*x[j]; rho++;
                }
                j--;
            }
            if (i<j){
                if (x[i] > lambda_S){
                    s_S+=x[i];q_S+=x[i]*x[i]; rho_S++;
                }
                else{
                    s+=x[i];q+=x[i]*x[i]; rho++;
                }
                
                if (x[j]> lambda_Q){
                    s_Q+=x[j]; q_Q+=x[j]*x[j]; rho_Q++;
                }
                
                temp=x[i]; x[i]=x[j];  x[j]=temp;
                i++; j--;
            }
	}/* end for i<=j */
        
        s_S+=s_2; q_S+=q_2; rho_S+=rho_2;
        s+=s_S; q+=q_S; rho+=rho_S;
        s_Q+=s; q_Q+=q; rho_Q+=rho;
        f_lambda_S=(rho_S-z2)*(rho_S*lambda_S-2*s_S)*lambda_S+s_S*s_S-z2*q_S;
        f_lambda=(rho-z2)*(rho*lambda-2*s)*lambda+s*s-z2*q;
        f_lambda_Q=(rho_Q-z2)*(rho_Q*lambda_Q-2*s_Q)*lambda_Q+s_Q*s_Q-z2*q_Q;
        
        /*printf("\n %d & %d  & %5.6f & %5.6f & %5.6f & %5.6f & %5.6f \\\\ \n \\hline ", iter_step, V_i, lambda_1, lambda_T, lambda, lambda_S, lambda_2);*/
                
        if ( fabs(f_lambda)< delta ){
            /*printf("\n lambda");*/
            flag=1;
            break;
        }  
        if ( fabs(f_lambda_Q)< delta){
            /*printf("\n lambda");*/
            flag=2;
            lambda=lambda_Q;
            break;
        } 
        
        /*
        printf("\n\n f_lambda_1=%5.6f, f_lambda_2=%5.6f, f_lambda=%5.6f",f_lambda_1,f_lambda_2, f_lambda);
        printf("\n lambda_1=%5.6f, lambda_2=%5.6f, lambda=%5.6f",lambda_1, lambda_2, lambda);
        printf("\n rho_1=%d, rho_2=%d, rho=%d ",rho_1, rho_2, rho);
         */
        
        if (f_lambda <0){
            lambda_2=lambda;  s_2=s; q_2=q; rho_2=rho;
            f_lambda_2=f_lambda;            
            
            lambda_1=lambda_Q; s_1=s_Q; q_1=q_Q; rho_1=rho_Q;
            f_lambda_1=f_lambda_Q;
            
            V_i_e=j;  i=V_i_b;
            while (i <= j){
                while( (i <= V_i_e) && (x[i] <= lambda_Q) ){
                    i++;
                }
                while( (j>=V_i_b) && (x[j] > lambda_Q) ){
                    j--;
                }
                if (i<j){                    
                    x[j]=x[i];
                    i++;   j--;
                }
            }            
            V_i_b=i; V_i=V_i_e-V_i_b+1;
        }
        else{
            lambda_1=lambda;  s_1=s;  q_1=q; rho_1=rho;
            f_lambda_1=f_lambda;
            
            lambda_2=lambda_S; s_2=s_S; q_2=q_S; rho_2=rho_S;
            f_lambda_2=f_lambda_S;
            
            V_i_b=i;  j=V_i_e;
            while (i <= j){
                while( (i <= V_i_e) && (x[i] <= lambda_S) ){
                    i++;
                }
                while( (j>=V_i_b) && (x[j] > lambda_S) ){
                    j--;
                }
                if (i<j){
                    x[i]=x[j];
                    i++;   j--;
                }
            }
            V_i_e=j; V_i=V_i_e-V_i_b+1;
        }
         /*if (V_i==0){
            lambda=(s_1-z*sqrt((rho_1*q_1-s_1*s_1)/(rho_1-z2)))/rho_1; flag=2;
            /*printf("\n V_i=0, lambda=%5.6f",lambda);
            break;
        }*/
        U[iter_step]=V_i;
    }/* end of while(!flag) */
   }
   /*printf("%f\n",lambda);*/
   norm2x=0;
    for(i=0;i<n;i++){
        if (v[i] > lambda){
            x[i]=v[i]-lambda;
            norm2x+=x[i]*x[i];
        }
        else{
            if (v[i]< -lambda){
                x[i]=v[i]+lambda;
                norm2x+=x[i]*x[i];
                }
            else
                x[i]=0;
        }
    }
  /* printf("norm2x=%f\n",1e6*norm2x);*/

    for(i=0;i<n;i++){
    x[i]=x[i]/sqrt(norm2x);
    }
    *root=lambda;
    *steps=iter_step;
    *fla=flag;
    return;
}




void mexFunction (int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    /*set up input arguments */
    double* v=              mxGetPr(prhs[0]);
    int     n=         (int)mxGetScalar(prhs[1]);
    double  z=              mxGetScalar(prhs[2]);
    
    double *U;
    double *x, *lambda;
    double *iter_step, *flag;
    int steps;
    int fla;
    double root;
    
    
    /* set up output arguments */
    plhs[0] = mxCreateDoubleMatrix(10,1,mxREAL);
    plhs[1] = mxCreateDoubleMatrix(n,1,mxREAL);
    plhs[2] = mxCreateDoubleMatrix(1,1,mxREAL);
    plhs[3] = mxCreateDoubleMatrix(1,1,mxREAL);
    plhs[4] = mxCreateDoubleMatrix(1,1,mxREAL);
    
    U=mxGetPr(plhs[0]);
    x=mxGetPr(plhs[1]);
    lambda=mxGetPr(plhs[2]);
    iter_step=mxGetPr(plhs[3]);
    flag=mxGetPr(plhs[4]);

    S1S2p_QASB(U, x, &root, &steps, &fla, v, n, z);
    *iter_step=steps;
    *flag=fla;
    *lambda=root;
}

