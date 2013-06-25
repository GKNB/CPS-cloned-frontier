#include<math.h>
#include<stdio.h>
#include<stdlib.h>
#include"essl.h"
#include<util/verbose.h>

USING_NAMESPACE_CPS

void tred2(double **a, int n, double *d, double *e);
int tqli(double *d, double *e, int n, double **z);

//int eigen_solver(double **A, int n, double *Eval, double *aux)
int eigen_solver(double *A, double *Evec, double *Eval, int n)
//use the same call as the old version of eigen solver, so do not need to change the function that call it
//double A[n*(n+1)/2], EV[n*n], E[n];
{
	int i,j;
	double **a=(double **)malloc(n*sizeof(double*));
	double *aux=(double *)malloc(n*sizeof(double));
	for(i=0;i<n;i++)
		a[i]=&Evec[i*n];
	for(i=0;i<n;i++)
		for(j=0;j<=i;j++)
			a[j][i]=a[i][j]=A[i*(i+1)/2+j];
	tred2(a,n,Eval,aux);
	int success = tqli(Eval,aux,n,a);
	free(a);
	free(aux);
	return success;
}
void tred2(double **a, int n, double *d, double *e)
{
    int     l, k, j, i;
    double  scale, hh, h, g, f;

    for (i = 0; i < n; ++i)
        (a[i])--;
    a--;
    d--;
    e--;

    for (i = n; i >= 2; i--)
    {
        l = i - 1;
        h = scale = 0.0;
        if (l > 1)
        {
            for (k = 1; k <= l; k++)
                scale += fabs(a[i][k]);
            if (scale == 0.0)
                e[i] = a[i][l];
            else
            {
                for (k = 1; k <= l; k++)
                {
                    a[i][k] /= scale;
                    h += a[i][k] * a[i][k];
                }
                f = a[i][l];
                g = f > 0 ? -sqrt(h) : sqrt(h);
                e[i] = scale * g;
                h -= f * g;
                a[i][l] = f - g;
                f = 0.0;
                for (j = 1; j <= l; j++)
                {
                    a[j][i] = a[i][j] / h;
                    g = 0.0;
                    for (k = 1; k <= j; k++)
                        g += a[j][k] * a[i][k];
                    for (k = j + 1; k <= l; k++)
                        g += a[k][j] * a[i][k];
                    e[j] = g / h;
                    f += e[j] * a[i][j];
                }
                hh = f / (h + h);
                for (j = 1; j <= l; j++)
                {
                    f = a[i][j];
                    e[j] = g = e[j] - hh * f;
                    for (k = 1; k <= j; k++)
                        a[j][k] -= (f * e[k] + g * a[i][k]);
                }
            }
        } else
            e[i] = a[i][l];
        d[i] = h;
    }
    d[1] = 0.0;
    e[1] = 0.0;
    for (i = 1; i <= n; i++)
    {
        l = i - 1;
        if (d[i])
        {
            for (j = 1; j <= l; j++)
            {
                g = 0.0;
                for (k = 1; k <= l; k++)
                    g += a[i][k] * a[k][j];
                for (k = 1; k <= l; k++)
                    a[k][j] -= g * a[k][i];
            }
        }
        d[i] = a[i][i];
        a[i][i] = 1.0;
        for (j = 1; j <= l; j++)
            a[j][i] = a[i][j] = 0.0;
    }

    a++;
    d++;
    e++;
    for (i = 0; i < n; ++i)
        (a[i])++;
}

#define SIGN(a,b) ((b)<0 ? -fabs(a) : fabs(a))

int tqli(double *d, double *e, int n, double **z)
{
    int     m, l, iter, i, k;
    double  s, r, p, g, f, dd, c, b;

    for (i = 0; i < n; ++i)
        (z[i])--;
    z--;
    d--;
    e--;

    for (i = 2; i <= n; i++)
        e[i - 1] = e[i];
    e[n] = 0.0;
    for (l = 1; l <= n; l++)
    {
        iter = 0;
        do
        {
            for (m = l; m <= n - 1; m++)
            {
                dd = fabs(d[m]) + fabs(d[m + 1]);
                if (fabs(e[m]) + dd == dd)
                    break;
            }
            if (m != l)
            {
                if (iter++ == 30)
                    return (0);
                g = (d[l + 1] - d[l]) / (2.0 * e[l]);
                r = sqrt((g * g) + 1.0);
                g = d[m] - d[l] + e[l] / (g + SIGN(r, g));
                s = c = 1.0;
                p = 0.0;
                for (i = m - 1; i >= l; i--)
                {
                    f = s * e[i];
                    b = c * e[i];
                    if (fabs(f) >= fabs(g))
                    {
                        c = g / f;
                        r = sqrt((c * c) + 1.0);
                        e[i + 1] = f * r;
                        c *= (s = 1.0 / r);
                    } else
                    {
                        s = f / g;
                        r = sqrt((s * s) + 1.0);
                        e[i + 1] = g * r;
                        s *= (c = 1.0 / r);
                    }
                    g = d[i + 1] - p;
                    r = (d[i] - g) * s + 2.0 * c * b;
                    p = s * r;
                    d[i + 1] = g + p;
                    g = c * r - b;
                    for (k = 1; k <= n; k++)
                    {
                        f = z[k][i + 1];
                        z[k][i + 1] = s * z[k][i] + c * f;
                        z[k][i] = c * z[k][i] - s * f;
                    }
                }
                d[l] = d[l] - p;
                e[l] = g;
                e[m] = 0.0;
            }
        } while (m != l);
    }

    z++;
    d++;
    e++;
    for (i = 0; i < n; ++i)
        (z[i])++;

    return (1);
}


void min_eig_index(int *INDEX, int nev,double *EIG, int n)
{
	//complexity=n*nev; can be reduced to n
	int max;
	int i,j;
	for(i=0;i<nev;i++)INDEX[i]=i;
	for(i=nev;i<n;i++)
	{
		//find max in INDEX
		max=0;
		for(j=1;j<nev;j++)if(EIG[INDEX[j]]>EIG[INDEX[max]])max=j;
		//change the max one
		if(EIG[i]<EIG[INDEX[max]])INDEX[max]=i;
	}
}

#include<complex>
void invert_H_matrix(std::complex<double> *data, int N)
{
    if (N <= 0) return;  // sanity check
    if (N == 1) {data[0]=1.0/data[0];return;}  // must be of dimension >= 2

    for (int i=1; i < N; i++) {
        data[i] /= data[0]; // normalize row 0
    }

    for (int i=1; i < N; i++) {
        for (int j=i; j < N; j++)  { // do a column of L
            std::complex<double> sum = 0.0;
            for (int k = 0; k < i; k++) {
                sum += data[j*N+k] * data[k*N+i];
            }

            data[j*N+i] -= sum;
        }
        if (i == N-1) continue;
        for (int j=i+1; j < N; j++)  {  // do a row of U
            std::complex<double> sum = 0.0;
            for (int k = 0; k < i; k++)
                sum += data[i*N+k]*data[k*N+j];
            data[i*N+j] = 
                (data[i*N+j]-sum) / data[i*N+i];
        }
    }

    for ( int i = 0; i < N; i++ ) { // invert L
        for ( int j = i; j < N; j++ )  {
            std::complex<double> x = 1.0;
            if ( i != j ) {
                x = 0.0;
                for ( int k = i; k < j; k++ ) 
                    x -= data[j*N+k]*data[k*N+i];
            }
            data[j*N+i] = x / data[j*N+j];
        }
    }

    for ( int i = 0; i < N; i++ ) {  // invert U
        for ( int j = i; j < N; j++ )  {
            if ( i == j ) continue;
            std::complex<double> sum = 0.0;
            for ( int k = i; k < j; k++ )
                sum += data[k*N+j]*( (i==k) ? 1.0 : data[i*N+k] );
            data[i*N+j] = -sum;
        }
    }

    for ( int i = 0; i < N; i++ ) {  // final inversion
        for ( int j = 0; j < N; j++ )  {
            std::complex<double> sum = 0.0;
            for ( int k = ((i>j)?i:j); k < N; k++ )  
                sum += ((j==k)?1.0:data[j*N+k])*data[k*N+i];
            data[j*N+i] = sum;
        }
    }
}

void invert_H_zpotri(std::complex<double> *data, int N)
{
    const char type = 'U';
    int info = 0;

    int *ipiv = new int[N];
    std::complex<double> *work = new std::complex<double>[N*N];

    // Sometimes zpotrf() fails, reporting non positive definite Hermitian matrix.
    // zpotrf(&type, N, (void*)data, N, info);

    // if(info != 0) {
    //     printf("zpotrf() failed!\n");
    //     exit(-1);
    // }

    // zpotri(&type, N, (void*)data, N, info);
    // if(info != 0) {
    //     printf("zpotri() failed!\n");
    //     exit(-1);
    // }

    // for(int i = 1; i < N; ++i) {
    //     for(int j = 0; j < i; ++j) {
    //         data[j*N+i]=conj(data[i*N+j]);
    //     }
    // }

    zgetrf(N, N, (void*)data, N, ipiv, info);
    zgetri(N, (void*)data, N, ipiv, work, N*N, info);
    
    delete[] ipiv;
    delete[] work;
}
