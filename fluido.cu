#include <cuda.h>
#include <stdio.h>
#include <iostream>
#include <pthread.h>

using namespace std;

int N;

__global__ void csimpson_elem_xy(double *arreglo_x, double *arreglo_y, double a, double b, int N)
{
    double x_2k_2, x_2k_1, x_2k, h;
    int k = threadIdx.x + blockIdx.x * blockDim.x;

    if (k == 0)
    {
        return;
    }

    h = (b - a) / N;
    x_2k_2 = (a + (((2 * k) - 2) * h));
    x_2k_1 = (a + (((2 * k) - 1) * h));
    x_2k = (a + ((2 * k) * h));
    arreglo_x[k] = pow(sin(x_2k_2), 2) + (4 * pow(sin(x_2k_1), 2)) + pow(sin(x_2k), 2);
    arreglo_y[k] = pow(cos(x_2k_2), 2) + (4 * pow(cos(x_2k_1), 2)) + pow(cos(x_2k), 2);

    // printf("%d :%f\n",k, arreglo[k]);
}

__global__ void csimpson_elem_x(double *arreglo, double a, double b, int N)
{
    double x_2k_2, x_2k_1, x_2k, h;
    int k = threadIdx.x + blockIdx.x * blockDim.x;

    if (k == 0)
    {
        return;
    }

    h = (b - a) / N;
    x_2k_2 = (a + (((2 * k) - 2) * h));
    x_2k_1 = (a + (((2 * k) - 1) * h));
    x_2k = (a + ((2 * k) * h));
    arreglo[k] = pow(sin(x_2k_2), 2) + (4 * pow(sin(x_2k_1), 2)) + pow(sin(x_2k), 2);
    // printf("%d :%f\n",k, arreglo[k]);
}

__global__ void csimpson_elem_y(double *arreglo, double a, double b, int N)
{
    double x_2k_2, x_2k_1, x_2k, h;
    int k = threadIdx.x + blockIdx.x * blockDim.x;

    if (k == 0)
    {
        return;
    }

    h = (b - a) / N;
    x_2k_2 = (a + (((2 * k) - 2) * h));
    x_2k_1 = (a + (((2 * k) - 1) * h));
    x_2k = (a + ((2 * k) * h));
    arreglo[k] = pow(cos(x_2k_2), 2) + (4 * pow(cos(x_2k_1), 2)) + pow(cos(x_2k), 2);
    // printf("%d :%f\n",k, arreglo[k]);
}

/*
Realiza la integración de la función sin^2(t) (i^) y cos^2(t) (j^) desde 0 hasta el instante t
Parameters
----------
t : double
    Instante de tiempo para el cálculo de la posición

Returns
----------
sumas : double*
    Array de 2 elementos siendo el primero la posición en x y el segundo en y
*/
double *integral(double t)
{
    // Definición de límites usados cómo parámetros de la integral
    int a = 0;
    int b = t;

    // Creación de arrays , el sufijo _d hace referencia a device y _h al host
    double *ar_x_d, *ar_x_h;
    cudaMalloc((void **)&ar_x_d, N * sizeof(double));
    ar_x_h = (double *)malloc(N * sizeof(double));

    csimpson_elem_x<<<N / 32 / 2, 32>>>(ar_x_d, a, b, (N));
    cudaMemcpy(ar_x_h, ar_x_d, N * sizeof(double), cudaMemcpyDeviceToHost);
    double suma_x = 0;
    for (int i = 1; i < N; i++)
    {
        suma_x += ar_x_h[i];
    }
    double h = (double)(b - a) / (N);
    suma_x *= h / 3;

    double *ar_y_d, *ar_y_h;
    cudaMalloc((void **)&ar_y_d, N * sizeof(double));
    ar_y_h = (double *)malloc(N * sizeof(double));

    csimpson_elem_y<<<N / 32 / 2, 32>>>(ar_y_d, a, b, (N));
    cudaMemcpy(ar_y_h, ar_y_d, N * sizeof(double), cudaMemcpyDeviceToHost);
    double suma_y = 0;
    for (int i = 1; i < N; i++)
    {
        suma_y += ar_y_h[i];
    }
    suma_y *= h / 3;

    double *sumas = new double[2];
    sumas[0] = suma_x;
    sumas[1] = suma_y;
    cout << sumas[1] << endl;
    cudaFree(ar_x_d);
    cudaFree(ar_y_d);

    return sumas;
}

struct posParticle
{
    double x;
    double y;
    double t;
};

int main()
{
    N = 32 * 10000;
    struct posParticle *pos1[10];
    struct posParticle *pos1_1;

    pos1_1 = (struct posParticle *)malloc(sizeof(struct posParticle));

    pos1_1->x = 10;
    pos1_1->y = 1;
    pos1_1->t = 0;
    pos1[0] = pos1_1;

    cout << "t: " << pos1[0]->t << " x: " << pos1[0]->x << " y: " << pos1[0]->y << endl;

    for (int i = 1; i < 10; i++)
    {
        double *pos = integral(i);
        struct posParticle *pos1_i = (struct posParticle *)malloc(sizeof(struct posParticle));
        pos1_i->x = pos1[i - 1]->x + pos[0];
        pos1_i->y = pos1[i - 1]->y + pos[1];
        pos1_i->t = i;
        pos1[i] = pos1_i;
        // free(pos1_i);
        cout << "t: " << i << " x: " << pos1[i]->x << " y: " << pos1[i]->y << endl;
    }

    return 0;
}