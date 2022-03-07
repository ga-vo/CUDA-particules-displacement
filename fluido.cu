#include <cuda.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <pthread.h>
#include <math.h>
#include "gnuplot-iostream.h"

using namespace std;
int N, t_max, n_particles;
double dt;

/*
Estructura posParticle
    Almacena posición x,y y el instante t
*/
struct posParticle
{
    double x;
    double y;
    double t;
};

/*
Kernel global
Realiza la n-ésima operación del método de simpson para integración
Parameters
----------
total_x : double
    Apuntador de la variable dónde se almacena la sumatoria de los valores de cada operación
    del método de simpson para la integral respecto a x
total_y : double
    Apuntador de la variable dónde se almacena la sumatoria de los valores de cada operación
    del método de simpson para la integral respecto a y
a : double
    Límite inferior de la integral
b : double
    Límite superior de la integral
N : int
    Número iteraciones en el método de simpson
Returns
----------
*/
__global__ void csimpson_elem_op(double *total_x, double *total_y, double a, double b, int N)
{
    // Definición de variables
    double x_2k_2, x_2k_1, x_2k, h;

    // Cálculo de identificador del hilo (para ubicar la iteración en la que se encuentra)
    int k = threadIdx.x + blockIdx.x * blockDim.x;

    if (k == 0)
    {
        return;
    }

    // Asignación de cálculos propios del método de simpson
    h = (b - a) / N;
    x_2k_2 = (a + (((2 * k) - 2) * h));
    x_2k_1 = (a + (((2 * k) - 1) * h));
    x_2k = (a + ((2 * k) * h));

    // Uso de la función atomicAdd de CUDA, para garantizar la atomicidad de la operación y evitar data race
    // Se suma a los apuntadores total_x y total_y, para después obtener estos valores en el host
    // Se realiza la operación f(x_2k_2)+4*f(x_2k_1)+f(x_2k) del método de simpson
    atomicAdd(total_x, pow(sin(x_2k_2), 2) + (4 * pow(sin(x_2k_1), 2)) + pow(sin(x_2k), 2));
    atomicAdd(total_y, pow(cos(x_2k_2), 2) + (4 * pow(cos(x_2k_1), 2)) + pow(cos(x_2k), 2));
}

/*
Kernel global
Realiza la asignación de la posición en x,y y el tiempo para las párticulas
Parameters
----------
particulas : posParticle*
    Apuntador a arreglo de estructuras posParticle a modificar
particulas_ini : posParticle*
    Apuntador a arreglo de estructuras posParticle con las posiciones en t=0
t : double
    Valor de tiempo t a asignar
b : double*
    Arreglo de tamaño 2 con los valores de x y y respectivamente
Returns
----------
*/
__global__ void caminado_ind(posParticle *particulas, posParticle *particulas_ini, double t, double *pos)
{
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    particulas[j].x = (particulas_ini)[j].x + pos[0];
    particulas[j].y = (particulas_ini)[j].y + pos[1];
    particulas[j].t = t;
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
    // Definición de arreglo para ser retornado
    double *sumas = new double[2];
    double h, suma_x, suma_y;

    // Definición de límites usados cómo parámetros de la integral
    int a = 0;
    int b = t;

    // Creación de arrays , el sufijo _d hace referencia a device y _h al host
    double *ar_x_d, ar_x_h, *ar_y_d, ar_y_h;

    // x
    cudaMalloc((void **)&ar_x_d, sizeof(double));
    // y
    cudaMalloc((void **)&ar_y_d, sizeof(double));
    ar_x_h = 0;
    ar_y_h = 0;

    // Copia desde el host al dispositivo de los acumuladores
    cudaMemcpy(ar_x_d, &ar_x_h, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(ar_y_d, &ar_y_h, sizeof(double), cudaMemcpyHostToDevice);

    // Lanzamiento del kernel con los parámetros
    /* El parámetro N /256 / 2 se realiza de esta manera debido a que el
        método de simpson requiere N/2 iteraciones */
    csimpson_elem_op<<<N / 256 / 2, 256>>>(ar_x_d, ar_y_d, a, b, (N));

    // Copia desde el dispositivo al host de los acumuladores
    cudaMemcpy(&ar_x_h, ar_x_d, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&ar_y_h, ar_y_d, sizeof(double), cudaMemcpyDeviceToHost);

    // Operación de cálculo final del método de simpson
    h = (double)(b - a) / (N);
    suma_x = ar_x_h * h / 3;
    suma_y = ar_y_h * h / 3;

    // Asignación de valores al arreglo
    sumas[0] = suma_x;
    sumas[1] = suma_y;

    // Liberación de memoria de apuntadores del dispositivo
    cudaFree(ar_x_d);
    cudaFree(ar_y_d);

    return sumas;
}

/*
Realiza asignación de las posiciones iniciales de las párticulas
Parameters
----------
seeds : posParticle**
    Apuntador a arreglo de dos dimensiones de la estructura posParticle
maxParticles: int
    Número de partículas

Returns
----------
*/
void generateSeeds(posParticle **seeds, const unsigned int maxParticles)
{

    unsigned int num_seeds = maxParticles;
    for (int i = 0; i < num_seeds; ++i)
    {
        const double radius = 0.1;
        const double alpha = 2.0f * M_PI * (double)i / (double)num_seeds;
        (seeds)[0][i].x = 0.5f + radius * cos(alpha);
        (seeds)[0][i].y = 0.5f + radius * sin(alpha);
        (seeds)[0][i].t = 0.0;
    }
}

/*
Realiza la graficación de datos de posiciones
Parameters
----------
seeds : posParticle**
    Apuntador a arreglo de dos dimensiones de la estructura posParticle
t_max: int
    Número de instantes de tiempo
n_particles: int
    Número de partículas

Returns
----------
*/
void graph(posParticle **particulas, int t_max, int n_particles)
{
    // Creación de una interfaz Gnuplot
    Gnuplot gp;

    // Definición del archivo de salida
    ofstream salida("output_text");

    // Recorrer la totalidad de los instantes de tiempo
    for (int j = 0; j < t_max; j++)
    {
        // Recorrer todas las partículas para el instante de tiempo j
        for (int i = 0; i < n_particles; i++)
        {
            // Escritura de las posiciones en el archivo de salida con estructura x,y
            salida << particulas[j][i].x << " " << particulas[j][i].y << endl;
        }
        // Escrituras de saltos de línea necesarios para separar los instantes de tiempo para GNUPlot
        salida << endl
               << endl;
    }

    salida.close();

    /* Paso de comandos a la interfaz de GNUPlot, cambiando la salida y el formato a gif y utilizando stats
        para separar los instantes de tiempo y graficación utilizando un ciclo for 
        * Se obtendrá un archivo de saluda .gif */
    gp << "set terminal gif animate delay 10; set output 'animation.gif'; stats 'output_text' nooutput; set xrange [-1:7]; set yrange [-1:7]; do for[i = 1 : int(STATS_blocks)]{ plot 'output_text' index(i - 1) with circles }" << endl;
}


/*
Función principal
Parameters
----------
argc : int
    Número de parámetros
argv: char**
    Arreglo de parámetros de línea de comandos
Returns
0 : int
    Si se ejecutó correctamente
----------
*/
int main(int argc, char **argv)
{
    // Definición de variables con valores por defecto
    N = 256 * 10000;
    n_particles = 500;
    int MAX_T = 5;
    double dt = 0.5;

    // Búsqueda de parámetros de linea de comandos
    if (argc > 0)
    {
        // Recorrido y asignación (de encontrarse el parámetro)
        for (int i = 0; i < argc; i++)
        {
            string param = argv[i];
            if (param == "-dt")
            {
                try
                {
                    dt = std::stod(argv[i + 1]);
                    i++;
                }
                catch (const std::invalid_argument &)
                {
                    std::cout << "If you're using -dt flag provide a value for dt: -dt <value>(DOUBLE)" << std::endl;
                }
            }
            else if (param == "-tmax")
            {
                try
                {
                    MAX_T = std::stoi(argv[i + 1]);
                    i++;
                }
                catch (const std::invalid_argument &)
                {
                    std::cout << "If you're using -tmax flag provide a value for max time: -tmax <value>(INT)" << std::endl;
                }
            }
            else if (param == "-np" || param == "-nparticles")
            {
                try
                {
                    n_particles = std::stoi(argv[i + 1]);
                    /* Debido a la arquitectura de las GPUs estas trabajan en bloques que sean potencias de 2
                        por esto se verifica que el número de partículas sea múltiplo de 32, ya que este número
                        fue el arbitrariamente elegido para el tamaño de los bloques */
                    if (n_particles % 32 != 0)
                    {
                        int div = n_particles / 32;
                        n_particles = 32 * (div + 1);
                        cout << "Adjusting the particles number to optimize for GPU processing" << endl;
                    }
                    i++;
                }
                catch (const std::invalid_argument &)
                {
                    std::cout << "If you're using -np flag provide a value for number of particles: -np <value>(INT)" << std::endl;
                }
            }
        }
    }

    // Cálculo del total de instantes de tiempo
    t_max = (MAX_T / dt) + 1;

    // Impresión de los parámetros del proceso
    cout << endl;
    cout << "|| N particles: " << n_particles << " dt: " << dt << " max time: " << MAX_T << " ||" << endl
         << endl;

    // Creación del arreglo de 2 dimensiones de estructuras posParticle del tamaño del total de pasos de tiempo
    posParticle **particulas = new posParticle *[t_max];

    // Recorrido de las posiciones del arreglo reservando el espacio de memoria del tamaño del número del partículas
    for (int i = 0; i < t_max; i++)
    {
        posParticle *unidad;
        unidad = (struct posParticle *)malloc(n_particles * sizeof(struct posParticle));
        particulas[i] = unidad;
    }

    cout << "Starting..." << endl;
    cout << "Generating initial positions..." << endl;

    // Llamado a la función generateSeeds() para la asignación de posiciones iniciales, pasándole el arreglo previamente definido
    generateSeeds(particulas, n_particles);

    cout << "Initial positions generated" << endl;

    // Ciclo recorriendo el número de instantes de tiempo
    for (int i = 1; i < t_max; i++)
    {
        cout << "- Calculating positions at time " << i * dt << endl;
        double *pos = (double *)malloc(2 * sizeof(double));
        double *pos_d;

        // Llamado a la función integral() para le cálculo de desplazamiento en x y y para el instante de tiempo
        /* *Se aprovecha el hecho de que la función de velocidad sólo depende del tiempo,
        entonces el desplazamiento será igual para todas las partículas en un instante de tiempo t */
        pos = integral(i * dt);
        // El i*dt se realiza para obtener el valor de t en ese instante


        // Definición de arreglos de una sola dimensión de estructuras posParticle
        posParticle *particulas_d, *particulas_h, *particulas_ini_d;

        // Reservación de espacio de memoria para los arreglos del dispositivo
        cudaError_t ret = cudaMalloc((void **)&particulas_d, n_particles * sizeof(struct posParticle));
        ret = cudaMalloc((void **)&particulas_ini_d, n_particles * sizeof(struct posParticle));

        cudaMalloc((void **)&pos_d, 2 * sizeof(double));

        // Reservación de espacio de memoria para el arreglo del host
        particulas_h = (struct posParticle *)malloc(n_particles * sizeof(struct posParticle));

        // Copia de la primera fila de particulas (el primer instante de tiempo) a particulas_h
        memcpy(particulas_h, particulas[0], n_particles * sizeof(struct posParticle));

        // Copia desde el host al dispositivo del arreglo con el instante inicial de tiempo
        ret = cudaMemcpy(particulas_ini_d, particulas_h, n_particles * sizeof(struct posParticle), cudaMemcpyHostToDevice);
        
        // Copia del arreglo pos con los valores de desplazamiento en x y y
        ret = cudaMemcpy(pos_d, pos, 2 * sizeof(double), cudaMemcpyHostToDevice);

        if (ret != cudaSuccess)
        {
            std::cout << "Primera copia: " << cudaGetErrorString(ret) << std::endl;
            return 1;
        }

        // Llamado al kernel con 32 bloques y los parámetros de particulas_d, particulas_ini_d, i * dt, pos_d
        caminado_ind<<<n_particles / 32, 32>>>(particulas_d, particulas_ini_d, i * dt, pos_d);

        // Copia desde el dispositivo al dispositivo del arreglo de estructuras en ese instante de tiempo i
        ret = cudaMemcpy(particulas_h, particulas_d, n_particles * sizeof(struct posParticle), cudaMemcpyDeviceToHost);

        if (ret != cudaSuccess)
        {
            std::cout << "Segunda copia: " << cudaGetErrorString(ret) << std::endl;
            return 1;
        }

        // Copia de la fila del resultado en el instante de tiempo al arreglo de 2 dimensiones particulas en la posición i
        memcpy(particulas[i], particulas_h, n_particles * sizeof(struct posParticle));

        // Liberación de memoria de punteros del dispositivo y el host
        cudaFree(particulas_d);
        cudaFree(pos_d);
        cudaFree(particulas_ini_d);
        free(particulas_h);
        free(pos);
    }

    cout << "Generating graph..." << endl;

    //Llamado a la función graph() para la creación del archivo de salida y la graficación

    graph(particulas, t_max, n_particles);
    cout << "Finished :)" << endl;
    return 0;
}
