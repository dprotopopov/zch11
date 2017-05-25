
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand_kernel.h>

#include <locale.h>
#include <iostream>

int n = 3; // размерность пространства
double a[] = { 0, 0, 0 }; // нижняя граница значения переменных
double b[] = { 100, 100, 100 }; // верхняя граница значения переменных
int m[] = { 10, 10, 10 }; // Количество элементов решётки
int R = 10000; // количество итераций

double p[] = { 10, 20, 30 };

// Исследуемая функция
// после вызова надо сложить элементы массива s
__global__ void f(double *fx, double *x, double *p, int n, int total)
{
	for (int id = threadIdx.x + blockIdx.x * blockDim.x; id<total; id += gridDim.x*blockDim.x)
	{
		fx[id] = 0;	
		for (int i = 0; i<n; i++)
		{
			fx[id] += (x[n*id + i] - p[i])*(x[n*id + i] - p[i]);				
		}
	}
}

// Инициализация генератора псевдослучайных чисел
__global__ void setuprand(curandState *state, int total)
{
	for (int id = threadIdx.x + blockIdx.x * blockDim.x; id<total; id += gridDim.x*blockDim.x)
		curand_init(1234, id, 0, &state[id]);
}

__global__ void fillindex(int *k, int *m, int n, int total)
{
	for (int id = threadIdx.x + blockIdx.x * blockDim.x; id<total; id += gridDim.x*blockDim.x)
	{
		for (int i = 0, j = id; i < n; i++)
		{
			k[n*id + i] = j%m[i];
			j /= m[i];
		}		
	}
}
// Генератор псевдослучайного вектора
__global__ void randvector(double *x, double *a, double *b, int *k, int *m, int n, curandState *state, int total)
{
	for (int id = threadIdx.x + blockIdx.x * blockDim.x; id < total; id += gridDim.x*blockDim.x)
	{
		curandState localState = state[id];
		for (int i = 0; i < n; i++)
		{
			double p = curand_uniform_double(&localState);
			x[n*id + i] = ((m[i] - k[n*id + i])*a[i] + k[n*id + i] * b[i] + p*(b[i] - a[i])) / m[i];
		}
		state[id] = localState;
	}
}

__global__ void getminimal(double *fx, double *fx1, double *x, double *x1, int n, int total)
{
	for (int id = threadIdx.x + blockIdx.x * blockDim.x; id < total; id += gridDim.x*blockDim.x)
	{
		if (fx1[id] < fx[id])
		{
			fx[id] = fx1[id];
			for (int i = 0; i < n; i++) x[n*id + i] = x1[n*id + i];
		}
	}
}

int main()
{
	// Поддержка кириллицы в консоли Windows
	// Функция setlocale() имеет два параметра, первый параметр - тип категории локали, в нашем случае LC_TYPE - набор символов, второй параметр — значение локали. 
	// Вместо второго аргумента можно писать "Russian", или оставлять пустые двойные кавычки, тогда набор символов будет такой же как и в ОС.
	setlocale(LC_ALL, "");

	int total = 1; for (auto i = 0; i < n; i++) total *= m[i];
	int N = (1 + sqrt(total)>255) ? 255 : (int)(1 + sqrt(total));

	curandState *devStates;
	cudaMalloc((void **)&devStates, total*sizeof(curandState));
	setuprand <<<1, N>>>(devStates, total);

	double *devA, *devB, *devX, *devX1, *devP, *devFX, *devFX1;
	double *x, *fx;
	int *devK, *devM;
	cudaMalloc((void **)&devA, n*sizeof(double));
	cudaMalloc((void **)&devB, n*sizeof(double));
	cudaMalloc((void **)&devP, n*sizeof(double));
	cudaMalloc((void **)&devM, n*sizeof(int));
	cudaMalloc((void **)&devK, n*total*sizeof(int));
	cudaMalloc((void **)&devX, n*total*sizeof(double));
	cudaMalloc((void **)&devX1, n*total*sizeof(double));
	cudaMalloc((void **)&devFX, total*sizeof(double));
	cudaMalloc((void **)&devFX1, total*sizeof(double));
	fx = (double *)malloc(total*sizeof(double));
	x = (double *)malloc(n*sizeof(double));

	cudaMemcpy(devA, a, n*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(devB, b, n*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(devP, p, n*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(devM, m, n*sizeof(int), cudaMemcpyHostToDevice);

	fillindex <<<1, N>>>(devK, devM, n, total);

	// Выбор начальной точки
	randvector <<<1, N>>>(devX, devA, devB, devK, devM, n, devStates, total);
	f <<<1, N>>>(devFX, devX, devP, n, total);

	for (auto r = 0; r < R; r++)
	{
		//std::clog << "Выбор следующей точки" << std::endl;
		randvector <<<1, N>>>(devX1, devA, devB, devK, devM, n, devStates, total);
		f <<<1, N>>>(devFX1, devX1, devP, n, total);
		getminimal <<<1, N>>>(devFX, devFX1, devX, devX1, n, total);
	}

	cudaMemcpy(fx, devFX, total*sizeof(double), cudaMemcpyDeviceToHost);
	// Нахождение наименьшего значения
	int index = 0;
	for (int id = 1; id < total; id++)
	{
		if (fx[id] < fx[index])
			index = id;
	}
	cudaMemcpy(x, &devX[n*index], n*sizeof(double), cudaMemcpyDeviceToHost);

	// Вывод результатов

	std::cout << "Точка минимума : ";
	for (auto i = 0; i < n; i++)
	{
		std::cout << x[i];
		if (i < n - 1) std::cout << ",";
	}
	std::cout << std::endl;

	std::cout << "Значение минимума : " << fx[index] << std::endl;

	free(x);
	free(fx);
	cudaFree(devX);
	cudaFree(devX1);
	cudaFree(devFX);
	cudaFree(devFX1);
	cudaFree(devA);
	cudaFree(devB);
	cudaFree(devP);
	cudaFree(devM);
	cudaFree(devK);
	cudaFree(devStates);

	getchar(); // Ожидание ввода с клавиатуры перед завершением программы
	return 0;
}

