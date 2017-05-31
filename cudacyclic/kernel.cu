#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand_kernel.h>

#include <locale.h>
#include <iostream>

int n = 3; // размерность пространства
double a[] = { 0, 0, 0 }; // нижняя граница значения переменных
double b[] = { 100, 100, 100 }; // верхняя граница значения переменных
int m[] = { 10, 10, 10 }; // Количество элементов решётки
double e1 = 1e-7; // точность вычисления 
double e2 = 1e-7; // точность вычисления 
double t = 0.5; // Величина шага

double p[] = { 10, 20, 30 };

// Исследуемая функция
__device__ double f(double *x, double *p, int n, int id)
{
	double s = 0.0;
	for (auto i = 0; i < n; i++) s += (x[i + n*id] - p[i])*(x[i + n*id] - p[i]);
	return s;
}

// Градиент исследуемой функции
__device__ void gradientvector(double *g, double *x, double *p, int n, int id)
{
	for (auto i = 0; i < n; i++)
	{
		g[i + n*id] = 2.0*(x[i + n*id] - p[i]);
	}
}

__device__ double l2(double *x, double *x1, int n, int id)
{
	double s = 0;
	for (auto i = 0; i < n; i++) s += (x[i + n*id] - x1[i + n*id])*(x[i + n*id] - x1[i + n*id]);
	return s;
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

__global__ void cyclicalgorithm(double *fx, double *x, double *g, double *fx1, double *x1, double *x2, double *a, double *b, double *p, int *k, int *m, int n, double t, double e1, double e2, curandState *state, int total)
{
	// Выбор начальной точки
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
	// Вычисление функции
	for (int id = threadIdx.x + blockIdx.x * blockDim.x; id < total; id += gridDim.x*blockDim.x)
	{
		fx[id] = f(x, p, n, id);
	}

	for (int id = threadIdx.x + blockIdx.x * blockDim.x; id < total; id += gridDim.x*blockDim.x)
	{
		for (;;)
		{
			for (int j = 0; j < n; j++) x2[n*id + j] = x[n*id + j];
				
			// Цикл по измерениям
			for (int i = 0; i < n; i++)
			{
				// Алгоритм одномерной оптимизации по направлению

				// Вычисление градиента
				gradientvector(g, x, p, n, id);

				double s = (g[n*id + i]>-g[n*id + i]) ? g[n*id + i] : -g[n*id + i];
				if (s < e1) continue;
				for (double t1 = t;; t1 /= 2)
				{
					for (int j = 0; j < n; j++) x1[n*id + j] = x[n*id + j];
					x1[n*id + i] = x[n*id + i] - t1*g[n*id + i];
					if (x1[n*id + i] < ((m[i] - k[n*id + i])*a[i] + k[n*id + i] * b[i]) / m[i])
						x1[n*id + i] = ((m[i] - k[n*id + i])*a[i] + k[n*id + i] * b[i]) / m[i];
					if (x1[n*id + i] > ((m[i] - k[n*id + i])*a[i] + k[n*id + i] * b[i] + b[i] - a[i]) / m[i])
						x1[n*id + i] = ((m[i] - k[n*id + i])*a[i] + k[n*id + i] * b[i] + b[i] - a[i]) / m[i];
					fx1[id] = f(x1, p, n, id);
					if (fx1[id] < fx[id])
					{
						x[n*id + i] = x1[n*id + i];
						fx[id] = fx1[id];
						break;
					}
					s = (x[n*id + i] > x1[n*id + i]) ? (x[n*id + i] - x1[n*id + i]) : (x1[n*id + i] - x[n*id + i]);
					if (s < e2) break;
				}
			}
			if (l2(x, x2, n, id) < e2*e2) break;
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

	double *devA, *devB, *devG, *devX, *devX1, *devX2, *devP, *devFX, *devFX1;
	double *x, *fx;
	int *devK, *devM;
	cudaMalloc((void **)&devA, n*sizeof(double));
	cudaMalloc((void **)&devB, n*sizeof(double));
	cudaMalloc((void **)&devP, n*sizeof(double));
	cudaMalloc((void **)&devM, n*sizeof(int));
	cudaMalloc((void **)&devK, n*total*sizeof(int));
	cudaMalloc((void **)&devG, n*total*sizeof(double));
	cudaMalloc((void **)&devX, n*total*sizeof(double));
	cudaMalloc((void **)&devX1, n*total*sizeof(double));
	cudaMalloc((void **)&devX2, n*total*sizeof(double));
	cudaMalloc((void **)&devFX, total*sizeof(double));
	cudaMalloc((void **)&devFX1, total*sizeof(double));
	fx = (double *)malloc(total*sizeof(double));
	x = (double *)malloc(n*sizeof(double));

	cudaMemcpy(devA, a, n*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(devB, b, n*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(devP, p, n*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(devM, m, n*sizeof(int), cudaMemcpyHostToDevice);

	fillindex <<<1, N>>>(devK, devM, n, total);

	cyclicalgorithm <<<1, N>>>(devFX, devX, devG, devFX1, devX1, devX2, devA, devB, devP, devK, devM, n, t, e1, e2, devStates, total);


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
	cudaFree(devG);
	cudaFree(devX);
	cudaFree(devX1);
	cudaFree(devX2);
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

