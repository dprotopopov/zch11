// gradient.cpp: ���������� ����� ����� ��� ����������� ����������.
//

#include "stdafx.h"
#include <vector>
#include <locale.h>
#include <ctime>
#include <ostream>
#include <iostream>

int n = 3; // ����������� ������������
std::vector<double> a = { 0, 0, 0 }; // ������ ������� �������� ����������
std::vector<double> b = { 100, 100, 100 }; // ������� ������� �������� ����������
std::vector<int> m = { 3, 3, 3 }; // ���������� ��������� �������
double e1 = 1e-7; // �������� ���������� 
double e2 = 1e-7; // �������� ���������� 
double t = 0.5; // �������� ����

std::vector<double> p = { 10, 20, 30 };

// ����������� �������
double f(const std::vector<double> &x, const std::vector<double> &p, int n, int id)
{
	double s = 0;
	for (auto i = 0; i < n; i++) s += (x[i + n*id] - p[i])*(x[i + n*id] - p[i]);
	return s;
}

// �������� ����������� �������
void gradientvector(std::vector<double> &g, const std::vector<double> &x, const std::vector<double> &p, int n, int id)
{
	for (auto i = 0; i < n; i++)
	{
		g[i + n*id] = 2.0*(x[i + n*id] - p[i]);
	}
}

// ��������� ��������������� ����� � ��������� [0;1]
// ���������� �� ������ ������������ ���������� ��������������� �����
double drand()
{
	double x = 0;
	double y = 1;
	for (auto i = 0; i < sizeof(double); i++)
	{
		x += (y /= 256)*(rand() % 256);
	}
	return x;
}


int main(int argc, char* argv[])
{
	// ��������� ��������� � ������� Windows
	// ������� setlocale() ����� ��� ���������, ������ �������� - ��� ��������� ������, � ����� ������ LC_TYPE - ����� ��������, ������ �������� � �������� ������. 
	// ������ ������� ��������� ����� ������ "Russian", ��� ��������� ������ ������� �������, ����� ����� �������� ����� ����� �� ��� � � ��.
	setlocale(LC_ALL, "");

	auto total = 1; for (auto i = 0; i < n; i++) total *= m[i];

	std::vector<int> k(n*total);
	std::vector<double> x(n*total);
	std::vector<double> g(n*total);
	std::vector<double> x1(n*total);
	std::vector<double> fx(total);
	std::vector<double> fx1(total);

#pragma omp parallel for
	for (auto id = 0; id < total; id++)
	{
		//std::clog << "��������������� ���������� ��������������� ����� ��������� �������� �������" << std::endl;
		srand(static_cast<int>(time(nullptr)) ^ id);

		// ���������� ��������� ������ �����
		for (auto i = 0, j = id; i < n; i++)
		{
			k[n*id + i] = j%m[i];
			j /= m[i];
		}

		//std::clog << "����� ��������� �����" << std::endl;
		for (auto i = 0; i < n; i++)
		{
			x[n*id + i] = ((m[i] - k[n*id + i])*a[i] + k[n*id + i] * b[i] + drand()*(b[i] - a[i])) / m[i];
		}
		fx[id] = f(x, p, n, id);

		for (;;)
		{
			gradientvector(g, x, p, n, id);
			double s = 0;
			for (auto i = 0; i < n; i++) s += g[n*id + i] * g[n*id + i];
			s = sqrt(s);
			if (s < e1) break;

			//std::clog << "����� ��������� �����" << std::endl;
			double l = 0;
			for (auto t1 = t;; t1 /= 2)
			{
				for (auto i = 0; i < n; i++)
				{
					x1[n*id + i] = x[n*id+i]-t1*g[n*id+i];
					if (x1[n*id + i] < ((m[i] - k[n*id + i])*a[i] + k[n*id + i] * b[i]) / m[i]) 
						x1[n*id + i] = ((m[i] - k[n*id + i])*a[i] + k[n*id + i] * b[i]) / m[i];
					if (x1[n*id + i] > ((m[i] - k[n*id + i])*a[i] + k[n*id + i] * b[i] + b[i] - a[i]) / m[i]) 
						x1[n*id + i] = ((m[i] - k[n*id + i])*a[i] + k[n*id + i] * b[i] + b[i] - a[i]) / m[i];
				}
				fx1[id] = f(x1, p, n, id);
				l = 0;
				for (auto i = 0; i < n; i++) l += (x[n*id + i] - x1[n*id + i]) * (x[n*id + i] - x1[n*id + i]);
				l = sqrt(l);
				if (fx1[id] < fx[id]) break;
				if (l < e2) break;
			}
			if (l < e2 && abs(fx[id] - fx1[id]) < e2)
			{
				std::copy_n(x1.begin() + n*id, n, x.begin() + n*id);
				fx[id] = fx1[id];
				break;
			}
			std::copy_n(x1.begin() + n*id, n, x.begin() + n*id);
			fx[id] = fx1[id];
		}
	}
	
	// ���������� ����������� ��������
	auto index = 0;
	for (auto id = 1; id < total; id++)
	{
		if (fx[id] < fx[index])
			index = id;
	}

	// ����� �����������

	std::cout << "����� �������� : ";
	for (auto i = 0; i < n; i++)
	{
		std::cout << x[n*index + i];
		if (i < n - 1) std::cout << ",";
	}
	std::cout << std::endl;

	std::cout << "�������� �������� : " << fx[index] << std::endl;

	getchar(); // �������� ����� � ���������� ����� ����������� ���������
	return 0;
}

