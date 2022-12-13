#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <fstream>
#include <algorithm>
#include <cstdio>
using namespace std;

bool IsCudaInstalled()
{
	FILE *fpipe;
std:
	string command = "nvcc --version";
	char c = 0;

	if (0 == (fpipe = (FILE *)popen(command.c_str(), "r")))
	{
		perror("popen() failed.");
		exit(EXIT_FAILURE);
	}

	std::string str;
	std::string tot_str;
	while (fread(&c, sizeof c, 1, fpipe))
	{
		str = c;
		tot_str = tot_str + str;
	}
	// std::cout << tot_str << std::endl;
	std::size_t found = tot_str.find("Cuda compiler driver");
	if (found != std::string::npos)
	{

		pclose(fpipe);
		return true;
	}
	else
	{
		pclose(fpipe);
		return false;
	}
}

int main()
{
	// std::cout << "hello world" << std::endl;

	if (IsCudaInstalled())
	{
		std::cout << "cuda found!" << std::endl;
	}
	else
	{
		std::cout << "cuda not found!" << std::endl;
	}

	return 0;
}