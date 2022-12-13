#include <iostream>
#include<stdlib.h>
#include<string.h>
#include <cstdio>
using namespace std;

int main() {
	
	
	char cmd[15];
	char cmd2[15];
	char cmd3[15];
	char cmd4[15];
	char cmd5[15];
	strcpy(cmd, "echo Installing Python");
	system(cmd);
	strcpy(cmd2, R"(python-3.9.12-amd64.exe InstallAllUsers=0 TargetDir="C:\Users\Nick\pytorch-cuda-installer\bin\src" Include_launcher=0 Include_test=0 SimpleInstall=1 SimpleInstallDescription="testing install")");
	system(cmd2);
	std::cout << "Python Installation Complete" << std::endl;

	std::cout << "Installing PyTorch" << std::endl;
	strcpy(cmd3, R"(C:\Users\Nick\pytorch-cuda-installer\bin\src\Scripts\pip3 install torch torchvision torchaudio --no-warn-script-location --extra-index-url https://download.pytorch.org/whl/cu113)");
	system(cmd3);
	std::cout << "Installing PyTorch Complete" << std::endl;

	std::cout << "Installing requirements.txt" << std::endl;
	strcpy(cmd4, R"(C:\Users\Nick\pytorch-cuda-installer\bin\src\Scripts\pip3 install -r C:\Users\Nick\pytorch-cuda-installer\requirements.txt)");
	system(cmd4);
	std::cout << "Installing requirements.txt complete" << std::endl;
	//strcpy(cmd5, "echo CUDA installation complete");
	//system(cmd5);
	//strcpy(cmd3,"python .\\bin\\cli.py");
	//system(cmd3);
	//strcpy(cmd4,"echo Press ENTER to exit...");
	//system(cmd4);
	getchar();
	return 0;
}