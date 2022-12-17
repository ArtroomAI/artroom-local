#include <iostream>
#include <sstream>
#include<stdlib.h>
#include <stdio.h>
#include<string.h>
#include <fstream>
#include <algorithm>
#include <cstdio>
#include<conio.h>
#include<windows.h>
using namespace std;

// Config file with installation path defaults to %UserProfile%
// User can select own installation path during nsis --> Updates the config file
// Program also moves, users have artroom in new folder
// Install to that directory. If user updates, should check install_dir from config and update that one

// Will include setting in artroom where user can select it (or provide tutorial for moving it)
// If they do it themselves just by moving it, will throw an alert that says "Hey not found! Please update settings"

bool isInstalled(std::string command, std::string searchTerm) 
{
    FILE *fpipe;
    //std::string command = "nvcc --version";
    char c = 0;
    if (0 == (fpipe = (FILE*)popen(command.c_str(), "r")))
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
    //std::cout << tot_str << std::endl;
    std::size_t found = tot_str.find(searchTerm.c_str());
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
std::string getCmdOutput(std::string command) 
{
    FILE *fpipe;
    //std::string command = "nvcc --version";
    char c = 0;
    if (0 == (fpipe = (FILE*)popen(command.c_str(), "r")))
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
	return tot_str;
}


int main(int argc, char** argv) {

    std::string install_loc;
    std::string torch_cuda_ver;
    std::string cuda_release;
    std::string stable_repo;
    std::ifstream cFile ("config.txt");
	
    if (cFile.is_open())
    {
        std::string line;
        std::cout << "Reading config setup: " << std::endl;
        while(getline(cFile, line))
       {
            //line.erase(std::remove_if(line.begin(), line.end(), isspace), line.end());
            if( line.empty() || line[0] == '#' )
            {
                continue;
            }
            auto delimiterPos = line.find("=");
            std::string name = line.substr(0, delimiterPos);
            std::string value = line.substr(delimiterPos + 1);
            std::cout << "   " << name << "    " << value << '\n';
            if(name == "INSTALL_LOCATION")
            {
                install_loc = value;
            }
            else if(name == "PYTORCH_CUDA_VERSION")
            {
                torch_cuda_ver = value;
            }
            else if(name == "CUDA_RELEASE")
            {
                cuda_release = value;
            }
            else if(name == "STABLE_DIFF_REPO")
            {
                stable_repo = value;
            }
        }
    }
    else 
    {
        // std::cerr << "Couldn't open config file for reading.\n";
    }
    
/*     std::string python_path = install_loc +  R"(\bin\src)";
    std::string python_install = R"(python-3.9.12-amd64.exe InstallAllUsers=0 TargetDir=")" + python_path + R"(" Include_launcher=0 Include_test=0 SimpleInstall=1 SimpleInstallDescription="Installing Python dependencies within ArtRoom generator app")";
    std::string pip_path = python_path +  R"(\Scripts\pip3)";
    std::string cu_path = "https://download.pytorch.org/whl/" + torch_cuda_ver;
    std::string req_path = install_loc + R"(\requirements.txt)";
    std::string torch_install = pip_path + " install torch torchvision torchaudio --no-warn-script-location --extra-index-url " + cu_path;
    std::string req_install = pip_path + " install -r " + req_path;
    std::string cuda_install = install_loc + R"(\cuda_release\)" + cuda_release; */
    
    //std::string stable_yaml_path = "conda env create -f " + install_loc +  R"(\stable-diffusion-main\environment.yaml)";
    
/*  std::cout << python_install <<'\n';
    std::cout << torch_install <<'\n';
    std::cout << req_install <<'\n'; 
    std::cout << cuda_install <<'\n';*/
    ////////////////////////////////////
    //////// python Installation ///////
    ////////////////////////////////////
/*  std::cout << "Installing Python" << std::endl;
    std::cout << "   Follow Install instructions in Python install application and close when done" << std::endl;
    system(python_install.c_str());
    std::cout << "   Python Installation Complete" << std::endl;
    std::cout << "-------------------------" << std::endl;
    ////////////////////////////////////
    /////// pytorch Installation ///////
    ////////////////////////////////////
    std::cout << "Installing PyTorch" << std::endl;
    system(torch_install.c_str());
    std::cout << "   PyTorch Installation Complete" << std::endl;
    std::cout << "-------------------------" << std::endl;
    ////////////////////////////////////
    ///requirements.txt Installation ///
    ////////////////////////////////////
    std::cout << "Installing requirements.txt" << std::endl;
    system(req_install.c_str());
    std::cout << "   requirements.txt installation complete" << std::endl;
    std::cout << "-------------------------" << std::endl; */
    
/* 	std::string install_ref_file1 = "echo %UserProfile%\\AppData\\Local\\artroom_install.log";
	std::string install_ref_file;
	install_ref_file = getCmdOutput(install_ref_file1);
	//remove \n from end
	install_ref_file.pop_back(); */
	
	std::string install_ref_file1 = "echo %UserProfile%";
	std::string install_ref_file;
	std::string user_profile;
	user_profile = getCmdOutput(install_ref_file1);
	//remove \n from end
	user_profile.pop_back();
	install_ref_file = user_profile + "\\AppData\\Local\\artroom_install.log";
	
	std::string install_path_raw;
	std::string install_path;
	std::string skipWeights = "0";
	if (argc >= 2)
	{
		skipWeights = argv[1];
	}
	
	if (argc > 2)
	{
		install_path_raw = "";
		for (int i = 2; i < argc; i++)
		{
			install_path_raw = install_path_raw + argv[i];
			if (i < argc - 1)
			{
				install_path_raw = install_path_raw + " ";
			}
		}
		install_path = install_path_raw + "\\artroom";
	}
	else
	{
		cout << "No Install path provided as input, try finding artroom path in \\AppData\\Local\\artroom_install.log\n";
		
		std::ifstream infile(install_ref_file);
		std::string line;
		if (infile.is_open())
		{
			while(std::getline(infile, line))
			{
/* 				std::istringstream iss(line);
				iss >> install_path_raw; */
				install_path_raw = line;
			}
			std::cout << "install path: " << install_path_raw << std::endl;
			install_path = install_path_raw + "\\artroom";
		}
		else
		{
			cout << "Cant find \\AppData\\Local\\artroom_install.log, defaulting to user profile area\n";
			install_path_raw = user_profile;
			install_path = install_path_raw + "\\artroom";
		}
		//system("pause");
		//return 0;
	}

	
/* 	std::cout << "install ref file: " << install_ref_file << "test" << std::endl;
	std::cout << "raw: " << "C:\\Users\\Nick\\AppData\\Local\\artroom_install.log" << "test" << std::endl;
	system("pause"); */
	
	
	//write install path to ref file in user profile appdata
	ofstream fw(install_ref_file, std::ofstream::out);
	
	//check if file was successfully opened for writing
	if (fw.is_open())
	{
	  //store install_path  to text file
		fw << install_path_raw << "\n";
		fw.close();
	}
	else
	{
		cout << "unable to open: " << install_ref_file << ", install failed\n";
		system("pause");
		return 0;
	}
	
	
	std::cout << "install path: " << install_path << std::endl;
	std::cout << "install ref file: " << install_ref_file << std::endl;
	
	std::string install_conda = "\"" + install_path + "\\miniconda3\\Scripts\\conda\"";
	std::string install_activate = "\"" + install_path + "\\miniconda3\\condabin\\activate\"";
	std::string install_ldm = "\"" + install_path + "\\miniconda3\\envs\\artroom-ldm\"";
	std::string temp;
	
	HANDLE hInput;
    DWORD prev_mode;
    hInput = GetStdHandle(STD_INPUT_HANDLE);
    GetConsoleMode(hInput, &prev_mode); 
    SetConsoleMode(hInput, ENABLE_EXTENDED_FLAGS);    
	std::cout << "------------------------------" << std::endl;
    std::cout << "Setting up Artroom App. Please wait until installer finished before using the Artroom App." << std::endl;
    std::cout << "Create artroom Directory..." << std::endl;
	temp = "mkdir \"" + install_path +"\"" + " > artroomlog.txt & type artroomlog.txt";
	system(temp.c_str());
	//system("mkdir \"%UserProfile%\\artroom\" > artroomlog.txt & type artroomlog.txt");
    std::cout << "------------------------------" << std::endl;
    std::cout << "Installing CONDA Environment..." << std::endl;
    //system("rmdir -s -q file_to_delte %UserProfile%\\artroom\\miniconda3");
    std::cout << "If it freezes at any point, please try pressing enter or trying again. " << std::endl;
    std::cout << "(Doesn't always happen, but sometimes could get hung up on something)" << std::endl;

	std::string str;
	std::string command = "\"" + install_conda + " info --envs\"";
	//std::string command = "\"\"%UserProfile%\\artroom\\miniconda3\\Scripts\\conda\" info --envs\"";
    std::string searchTerm ="conda environments";
    if (isInstalled(command, searchTerm))
    {
		std::cout << "miniconda3 already found in Environment PATH" << std::endl;
    }
    else
    {
		std::cout << "Installing localized miniconda" << std::endl;
		// system("powershell -command \"Invoke-WebRequest -Uri https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe -OutFile ~\\artroom\\miniconda.exe\"");
		std::cout << "Setting up CONDA Environment..." << std::endl;
		temp = "\"start \"test\" /B /WAIT \"miniconda.exe\" /InstallationType=JustMe /AddToPath=0 /RegisterPython=0 /S /D=" + install_path + "\\miniconda3\"" + " >> artroomlog.txt & type artroomlog.txt";
		system(temp.c_str());
		//system("\"start \"test\" /B /WAIT \"miniconda.exe\" /InstallationType=JustMe /AddToPath=0 /RegisterPython=0 /S /D=%UserProfile%\\artroom\\miniconda3\" >> artroomlog.txt & type artroomlog.txt");	
		std::cout << "Updating CONDA Environment..." << std::endl;
		temp = "\"" + install_activate + " && " + install_conda + " update -n base -c defaults conda -y\"" + " >> artroomlog.txt & type artroomlog.txt";
		system(temp.c_str());
		//system("\"\"%UserProfile%\\artroom\\miniconda3\\condabin\\activate\" && \"%UserProfile%\\artroom\\miniconda3\\Scripts\\conda\" update -n base -c defaults conda -y\" >> artroomlog.txt & type artroomlog.txt");
    }
    
    std::cout << "Create artroom-ldm conda environment" << std::endl;
    command = "\"" + install_activate + " && " + install_conda + " info --envs\"";
	//command = "\"\"%UserProfile%\\artroom\\miniconda3\\condabin\\activate\" && \"%USERPROFILE%\\artroom\\miniconda3\\Scripts\\conda\" info --envs\"";
    searchTerm =R"(\envs\artroom-ldm)";
    if (isInstalled(command, searchTerm))
    {
		//std::cout << "Updating Existing artroom-ldm  environment" << std::endl;
		//system("\"\"%UserProfile%\\artroom\\miniconda3\\condabin\\activate\" && \"%UserProfile%\\artroom\\miniconda3\\Scripts\\conda\" env update --name artroom-ldm --file stable-diffusion/environment.yaml --prune\"");
		std::cout << "Installing clean Stable Diffusion in environment" << std::endl;
		temp = "\"" + install_activate + " && " + install_conda + " env remove -n artroom-ldm\"";
		system(temp.c_str());
        //system("\"\"%UserProfile%\\artroom\\miniconda3\\condabin\\activate\" && \"%UserProfile%\\artroom\\miniconda3\\Scripts\\conda\" env remove -n artroom-ldm\"");
        std::cout << "Installing Stable Diffusion libraries. Please wait, this takes a while :)" << std::endl;
		temp = "\"" + install_activate + " && " + install_conda + " env create -f stable-diffusion/environment.yaml\"";
		system(temp.c_str());
        //system("\"\"%UserProfile%\\artroom\\miniconda3\\condabin\\activate\" && \"%UserProfile%\\artroom\\miniconda3\\Scripts\\conda\" env create -f stable-diffusion/environment.yaml\"");
    }
    else
    {       
        std::cout << "Installing Stable Diffusion libraries. Please wait, this takes a while :)" << std::endl;
		temp = "\"" + install_conda + " config --set ssl_verify true\"";
		system(temp.c_str());
		temp = "\"" + install_activate + " && " + install_conda + " env create -f stable-diffusion/environment.yaml\"";
		system(temp.c_str());
		//system("\"\"%UserProfile%\\artroom\\miniconda3\\Scripts\\conda\" config --set ssl_verify true\"");
        //system("\"\"%UserProfile%\\artroom\\miniconda3\\condabin\\activate\" && \"%UserProfile%\\artroom\\miniconda3\\Scripts\\conda\" env create -f stable-diffusion/environment.yaml\"");
    }
	temp = "\"" + install_activate + " && " + install_conda + " clean --all --yes\"";
	system(temp.c_str());
	//system("\"\"%UserProfile%\\artroom\\miniconda3\\condabin\\activate\" && \"%UserProfile%\\artroom\\miniconda3\\Scripts\\conda\" clean --all --yes\"");
	
    // std::cout << "Downloading model weights (these are a few gigs, so may take a few minutes)" << std::endl;
    // system("%UserProfile%\\artroom\\miniconda3\\condabin\\activate && conda run -p %UserProfile%/artroom/miniconda3/envs/artroom-ldm python model_downloader.py");
    ////////////////////////////////////
    ///////// CUDA Installation ////////
    ////////////////////////////////////
    // std::cout << "-------------------------" << std::endl;
    // std::cout << "Installing CUDA Drivers" << std::endl;
    // command = "nvcc --version";
    // searchTerm ="Cuda compiler driver";
    // if (isInstalled(command, searchTerm))
    // {
    //  std::cout << "   cuda already found!" << std::endl;
    // }
    // else
    // {
    //  std::cout << "   cuda not found!" << std::endl;
    //  std::cout << "   Follow Install instructions in Nvidia CUDA install application and close when done" << std::endl;
    //  system(cuda_install.c_str());
    // }

    std::cout << "Installing Stable Diffusion Model Weights (these are a few gigs, so may take a few minutes)." << std::endl;
    // system("\"\"%UserProfile%\\artroom\\miniconda3\\envs\\artroom-ldm\\python\"  model_downloader.py\"");
	
	temp = "\"" + install_conda + " run --no-capture-output -p " + install_ldm + " python artroom_helpers/model_downloader.py\" " + skipWeights;
	system(temp.c_str());
    //system("\"\"%UserProfile%\\artroom\\miniconda3\\Scripts\\conda\" run --no-capture-output -p \"%UserProfile%/artroom/miniconda3/envs/artroom-ldm\" python artroom_helpers/model_downloader.py\"");   
    
    std::cout << "Artroom Installation Complete" << std::endl;
    std::cout << "-----------------------------" << std::endl;
    
    //system("pause");
    return 0;
}