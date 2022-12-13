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
   strcpy(cmd, "echo Installing CUDA");
   system(cmd);
   strcpy(cmd2, ".\\bin\\cuda.exe -s");
   system(cmd2);
   strcpy(cmd5, "echo CUDA installation complete");
   system(cmd5);
   strcpy(cmd3,"python .\\bin\\cli.py");
   system(cmd3);
   strcpy(cmd4,"echo Press ENTER to exit...");
   system(cmd4);
   getchar();
   return 0;
}