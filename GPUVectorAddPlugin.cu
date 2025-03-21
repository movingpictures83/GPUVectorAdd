#include <emmintrin.h>
#include <sys/time.h> 
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <iostream>
#include <iomanip>
#include <fstream>

#include "GPUVectorAddPlugin.h"

void GPUVectorAddPlugin::input(std::string file) {
 inputfile = file;
 readParameterFile(file);
 N = atoi(myParameters["N"].c_str());
 A = (float*) malloc(N*sizeof(float));
 B = (float*) malloc(N*sizeof(float));
 C = (float*) malloc(N*sizeof(float));
 std::ifstream myinput((std::string(PluginManager::prefix())+myParameters["vector1"]).c_str(), std::ios::in);
 int i;
 for (i = 0; i < N; ++i) {
	float k;
	myinput >> k;
        A[i] = k;
 }
 std::ifstream myinput2((std::string(PluginManager::prefix())+myParameters["vector2"]).c_str(), std::ios::in);
 for (i = 0; i < N; ++i) {
	float k;
	myinput2 >> k;
        B[i] = k;
 }
}




void GPUVectorAddPlugin::run() {
	float *pA;
	float *pB;
	float *pC;
cudaMalloc((void**)&pA, (N)*sizeof(float));
cudaMalloc((void**)&pB, (N)*sizeof(float));
cudaMalloc((void**)&pC, (N)*sizeof(float));
cudaMemcpy(pA, A, (N)*sizeof(float), cudaMemcpyHostToDevice);
cudaMemcpy(pB, B, (N)*sizeof(float), cudaMemcpyHostToDevice);
printf("***Add on %d x %d Matrix on GPU***\n",N,N);
vecAdd<<<1,N>>>(pA, pB, pC, N);
cudaMemcpy(C, pC, (N)*sizeof(float), cudaMemcpyDeviceToHost);

cudaFree(pA);
cudaFree(pB);
cudaFree(pC);

}

void GPUVectorAddPlugin::output(std::string file) {
	std::ofstream outfile(file.c_str(), std::ios::out);
	int i, j;
        for (i = 0; i < N; ++i){
		outfile << C[i];//std::setprecision(0) << a[i*N+j];
		outfile << "\n";
	}
	free(A);
	free(B);
	free(C);
}



PluginProxy<GPUVectorAddPlugin> GPUVectorAddPluginProxy = PluginProxy<GPUVectorAddPlugin>("GPUVectorAdd", PluginManager::getInstance());


