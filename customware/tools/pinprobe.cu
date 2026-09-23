// Can this Windows box pin tens of GB of host memory for CUDA DMA?
// usage: pinprobe <gb_to_cudaMallocHost> [file_to_map_and_cudaHostRegister_first_1GB]
#include <cuda_runtime.h>
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static double now() { LARGE_INTEGER f, t; QueryPerformanceFrequency(&f); QueryPerformanceCounter(&t); return (double) t.QuadPart / f.QuadPart; }

int main(int argc, char ** argv) {
    if (argc > 1 && strcmp(argv[1], "loop") == 0) {
        // how much can be pinned in total with 2 GB allocations?
        size_t chunk = (size_t) 2 << 30, total = 0;
        for (int i = 0; i < 64; i++) {
            void * q = NULL;
            if (cudaMallocHost(&q, chunk) != cudaSuccess) { printf("pin loop: failed at %zu GB total: %s\n", total >> 30, cudaGetErrorString(cudaGetLastError())); break; }
            memset(q, 1, chunk);
            total += chunk;
        }
        printf("pin loop: %zu GB pinned in 2 GB chunks (not freed until exit)\n", total >> 30);
        return 0;
    }
    size_t gb = argc > 1 ? (size_t) atoi(argv[1]) : 8;
    size_t bytes = gb << 30;
    void * p = NULL;
    double t0 = now();
    cudaError_t err = cudaMallocHost(&p, bytes);
    double t1 = now();
    printf("cudaMallocHost(%zu GB): %s (%.1f s)\n", gb, cudaGetErrorString(err), t1 - t0);
    if (err == cudaSuccess) {
        // touch + time a device copy of 1 GB out of it
        memset(p, 1, bytes);
        void * d = NULL;
        size_t chunk = (size_t) 1 << 30;
        if (cudaMalloc(&d, chunk) == cudaSuccess) {
            cudaMemcpy(d, p, chunk, cudaMemcpyHostToDevice);
            double a = now();
            for (int i = 0; i < 4; i++) cudaMemcpy(d, (char *) p + i * chunk, chunk, cudaMemcpyHostToDevice);
            cudaDeviceSynchronize();
            double b = now();
            printf("H2D from pinned: %.1f GB/s\n", 4.0 / (b - a));
            // scattered 1 MB chunks, like expert streaming
            a = now();
            for (int i = 0; i < 512; i++) cudaMemcpyAsync((char *) d + (i % 1024) * ((size_t) 1 << 20), (char *) p + (size_t) (i * 7919 % 4096) * ((size_t) 1 << 20), 1 << 20, cudaMemcpyHostToDevice, 0);
            cudaDeviceSynchronize();
            b = now();
            printf("H2D 512 x 1 MB scattered from pinned: %.1f GB/s\n", 0.5 / (b - a));
            cudaFree(d);
        }
        cudaFreeHost(p);
    }
    // pageable baseline for the scattered pattern
    {
        size_t pb = (size_t) 4 << 30;
        char * q = (char *) malloc(pb);
        memset(q, 1, pb);
        void * d = NULL;
        if (cudaMalloc(&d, (size_t) 1 << 30) == cudaSuccess) {
            double a = now();
            for (int i = 0; i < 512; i++) cudaMemcpyAsync((char *) d + (i % 1024) * ((size_t) 1 << 20), q + (size_t) (i * 7919 % 4096) * ((size_t) 1 << 20), 1 << 20, cudaMemcpyHostToDevice, 0);
            cudaDeviceSynchronize();
            double b = now();
            printf("H2D 512 x 1 MB scattered from pageable: %.1f GB/s\n", 0.5 / (b - a));
            cudaFree(d);
        }
        free(q);
    }
    if (argc > 2) {
        HANDLE h = CreateFileA(argv[2], GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
        HANDLE m = CreateFileMappingA(h, NULL, PAGE_READONLY, 0, 0, NULL);
        void * v = MapViewOfFile(m, FILE_MAP_READ, 0, 0, 0);
        printf("mapped %s at %p\n", argv[2], v);
        size_t reg = (size_t) 1 << 30;
        err = cudaHostRegister(v, reg, cudaHostRegisterPortable | cudaHostRegisterReadOnly);
        printf("cudaHostRegister(1 GB of file mapping): %s\n", cudaGetErrorString(err));
        if (err == cudaSuccess) cudaHostUnregister(v);
    }
    return 0;
}
