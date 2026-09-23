// DRAM read-bandwidth probe for the expert-streaming pattern: N threads each stream their slice of a big
// buffer with AVX2 loads. Compares 4 KB pages with 2 MB buf2m pages (needs SeLockMemoryPrivilege).
// usage: membw [size_mb] [threads-csv]
#include <windows.h>
#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

typedef struct { const uint8_t * base; size_t bytes; float sink; } job_t;

static DWORD WINAPI worker(LPVOID arg) {
    job_t * j = (job_t *) arg;
    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps(), acc2 = _mm256_setzero_ps(), acc3 = _mm256_setzero_ps();
    const uint8_t * p = j->base;
    const uint8_t * end = j->base + j->bytes;
    for (; p + 128 <= end; p += 128) {
        acc0 = _mm256_add_ps(acc0, _mm256_load_ps((const float *) (p +  0)));
        acc1 = _mm256_add_ps(acc1, _mm256_load_ps((const float *) (p + 32)));
        acc2 = _mm256_add_ps(acc2, _mm256_load_ps((const float *) (p + 64)));
        acc3 = _mm256_add_ps(acc3, _mm256_load_ps((const float *) (p + 96)));
    }
    __m256 s = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
    float tmp[8];
    _mm256_storeu_ps(tmp, s);
    j->sink = tmp[0] + tmp[3] + tmp[7];
    return 0;
}

static int enable_lock_privilege(void) {
    HANDLE tok;
    if (!OpenProcessToken(GetCurrentProcess(), TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, &tok)) return 0;
    TOKEN_PRIVILEGES tp;
    tp.PrivilegeCount = 1;
    tp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;
    if (!LookupPrivilegeValueA(NULL, "SeLockMemoryPrivilege", &tp.Privileges[0].Luid)) return 0;
    AdjustTokenPrivileges(tok, FALSE, &tp, 0, NULL, NULL);
    return GetLastError() == ERROR_SUCCESS;
}

static void run(const char * label, uint8_t * buf, size_t bytes, const char * threads_csv) {
    char csv[256];
    snprintf(csv, sizeof(csv), "%s", threads_csv);
    for (char * tok = strtok(csv, ","); tok; tok = strtok(NULL, ",")) {
        int nt = atoi(tok);
        if (nt < 1 || nt > 64) continue;
        double best = 0;
        for (int rep = 0; rep < 3; rep++) {
            job_t jobs[64];
            HANDLE th[64];
            size_t slice = (bytes / nt) & ~(size_t) 4095;
            LARGE_INTEGER f, t0, t1;
            QueryPerformanceFrequency(&f);
            QueryPerformanceCounter(&t0);
            for (int i = 0; i < nt; i++) {
                jobs[i].base = buf + (size_t) i * slice; jobs[i].bytes = slice;
                th[i] = CreateThread(NULL, 0, worker, &jobs[i], 0, NULL);
            }
            WaitForMultipleObjects(nt, th, TRUE, INFINITE);
            QueryPerformanceCounter(&t1);
            for (int i = 0; i < nt; i++) CloseHandle(th[i]);
            double s = (double) (t1.QuadPart - t0.QuadPart) / (double) f.QuadPart;
            double gbs = (double) slice * nt / s / 1e9;
            if (gbs > best) best = gbs;
        }
        printf("%-12s threads=%2d  %6.2f GB/s\n", label, nt, best);
    }
}

int main(int argc, char ** argv) {
    size_t mb = argc > 1 ? (size_t) atoi(argv[1]) : 4096;
    const char * threads = argc > 2 ? argv[2] : "1,2,4,6,8";
    size_t bytes = mb << 20;

    uint8_t * buf4k = (uint8_t *) VirtualAlloc(NULL, bytes, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    if (!buf4k) { printf("VirtualAlloc failed\n"); return 1; }
    for (size_t i = 0; i < bytes; i += 4096) buf4k[i] = (uint8_t) i;
    run("4K pages", buf4k, bytes, threads);
    VirtualFree(buf4k, 0, MEM_RELEASE);

    int priv = enable_lock_privilege();
    size_t lp = GetLargePageMinimum();
    size_t lbytes = (bytes + lp - 1) / lp * lp;
    uint8_t * buf2m = priv ? (uint8_t *) VirtualAlloc(NULL, lbytes, MEM_COMMIT | MEM_RESERVE | MEM_LARGE_PAGES, PAGE_READWRITE) : NULL;
    if (!buf2m) {
        printf("buf2m pages unavailable (privilege %d, minimum %zu, error %lu)\n", priv, lp, GetLastError());
        return 0;
    }
    for (size_t i = 0; i < lbytes; i += 4096) buf2m[i] = (uint8_t) i;
    run("2M pages", buf2m, lbytes, threads);
    VirtualFree(buf2m, 0, MEM_RELEASE);
    return 0;
}
