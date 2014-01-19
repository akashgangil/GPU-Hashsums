#ifdef __cplusplus
extern "C" {
#endif

void initCudaArray(char** d_A, char* h_A, unsigned int N);
void cudaCRC(char* A, unsigned int N, char* ret);

#ifdef __cplusplus
}
#endif
