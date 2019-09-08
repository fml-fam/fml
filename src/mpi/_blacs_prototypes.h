#ifndef FML_MPI__BLACS_PROTOTYPES_H
#define FML_MPI__BLACS_PROTOTYPES_H


#ifdef __cplusplus
extern "C" {
#endif

extern void Cblacs_exit(int NotDone);
extern void Cblacs_get(int ictxt, int what, int *val);
extern void Cblacs_gridexit(int Contxt);
extern void Cblacs_gridinfo(int ConTxt, int *nprow, int *npcol, int *myrow, int *mycol);
extern void Cblacs_gridinit(int *ConTxt, char *order, int nprow, int npcol);
extern void Cblacs_barrier(int ictxt, const char *scope);
extern void Cblacs_pinfo(int *mypnum, int *nprocs);
extern void Cigsum2d(int ConTxt, const char *scope, char *top, int m, int n, int *A, int lda, int rdest, int cdest);
extern void Csgsum2d(int ConTxt, const char *scope, char *top, int m, int n, float *A, int lda, int rdest, int cdest);
extern void Cdgsum2d(int ConTxt, const char *scope, char *top, int m, int n, double *A, int lda, int rdest, int cdest);
extern void Cigesd2d(int ConTxt, int m, int n, const int *A, int lda, int rdest, int cdest);
extern void Csgesd2d(int ConTxt, int m, int n, const float *A, int lda, int rdest, int cdest);
extern void Cdgesd2d(int ConTxt, int m, int n, const double *A, int lda, int rdest, int cdest);
extern void Cigerv2d(int ConTxt, int m, int n, int *A, int lda, int rdest, int cdest);
extern void Csgerv2d(int ConTxt, int m, int n, float *A, int lda, int rdest, int cdest);
extern void Cdgerv2d(int ConTxt, int m, int n, double *A, int lda, int rdest, int cdest);
extern void Cigebs2d(int ConTxt, const char *scope, char *top, int m, int n, int *A, int lda);
extern void Csgebs2d(int ConTxt, const char *scope, char *top, int m, int n, float *A, int lda);
extern void Cdgebs2d(int ConTxt, const char *scope, char *top, int m, int n, double *A, int lda);
extern void Cigebr2d(int ConTxt, const char *scope, char *top, int m, int n, int *A, int lda, int rsrc, int csrc);
extern void Csgebr2d(int ConTxt, const char *scope, char *top, int m, int n, float *A, int lda, int rsrc, int csrc);
extern void Cdgebr2d(int ConTxt, const char *scope, char *top, int m, int n, double *A, int lda, int rsrc, int csrc);


#ifdef __cplusplus
}
#endif


#endif
