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
extern void Cblacs_barrier(int ictxt, char *scope);
extern void Cblacs_pinfo(int *mypnum, int *nprocs);


#ifdef __cplusplus
}
#endif


#endif
