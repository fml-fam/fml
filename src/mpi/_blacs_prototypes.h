#ifndef FML_MPI__BLACS_PROTOTYPES_H
#define FML_MPI__BLACS_PROTOTYPES_H


extern void Cblacs_exit(int NotDone);
extern void Cblacs_get(int ictxt, int what, int *val);
extern void Cblacs_gridexit(int Contxt);
extern void Cblacs_gridinfo(int ConTxt, int *nprow, int *npcol, int *myrow, int *mycol);
extern void Cblacs_gridinit(int *ConTxt, char *order, int nprow, int npcol);


#endif
