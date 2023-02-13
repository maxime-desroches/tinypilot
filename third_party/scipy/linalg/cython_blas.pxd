# This file was generated by _generate_pyx.py.
# Do not edit this file directly.

# Within scipy, these wrappers can be used via relative or absolute cimport.
# Examples:
# from ..linalg cimport cython_blas
# from scipy.linalg cimport cython_blas
# cimport scipy.linalg.cython_blas as cython_blas
# cimport ..linalg.cython_blas as cython_blas

# Within SciPy, if BLAS functions are needed in C/C++/Fortran,
# these wrappers should not be used.
# The original libraries should be linked directly.

ctypedef float s
ctypedef double d
ctypedef float complex c
ctypedef double complex z

cdef void caxpy(int *n, c *ca, c *cx, int *incx, c *cy, int *incy) nogil

cdef void ccopy(int *n, c *cx, int *incx, c *cy, int *incy) nogil

cdef c cdotc(int *n, c *cx, int *incx, c *cy, int *incy) nogil

cdef c cdotu(int *n, c *cx, int *incx, c *cy, int *incy) nogil

cdef void cgbmv(char *trans, int *m, int *n, int *kl, int *ku, c *alpha, c *a, int *lda, c *x, int *incx, c *beta, c *y, int *incy) nogil

cdef void cgemm(char *transa, char *transb, int *m, int *n, int *k, c *alpha, c *a, int *lda, c *b, int *ldb, c *beta, c *c, int *ldc) nogil

cdef void cgemv(char *trans, int *m, int *n, c *alpha, c *a, int *lda, c *x, int *incx, c *beta, c *y, int *incy) nogil

cdef void cgerc(int *m, int *n, c *alpha, c *x, int *incx, c *y, int *incy, c *a, int *lda) nogil

cdef void cgeru(int *m, int *n, c *alpha, c *x, int *incx, c *y, int *incy, c *a, int *lda) nogil

cdef void chbmv(char *uplo, int *n, int *k, c *alpha, c *a, int *lda, c *x, int *incx, c *beta, c *y, int *incy) nogil

cdef void chemm(char *side, char *uplo, int *m, int *n, c *alpha, c *a, int *lda, c *b, int *ldb, c *beta, c *c, int *ldc) nogil

cdef void chemv(char *uplo, int *n, c *alpha, c *a, int *lda, c *x, int *incx, c *beta, c *y, int *incy) nogil

cdef void cher(char *uplo, int *n, s *alpha, c *x, int *incx, c *a, int *lda) nogil

cdef void cher2(char *uplo, int *n, c *alpha, c *x, int *incx, c *y, int *incy, c *a, int *lda) nogil

cdef void cher2k(char *uplo, char *trans, int *n, int *k, c *alpha, c *a, int *lda, c *b, int *ldb, s *beta, c *c, int *ldc) nogil

cdef void cherk(char *uplo, char *trans, int *n, int *k, s *alpha, c *a, int *lda, s *beta, c *c, int *ldc) nogil

cdef void chpmv(char *uplo, int *n, c *alpha, c *ap, c *x, int *incx, c *beta, c *y, int *incy) nogil

cdef void chpr(char *uplo, int *n, s *alpha, c *x, int *incx, c *ap) nogil

cdef void chpr2(char *uplo, int *n, c *alpha, c *x, int *incx, c *y, int *incy, c *ap) nogil

cdef void crotg(c *ca, c *cb, s *c, c *s) nogil

cdef void cscal(int *n, c *ca, c *cx, int *incx) nogil

cdef void csrot(int *n, c *cx, int *incx, c *cy, int *incy, s *c, s *s) nogil

cdef void csscal(int *n, s *sa, c *cx, int *incx) nogil

cdef void cswap(int *n, c *cx, int *incx, c *cy, int *incy) nogil

cdef void csymm(char *side, char *uplo, int *m, int *n, c *alpha, c *a, int *lda, c *b, int *ldb, c *beta, c *c, int *ldc) nogil

cdef void csyr2k(char *uplo, char *trans, int *n, int *k, c *alpha, c *a, int *lda, c *b, int *ldb, c *beta, c *c, int *ldc) nogil

cdef void csyrk(char *uplo, char *trans, int *n, int *k, c *alpha, c *a, int *lda, c *beta, c *c, int *ldc) nogil

cdef void ctbmv(char *uplo, char *trans, char *diag, int *n, int *k, c *a, int *lda, c *x, int *incx) nogil

cdef void ctbsv(char *uplo, char *trans, char *diag, int *n, int *k, c *a, int *lda, c *x, int *incx) nogil

cdef void ctpmv(char *uplo, char *trans, char *diag, int *n, c *ap, c *x, int *incx) nogil

cdef void ctpsv(char *uplo, char *trans, char *diag, int *n, c *ap, c *x, int *incx) nogil

cdef void ctrmm(char *side, char *uplo, char *transa, char *diag, int *m, int *n, c *alpha, c *a, int *lda, c *b, int *ldb) nogil

cdef void ctrmv(char *uplo, char *trans, char *diag, int *n, c *a, int *lda, c *x, int *incx) nogil

cdef void ctrsm(char *side, char *uplo, char *transa, char *diag, int *m, int *n, c *alpha, c *a, int *lda, c *b, int *ldb) nogil

cdef void ctrsv(char *uplo, char *trans, char *diag, int *n, c *a, int *lda, c *x, int *incx) nogil

cdef d dasum(int *n, d *dx, int *incx) nogil

cdef void daxpy(int *n, d *da, d *dx, int *incx, d *dy, int *incy) nogil

cdef d dcabs1(z *z) nogil

cdef void dcopy(int *n, d *dx, int *incx, d *dy, int *incy) nogil

cdef d ddot(int *n, d *dx, int *incx, d *dy, int *incy) nogil

cdef void dgbmv(char *trans, int *m, int *n, int *kl, int *ku, d *alpha, d *a, int *lda, d *x, int *incx, d *beta, d *y, int *incy) nogil

cdef void dgemm(char *transa, char *transb, int *m, int *n, int *k, d *alpha, d *a, int *lda, d *b, int *ldb, d *beta, d *c, int *ldc) nogil

cdef void dgemv(char *trans, int *m, int *n, d *alpha, d *a, int *lda, d *x, int *incx, d *beta, d *y, int *incy) nogil

cdef void dger(int *m, int *n, d *alpha, d *x, int *incx, d *y, int *incy, d *a, int *lda) nogil

cdef d dnrm2(int *n, d *x, int *incx) nogil

cdef void drot(int *n, d *dx, int *incx, d *dy, int *incy, d *c, d *s) nogil

cdef void drotg(d *da, d *db, d *c, d *s) nogil

cdef void drotm(int *n, d *dx, int *incx, d *dy, int *incy, d *dparam) nogil

cdef void drotmg(d *dd1, d *dd2, d *dx1, d *dy1, d *dparam) nogil

cdef void dsbmv(char *uplo, int *n, int *k, d *alpha, d *a, int *lda, d *x, int *incx, d *beta, d *y, int *incy) nogil

cdef void dscal(int *n, d *da, d *dx, int *incx) nogil

cdef d dsdot(int *n, s *sx, int *incx, s *sy, int *incy) nogil

cdef void dspmv(char *uplo, int *n, d *alpha, d *ap, d *x, int *incx, d *beta, d *y, int *incy) nogil

cdef void dspr(char *uplo, int *n, d *alpha, d *x, int *incx, d *ap) nogil

cdef void dspr2(char *uplo, int *n, d *alpha, d *x, int *incx, d *y, int *incy, d *ap) nogil

cdef void dswap(int *n, d *dx, int *incx, d *dy, int *incy) nogil

cdef void dsymm(char *side, char *uplo, int *m, int *n, d *alpha, d *a, int *lda, d *b, int *ldb, d *beta, d *c, int *ldc) nogil

cdef void dsymv(char *uplo, int *n, d *alpha, d *a, int *lda, d *x, int *incx, d *beta, d *y, int *incy) nogil

cdef void dsyr(char *uplo, int *n, d *alpha, d *x, int *incx, d *a, int *lda) nogil

cdef void dsyr2(char *uplo, int *n, d *alpha, d *x, int *incx, d *y, int *incy, d *a, int *lda) nogil

cdef void dsyr2k(char *uplo, char *trans, int *n, int *k, d *alpha, d *a, int *lda, d *b, int *ldb, d *beta, d *c, int *ldc) nogil

cdef void dsyrk(char *uplo, char *trans, int *n, int *k, d *alpha, d *a, int *lda, d *beta, d *c, int *ldc) nogil

cdef void dtbmv(char *uplo, char *trans, char *diag, int *n, int *k, d *a, int *lda, d *x, int *incx) nogil

cdef void dtbsv(char *uplo, char *trans, char *diag, int *n, int *k, d *a, int *lda, d *x, int *incx) nogil

cdef void dtpmv(char *uplo, char *trans, char *diag, int *n, d *ap, d *x, int *incx) nogil

cdef void dtpsv(char *uplo, char *trans, char *diag, int *n, d *ap, d *x, int *incx) nogil

cdef void dtrmm(char *side, char *uplo, char *transa, char *diag, int *m, int *n, d *alpha, d *a, int *lda, d *b, int *ldb) nogil

cdef void dtrmv(char *uplo, char *trans, char *diag, int *n, d *a, int *lda, d *x, int *incx) nogil

cdef void dtrsm(char *side, char *uplo, char *transa, char *diag, int *m, int *n, d *alpha, d *a, int *lda, d *b, int *ldb) nogil

cdef void dtrsv(char *uplo, char *trans, char *diag, int *n, d *a, int *lda, d *x, int *incx) nogil

cdef d dzasum(int *n, z *zx, int *incx) nogil

cdef d dznrm2(int *n, z *x, int *incx) nogil

cdef int icamax(int *n, c *cx, int *incx) nogil

cdef int idamax(int *n, d *dx, int *incx) nogil

cdef int isamax(int *n, s *sx, int *incx) nogil

cdef int izamax(int *n, z *zx, int *incx) nogil

cdef bint lsame(char *ca, char *cb) nogil

cdef s sasum(int *n, s *sx, int *incx) nogil

cdef void saxpy(int *n, s *sa, s *sx, int *incx, s *sy, int *incy) nogil

cdef s scasum(int *n, c *cx, int *incx) nogil

cdef s scnrm2(int *n, c *x, int *incx) nogil

cdef void scopy(int *n, s *sx, int *incx, s *sy, int *incy) nogil

cdef s sdot(int *n, s *sx, int *incx, s *sy, int *incy) nogil

cdef s sdsdot(int *n, s *sb, s *sx, int *incx, s *sy, int *incy) nogil

cdef void sgbmv(char *trans, int *m, int *n, int *kl, int *ku, s *alpha, s *a, int *lda, s *x, int *incx, s *beta, s *y, int *incy) nogil

cdef void sgemm(char *transa, char *transb, int *m, int *n, int *k, s *alpha, s *a, int *lda, s *b, int *ldb, s *beta, s *c, int *ldc) nogil

cdef void sgemv(char *trans, int *m, int *n, s *alpha, s *a, int *lda, s *x, int *incx, s *beta, s *y, int *incy) nogil

cdef void sger(int *m, int *n, s *alpha, s *x, int *incx, s *y, int *incy, s *a, int *lda) nogil

cdef s snrm2(int *n, s *x, int *incx) nogil

cdef void srot(int *n, s *sx, int *incx, s *sy, int *incy, s *c, s *s) nogil

cdef void srotg(s *sa, s *sb, s *c, s *s) nogil

cdef void srotm(int *n, s *sx, int *incx, s *sy, int *incy, s *sparam) nogil

cdef void srotmg(s *sd1, s *sd2, s *sx1, s *sy1, s *sparam) nogil

cdef void ssbmv(char *uplo, int *n, int *k, s *alpha, s *a, int *lda, s *x, int *incx, s *beta, s *y, int *incy) nogil

cdef void sscal(int *n, s *sa, s *sx, int *incx) nogil

cdef void sspmv(char *uplo, int *n, s *alpha, s *ap, s *x, int *incx, s *beta, s *y, int *incy) nogil

cdef void sspr(char *uplo, int *n, s *alpha, s *x, int *incx, s *ap) nogil

cdef void sspr2(char *uplo, int *n, s *alpha, s *x, int *incx, s *y, int *incy, s *ap) nogil

cdef void sswap(int *n, s *sx, int *incx, s *sy, int *incy) nogil

cdef void ssymm(char *side, char *uplo, int *m, int *n, s *alpha, s *a, int *lda, s *b, int *ldb, s *beta, s *c, int *ldc) nogil

cdef void ssymv(char *uplo, int *n, s *alpha, s *a, int *lda, s *x, int *incx, s *beta, s *y, int *incy) nogil

cdef void ssyr(char *uplo, int *n, s *alpha, s *x, int *incx, s *a, int *lda) nogil

cdef void ssyr2(char *uplo, int *n, s *alpha, s *x, int *incx, s *y, int *incy, s *a, int *lda) nogil

cdef void ssyr2k(char *uplo, char *trans, int *n, int *k, s *alpha, s *a, int *lda, s *b, int *ldb, s *beta, s *c, int *ldc) nogil

cdef void ssyrk(char *uplo, char *trans, int *n, int *k, s *alpha, s *a, int *lda, s *beta, s *c, int *ldc) nogil

cdef void stbmv(char *uplo, char *trans, char *diag, int *n, int *k, s *a, int *lda, s *x, int *incx) nogil

cdef void stbsv(char *uplo, char *trans, char *diag, int *n, int *k, s *a, int *lda, s *x, int *incx) nogil

cdef void stpmv(char *uplo, char *trans, char *diag, int *n, s *ap, s *x, int *incx) nogil

cdef void stpsv(char *uplo, char *trans, char *diag, int *n, s *ap, s *x, int *incx) nogil

cdef void strmm(char *side, char *uplo, char *transa, char *diag, int *m, int *n, s *alpha, s *a, int *lda, s *b, int *ldb) nogil

cdef void strmv(char *uplo, char *trans, char *diag, int *n, s *a, int *lda, s *x, int *incx) nogil

cdef void strsm(char *side, char *uplo, char *transa, char *diag, int *m, int *n, s *alpha, s *a, int *lda, s *b, int *ldb) nogil

cdef void strsv(char *uplo, char *trans, char *diag, int *n, s *a, int *lda, s *x, int *incx) nogil

cdef void zaxpy(int *n, z *za, z *zx, int *incx, z *zy, int *incy) nogil

cdef void zcopy(int *n, z *zx, int *incx, z *zy, int *incy) nogil

cdef z zdotc(int *n, z *zx, int *incx, z *zy, int *incy) nogil

cdef z zdotu(int *n, z *zx, int *incx, z *zy, int *incy) nogil

cdef void zdrot(int *n, z *cx, int *incx, z *cy, int *incy, d *c, d *s) nogil

cdef void zdscal(int *n, d *da, z *zx, int *incx) nogil

cdef void zgbmv(char *trans, int *m, int *n, int *kl, int *ku, z *alpha, z *a, int *lda, z *x, int *incx, z *beta, z *y, int *incy) nogil

cdef void zgemm(char *transa, char *transb, int *m, int *n, int *k, z *alpha, z *a, int *lda, z *b, int *ldb, z *beta, z *c, int *ldc) nogil

cdef void zgemv(char *trans, int *m, int *n, z *alpha, z *a, int *lda, z *x, int *incx, z *beta, z *y, int *incy) nogil

cdef void zgerc(int *m, int *n, z *alpha, z *x, int *incx, z *y, int *incy, z *a, int *lda) nogil

cdef void zgeru(int *m, int *n, z *alpha, z *x, int *incx, z *y, int *incy, z *a, int *lda) nogil

cdef void zhbmv(char *uplo, int *n, int *k, z *alpha, z *a, int *lda, z *x, int *incx, z *beta, z *y, int *incy) nogil

cdef void zhemm(char *side, char *uplo, int *m, int *n, z *alpha, z *a, int *lda, z *b, int *ldb, z *beta, z *c, int *ldc) nogil

cdef void zhemv(char *uplo, int *n, z *alpha, z *a, int *lda, z *x, int *incx, z *beta, z *y, int *incy) nogil

cdef void zher(char *uplo, int *n, d *alpha, z *x, int *incx, z *a, int *lda) nogil

cdef void zher2(char *uplo, int *n, z *alpha, z *x, int *incx, z *y, int *incy, z *a, int *lda) nogil

cdef void zher2k(char *uplo, char *trans, int *n, int *k, z *alpha, z *a, int *lda, z *b, int *ldb, d *beta, z *c, int *ldc) nogil

cdef void zherk(char *uplo, char *trans, int *n, int *k, d *alpha, z *a, int *lda, d *beta, z *c, int *ldc) nogil

cdef void zhpmv(char *uplo, int *n, z *alpha, z *ap, z *x, int *incx, z *beta, z *y, int *incy) nogil

cdef void zhpr(char *uplo, int *n, d *alpha, z *x, int *incx, z *ap) nogil

cdef void zhpr2(char *uplo, int *n, z *alpha, z *x, int *incx, z *y, int *incy, z *ap) nogil

cdef void zrotg(z *ca, z *cb, d *c, z *s) nogil

cdef void zscal(int *n, z *za, z *zx, int *incx) nogil

cdef void zswap(int *n, z *zx, int *incx, z *zy, int *incy) nogil

cdef void zsymm(char *side, char *uplo, int *m, int *n, z *alpha, z *a, int *lda, z *b, int *ldb, z *beta, z *c, int *ldc) nogil

cdef void zsyr2k(char *uplo, char *trans, int *n, int *k, z *alpha, z *a, int *lda, z *b, int *ldb, z *beta, z *c, int *ldc) nogil

cdef void zsyrk(char *uplo, char *trans, int *n, int *k, z *alpha, z *a, int *lda, z *beta, z *c, int *ldc) nogil

cdef void ztbmv(char *uplo, char *trans, char *diag, int *n, int *k, z *a, int *lda, z *x, int *incx) nogil

cdef void ztbsv(char *uplo, char *trans, char *diag, int *n, int *k, z *a, int *lda, z *x, int *incx) nogil

cdef void ztpmv(char *uplo, char *trans, char *diag, int *n, z *ap, z *x, int *incx) nogil

cdef void ztpsv(char *uplo, char *trans, char *diag, int *n, z *ap, z *x, int *incx) nogil

cdef void ztrmm(char *side, char *uplo, char *transa, char *diag, int *m, int *n, z *alpha, z *a, int *lda, z *b, int *ldb) nogil

cdef void ztrmv(char *uplo, char *trans, char *diag, int *n, z *a, int *lda, z *x, int *incx) nogil

cdef void ztrsm(char *side, char *uplo, char *transa, char *diag, int *m, int *n, z *alpha, z *a, int *lda, z *b, int *ldb) nogil

cdef void ztrsv(char *uplo, char *trans, char *diag, int *n, z *a, int *lda, z *x, int *incx) nogil
