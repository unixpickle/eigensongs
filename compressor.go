package eigensongs

import (
	"github.com/gonum/blas"
	"github.com/gonum/blas/blas64"
	"github.com/unixpickle/num-analysis/linalg"
)

// A Compressor can compress large vectors (representing
// sample data) into lower-dimensional vectors.
type Compressor struct {
	rowBasis *linalg.Matrix
}

// Dims returns two dimensions: the size of vectors that
// c can compress, and the size of compressed outputs.
func (c *Compressor) Dims() (full, compressed int) {
	return c.rowBasis.Cols, c.rowBasis.Rows
}

// Compress compresses one or more samples, where each
// sample vector is laid out as one row in the matrix.
func (c *Compressor) Compress(samples *linalg.Matrix) *linalg.Matrix {
	resMat := linalg.NewMatrix(samples.Rows, c.rowBasis.Rows)
	output := blasMatrix(resMat)
	blas64.Gemm(blas.NoTrans, blas.Trans, 1, blasMatrix(samples),
		blasMatrix(c.rowBasis), 0, output)
	return resMat
}

// Decompress decompresses one or more samples, where
// each sample is a row in the matrix.
func (c *Compressor) Decompress(samples *linalg.Matrix) *linalg.Matrix {
	resMat := linalg.NewMatrix(samples.Rows, c.rowBasis.Cols)
	output := blasMatrix(resMat)
	blas64.Gemm(blas.NoTrans, blas.NoTrans, 1, blasMatrix(samples),
		blasMatrix(c.rowBasis), 0, output)
	return resMat
}

func blasMatrix(m *linalg.Matrix) blas64.General {
	return blas64.General{
		Rows:   m.Rows,
		Cols:   m.Cols,
		Stride: m.Cols,
		Data:   m.Data,
	}
}
