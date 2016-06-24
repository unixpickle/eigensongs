package eigensongs

import (
	"encoding/json"

	"github.com/gonum/blas"
	"github.com/gonum/blas/blas64"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/serializer"
)

const compressorSerializerType = "github.com/unixpickle/eigensongs.Compressor"

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

// Serialize encodes the Compressor as binary data.
func (c *Compressor) Serialize() ([]byte, error) {
	return json.Marshal(c.rowBasis)
}

// SerializerType returns serializer ID for Compressors.
func (c *Compressor) SerializerType() string {
	return compressorSerializerType
}

func blasMatrix(m *linalg.Matrix) blas64.General {
	return blas64.General{
		Rows:   m.Rows,
		Cols:   m.Cols,
		Stride: m.Cols,
		Data:   m.Data,
	}
}

func init() {
	serializer.RegisterDeserializer(compressorSerializerType,
		func(d []byte) (serializer.Serializer, error) {
			var mat linalg.Matrix
			if err := json.Unmarshal(d, &mat); err != nil {
				return nil, err
			}
			return &Compressor{rowBasis: &mat}, nil
		})
}
