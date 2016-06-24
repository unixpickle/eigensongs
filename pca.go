package eigensongs

import (
	"errors"
	"math/rand"

	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/num-analysis/linalg/qrdecomp"
	"github.com/unixpickle/wav"
)

const (
	stochasticIterations = 10
	powerIterations      = 10
)

// SolveCompressor computes an efficient compressor for
// a given set of sounds, given the desired input and
// output vector sizes.
func SolveCompressor(sounds []wav.Sound, bigSize, smallSize int) (*Compressor, error) {
	basisMat := linalg.NewMatrix(bigSize, smallSize)
	for i := 0; i < stochasticIterations; i++ {
		mat, err := stochasticNormalMatrix(sounds, bigSize, bigSize)
		if err != nil {
			return nil, err
		}
		for i := 0; i < powerIterations; i++ {
			basisMat = mat.Mul(basisMat)
			basisMat, _ = qrdecomp.Householder(basisMat)
		}
	}
	return &Compressor{
		rowBasis: basisMat.Transpose(),
	}, nil
}

func stochasticNormalMatrix(sounds []wav.Sound, bigSize, count int) (*linalg.Matrix, error) {
	var totalOptions int
	for _, s := range sounds {
		totalOptions += len(s.Samples()) / bigSize
	}
	if totalOptions == 0 {
		return nil, errors.New("not enough samples in any Sound")
	}

	rowMat := linalg.NewMatrix(count, bigSize)
	for i := 0; i < count; i++ {
		sampleIdx := rand.Intn(totalOptions)
		var sampleVec []wav.Sample
		for _, s := range sounds {
			contained := len(s.Samples()) / bigSize
			if contained > sampleIdx {
				sampleVec = s.Samples()[sampleIdx*bigSize : (sampleIdx+1)*bigSize]
				break
			} else {
				sampleIdx -= contained
			}
		}
		dst := rowMat.Data[i*bigSize:]
		for i, x := range sampleVec {
			dst[i] = float64(x)
		}
	}

	return rowMat.Transpose().Mul(rowMat), nil
}
