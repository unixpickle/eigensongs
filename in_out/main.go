package main

import (
	"fmt"
	"io/ioutil"
	"os"

	"github.com/unixpickle/eigensongs"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/serializer"
	"github.com/unixpickle/wav"
)

func main() {
	if len(os.Args) != 4 {
		fmt.Fprintln(os.Stderr, "Usage: in_out <compressor> <in.wav> <out.wav>")
		os.Exit(2)
	}
	compressorPath := os.Args[1]
	inPath := os.Args[2]
	outPath := os.Args[3]

	compressor, err := readCompressor(compressorPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, "Failed to read compressor:", err)
		os.Exit(1)
	}

	soundFile, err := wav.ReadSoundFile(inPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, "Failed to read sound file:", err)
		os.Exit(1)
	}

	compSize, _ := compressor.Dims()

	samples := soundFile.Samples()
	for i := 0; i < len(samples)-compSize; i += compSize {
		subSet := samples[i : i+compSize]
		mat := &linalg.Matrix{
			Rows: 1,
			Cols: len(subSet),
			Data: samplesToVec(subSet),
		}
		compressed := compressor.Compress(mat)
		result := compressor.Decompress(compressed).Data
		copyVectorToSamples(subSet, result)
	}
	soundFile.SetSamples(samples)

	outFile, err := os.Create(outPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, "Failed to open output file:", err)
		os.Exit(1)
	}
	defer outFile.Close()
	if err := soundFile.Write(outFile); err != nil {
		fmt.Fprintln(os.Stderr, "Failed to write output:", err)
		os.Exit(1)
	}
}

func readCompressor(filePath string) (*eigensongs.Compressor, error) {
	compressorData, err := ioutil.ReadFile(filePath)
	if err != nil {
		return nil, err
	}
	compressorS, err := serializer.DeserializeWithType(compressorData)
	if err != nil {
		return nil, err
	}
	compressor, ok := compressorS.(*eigensongs.Compressor)
	if !ok {
		return nil, fmt.Errorf("unexpected type: %T", compressor)
	}
	return compressor, nil
}

func samplesToVec(s []wav.Sample) linalg.Vector {
	res := make(linalg.Vector, len(s))
	for i, x := range s {
		res[i] = float64(x)
	}
	return res
}

func copyVectorToSamples(s []wav.Sample, v linalg.Vector) {
	for i, x := range v {
		s[i] = wav.Sample(x)
	}
}
