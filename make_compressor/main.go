package main

import (
	"errors"
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/unixpickle/eigensongs"
	"github.com/unixpickle/serializer"
	"github.com/unixpickle/wav"
)

const outputPerms = 0755

func main() {
	if len(os.Args) != 5 {
		fmt.Fprintln(os.Stderr, "Usage: make_compressor <compressor out> <wav dir> "+
			"<in dim> <out dim>")
		os.Exit(2)
	}
	outputFile := os.Args[1]
	wavDir := os.Args[2]
	inDim, err := strconv.Atoi(os.Args[3])
	if err != nil {
		fmt.Fprintln(os.Stderr, "Invalid input dimension:", os.Args[3])
		os.Exit(2)
	}
	outDim, err := strconv.Atoi(os.Args[4])
	if err != nil {
		fmt.Fprintln(os.Stderr, "Invalid output dimension:", os.Args[3])
		os.Exit(2)
	}

	log.Println("Reading sounds...")
	sounds, err := readSounds(wavDir)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	log.Println("Running PCA (this may take some time)...")
	compressor, err := eigensongs.SolveCompressor(sounds, inDim, outDim)
	if err != nil {
		fmt.Fprintln(os.Stderr, "Failed to solve compressor:", err)
		os.Exit(1)
	}

	log.Println("Saving...")
	outData, err := serializer.SerializeWithType(compressor)
	if err != nil {
		fmt.Fprintln(os.Stderr, "Failed to serialize:", err)
		os.Exit(1)
	}
	if err := ioutil.WriteFile(outputFile, outData, outputPerms); err != nil {
		fmt.Fprintln(os.Stderr, "Failed to save output:", err)
		os.Exit(1)
	}
}

func readSounds(wavDir string) ([]wav.Sound, error) {
	contents, err := ioutil.ReadDir(wavDir)
	if err != nil {
		return nil, err
	}

	var sounds []wav.Sound
	for _, obj := range contents {
		if strings.HasPrefix(obj.Name(), ".") {
			continue
		}
		p := filepath.Join(wavDir, obj.Name())
		sound, err := wav.ReadSoundFile(p)
		if err != nil {
			return nil, errors.New("file " + p + ": " + err.Error())
		}
		sounds = append(sounds, sound)
	}

	return sounds, nil
}
