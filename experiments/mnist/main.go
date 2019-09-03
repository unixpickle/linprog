package main

import (
	"fmt"
	"image"
	"image/color"
	"image/png"
	"log"
	"math"
	"os"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/essentials"

	"github.com/unixpickle/anynet"
	"github.com/unixpickle/linprog"
	"github.com/unixpickle/mnist"
)

const MaxDelta = 0.2

func main() {
	log.Println("Training/loading classifier...")
	classifier := TrainClassifier()

	data := mnist.LoadTestingDataSet()
	sample := data.Samples[0]

	targetLabel := (sample.Label + 1) % 10
	intensities := GenerateAdversarial(classifier, sample, targetLabel)
	SaveImage("adversarial.png", intensities)
}

func GenerateAdversarial(classifier anynet.Net, sample mnist.Sample, targetLabel int) []float64 {
	log.Printf("Turning a %d into a %d", sample.Label, targetLabel)

	v := &anydiff.Var{Vector: Creator.MakeVectorData(Creator.MakeNumericList(sample.Intensities))}
	activations := classifier[:1].Apply(v, 1)
	outs := classifier[1:].Apply(activations, 1)
	oldProb := math.Exp(Creator.Float64Slice(outs.Output().Data())[targetLabel])
	out := anydiff.Slice(outs, targetLabel, targetLabel+1)
	gradient := anydiff.NewGrad(v)
	out.Propagate(anyvec.Ones(Creator, 1), gradient)

	gradVec := linprog.Vector(Creator.Float64Slice(gradient[v].Data()))
	activationsVec := linprog.Vector(Creator.Float64Slice(activations.Output().Data()))
	weights, biases := ConvertLayer(classifier[0].(*anynet.FC))
	system := CreateLinearProgram(sample.Intensities, gradVec, activationsVec, biases, weights)
	log.Println("Solving linear program...")
	solution, solved := linprog.Simplex(system, linprog.GreedyPivotRule{}, true)
	if !solved {
		essentials.Die("unsolvable system")
	}
	solution = solution[:28*28]

	outs = classifier.Apply(anydiff.NewConst(anyvec.Make(Creator, solution)), 1)
	newProb := math.Exp(Creator.Float64Slice(outs.Output().Data())[targetLabel])

	fmt.Println("Went from", oldProb, "to", newProb)

	return solution
}

func ConvertLayer(layer *anynet.FC) (linprog.Matrix, linprog.Vector) {
	data := Creator.Float64Slice(layer.Weights.Vector.Data())
	return &linprog.DenseMatrix{
		NumRows: layer.OutCount,
		NumCols: layer.InCount,
		Data:    data,
	}, linprog.Vector(Creator.Float64Slice(layer.Biases.Vector.Data()))
}

func CreateLinearProgram(inputs, gradient, activations, biases linprog.Vector,
	weights linprog.Matrix) *linprog.StandardLP {
	numVars := len(gradient)
	constraintRows := []linprog.Vector{}
	constraintValues := linprog.Vector{}
	for i, x := range inputs {
		if x > MaxDelta {
			row := make(linprog.Vector, numVars+1)
			row[i] = 1
			row[numVars] = -1
			numVars++
			constraintRows = append(constraintRows, row)
			constraintValues = append(constraintValues, x-MaxDelta)
		}
		maximum := math.Min(1, x+MaxDelta)
		row := make(linprog.Vector, numVars+1)
		row[i] = 1
		row[numVars] = 1
		numVars++
		constraintRows = append(constraintRows, row)
		constraintValues = append(constraintValues, maximum)
	}
	for i, activation := range activations {
		row := make(linprog.Vector, numVars+1)
		copy(row, weights.CopyRow(i))
		if activation < 0 {
			row[numVars] = 1
		} else {
			row[numVars] = -1
		}
		numVars++
		constraintValues = append(constraintValues, -biases[i])
		constraintRows = append(constraintRows, row)
	}

	matrix := linprog.NewDenseMatrix(len(constraintRows), numVars)
	for i, row := range constraintRows {
		copy(matrix.Data[i*matrix.NumCols:(i+1)*matrix.NumCols], row)
	}

	return &linprog.StandardLP{
		Objective:        append(gradient, make([]float64, numVars-len(gradient))...),
		ConstraintMatrix: matrix,
		ConstraintVector: constraintValues,
	}
}

func SaveImage(name string, intensities []float64) {
	img := image.NewGray(image.Rect(0, 0, 28, 28))
	for i, x := range intensities {
		img.SetGray(i%28, i/28, color.Gray{Y: uint8(x * 255)})
	}
	f, _ := os.Create(name)
	defer f.Close()
	png.Encode(f, img)
}
