package main

import (
	"log"
	"os"

	"github.com/unixpickle/essentials"
	"github.com/unixpickle/serializer"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anynet/anyff"
	"github.com/unixpickle/anynet/anysgd"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/anyvec/anyvec32"
	"github.com/unixpickle/mnist"
)

var Creator anyvec.Creator

const ModelPath = "trained_model"

func TrainClassifier() anynet.Net {
	Creator = anyvec32.CurrentCreator()

	if _, err := os.Stat(ModelPath); err == nil {
		var net anynet.Net
		essentials.Must(serializer.LoadAny(ModelPath, &net))
		return net
	}

	network := anynet.Net{
		anynet.NewFC(Creator, 28*28, 200),
		anynet.ReLU,
		anynet.NewFC(Creator, 200, 10),
		anynet.LogSoftmax,
	}

	t := &anyff.Trainer{
		Net:     network,
		Cost:    anynet.DotCost{},
		Params:  network.Parameters(),
		Average: true,
	}

	doneChan := make(chan struct{})

	var iterNum int
	s := &anysgd.SGD{
		Fetcher:     t,
		Gradienter:  t,
		Transformer: &anysgd.Adam{},
		Samples:     mnist.LoadTrainingDataSet().AnyNetSamples(Creator),
		Rater:       anysgd.ConstRater(0.001),
		StatusFunc: func(b anysgd.Batch) {
			iterNum++
			if iterNum == 1200 {
				close(doneChan)
			}
		},
		BatchSize: 100,
	}

	s.Run(doneChan)

	serializer.SaveAny(ModelPath, network)

	return network
}

func printStats(net anynet.Net) {
	ts := mnist.LoadTestingDataSet()
	cf := func(in []float64) int {
		vec := Creator.MakeVectorData(Creator.MakeNumericList(in))
		inRes := anydiff.NewConst(vec)
		res := net.Apply(inRes, 1).Output()
		return anyvec.MaxIndex(res)
	}
	log.Println("Validation:", ts.NumCorrect(cf))
	log.Println("Histogram:", ts.CorrectnessHistogram(cf))
}
