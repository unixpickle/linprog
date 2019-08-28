package linprog

// A Vector is an n-dimensional list of numbers.
type Vector []float64

// A Matrix is a (potentially sparse) matrix.
type Matrix interface {
	Rows() int
	Cols() int
	At(i, j int) float64
	Set(i, j int, value float64)
	ScaleRow(i int, s float64)
	AddRow(source, dest int)
}

// A DenseMatrix is a Matrix that stores every entry
// explicitly in memory.
type DenseMatrix struct {
	NumRows int
	NumCols int
	Data    []float64
}

func NewDenseMatrix(rows, cols int) *DenseMatrix {
	return &DenseMatrix{
		NumRows: rows,
		NumCols: cols,
		Data:    make([]float64, rows*cols),
	}
}

func (d *DenseMatrix) Rows() int {
	return d.NumRows
}

func (d *DenseMatrix) Cols() int {
	return d.NumCols
}

func (d *DenseMatrix) At(i, j int) float64 {
	if i < 0 || i >= d.NumRows || j < 0 || j >= d.NumCols {
		panic("index out of bounds")
	}
	return d.Data[j+i*d.NumCols]
}

func (d *DenseMatrix) Set(i, j int, value float64) {
	if i < 0 || i >= d.NumRows || j < 0 || j >= d.NumCols {
		panic("index out of bounds")
	}
	d.Data[j+i*d.NumCols] = value
}

func (d *DenseMatrix) ScaleRow(i int, s float64) {
	if i < 0 || i >= d.NumRows {
		panic("index out of bounds")
	}
	for j := d.NumCols * i; j < d.NumCols*(i+1); j++ {
		d.Data[j] *= s
	}
}

func (d *DenseMatrix) AddRow(source, dest int) {
	if source < 0 || source >= d.NumRows || dest < 0 || dest >= d.NumRows {
		panic("index out of bounds")
	}
	sourceRow := d.Data[source*d.NumCols : (source+1)*d.NumCols]
	destRow := d.Data[dest*d.NumCols : (dest+1)*d.NumCols]
	for i, x := range sourceRow {
		destRow[i] += x
	}
}
