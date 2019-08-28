package linprog

// A Vector is an n-dimensional list of numbers.
type Vector []float64

// Scale multiplies v by s in place.
func (v Vector) Scale(s float64) {
	for i, x := range v {
		v[i] = x * s
	}
}

// Add adds other * scale to v in place.
func (v Vector) Add(other Vector, scale float64) {
	for i, x := range other {
		v[i] += x * scale
	}
}

// A Matrix is a (potentially sparse) matrix.
type Matrix interface {
	Rows() int
	Cols() int
	At(i, j int) float64
	Set(i, j int, value float64)
	ScaleRow(i int, s float64)
	AddRow(source, dest int, sourceScale float64)
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
	d.Row(i).Scale(s)
}

func (d *DenseMatrix) AddRow(source, dest int, sourceScale float64) {
	d.Row(dest).Add(d.Row(source), sourceScale)
}

func (d *DenseMatrix) Row(i int) Vector {
	if i < 0 || i >= d.NumRows {
		panic("index out of bounds")
	}
	return d.Data[i*d.NumCols : (i+1)*d.NumCols]
}

// A ColumnBlockMatrix is a Matrix composed of one or more
// matrices arranged from left to right. All contained
// matrices must have the same number of rows.
type ColumnBlockMatrix []Matrix

func (c ColumnBlockMatrix) Rows() int {
	return c[0].Rows()
}

func (c ColumnBlockMatrix) Cols() int {
	var total int
	for _, m := range c {
		total += m.Cols()
	}
	return total
}

func (c ColumnBlockMatrix) At(i, j int) float64 {
	for _, m := range c {
		if j < m.Cols() {
			return m.At(i, j)
		}
		j -= m.Cols()
	}
	panic("index out of range")
}

func (c ColumnBlockMatrix) Set(i, j int, value float64) {
	for _, m := range c {
		if j < m.Cols() {
			m.Set(i, j, value)
			return
		}
		j -= m.Cols()
	}
	panic("index out of range")
}

func (c ColumnBlockMatrix) ScaleRow(i int, s float64) {
	for _, m := range c {
		m.ScaleRow(i, s)
	}
}

func (c ColumnBlockMatrix) AddRow(source, dest int, sourceScale float64) {
	for _, m := range c {
		m.AddRow(source, dest, sourceScale)
	}
}

// A RowBlockMatrix is a Matrix composed of one or more
// matrices arranged from top to bottom. All contained
// matrices must have the same number of columns.
type RowBlockMatrix []Matrix

func (r RowBlockMatrix) Rows() int {
	var total int
	for _, m := range r {
		total += m.Rows()
	}
	return total
}

func (r RowBlockMatrix) Cols() int {
	return r[0].Cols()
}

func (r RowBlockMatrix) At(i, j int) float64 {
	for _, m := range r {
		if i < m.Rows() {
			return m.At(i, j)
		}
		i -= m.Rows()
	}
	panic("index out of range")
}

func (r RowBlockMatrix) Set(i, j int, value float64) {
	for _, m := range r {
		if i < m.Rows() {
			m.Set(i, j, value)
			return
		}
		i -= m.Rows()
	}
	panic("index out of range")
}

func (r RowBlockMatrix) ScaleRow(i int, s float64) {
	for _, m := range r {
		if i < m.Rows() {
			m.ScaleRow(i, s)
			return
		}
		i -= m.Rows()
	}
	panic("index out of range")
}

func (r RowBlockMatrix) AddRow(source, dest int, sourceScale float64) {
	var sourceMat Matrix
	var destMat Matrix
	for _, m := range r {
		if sourceMat == nil && source < m.Rows() {
			sourceMat = m
		} else if sourceMat == nil {
			source -= m.Rows()
		}
		if destMat == nil && dest < m.Rows() {
			destMat = m
		} else if destMat == nil {
			dest -= m.Rows()
		}
	}
	if sourceMat == destMat {
		sourceMat.AddRow(source, dest, sourceScale)
	} else {
		cols := sourceMat.Cols()
		for i := 0; i < cols; i++ {
			value := sourceMat.At(source, i)
			if value != 0 {
				destMat.Set(dest, i, destMat.At(dest, i)+value*sourceScale)
			}
		}
	}
}
