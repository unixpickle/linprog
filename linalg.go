package linprog

import "math/rand"

// A Vector is an n-dimensional list of numbers.
type Vector []float64

// NewVectorRandom creates a vector with normally
// distributed entries.
func NewVectorRandom(size int) Vector {
	res := make(Vector, size)
	for i := range res {
		res[i] = rand.NormFloat64()
	}
	return res
}

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

// Col creates a column matrix from the vector.
func (v Vector) Col() *DenseMatrix {
	return &DenseMatrix{
		NumRows: len(v),
		NumCols: 1,
		Data:    v,
	}
}

// Row creates a row matrix from the vector.
func (v Vector) Row() *DenseMatrix {
	return &DenseMatrix{
		NumRows: 1,
		NumCols: len(v),
		Data:    v,
	}
}

// AbsMax gets the maximum absoute value in the vector.
func (v Vector) AbsMax() float64 {
	var res float64
	for _, x := range v {
		if x > res {
			res = x
		} else if -x > res {
			res = -x
		}
	}
	return res
}

// A Matrix is a (potentially sparse) matrix.
type Matrix interface {
	Rows() int
	Cols() int
	At(i, j int) float64
	Set(i, j int, value float64)
	ScaleRow(i int, s float64)
	AddRow(source, dest int, sourceScale float64)
	AbsMax() float64
	Copy() Matrix
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

func (d *DenseMatrix) AbsMax() float64 {
	res := 0.0
	for _, x := range d.Data {
		if x > res {
			res = x
		} else if -x > res {
			res = -x
		}
	}
	return res
}

func (d *DenseMatrix) Copy() Matrix {
	return &DenseMatrix{
		NumRows: d.NumRows,
		NumCols: d.NumCols,
		Data:    append([]float64{}, d.Data...),
	}
}

// A SparseMatrix is a Matrix that lazily populates itself
// as more and more entries get filled.
type SparseMatrix struct {
	NumRows int
	NumCols int
	RowData []map[int]float64
}

func NewSparseMatrix(rows, cols int) *SparseMatrix {
	res := &SparseMatrix{
		NumRows: rows,
		NumCols: cols,
		RowData: make([]map[int]float64, rows),
	}
	for i := range res.RowData {
		res.RowData[i] = map[int]float64{}
	}
	return res
}

func NewSparseMatrixIdentity(size int) *SparseMatrix {
	res := NewSparseMatrix(size, size)
	for i := 0; i < size; i++ {
		res.Set(i, i, 1)
	}
	return res
}

func (s *SparseMatrix) Rows() int {
	return s.NumRows
}

func (s *SparseMatrix) Cols() int {
	return s.NumCols
}

func (s *SparseMatrix) At(i, j int) float64 {
	return s.RowData[i][j]
}

func (s *SparseMatrix) Set(i, j int, value float64) {
	if value == 0 {
		delete(s.RowData[i], j)
	} else {
		s.RowData[i][j] = value
	}
}

func (s *SparseMatrix) ScaleRow(i int, scale float64) {
	row := s.RowData[i]
	for k := range row {
		row[k] *= scale
	}
}

func (s *SparseMatrix) AddRow(source, dest int, sourceScale float64) {
	sourceRow := s.RowData[source]
	destRow := s.RowData[dest]
	for i, x := range sourceRow {
		destRow[i] += x * sourceScale
	}
}

func (s *SparseMatrix) AbsMax() float64 {
	res := 0.0
	for _, row := range s.RowData {
		for _, x := range row {
			if x > res {
				res = x
			} else if -x > res {
				res = -x
			}
		}
	}
	return res
}

func (s *SparseMatrix) Copy() Matrix {
	res := &SparseMatrix{
		NumRows: s.NumRows,
		NumCols: s.NumCols,
		RowData: make([]map[int]float64, s.NumRows),
	}
	for i, row := range s.RowData {
		newRow := map[int]float64{}
		for k, v := range row {
			newRow[k] = v
		}
		res.RowData[i] = newRow
	}
	return res
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

func (c ColumnBlockMatrix) AbsMax() float64 {
	res := 0.0
	for _, m := range c {
		if max := m.AbsMax(); max > res {
			res = max
		}
	}
	return res
}

func (c ColumnBlockMatrix) Copy() Matrix {
	var res ColumnBlockMatrix
	for _, m := range c {
		res = append(res, m.Copy())
	}
	return res
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
	sourceIdx := -1
	destIdx := -1
	for i, m := range r {
		if sourceMat == nil && source < m.Rows() {
			sourceMat = m
			sourceIdx = i
		} else if sourceMat == nil {
			source -= m.Rows()
		}
		if destMat == nil && dest < m.Rows() {
			destMat = m
			destIdx = i
		} else if destMat == nil {
			dest -= m.Rows()
		}
	}
	if sourceIdx == destIdx {
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

func (r RowBlockMatrix) AbsMax() float64 {
	res := 0.0
	for _, m := range r {
		if max := m.AbsMax(); max > res {
			res = max
		}
	}
	return res
}

func (r RowBlockMatrix) Copy() Matrix {
	var res RowBlockMatrix
	for _, m := range r {
		res = append(res, m.Copy())
	}
	return res
}
