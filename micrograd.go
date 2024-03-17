package micrograd

import (
	"bytes"
	"fmt"
	"io"
	"math"
)

type scalar = float64

type Value struct {
	data     scalar
	grad     scalar
	prev     [2]*Value
	op       string
	label    string
	backward func()
}

func New(data scalar, label string) *Value {
	return &Value{
		data:  data,
		label: label,
	}
}

func (self *Value) Add(other *Value, label string) *Value {
	out := &Value{
		data:  self.data + other.data,
		prev:  [2]*Value{self, other},
		op:    "+",
		label: label,
	}
	out.backward = func() {
		self.grad = 1.0 * out.grad
		other.grad = 1.0 * out.grad
	}
	return out
}

func (self *Value) Mul(other *Value, label string) *Value {
	out := &Value{
		data:  self.data * other.data,
		prev:  [2]*Value{self, other},
		op:    "*",
		label: label,
	}
	out.backward = func() {
		self.grad = other.data * out.grad
		other.grad = self.data * out.grad
	}
	return out
}

func (self *Value) Tanh(label string) *Value {
	x := self.data
	t := math.Tanh(x)
	out := &Value{
		data:  t,
		prev:  [2]*Value{self, nil},
		op:    "tanh",
		label: label,
	}
	out.backward = func() {
		self.grad = (1 - math.Pow(t, 2)) * out.grad
	}
	return out
}

func (self *Value) Backprop() {
	if self.grad == 0.0 {
		self.grad = 1.0 // initialize it
	}
	// self.backpropRecurse()
	self.backpropTopoSort()
}

func (self *Value) backpropRecurse() {
	if self.backward != nil {
		self.backward()
	}
	for _, prev := range self.prev {
		if prev != nil {
			prev.backpropRecurse()
		}
	}
}

func (self *Value) backpropTopoSort() {
	topo := TopoSort(self)

	for i := len(topo) - 1; i >= 0; i-- {
		v := topo[i]
		if v.backward != nil {
			v.backward()
		}
	}
}

func TopoSort(v *Value) []*Value {
	var topo []*Value
	visited := make(map[*Value]struct{})

	var buildTopo func(v *Value)
	buildTopo = func(v *Value) {
		if _, ok := visited[v]; ok {
			return
		}
		visited[v] = struct{}{}
		for _, child := range v.prev {
			if child == nil {
				continue
			}
			buildTopo(child)
		}
		topo = append(topo, v)
	}
	buildTopo(v)
	return topo
}

func (self *Value) String() string {
	return fmt.Sprintf("Value(data=%f)", self.data)
}

func (self *Value) dotvisit(parent *Value, parentname string, nodes, edges *bytes.Buffer) {
	name := fmt.Sprintf("%p", &self)
	fmt.Fprintf(nodes, "\t"+`%q [label="%s | data %.4f | grad %.4f", shape=record]`+"\n", name, self.label, self.data, self.grad)

	linkname := name
	if self.op != "" {
		opname := name + self.op
		fmt.Fprintf(nodes, "\t%q [label=%q]\n", name+self.op, self.op)
		fmt.Fprintf(edges, "\t%q -> %q /* link from op %q to value %q */\n", opname, name, self.op, self.label)
		linkname = opname
	}

	if parent != nil {
		fmt.Fprintf(edges, "\t%q -> %q /* parent link to %q */\n", name, parentname, parent.label)
	}

	for _, prev := range self.prev {
		if prev != nil {
			prev.dotvisit(self, linkname, nodes, edges)
		}
	}
}

func DotGraph(v *Value, out io.Writer) (int, error) {
	nodes := bytes.NewBuffer(nil)
	edges := bytes.NewBuffer(nil)

	fmt.Fprintf(nodes, "digraph {\n\trankdir=LR\n")
	v.dotvisit(nil, "", nodes, edges)

	edges.WriteTo(nodes)

	fmt.Fprintf(nodes, "}\n")

	return out.Write(nodes.Bytes())
}
