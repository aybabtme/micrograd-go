package micrograd

import (
	"bytes"
	"fmt"
	"io"
	"math"
	"strconv"
	"sync/atomic"
)

type scalar = float64

var valueID uint64

func nextID() uint64 {
	return atomic.AddUint64(&valueID, 1)
}

type Value struct {
	id       uint64
	data     scalar
	grad     scalar
	prev     [2]*Value
	op       string
	label    string
	backward func()
}

func New(data scalar, label string) *Value {
	return &Value{
		id:    nextID(),
		data:  data,
		label: label,
	}
}

func (self *Value) Add(other *Value, label string) *Value {
	out := &Value{
		id:    nextID(),
		data:  self.data + other.data,
		prev:  [2]*Value{self, other},
		op:    "+",
		label: label,
	}
	out.backward = func() {
		self.grad += 1.0 * out.grad
		other.grad += 1.0 * out.grad
	}
	return out
}

var one = New(-1, "-1")

func (self *Value) Neg(label string) *Value {
	return self.Mul(one, label)
}

func (self *Value) Sub(other *Value, label string) *Value {
	return self.Add(other.Neg("-"+other.label), label)
}

func (self *Value) Mul(other *Value, label string) *Value {
	out := &Value{
		id:    nextID(),
		data:  self.data * other.data,
		prev:  [2]*Value{self, other},
		op:    "*",
		label: label,
	}
	out.backward = func() {
		self.grad += other.data * out.grad
		other.grad += self.data * out.grad
	}
	return out
}

func (self *Value) Div(other *Value, label string) *Value {
	out := self.Mul(other.Pow(-1, ""), "")
	out.label = label
	return out
}

func (self *Value) Pow(other scalar, label string) *Value {
	out := &Value{
		id:    nextID(),
		data:  math.Pow(self.data, other),
		prev:  [2]*Value{self},
		op:    "^" + strconv.FormatFloat(other, 'f', -1, 64),
		label: label,
	}
	out.backward = func() {
		// g(x)  = x^n
		// g(x)' = n * x^(n-1)
		self.grad = other * math.Pow(self.data, other-1) * out.grad
	}
	return out
}

func (self *Value) Tanh(label string) *Value {
	x := self.data
	t := math.Tanh(x)
	out := &Value{
		id:    nextID(),
		data:  t,
		prev:  [2]*Value{self, nil},
		op:    "tanh",
		label: label,
	}
	out.backward = func() {
		self.grad += (1 - math.Pow(t, 2)) * out.grad
	}
	return out
}

func (self *Value) Exp(label string) *Value {
	x := self.data
	t := math.Exp(x)
	out := &Value{
		id:    nextID(),
		data:  t,
		prev:  [2]*Value{self, nil},
		op:    "exp",
		label: label,
	}
	out.backward = func() {
		self.grad += out.data * out.grad
	}
	return out
}

func (self *Value) Backward() {
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
	topo := topoSort(self)

	for i := len(topo) - 1; i >= 0; i-- {
		v := topo[i]
		if v.backward != nil {
			v.backward()
		}
	}
}

func topoSort(v *Value) []*Value {
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

type edge struct{ from, to *Value }

func flattenGraph(v *Value) (nodes []*Value, edges []edge) {
	visited := make(map[*Value]struct{})
	var visit func(v *Value)
	visit = func(v *Value) {
		nodes = append(nodes, v)
		visited[v] = struct{}{}
		for _, child := range v.prev {
			if child == nil {
				continue
			}
			edges = append(edges, edge{from: child, to: v})
			if _, ok := visited[child]; ok {
				continue
			}
			visit(child)
		}
	}
	visit(v)
	return nodes, edges
}

func DrawDotGraph(v *Value, out io.Writer) (int, error) {
	fmt.Fprintf(out, "digraph {\n\trankdir=LR\n")

	nodes := bytes.NewBuffer(nil)
	edges := bytes.NewBuffer(nil)

	n, e := flattenGraph(v)
	for _, n := range n {
		name := strconv.FormatUint(n.id, 10)
		fmt.Fprintf(nodes, "\t"+`%q [label="%s | data %.4f | grad %.4f", shape=record]`+"\n", name, n.label, n.data, n.grad)
		if n.op != "" {
			opname := name + n.op
			fmt.Fprintf(nodes, "\t%q [label=%q]\n", name+n.op, n.op)
			fmt.Fprintf(edges, "\t%q -> %q /* link from op %q to value %q */\n", opname, name, n.op, n.label)
		}
	}

	for _, e := range e {
		from := e.from
		fromName := strconv.FormatUint(from.id, 10)
		to := e.to
		toName := strconv.FormatUint(to.id, 10)
		if to.op != "" {
			toName += to.op
		}
		fmt.Fprintf(edges, "\t%q -> %q /* parent link to %q */\n", fromName, toName, to.label)
	}

	nodes.WriteTo(out)
	edges.WriteTo(out)

	return fmt.Fprintf(out, "}\n")
}
