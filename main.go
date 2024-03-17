//go:build ignore

package main

import (
	"bytes"
	"fmt"
	"io"
	"log"
	"os/exec"

	mg "github.com/aybabtme/micrograd"
	"github.com/pkg/browser"
)

func main() {

	dot := bytes.NewBuffer(nil)

	mg.DrawDotGraph(
		example5(),
		dot,
	)

	log.Printf(dot.String())
	if err := openDot(dot); err != nil {
		log.Fatal(err)
	}
}

func example1() *mg.Value {
	a := mg.New(2.0, "a")
	b := mg.New(-3.0, "b")
	c := mg.New(10.0, "c")
	e := a.Mul(b, "e")
	d := e.Add(c, "d")
	f := mg.New(-2.0, "f")
	L := d.Mul(f, "L")
	L.Backward()
	return L
}

func example2() *mg.Value {
	x1 := mg.New(2.0, "x1")
	x2 := mg.New(0.0, "x2")
	w1 := mg.New(-3.0, "w1")
	w2 := mg.New(1.0, "w2")

	b := mg.New(6.8813735870195432, "b")

	x1w1 := x1.Mul(w1, "x1*w1")
	x2w2 := x2.Mul(w2, "x2*w2")

	x1w1x2w2 := x1w1.Add(x2w2, "x1*w1 + x2*w2")

	n := x1w1x2w2.Add(b, "n")

	o := n.Tanh("o")
	o.Backward()
	return o
}

func example3() *mg.Value {
	a := mg.New(3.0, "a")
	b := a.Add(a, "b")

	b.Backward()
	return b
}

func example4() *mg.Value {
	a := mg.New(-2, "a")
	b := mg.New(3, "b")
	d := a.Mul(b, "d")
	e := a.Add(b, "e")
	f := d.Mul(e, "f")

	f.Backward()
	return f
}

func example5() *mg.Value {
	x1 := mg.New(2.0, "x1")
	x2 := mg.New(0.0, "x2")
	w1 := mg.New(-3.0, "w1")
	w2 := mg.New(1.0, "w2")

	b := mg.New(6.8813735870195432, "b")

	x1w1 := x1.Mul(w1, "x1*w1")
	x2w2 := x2.Mul(w2, "x2*w2")

	x1w1x2w2 := x1w1.Add(x2w2, "x1*w1 + x2*w2")

	n := x1w1x2w2.Add(b, "n")

	one := mg.New(1, "1")
	two := mg.New(2, "2")
	e := two.Mul(n, "2n").Exp("e^2n")

	o := (e.Sub(one, "e-1")).Div(e.Add(one, "e+1"), "o")

	o.Backward()
	return o
}

func openDot(r io.Reader) error {
	cmd := exec.Command("dot", "-Tsvg")
	cmd.Stdin = r
	data, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("turning dot into svg: %v", err)
	}
	svg := bytes.NewBuffer(data)
	if err := browser.OpenReader(svg); err != nil {
		return fmt.Errorf("opening browser: %v", err)
	}
	return nil
}
