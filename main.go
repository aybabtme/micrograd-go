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
	// a := mg.New(2.0, "a")
	// b := mg.New(-3.0, "b")
	// c := mg.New(10.0, "c")
	// e := a.Mul(b, "e")
	// d := e.Add(c, "d")
	// f := mg.New(-2.0, "f")
	// L := d.Mul(f, "L")
	// L.Backprop()
	// mg.DotGraph(L, dot)

	// log.Print(L)

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
	o.Backprop()

	dot := bytes.NewBuffer(nil)

	mg.DotGraph(o, dot)

	log.Printf(dot.String())
	if err := openDot(dot); err != nil {
		log.Fatal(err)
	}
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
