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
	a := mg.New(2.0, "a")
	b := mg.New(-3.0, "b")
	c := mg.New(10.0, "c")
	e := a.Mul(b, "e")
	d := e.Add(c, "d")
	f := mg.New(-2.0, "f")
	L := d.Mul(f, "L")

	dot := bytes.NewBuffer(nil)

	L.Backprop()

	mg.DotGraph(L, dot)

	log.Print(L)
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
