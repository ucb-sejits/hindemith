Implementation of the [Caffe](http://caffe.berkeleyvision.org/) framework in
Python using Hindemith.

# Setup
Install Caffe with it's Python bindings ([instructions](http://caffe.berkeleyvision.org/installation.html)) and be sure to add it to your PYTHONPATH.

Install Hindemith's main directory either with a sym-linked egg (`pip install
-e /path/to/hindemith`) or `python setup.py install`.


# Benchmarks
```
python benchmarks/alexnet.py
```

# Demo
Requires OpenCV.
```
python demo/webcam.py
```
