(require :clml-svm)
(require :clml-read-data)

(in-package :svm)

(defparameter a1a (read-data:read-libsvm-data-from-file "/home/wiz/datasets/a1a"))
(defparameter a1a.t (read-data:read-libsvm-data-from-file "/home/wiz/datasets/a1a.t"))
(grid-search-by-cv 3 a1a)
(grid-search a1a a1a.t)

(defparameter a1a.model (make-svm-model a1a (make-rbf-kernel :gamma (expt 2d0 -2)) :c 1d0))


(multiple-value-bind (scaled-vector scale-params)
    (autoscale diabetes)
  (defparameter diabetes.scaled scaled-vector)
  (defparameter scale-parameters scale-params))

(defparameter diabetes.model (make-svm-model diabetes.scaled (make-rbf-kernel :gamma (expt 2d0 -2)) :c 1d0))
