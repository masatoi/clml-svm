;;; -*- Coding: utf-8; Mode: Lisp; Syntax: Common-Lisp; -*-

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;; Hard Margin SVM ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(in-package :svm.mu)

;; Obtain training dataset. Note that number must be double-float.
(defparameter *positive-set*
  '((8.0d0 8.0d0) (8.0d0 20.0d0) (8.0d0 44.0d0) (8.0d0 56.0d0) (12.0d0 32.0d0) (16.0d0 16.0d0) (16.0d0 48.0d0)
    (24.0d0 20.0d0) (24.0d0 32.0d0) (24.0d0 44.0d0) (28.0d0 8.0d0) (32.0d0 52.0d0) (36.0d0 16.0d0)))

(defparameter *negative-set*
  '((36.0d0 24.0d0) (36.0d0 36.0d0) (44.0d0 8.0d0) (44.0d0 44.0d0) (44.0d0 56.0d0)
    (48.0d0 16.0d0) (48.0d0 28.0d0) (56.0d0 8.0d0) (56.0d0 44.0d0) (56.0d0 52.0d0)))

;; Train SVM model with linear kernel.
(defparameter linear-fcn
  (svm +linear-kernel+ *positive-set* *negative-set*))

;; Make prediction with the trained SVM model.
(funcall linear-fcn (car *positive-set*))
(mapcar linear-fcn (append *positive-set* *negative-set*))


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;; Soft Margin SVM ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(in-package :svm.wss3)

;; Vector which used in CLML have to be specialized to double-float.
;; This function makes CLML style vector from list.
(defun make-clml-vector (list)
  (let* ((len (length list))
	 (vec (vector:make-dvec len)))
    (loop for i from 0 to (1- len) do
      (setf (aref vec i) (* (nth i list) 1.0d0)))
    vec))


(defun make-dataset (positive-set negative-set)
  (let* ((n-of-posi-data (length positive-set))
	 (n-of-nega-data (length negative-set))
	 (product-array (make-array (+ n-of-posi-data n-of-nega-data)))
	 (i 0))
    (loop for elem in positive-set do
      (setf (aref product-array i) (make-clml-vector (append elem '(1.0d0))))
      (incf i))
    (loop for elem in negative-set do
      (setf (aref product-array i) (make-clml-vector (append elem '(-1.0d0))))
      (incf i))
    product-array))

(require :wiz-util)

(defparameter *positive-set*
  (mapcar #'list
	  (wiz:n-times-collect 100 (wiz:random-normal :mean 0.0d0 :sd 1.0d0))
	  (wiz:n-times-collect 100 (wiz:random-normal :mean 1.0d0 :sd 1.0d0))))

(defparameter *negative-set*
  (mapcar #'list
	  (wiz:n-times-collect 100 (wiz:random-normal :mean 1.0d0 :sd 1.0d0))
	  (wiz:n-times-collect 100 (wiz:random-normal :mean 0.0d0 :sd 1.0d0))))

;; Plot training dataset
(wiz:plot-lists (list (mapcar #'cadr *positive-set*)
		      (mapcar #'cadr *negative-set*))
		:x-lists (list (mapcar #'car *positive-set*)
			       (mapcar #'car *negative-set*))
		:style 'points)

(defparameter *positive-set-test*
  (mapcar #'list
	  (wiz:n-times-collect 100 (wiz:random-normal :mean 0.0d0 :sd 1.0d0))
	  (wiz:n-times-collect 100 (wiz:random-normal :mean 1.0d0 :sd 1.0d0))))

(defparameter *negative-set-test*
  (mapcar #'list
	  (wiz:n-times-collect 100 (wiz:random-normal :mean 1.0d0 :sd 1.0d0))
	  (wiz:n-times-collect 100 (wiz:random-normal :mean 0.0d0 :sd 1.0d0))))


(defparameter training-vector (make-dataset *positive-set* *negative-set*))

(defparameter trained-svm (make-svm-learner training-vector
					    (make-rbf-kernel :gamma 0.05)
					    :c 10))

(defparameter predicted-positive-set
  (remove-if-not (lambda (datapoint)
		   (> (funcall trained-svm datapoint) 0))
		 (mapcar #'make-clml-vector (append *positive-set-test* *negative-set-test*))))

(defparameter predicted-negative-set
  (remove-if-not (lambda (datapoint)
		   (< (funcall trained-svm datapoint) 0))
		 (mapcar #'make-clml-vector (append *positive-set-test* *negative-set-test*))))

;; Plot test dataset.
(wiz:plot-lists (list (mapcar #'cadr *positive-set-test*)
		      (mapcar #'cadr *negative-set-test*))
		:x-lists (list (mapcar #'car *positive-set-test*)
			       (mapcar #'car *negative-set-test*))
		:style 'points)

;; Plot prediction result with respect to test dataset.

(wiz:plot-lists (list (mapcar (lambda (p) (aref p 1)) predicted-positive-set)
		      (mapcar (lambda (p) (aref p 1)) predicted-negative-set))
		:x-lists (list (mapcar (lambda (p) (aref p 0)) predicted-positive-set)
			       (mapcar (lambda (p) (aref p 0)) predicted-negative-set))
		:style 'points)
