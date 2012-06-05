;;; -*- Coding: utf-8; Mode: Lisp; Syntax: Common-Lisp; -*-

(in-package :svm.wss3)

;; Vector which used in CLML have to be specialized to double-float.
;; This function makes CLML style vector from list.
(defun list->clml-vector (list)
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
      (setf (aref product-array i) (list->clml-vector (append elem '(1.0d0))))
      (incf i))
    (loop for elem in negative-set do
      (setf (aref product-array i) (list->clml-vector (append elem '(-1.0d0))))
      (incf i))
    product-array))

(defun prediction-dataset (trained-svm dataset)
  (mapcar (lambda (datapoint)
	    (funcall trained-svm datapoint))
	  dataset))

(defun test (trained-svm positive-set negative-set)
  (let* ((count-positive-correct-prediction
	  (count-if (lambda (datapoint)
		      (> (funcall trained-svm (list->clml-vector datapoint)) 0))
		    positive-set))
	 (count-negative-correct-prediction
	  (count-if (lambda (datapoint)
		      (< (funcall trained-svm (list->clml-vector datapoint)) 0))
		    negative-set))
	 (total-correct-count (+ count-positive-correct-prediction
				 count-negative-correct-prediction))
	 (total-sample-size  (+ (length positive-set)
				(length negative-set)))
	 (correct-rate  (* (/ total-correct-count total-sample-size) 100d0)))
    (format t "~A / ~A (~f %) cases are correct.~%"
	    total-correct-count total-sample-size correct-rate)
    correct-rate))

(defun cross-validation (n positive-set negative-set kernel &key (c 10) (weight 1.0d0))
  (let* ((splited-posi-set (wiz:split-equally positive-set n))
	 (splited-nega-set (wiz:split-equally negative-set n))
	 (average-validity
	  (/ (loop for i from 0 to (1- n)
		   summing
		   (let* ((positive-set (apply #'append (wiz:remove-nth i splited-posi-set)))
			  (negative-set (apply #'append (wiz:remove-nth i splited-nega-set)))
			  (positive-test-set (nth i splited-posi-set))
			  (negative-test-set (nth i splited-nega-set))
			  (training-vector (make-dataset positive-set negative-set))
			  (trained-svm (make-svm-learner training-vector kernel
							 :c c :weight weight)))
		     (test trained-svm positive-test-set negative-test-set)))
	     n)))
    (format t "Average validity: ~f~%" average-validity)
    average-validity))
