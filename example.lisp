;;; -*- Coding: utf-8; Mode: Lisp; -*-

(in-package :svm)

(defun random-normal (&key (mean 0d0) (sd 1d0))
  (let ((alpha (random 1.0d0))
	(beta  (random 1.0d0)))
    (+ (* sd
	  (sqrt (* -2 (log alpha)))
	  (sin (* 2 pi beta)))
       mean)))

(defun make-training-vector (positive-set negative-set)
  (let ((training-vector (make-array (+ (length positive-set) (length negative-set)))))
    (loop for i from 0 to (1- (length positive-set))
	  for v in positive-set do
      (setf (aref training-vector i) (make-array (1+ (length v)) :element-type 'double-float :initial-contents (wiz:append1 v 1.0d0))))
    (loop for i from (length positive-set) to (1- (+ (length positive-set) (length negative-set)))
	  for v in negative-set do
      (setf (aref training-vector i) (make-array (1+ (length v)) :element-type 'double-float :initial-contents (wiz:append1 v -1.0d0))))
    training-vector))

(defparameter *scale* 20000)

(defparameter *positive-set*
  (append
   (loop repeat (/ *scale* 4) collect
     (list (random-normal :mean 2.5d0 :sd 2d0)
	   (random-normal :mean 5d0 :sd 3d0)))
   (loop repeat (/ *scale* 4) collect
     (list (random-normal :mean 17.5d0 :sd 2d0)
	   (random-normal :mean 5d0 :sd 3d0)))
   (loop repeat (/ *scale* 4) collect
     (list (random-normal :mean 10d0 :sd 3d0)
	   (random-normal :mean 0d0 :sd 2d0)))
   (loop repeat (/ *scale* 4) collect
     (list (random-normal :mean 10d0 :sd 3d0)
	   (random-normal :mean 10d0 :sd 2d0)))))

(defparameter *negative-set*
  (append
   (loop repeat *scale* collect
     (list (random-normal :mean 10d0 :sd 2d0)
	   (random-normal :mean 5d0 :sd 2d0)))))

(defparameter training-vector (make-training-vector *positive-set* *negative-set*))

(require 'sb-sprof)

(sb-sprof:with-profiling (:max-samples 1000 :report :flat :loop nil)
  (defparameter trained-svm (make-svm-model training-vector
					    (make-rbf-kernel :gamma 0.05)
					    :c 10)))

  36.864 seconds of real time

;; test by training-vector
(svm-validation trained-svm training-vector)

(ql:quickload :wiz-util)
;; Plot training dataset
(wiz:plot-lists (list (mapcar #'cadr *positive-set*)
		      (mapcar #'cadr *negative-set*))
		:x-lists (list (mapcar #'car *positive-set*)
			       (mapcar #'car *negative-set*))
		:style 'points)


(defparameter predicted-positive-set
  (remove-if-not (lambda (datapoint)
		   (> (funcall trained-svm datapoint) 0))
		 (mapcar #'list->clml-vector (append *positive-set-test* *negative-set-test*))))

(defparameter predicted-negative-set
  (remove-if-not (lambda (datapoint)
		   (< (funcall trained-svm datapoint) 0))
		 (mapcar #'list->clml-vector (append *positive-set-test* *negative-set-test*))))

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

;; test
(test trained-svm *positive-set* *negative-set*)
(test trained-svm *positive-set-test* *negative-set-test*)

;; cross-validation
(defparameter *positive-set*
  (mapcar #'list
	  (wiz:n-times-collect 1000 (wiz:random-normal :mean 0.0d0 :sd 0.5d0))
	  (wiz:n-times-collect 1000 (wiz:random-normal :mean 1.0d0 :sd 0.5d0))))

(defparameter *negative-set*
  (mapcar #'list
	  (wiz:n-times-collect 1000 (wiz:random-normal :mean 1.0d0 :sd 0.5d0))
	  (wiz:n-times-collect 1000 (wiz:random-normal :mean 0.0d0 :sd 0.5d0))))

(cross-validation 5 *positive-set* *negative-set*
		  (make-rbf-kernel :gamma 1d0) :c 10d0)

(cross-validation 5 *positive-set* *negative-set*
		  (make-rbf-kernel :gamma 1d0) :c 10d0)

(defun plot-grid-search

;;; svm-validation は test と同じと考えていい。 svm-validation の方が多少効率はいい。
;;; クロスバリデーションもvectorのままやった方がいいに決まっているが・・・

;;; positive-set negative-setの二つに分けてからn分割する方が精度は良いと予想されるが、実際にはデータはシーケンシャルに与えられる。
;;; なので実際にデータが与えられる方法で訓練データを作らないといけない。

;;; list-dataset->vector-dataset が必要。
;; ((input1 input2 input3 ... 1.0d0) ; positive-datapoint
;;  (input1' input2' input3' ... -1.0d0))
;; この形式のリストからdouble-floatで型付けされたベクタの単純ベクタの形に変換する関数

(defun list-dataset->vector-dataset (list-dataset)
  (let ((product-array (make-array (length list-dataset)))
	(i 0))
    (loop for elem in list-dataset do
	 (setf (aref product-array i) (list->clml-vector elem))
	 (incf i))
    product-array))

(defparameter list-dataset
  '((8.0d0 8.0d0 1.0d0) (8.0d0 20.0d0 1.0d0) (8.0d0 44.0d0 1.0d0)
    (8.0d0 56.0d0 1.0d0) (12.0d0 32.0d0 1.0d0) (16.0d0 16.0d0 1.0d0)
    (36.0d0 24.0d0 -1.0d0) (36.0d0 36.0d0 -1.0d0) (44.0d0 8.0d0 -1.0d0)
    (16.0d0 48.0d0 1.0d0) (24.0d0 20.0d0 1.0d0) (24.0d0 32.0d0 1.0d0)
    (44.0d0 44.0d0 -1.0d0) (44.0d0 56.0d0 -1.0d0) (48.0d0 16.0d0 -1.0d0)
    (24.0d0 44.0d0 1.0d0) (28.0d0 8.0d0 1.0d0) (32.0d0 52.0d0 1.0d0)
    (36.0d0 16.0d0 1.0d0)
    (48.0d0 28.0d0 -1.0d0) (56.0d0 8.0d0 -1.0d0) (56.0d0 44.0d0 -1.0d0)
    (56.0d0 52.0d0 -1.0d0)))

(defparameter vector-dataset (list-dataset->vector-dataset list-dataset))

(defparameter trained-svm (make-svm-learner vector-dataset (make-rbf-kernel :gamma 0.05) :c 10))

(svm-validation trained-svm vector-dataset)

(defun cross-validation-from-list-dataset (n list-dataset kernel &key (c 10) (weight 1.0d0))
  (let* ((splited-set (wiz:split-equally list-dataset n))
	 (average-validity
	  (/ (loop for i from 0 to (1- n)
		   summing
		   (let* ((training-set (apply #'append (wiz:remove-nth i splited-set)))
			  (test-set (nth i splited-set))
			  (training-vector (list-dataset->vector-dataset training-set))
			  (test-vector (list-dataset->vector-dataset test-set))
			  (trained-svm (make-svm-learner training-vector kernel
							 :c c :weight weight)))
		     (multiple-value-bind (sum-up-list accuracy)
			 (svm-validation trained-svm test-vector)
		       (declare (ignore sum-up-list))
		       (format t "accuracy: ~f %~%" accuracy)
		       accuracy)))
	     n)))
    (format t "Average validity: ~f~%" average-validity)
    average-validity))

;;; 上のだとメモリ消費がすごい。訓練セットとテストセットの配列を2つ作って、その要素を破壊的に書き換えていくようにする。

(cross-validation-from-list-dataset 2 list-dataset (make-rbf-kernel :gamma 0.05) :c 10)

(defun make-dataset-vector (number-of-data input-dimension)
  (let ((arr (make-array number-of-data)))
    (loop for i from 0 to (1- number-of-data) do
      (setf (aref arr i) (make-dvec input-dimension)))
    arr))

(defun set-dataset! (list-dataset vector-dataset)
  (let ((i 0))
    (loop for data in list-dataset do
      (let ((j 0))
	(loop for elem in data do
	  (setf (aref (aref vector-dataset i) j) elem)
	  (incf j)))
      (incf i)))
  vector-dataset)

(defun truncate-by-mod (list n)
  (let ((red (mod (length list) n)))
    (if (zerop red)
	list
	(wiz:nthcar (- (length list) red) list))))

;;; とりあえずlist-datasetはnで割り切れる長さのリストとしておく
(defun cross-validation-from-list-dataset (n list-dataset kernel &key (c 10) (weight 1.0d0) (stream *standard-output*))
  (let* ((list-dataset (truncate-by-mod list-dataset n))
	 (splited-set (wiz:split-equally list-dataset n))
	 (number-of-data (length list-dataset))
	 (input-dimension (length (car list-dataset)))
	 (training-vector (make-dataset-vector (- number-of-data (/ number-of-data n)) input-dimension))
	 (test-vector (make-dataset-vector (/ number-of-data n) input-dimension))	 
	 (average-validity
	  (/ (loop for i from 0 to (1- n)
		   summing
		   (let* ((training-set (apply #'append (wiz:remove-nth i splited-set)))
			  (test-set (nth i splited-set))
			  (training-vector (set-dataset! training-set training-vector))
			  (test-vector (set-dataset! test-set test-vector))
			  (trained-svm (make-svm-learner training-vector kernel :c c :weight weight)))
		     (multiple-value-bind (sum-up-list accuracy)
			 (svm-validation trained-svm test-vector)
		       (declare (ignore sum-up-list))
		       (format stream "accuracy: ~f %~%" accuracy)
		       accuracy)))
	     n)))
    (format stream "Average validity: ~f~%" average-validity)
    average-validity))

(defparameter a1a-train (read-libsvm-data "/home/wiz/tmp/a1a" 123 1605))
(defparameter a1a-test (read-libsvm-data  "/home/wiz/tmp/a1a.t" 123 30956))
(defparameter kernel (make-rbf-kernel :gamma (expt 2d0 -7d0)))

;; Training
(defparameter a1a-model (make-svm-model a1a-train kernel :c (expt 2d0 3d0)))

;; Test
(svm-validation a1a-model a1a-test)
;; (((-1.0d0 . 1.0d0) . 3186) ((1.0d0 . 1.0d0) . 4260) ((-1.0d0 . -1.0d0) . 21871)
;;  ((1.0d0 . -1.0d0) . 1639))
;; 84.413360899341d0

;; Search hyper parameter (gamma, C)
(grid-search a1a-train a1a-test)
;; # gamma	C	accuracy	time
;; -7.0	        3.0	84.413360899341	2.997
