;;; -*- coding:utf-8; mode: lisp;  -*-

;;;Support Vector Machine Package using SMO-type algorithm
;;;Abe Yusuke,Jianshi Huang. 2010 June
;;;Reference: Working Set Selection Using Second Order Information for Training SVM.
;;;Chih-Jen Lin. Department of Computer Science. National Taiwan University. 
;;;Joint work with Rong-En Fan and Pai-Hsuen Chen.

(in-package :cl-user)

(defpackage :svm
  (:use :cl
	:hjs.util.meta
	:hjs.util.vector
	)
  (:export #:make-svm-model
	   #:load-svm-model
	   #:save-svm-model
	   #:make-linear-kernel
	   #:make-rbf-kernel
	   #:make-polynomial-kernel
	   #:make-one-class-svm-kernel
	   #:discriminate
	   #:svm-validation
	   #:make-scale-parameters
	   #:autoscale
	   #:autoscale-datum
	   #:cross-validation
	   #:grid-search
	   #:grid-search-by-cv
	   #:read-libsvm-data
	   ))

(in-package svm)

;#-sbcl (declaim (optimize speed (safety 0) (debug 1)))
(declaim (optimize speed (safety 0) (debug 1)))

(defparameter *eps* 1d-3)
(defparameter *tau* 1d-12)
(defparameter *training-size* 0)
(defparameter *label-index* 0)
(defparameter *alpha-array* (make-array 0 :element-type 'double-float))
(defparameter *gradient-array* (make-array 0 :element-type 'double-float))
(defparameter *kernel-function-result* (make-array 1 :element-type 'double-float :initial-element 0d0))
(defparameter *kernel-cache* nil)
(defparameter *kernel-vec-d* (make-dvec 0))
(defparameter *iteration* 0)

(declaim (type double-float *eps* *tau*)
         (type fixnum *training-size* *label-index* *iteration*)
         (type dvec *alpha-array* *gradient-array* *kernel-vec-d*)
         (type (simple-array double-float (1)) *kernel-function-result*)
         ;; (type (or null cache) *kernel-cache*) ; cache is not declared yet
         )

(declaim (inline eta eta-cached sign update-gradient select-i select-j)
         (notinline get-cached-values))

;;;;
(defstruct kernel-function
  name
  scalar
  vectorized)

(defmacro call-kernel-function-uncached (kernel-function point1 point2)
  `(progn
     (funcall (the function (kernel-function-scalar ,kernel-function)) ,point1 ,point2)
     (the double-float
       (aref *kernel-function-result* 0))))

(defmacro call-kernel-function-vectorized-uncached (kernel-function point1 point2s result &optional start end)
  `(progn
     (funcall (the function (kernel-function-vectorized ,kernel-function)) ,point1 ,point2s ,result ,start ,end)))

(defmacro call-kernel-function (kernel-function point1 point2)
  `(call-kernel-function-uncached ,kernel-function ,point1 ,point2))

(defmacro call-kernel-function-vectorized (kernel-function point1 point2s result &optional start end)
  `(call-kernel-function-vectorized-uncached ,kernel-function ,point1 ,point2s ,result ,start ,end))

(defmacro define-kernel-function ((point1-var point2-var &optional (name :unknown)) &body body)
  (check-type point1-var symbol)
  (check-type point2-var symbol)
  (let ((point2-vec-var (intern (concatenate 'string (string point2-var) "-ARRAY"))))
    (with-unique-names (result i start end)
      `(make-kernel-function
        :name ,name
        :scalar
        (lambda (,point1-var ,point2-var)
          (declare (type dvec ,point1-var ,point2-var)
                   (optimize speed (safety 0)))
          (let ((,result (locally ,@body)))
            (declare (type double-float ,result))
            (setf (aref *kernel-function-result* 0) ,result)
            nil))
        :vectorized
        (lambda (,point1-var ,point2-vec-var ,result &optional ,start ,end)
          (declare (type dvec ,point1-var ,result)
                   (type (simple-array dvec (*)) ,point2-vec-var)
                   (optimize speed (safety 0))
                   (type (or null array-index) ,start ,end))
          (assert (<= (length ,point2-vec-var) (length ,result)))
          (loop for ,i of-type array-index from (or ,start 0) below (or ,end (length ,point2-vec-var))
                for ,point2-var of-type dvec = (aref ,point2-vec-var ,i)
                do
             (setf (aref ,result ,i) (locally ,@body))
                finally
             (return ,result)))))))
;;  e.g.

;; (define-kernel-function (z-i z-j :linear)
;;   (loop
;;     for k of-type array-index below (1- (length z-i))
;;     sum (* (aref z-i k) (aref z-j k))
;;       into result of-type double-float
;;     finally (return result)))

;; (defun make-rbf-kernel (&key gamma)
;;   (declare (type double-float gamma))
;;   (assert (> gamma 0.0d0))
;;   (define-kernel-function (z-i z-j :rbf)
;;     (loop
;;       for k of-type array-index below (1- (length z-i))
;;       sum (expt (- (aref z-i k) (aref z-j k)) 2)
;;         into result of-type double-float
;;       finally (return (d-exp (* (- gamma) result))))))

;;;; a circular list
(defconstant +double-float-in-bytes+ 8)



(defstruct head
  prev
  next
  data                          ; data[0, len) is cached in this entry
  (len 0 :type fixnum))

(defstruct (cache (:constructor %make-cache (total size heads lru-head)))
  (total 0 :type fixnum)
  (size #.(* 100 1024 1024) :type fixnum) ; size of free space (bytes)
  (heads #() :type (simple-array head (*)))
  (lru-head (make-head) :type head))

(defun make-cache (total size)
  (let* ((heads (coerce (loop repeat total collect (make-head)) 'vector))
         (lru-head (make-head))
         (size (max (/ size +double-float-in-bytes+) (* 2 total))))
    (setf (head-next lru-head) lru-head)
    (setf (head-prev lru-head) lru-head)
    (%make-cache total size heads lru-head)))

;;
(defmacro swap (a b)
  ;; `(psetf ,a ,b ,b ,a)
  (with-unique-names (va vb)
    `(let ((,va ,a)
           (,vb ,b))
       (setf ,a ,vb)
       (setf ,b ,va))))

;;
(declaim (type (function (cache head) cache) lru-delete lru-insert)
         (type (function (cache simple-vector array-index array-index) dvec) get-cached-values)
         (inline lru-delete lru-insert))

(locally (declare (optimize speed (safety 0)))
  (defun lru-delete (cache head)
    (declare (ignorable cache)
             (type cache cache)
             (type head head))
    (let ((next (head-next head))
          (prev (head-prev head)))
      (setf (head-next prev) next
            (head-prev next) prev))
    cache)

  (defun lru-insert (cache head)
    (declare (type cache cache)
             (type head head))
    (with-slots (lru-head) cache
      (let ((old-last (head-prev lru-head)))
        (setf (head-next head) lru-head
              (head-prev head) old-last
              (head-next old-last) head
              (head-prev lru-head) head)))
    cache)

  (defun get-cached-values (cache training-vector index len kernel-function)
    (declare (type cache cache)
             (type array-index index)
             (type fixnum len)
             (type simple-vector training-vector)
             (type kernel-function kernel-function))
    (with-slots (heads lru-head size) cache
      (declare (type (simple-array head (*)) heads)
               (type head lru-head)
               (type fixnum size))
      (let* ((h (aref heads index))
             (h-len (head-len (the head h)))
             (more (- len (the fixnum h-len))))
        (declare (type head h)
                 (type fixnum h-len more))
        (when (not (zerop h-len))
          (lru-delete cache h))
        (when (> more 0)
          ;; free old space
          (let (biggest-data-vec
                (biggest-data-vec-length 0))
            (loop while (< size more)
                  do
               (let* ((old (head-next lru-head)))
                 (lru-delete cache old)
                 (incf size (head-len old))
                 ;; reuse released data vec
                 (let ((data-size (length (the dvec (head-data old)))))
                   (when (> data-size biggest-data-vec-length)
                     (setf biggest-data-vec (head-data old))
                     (setf biggest-data-vec-length data-size)))
                 (setf (head-data old) nil)
                 (setf (head-len old) 0)))
            ;; allocate new space
            (let ((new-data (if (and biggest-data-vec (>= biggest-data-vec-length len))
                                biggest-data-vec
                                (make-dvec len)))
                  (h-data (head-data h)))
              (when h-data
                (locally
                    (declare (type dvec new-data h-data))
                  (replace new-data h-data)))
              (setf (head-data h) new-data)))
          (decf size more)
          (setf (head-len h) len)
          #+nil
          (loop with data of-type dvec = (head-data h)
                for j of-type array-index from h-len below len
                do
             (setf (aref data j)
                   (call-kernel-function kernel-function (aref training-vector index) (aref training-vector j))))
          (call-kernel-function-vectorized kernel-function (aref training-vector index)
                                           training-vector (head-data h) h-len len))
        (lru-insert cache h)
        (head-data h))))

  ;; (0) h includes neither i nor j: do nothing.
  ;; (1) h includes both i and j: the corresponding data would be swapped.
  ;; (2) h contains i but not j (recall that i < j): the column would be thrown away.
  (defun swap-index (cache i j)
    (declare (type cache cache)
             (type array-index i j))
    ;; 
    (when (= i j)
      (return-from swap-index))
    ;; 
    (with-slots (heads lru-head size) cache
      (declare (type (simple-array head (*)) heads)
               (type head lru-head)
               (type fixnum size))
      (let* ((head-i (aref heads i))
             (head-j (aref heads j)))
        (declare (type head head-i head-j))
        (unless (zerop (head-len head-i))
          (lru-delete cache head-i))
        (unless (zerop (head-len head-j))
          (lru-delete cache head-j))
        (swap (head-data head-i) (head-data head-j))
        (swap (head-len head-i) (head-len head-j))
        (unless (zerop (head-len head-i))
          (lru-insert cache head-i))
        (unless (zerop (head-len head-j))
          (lru-insert cache head-j))
        ;;
        (when (> i j)
          (swap i j))
        ;;
        #+ccl (loop for h = lru-head then (head-next h)
		    until (eq h lru-head)
		    when (> (head-len h) i)
		      do
		   (let ((h-data (head-data h)))
		     (declare (type dvec h-data))
		     (if (> (head-len h) j)
		       (swap (aref h-data i) (aref h-data j))
		       (progn
			 (lru-delete cache h)
			 (incf size (head-len h))
			 (setf (head-data h) nil)
			 (setf (head-len h) 0)))))

	#-ccl (loop for h of-type head = lru-head then (head-next h)
		    until (eq h lru-head)
		    when (> (head-len h) i)
		      do
		   (let ((h-data (head-data h)))
		     (declare (type dvec h-data))
		     (if (> (head-len h) j)
		       (swap (aref h-data i) (aref h-data j))
		       (progn
			 (lru-delete cache h)
			 (incf size (head-len h))
			 (setf (head-data h) nil)
			 (setf (head-len h) 0)))))
	)))
  )

;;;;

(locally (declare (optimize speed (safety 0)))

  (declaim (ftype (function (simple-vector kernel-function fixnum fixnum) double-float) eta)
           (ftype (function (dvec dvec fixnum fixnum) double-float) eta-cached))

  #+allegro
  (eval-when (:compile-toplevel :load-toplevel :execute)
    (setf (get 'eta 'sys::immed-args-call)
          '((:lisp :lisp :lisp :lisp) double-float)))
  (defun eta (training-vector kernel-function i j)
    (declare (type simple-vector training-vector)
             (type kernel-function kernel-function)
             (type array-index i j)
             (ignorable kernel-function training-vector))
    
    (let ((point-i (svref training-vector i))
          (point-j (svref training-vector j)))
      
      (declare (type dvec point-i point-j))
      
      (+ (call-kernel-function kernel-function point-i point-i)
         (call-kernel-function kernel-function point-j point-j)
         (* -2.0d0 (call-kernel-function kernel-function point-i point-j)))))

  #+allegro
  (eval-when (:compile-toplevel :load-toplevel :execute)
    (setf (get 'eta-cached 'sys::immed-args-call)
          '((:lisp :lisp :lisp :lisp) double-float)))
  (defun eta-cached (kernel-vec-i kernel-vec-d i j)
    (declare (type dvec kernel-vec-i kernel-vec-d)
             (type fixnum i j))
    (the double-float
      (+ (aref kernel-vec-d i)
         (aref kernel-vec-d j)
         (* -2.0d0 (aref kernel-vec-i j)))))


  (defun update-gradient (training-vector kernel-vec-i kernel-vec-j i j old-a-i old-a-j)
    (declare (type simple-vector training-vector) 
             (type double-float old-a-i old-a-j))
    
    (let* ((alpha-array *alpha-array*)
           (gradient-array *gradient-array*)
           (label-index *label-index*)
           (training-size *training-size*))
      (declare (type fixnum i j training-size label-index)
               (type dvec alpha-array gradient-array kernel-vec-i kernel-vec-j))
      (let ((delta-a-i (- (aref alpha-array i) old-a-i))
            (delta-a-j (- (aref alpha-array j) old-a-j)))
        (declare (type double-float delta-a-i delta-a-j))
        (loop
          for k of-type array-index below training-size
          with point-i of-type dvec = (svref training-vector i)
          with point-j of-type dvec = (svref training-vector j)
          with y-i of-type double-float = (aref point-i label-index)
          with y-j of-type double-float = (aref point-j label-index)
          as point-k of-type dvec = (svref training-vector k)
          as y-k of-type double-float = (aref point-k label-index)
          as s-i of-type double-float = (* y-k y-i)
          as s-j of-type double-float = (* y-k y-j)
          do
       (progn
         (incf (aref gradient-array k)
               (+ (* s-i (aref kernel-vec-i k) delta-a-i)
                  (* s-j (aref kernel-vec-j k) delta-a-j))))))
      nil))

  (defun qp-solver (training-vector kernel-function c weight cache-size-in-bytes)
    
    (declare (type simple-vector training-vector)
             (type kernel-function kernel-function)
             (type double-float c weight))

    (setf *iteration* 0)
    (setf *training-size* (length training-vector))
    (setf *label-index* (1- (length (the simple-array (aref training-vector 0)))))
    (setf *alpha-array* (make-array *training-size* :element-type 'double-float :initial-element 0.0d0))
    (setf *gradient-array* (make-array *training-size* :element-type 'double-float :initial-element -1.0d0))
    (setf *kernel-vec-d* (make-dvec *training-size*))
    (setf *kernel-cache* (make-cache *training-size* (or cache-size-in-bytes (* 100 1024 1024))))
    
    (let ((tau *tau*)
          (training-size *training-size*)
          (label-index *label-index*)
          (alpha-array *alpha-array*)
          (gradient-array *gradient-array*)
          (kernel-vec-d *kernel-vec-d*)
          (kernel-cache *kernel-cache*))
      
      (declare (type double-float tau)
               (type fixnum training-size)
               (type array-index label-index)
               (type dvec alpha-array gradient-array kernel-vec-d)
               (type cache kernel-cache))

      (loop for k of-type array-index below training-size
            for point-k = (aref training-vector k)
            do
         (setf (aref kernel-vec-d k) (call-kernel-function kernel-function point-k point-k)))
      
      (loop
        while t
        do (multiple-value-bind (i j) 
               (working-set-selection3 training-vector kernel-function c weight)
             (declare (type fixnum i j))
             
             (incf *iteration*)
             
             (when (= -1 j)
               ;; release memory
               (setf *kernel-cache* nil)
               (return-from qp-solver *alpha-array*))
             
             (let ((y-i (aref (the dvec (svref training-vector i)) label-index))
                   (y-j (aref (the dvec (svref training-vector j)) label-index))
                   (kernel-vec-i (get-cached-values kernel-cache training-vector i training-size kernel-function)))

               (declare (type double-float y-i y-j)
                        (type dvec kernel-vec-i))
               
               (let ((a (eta-cached kernel-vec-i kernel-vec-d i j))
                     (b (- (* y-j (aref gradient-array j))
                           (* y-i (aref gradient-array i)))))
                 
                 (declare (type double-float a b))

                 (when (<= a 0.0d0)
                   (setf a tau))
                 
                 ;;update alpha
                 (let ((old-a-i (aref alpha-array i))
                       (old-a-j (aref alpha-array j)))
                   
                   (declare (type double-float old-a-i old-a-j))
                   
                   (incf (aref alpha-array i) (/ (* y-i b) a))
                   (decf (aref alpha-array j) (/ (* y-j b) a))
                   
                   ;;clipping
                   (let ((diff (- old-a-i old-a-j))
                         (sum (+ old-a-i old-a-j))
                         (new-a-i (aref alpha-array i))
                         (new-a-j (aref alpha-array j))
                         (c-i (if (plusp y-i)
                                  c
                                  (* weight c)))
                         (c-j (if (plusp y-j)
                                  c
                                  (* weight c))))
                     
                     (declare (type double-float diff sum new-a-i new-a-j c-i c-j))
                     
                     (if (/= y-i y-j)
                         (progn
                           (if (> diff 0.0d0)
                               (when (< new-a-j 0.0d0)
                                 (setf (aref alpha-array j) 0.0d0)
                                 (setf (aref alpha-array i) diff))
                               (when (< new-a-i 0.0d0)
                                 (setf (aref alpha-array i) 0.0d0)
                                 (setf (aref alpha-array j) (- diff))))
                           
                           (if (> diff (- c-i c-j))
                               (when (> new-a-i c-i)
                                 (setf (aref alpha-array i) c-i)
                                 (setf (aref alpha-array j) (- c-i diff)))
                               (when (> new-a-j c-j)
                                 (setf (aref alpha-array j) c-j)
                                 (setf (aref alpha-array i) (+ c-j diff)))))
                         (progn
                           (if (> sum c-i)
                               (when (> new-a-i c-i)
                                 (setf (aref alpha-array i) c-i)
                                 (setf (aref alpha-array j) (- sum c-i)))
                               (when (< new-a-j 0.0d0)
                                 (setf (aref alpha-array j) 0.0d0)
                                 (setf (aref alpha-array i) sum)))
                           
                           (if (> sum c-j)
                               (when (> new-a-j c-j)
                                 (setf (aref alpha-array j) c-j)
                                 (setf (aref alpha-array i) (- sum c-j)))
                               (when (< new-a-i 0.0d0)
                                 (setf (aref alpha-array i) 0.0d0)
                                 (setf (aref alpha-array j) sum)))))
                     
                     ;;update gradient
                     (let ((kernel-vec-i (get-cached-values kernel-cache training-vector i training-size kernel-function))
                           (kernel-vec-j (get-cached-values kernel-cache training-vector j training-size kernel-function)))
                       (declare (type dvec kernel-vec-i kernel-vec-j))

                       #+nil
                       (let ((delta-a-i (- (aref alpha-array i) old-a-i))
                             (delta-a-j (- (aref alpha-array j) old-a-j)))
                         (declare (type double-float delta-a-i delta-a-j))
                         (loop
                           for k of-type array-index below training-size
                           as point-k of-type dvec = (svref training-vector k)
                           as y-k of-type double-float = (aref point-k label-index)
                           as s-i of-type double-float = (* y-k y-i)
                           as s-j of-type double-float = (* y-k y-j)
                           ;; branch is slower
                           ;; if (and (/= 0.0d0 delta-a-i) (/= 0.0d0 delta-a-j))
                           do (incf (aref gradient-array k)
                                    (+ (* s-i (aref kernel-vec-i k) delta-a-i)
                                       (* s-j (aref kernel-vec-j k) delta-a-j)))))
                       (update-gradient training-vector kernel-vec-i kernel-vec-j i j old-a-i old-a-j)
                       )))))))))

  (defun select-i (training-vector c)
    (declare (type simple-vector training-vector)
             (type double-float c))
    
    (let ((training-size *training-size*)
          (label-index *label-index*)
          (alpha-array *alpha-array*)
          (gradient-array *gradient-array*)
          (i -1)
          (g-max most-negative-double-float))
      
      (declare (type fixnum i training-size label-index)
               (type dvec alpha-array gradient-array)
               (type double-float g-max))
      (loop
        for k of-type array-index below training-size
        as y-k of-type double-float = (aref (the dvec (svref training-vector k)) label-index)
        as a-k of-type double-float = (aref alpha-array k)
        as g-k of-type double-float = (aref gradient-array k)
        as g-temp of-type double-float = (- (* y-k g-k))
        if (and (>= g-temp g-max)
                (or (and (= y-k 1.0d0) (< a-k c))
                    (and (= y-k -1.0d0) (> a-k 0d0))))
          do (progn
               (setf g-max g-temp)
               (setf i k)))
      (values i g-max)))

  (defun select-j (training-vector kernel-function c weight i g-max)
    (declare (type simple-vector training-vector)
             (type kernel-function kernel-function)
             (type double-float c weight g-max)
             (type array-index i)
             (ignorable kernel-function))
    
    (let* ((training-size *training-size*)
           (label-index *label-index*)
           (alpha-array *alpha-array*)
           (gradient-array *gradient-array*)
           (tau *tau*)
           (j -1)
           (g-min most-positive-double-float)
           (obj-min most-positive-double-float)
           (kernel-cache *kernel-cache*)
           (kernel-vec-d *kernel-vec-d*)
           (kernel-vec-i (get-cached-values kernel-cache training-vector i training-size kernel-function)))
      
      (declare (type fixnum i j training-size label-index)
               (type dvec alpha-array gradient-array kernel-vec-i kernel-vec-d)
               (type double-float tau g-max g-min obj-min)
               (dynamic-extent g-min obj-min))
      
      (loop
        for k of-type array-index below training-size
        as y-k of-type double-float = (aref (the dvec (svref training-vector k)) label-index)
        as a-k of-type double-float = (aref alpha-array k)
        as g-k of-type double-float = (aref gradient-array k)
        as g-temp of-type double-float = (- (* y-k g-k))
        with a of-type double-float = 0.0d0	      
        with b of-type double-float  = 0.0d0
        if (or (and (= y-k 1.0d0) (> a-k 0.0d0))
               (and (= y-k -1.0d0) (< a-k (* weight c))))
          do (setf b (- g-max g-temp))
             (when (> b 0.0d0)
               (setf a (eta-cached kernel-vec-i kernel-vec-d i k))
               (when (<= a 0.0d0)
                 (setf a tau))
               (let ((temp (/ (- (* b b)) a)))
                 (declare (type double-float temp))
                 (when (<= temp obj-min)
                   (setf obj-min temp)
                   (setf j k))))
             (when (<= g-temp g-min)
               (setf g-min g-temp)))
      (values j g-min)))
  
  (defun working-set-selection3 (training-vector kernel-function c weight)
    (declare (type simple-vector training-vector)
             (type kernel-function kernel-function)
             (type double-float c weight))
    (let ((i -1)
          (j -1)
          (eps *eps*)
          (tau *tau*)
          (training-size *training-size*)
          (label-index *label-index*)
          (alpha-array *alpha-array*)
          (gradient-array *gradient-array*))
      
      (declare (type fixnum i j)
               (type double-float eps tau)
               (type fixnum training-size)
               (type array-index label-index)
               (type dvec alpha-array gradient-array))
      
      (let ((g-max most-negative-double-float)
            (g-min most-positive-double-float))
        (declare (type double-float g-max g-min))
        
        ;;select i
        ;; (multiple-value-setq (i g-max) (select-i training-vector c))
        (loop
          for k of-type array-index below training-size
          as y-k of-type double-float = (aref (the dvec (svref training-vector k)) label-index)
          as a-k of-type double-float = (aref alpha-array k)
          as g-k of-type double-float = (aref gradient-array k)
          as g-temp of-type double-float = (- (* y-k g-k))
          if (and (>= g-temp g-max)
                  (or (and (= y-k 1.0d0) (< a-k c))
                      (and (= y-k -1.0d0) (> a-k 0.0d0))))
            do (progn 
                 (setf g-max g-temp)
                 (setf i k)))

        ;;select j
        ;; (multiple-value-setq (j g-min) (select-j training-vector kernel-function c weight i g-max))
        (let ((obj-min most-positive-double-float))
          (declare (type double-float obj-min))

          (let* ((kernel-cache *kernel-cache*)
                 (kernel-vec-d *kernel-vec-d*)
                 (kernel-vec-i (get-cached-values kernel-cache training-vector i training-size kernel-function)))
            (declare (type dvec kernel-vec-i kernel-vec-d))
            (loop
              for k of-type array-index below training-size
              as y-k of-type double-float = (aref (the dvec (svref training-vector k)) label-index)
              as a-k of-type double-float = (aref alpha-array k)
              as g-k of-type double-float = (aref gradient-array k)
              as g-temp of-type double-float = (- (* y-k g-k))
              with a of-type double-float = 0.0d0	      
              with b of-type double-float  = 0.0d0
              if (or (and (= y-k 1.0d0) (> a-k 0.0d0))
                     (and (= y-k -1.0d0) (< a-k (* weight c))))
                do (setf b (- g-max g-temp))
                   (when (> b 0.0d0)
                     (setf a (eta-cached kernel-vec-i kernel-vec-d i k))
                     (when (<= a 0.0d0)
                       (setf a tau))
                     (let ((temp (/ (- (* b b)) a)))
                       (declare (type double-float temp))
                       (when (<= temp obj-min)
                         (setf obj-min temp)
                         (setf j k))))
                   (when (<= g-temp g-min)
                     (setf g-min g-temp)))))
        
        (when (< (- g-max g-min) eps)
          (return-from working-set-selection3 (values -1 -1)))
        
        (values i j))))
  )

(defun compute-b (training-vector kernel-function c weight alpha-array)
  (declare (type simple-vector training-vector)
           (type dvec alpha-array)
	   (type kernel-function kernel-function)
	   (type double-float c weight)
           (ignorable kernel-function))

  (let ((label-index (1- (length (aref training-vector 0))))
	(n (length alpha-array)))
    
    (declare (type fixnum label-index n))
    
    (let ((result 0.0d0))
      (declare (type double-float result))
      (loop
        for i of-type fixnum below n
        as alpha-i of-type double-float = (aref alpha-array i)
        as y-i of-type double-float  = (aref (the dvec (svref training-vector i)) label-index)
        as c-i of-type double-float = (if (plusp y-i) c (* weight c))	 
        with count = 0
        if (< 0.0d0 alpha-i c-i)
	  do (incf count 1)
	     (incf result
		   (- y-i 
		      (let ((result2 0.0d0))
			(declare (type double-float result2))
			(loop 
                          for j of-type fixnum below n
                          as alpha-j of-type double-float = (aref alpha-array j)  
                          as y-j of-type double-float = (aref (the dvec (svref training-vector j)) label-index) 
                          unless (= 0.0d0 alpha-j)
			    do (incf result2 
				     (* alpha-j y-j
                                                (call-kernel-function kernel-function
                                                                      (svref training-vector i)
                                                                      (svref training-vector j))))
                          finally (return result2)))))
        finally (return (if (zerop count) 0d0 (/ result count)))))))

;;for check
(defun print-b (training-vector kernel-function c weight alpha-array)
  (declare (ignorable kernel-function))
  (let ((label-index (1- (length (aref training-vector 0)))))
    (loop
      for i below (length training-vector)
      as a-i = (aref alpha-array i)
      as point-i = (svref training-vector i)
      as y-i = (aref point-i label-index)
      as c-i of-type double-float = (if (plusp y-i) c (* weight c))
      if (< 0.0d0 a-i c-i)
	do (print (- y-i 
		     (loop
                       for j below (length training-vector)
                       as a-j = (aref alpha-array j)
                       as y-j = (aref (aref training-vector j) label-index)
                       unless (= 0.0d0 a-j)
			 sum (* a-j y-j
                                    (call-kernel-function kernel-function
                                                          (svref training-vector i)
                                                          (svref training-vector j)))))))))


(defun make-linear-kernel ()
  (define-kernel-function (z-i z-j :linear) 
    (loop
      for k of-type array-index below (1- (length z-i))
      sum (* (aref z-i k) (aref z-j k))
        into result of-type double-float
      finally (return result))))

(defun make-rbf-kernel (&key gamma)
  (let ((gamma (coerce gamma 'double-float)))
    (declare (type double-float gamma))
    (assert (> gamma 0.0d0))
    (define-kernel-function (z-i z-j :rbf)
      (loop
        for k of-type array-index below (1- (length z-i))
        sum (expt (- (aref z-i k) (aref z-j k)) 2)
          into result of-type double-float
        finally (return (d-exp (* (- gamma) result)))))))

(defun make-polynomial-kernel (&key gamma r d)
  (assert (> gamma 0.0d0))
  (assert (and (integerp d) (> d 0)))
  (let ((gamma (coerce gamma 'double-float))
        (r (coerce r 'double-float))
        (d (coerce d 'double-float)))
    (declare (type double-float gamma r d))
    (let ((linear-kernel (make-linear-kernel)))
      (define-kernel-function (z-i z-j :polynomial) 
        (d-expt (the (double-float 0d0)
                  (+ (* gamma (call-kernel-function-uncached linear-kernel z-i z-j)) r)) d)))))


;;for comparison
(declaim (inline sign))
#+allegro
(eval-when (:compile-toplevel :load-toplevel :execute)
  (setf (get 'sign 'sys::immed-args-call)
        '((double-float) double-float)))
(defun sign (x)
  (declare (type double-float x))
  (if (>= x 0.0d0)
      1.0d0
      -1.0d0))

(defclass svm-model ()
  ((training-vector :accessor training-vector-of :initarg :training-vector :type simple-vector)
   (kernel-function :accessor kernel-function-of :initarg :kernel-function :type kernel-function)
   (alpha-array :accessor alpha-array-of :initarg :alpha-array :type dvec)
   (b :accessor b-of :initarg :b :type double-float)))

(defun make-svm-model (training-vector kernel-function &key (c 1.0d0) (weight 1.0d0) cache-size-in-MB)
  (assert (plusp c))
  (assert (plusp weight))
  (let* ((c (coerce c 'double-float))
	 (weight (coerce weight 'double-float))
	 (alpha-array (qp-solver training-vector kernel-function c weight (* (or cache-size-in-MB 100) 1024 1024)))
	 (b (compute-b training-vector kernel-function c weight alpha-array)))
    (make-instance 'svm-model
       :training-vector training-vector
       :kernel-function kernel-function
       :alpha-array alpha-array
       :b b)))

(defun discriminate (model point)
  (let ((training-vector (training-vector-of model))
	(kernel-function (kernel-function-of model))
	(alpha-array (alpha-array-of model))
	(b (b-of model))
	(label-index (1- (length (svref (training-vector-of model) 0)))))
    (declare (type simple-vector training-vector)
	     (type kernel-function kernel-function)
	     (type dvec alpha-array)
	     (type double-float b)
	     (ignorable kernel-function)
	     (type fixnum label-index))
    (sign (+ (let ((result 0.0d0))
	       (declare (type double-float result))  
	       (loop 
		 for i of-type fixnum below (length alpha-array)
		 as a-i of-type double-float = (aref alpha-array i)
		 unless (= 0.0d0 a-i)
		   do (incf result
			    (* a-i
			       (aref (the dvec (svref training-vector i)) label-index)
			       (call-kernel-function-uncached kernel-function (svref training-vector i) point))))
   	       result)
	     b))))

(defun load-svm-model (file-name kernel-function)
  (let* ((material-list 
	  (with-open-file (in file-name :direction :input)
	    (read in)))
	 (training-vector (first material-list))
	 (alpha-array (specialize-vec (second material-list)))
	 (b (third material-list)))
    (loop 
      for i of-type fixnum below (length training-vector)
      do (setf (aref training-vector i) (specialize-vec (aref training-vector i))))
    (make-instance 'svm-model
       :training-vector training-vector
       :kernel-function kernel-function
       :alpha-array alpha-array
       :b b)))

(defun save-svm-model (file-name model)
  (with-open-file (out file-name
		       :direction :output
		       :if-exists :supersede
		       :if-does-not-exist :create)
    (write (list (training-vector-of model)
		 (alpha-array-of model)
		 (b-of model)) :stream out)
    nil))

(defun sum-up (lst)
  (loop with alist
      for obj in lst
      as sub-alist = (assoc obj alist :test #'equal)
      do (if sub-alist
	     (incf (cdr sub-alist))
	   (push (cons obj 1) alist))
      finally (return alist)))

(defun accuracy (sum-up-list)
  (loop for obj in sum-up-list
        as type = (first obj)
        sum (cdr obj) into m
        if (= (car type) (cdr type))
          sum (cdr obj) into n
        finally (return (* 100.0d0 (/ n m)))))


(defun svm-validation (svm-model test-vector)
  (let* ((n (length test-vector))
	 (label-index (1- (length (svref test-vector 0))))
	 (sum-up-list
	  (sum-up (loop for i of-type fixnum below n
                        collect (cons (discriminate svm-model (svref test-vector i))
                                      (aref (the dvec (svref test-vector i)) label-index))))))
    (values sum-up-list (accuracy sum-up-list))))


;;for test
(defun sample-vector (n)
  (let ((x (make-array n :initial-element 0.0d0 :element-type 'double-float)))
    (loop for i below n 
          do (setf (aref x i) (coerce (random 10) 'double-float))
          finally (return x)))) 


(defun make-one-class-svm-kernel (&key gamma)
  (declare (type double-float gamma))
  (assert (> gamma 0.0d0))
  (define-kernel-function (z-i z-j :rbf)
    (loop
      for k of-type array-index below (length z-i)
      sum (expt (- (aref z-i k) (aref z-j k)) 2)
        into result of-type double-float
      finally (return (d-exp (* (- gamma) result))))))

;;; Autoscale

(defun mean-vector (training-vector)
  (let* ((data-dim (1- (array-dimension (aref training-vector 0) 0)))
	 (data-size (length training-vector))
	 (sum-v (make-array data-dim :element-type 'double-float :initial-element 0d0)))
    (loop for datum across training-vector do
      (loop for i from 0 to (1- data-dim) do
	(incf (aref sum-v i) (aref datum i))))
    (loop for i from 0 to (1- data-dim) do
      (setf (aref sum-v i) (/ (aref sum-v i) data-size)))
    sum-v))

(defmacro square (x)
  (let ((val (gensym)))
    `(let ((,val ,x))
       (* ,val ,val))))

(defun standard-deviation-vector (training-vector)
  (let* ((data-dim (1- (array-dimension (aref training-vector 0) 0)))
	 (data-size (length training-vector))
	 (sum-v (make-array data-dim :element-type 'double-float :initial-element 0d0))
	 (ave-v (mean-vector training-vector)))
    (loop for datum across training-vector do
      (loop for i from 0 to (1- data-dim) do
	(incf (aref sum-v i) (square (- (aref datum i) (aref ave-v i))))))
    (loop for i from 0 to (1- data-dim) do
      (setf (aref sum-v i) (sqrt (/ (aref sum-v i) data-size))))
    sum-v))

(defun min-max-vector (training-vector)
  (let* ((data-dim (1- (array-dimension (aref training-vector 0) 0)))
	 (data-size (length training-vector))
	 (max-v (make-array data-dim :element-type 'double-float :initial-element 0d0))
	 (min-v (make-array data-dim :element-type 'double-float :initial-element 0d0))
	 (cent-v (make-array data-dim :element-type 'double-float :initial-element 0d0))
	 (scale-v (make-array data-dim :element-type 'double-float :initial-element 0d0)))
    ;; init
    (loop for j from 0 to (1- data-dim) do
      (let ((elem-0j (aref (aref training-vector 0) j)))
	(setf (aref max-v j) elem-0j
	      (aref min-v j) elem-0j)))
    ;; calc min-v, max-v
    (loop for i from 1 to (1- data-size) do
      (loop for j from 0 to (1- data-dim) do
	(let ((elem-ij (aref (aref training-vector i) j)))
	  (when (< (aref max-v j) elem-ij) (setf (aref max-v j) elem-ij))
	  (when (> (aref min-v j) elem-ij) (setf (aref min-v j) elem-ij)))))
    ;; calc cent-v, scale-v
    (loop for j from 0 to (1- data-dim) do
      (setf (aref cent-v j) (/ (+ (aref max-v j) (aref min-v j)) 2.0d0)
	    (aref scale-v j) (if (= (aref max-v j) (aref min-v j))
			       1d0
			       (/ (abs (- (aref max-v j) (aref min-v j))) 2d0))))
    (values cent-v scale-v)))

(defclass scale-parameters ()
  ((centre-vector :accessor centre-vector-of :initarg :centre-vector :type dvec)
   (scale-vector :accessor scale-vector-of :initarg :scale-vector :type dvec )))

(defun make-scale-parameters (training-vector &key scaling-method)
  (cond ((eq scaling-method :unit-standard-deviation)
	 (make-instance 'scale-parameters
	    :centre-vector (mean-vector training-vector)
	    :scale-vector (standard-deviation-vector training-vector)))
	(t
	 (multiple-value-bind (cent-v scale-v)
	     (min-max-vector training-vector)
	   (make-instance 'scale-parameters
	      :centre-vector cent-v
	      :scale-vector scale-v)))))

(defun autoscale (training-vector &key scale-parameters)
  (let* ((scale-parameters (if scale-parameters scale-parameters
			     (make-scale-parameters training-vector)))
	 (data-dim (1- (array-dimension (aref training-vector 0) 0)))
	 (data-size (length training-vector))	  
	 (new-v (make-array data-size))
	 (cent-v (centre-vector-of scale-parameters))
	 (scale-v (scale-vector-of scale-parameters)))
    (loop for i from 0 to (1- data-size) do
      (let ((new-data-v (make-array (1+ data-dim) :element-type 'double-float :initial-element 0d0))
	    (data-v (aref training-vector i)))
	(loop for j from 0 to (1- data-dim) do	  
	  (setf (aref new-data-v j) (/ (- (aref data-v j) (aref cent-v j)) (aref scale-v j))))
	(setf (aref new-data-v data-dim) (aref data-v data-dim))
	(setf (aref new-v i) new-data-v)))
    (values new-v scale-parameters)))

;; dimension of datum contains the target column so that it can be input of discriminate function. 
;; and target column will be ignored in this function.
(defun autoscale-datum (datum scale-parameters)
  (let* ((datum-dim (length datum))
	 (new-datum (make-array datum-dim :element-type 'double-float))
	 (cent-v (centre-vector-of scale-parameters))
	 (scale-v (scale-vector-of scale-parameters)))
    (loop for i from 0 to (- datum-dim 2) do
      (setf (aref new-datum i) (/ (- (aref datum i) (aref cent-v i)) (aref scale-v i))))
    (setf (aref new-datum (1- datum-dim)) (aref datum (1- datum-dim)))
    new-datum))

;;; Cross-Validation (N-fold)

(defun split-training-vector-2part (test-start test-end training-vector sub-training-vector sub-test-vector)
    (loop for i from 0 to (1- (length training-vector)) do
      (cond ((< i test-start)
	     (setf (aref sub-training-vector i) (aref training-vector i)))
	    ((and (>= i test-start) (<= i test-end))
	     (setf (aref sub-test-vector (- i test-start)) (aref training-vector i)))
	    (t
	     (setf (aref sub-training-vector (- i (1+ (- test-end test-start)))) (aref training-vector i))))))

(defun average (list)
  (/ (loop for i in list sum i) (length list)))

(defun cross-validation (n training-vector kernel &key (c 10) (weight 1.0d0))
  (let* ((bin-size (truncate (length training-vector) n))
	 (sub-training-vector (make-array (- (length training-vector) bin-size)))
	 (sub-test-vector (make-array bin-size))
	 (accuracy-percentage-list
	  (loop for i from 0 to (1- n) collect
	    (progn
	      (split-training-vector-2part (* i bin-size) (1- (* (1+ i) bin-size))
					   training-vector sub-training-vector sub-test-vector)
	      (let ((trained-svm (make-svm-model sub-training-vector kernel :c c :weight weight)))
		(multiple-value-bind (useless accuracy-percentage)
		    (svm-validation trained-svm sub-test-vector)
		  (declare (ignore useless))
		  accuracy-percentage))))))
    (values (average accuracy-percentage-list)
	    accuracy-percentage-list)))

;;; Slide-window-Validation

(defun slide-window-validation (training-window-size test-window-size shift-size
				training-vector kernel &key (c 10) (weight 1.0d0))
  (assert (> (length training-vector) (+ training-window-size test-window-size)))
  (let* ((sub-training-vector (make-array training-window-size))
	 (sub-test-vector (make-array test-window-size)))
    (labels ((iter (accuracy-percentage-list current-index)
	       (if (> (+ current-index training-window-size test-window-size) (length training-vector))
		 (nreverse accuracy-percentage-list)
		 (progn
		   (loop for i from 0 to (1- training-window-size) do
		     (setf (aref sub-training-vector i) (aref training-vector (+ i current-index))))
		   (loop for i from 0 to (1- test-window-size) do
		     (setf (aref sub-test-vector i) (aref training-vector (+ i current-index training-window-size))))
		   (let ((trained-svm (make-svm-model sub-training-vector kernel :c c :weight weight)))
		     (multiple-value-bind (useless accuracy-percentage)
			 (svm-validation trained-svm sub-test-vector)
		       (declare (ignore useless))
		       (iter (cons accuracy-percentage accuracy-percentage-list)
			     (+ current-index shift-size))))))))
      (iter nil 0))))

;;; Parameter range cited from Table 2 in paper "Working Set Selection Using Second Order Information for Training Support Vector Machines".
;;; http://www.csie.ntu.edu.tw/~cjlin/papers/quadworkset.pdf
(defun grid-search (training-vector test-vector)
  (let ((accuracy-max 0d0) gamma-max C-max)
    (format t "# gamma~AC~Aaccuracy~Atime~%" #\tab #\tab #\tab)
    (loop for C-log from -5d0 to 15d0 by 2d0 do
      (loop for gamma-log from 3d0 downto -15d0 by 2d0 do
	(let* ((start-time (get-internal-real-time))
	       (model (make-svm-model training-vector (make-rbf-kernel :gamma (expt 2d0 gamma-log)) :C (expt 2d0 C-log))))
	  (multiple-value-bind (useless accuracy)
	      (svm-validation model test-vector)
	    (declare (ignore useless))
	    (format t "~f~A~f~A~f~A~f~%" gamma-log #\tab C-log #\tab accuracy #\tab (/ (- (get-internal-real-time) start-time) 1000.0))
	    (when (> accuracy accuracy-max)
	      (setf accuracy-max accuracy
		    gamma-max (expt 2d0 gamma-log)
		    C-max (expt 2d0 C-log))))))
      (format t "~%"))
  (values accuracy-max gamma-max C-max)))

(defun grid-search-by-cv (n training-vector)
  (let ((cv-max 0d0) gamma-max C-max)
    (format t "# gamma~AC~Aaccuracy~Atime~%" #\tab #\tab #\tab)
    (loop for C-log from -5d0 to 15d0 by 2d0 do
      (loop for gamma-log from 3d0 downto -15d0 by 2d0 do
	(let* ((start-time (get-internal-real-time))
	       (cv-result (cross-validation n training-vector (make-rbf-kernel :gamma (expt 2d0 gamma-log)) :C (expt 2d0 C-log))))
	  (format t "~f~A~f~A~f~A~f~%" gamma-log #\tab C-log #\tab cv-result #\tab (/ (- (get-internal-real-time) start-time) 1000.0))
	  (when (> cv-result cv-max)
	    (setf cv-max cv-result
		  gamma-max (expt 2d0 gamma-log)
		  C-max (expt 2d0 C-log)))))
      (format t "~%"))
    (values cv-max gamma-max C-max)))

;;; Read libsvm data
(defun read-libsvm-data (data-path data-dimension data-size)
  (let ((v (make-array data-size)))
    (with-open-file (f data-path :direction :input)
      (loop for i from 0 to (1- data-size) do
        (let* ((read-data (read-line f))
               (dv (make-array (1+ data-dimension) :element-type 'double-float :initial-element 0d0))
               (d (ppcre:split "\\s+" read-data))
               (index-num-alist
                (mapcar (lambda (index-num-pair-str)
                          (let ((index-num-pair (ppcre:split #\: index-num-pair-str)))
                            (list (parse-integer (car index-num-pair))
                                  (coerce (parse-number:parse-number (cadr index-num-pair)) 'double-float))))
                        (cdr d))))
          (setf (aref dv data-dimension) (coerce (parse-integer (car d)) 'double-float))
          (dolist (index-num-pair index-num-alist)
            (setf (aref dv (1- (car index-num-pair))) (cadr index-num-pair)))
          (setf (aref v i) dv))))
    v))
