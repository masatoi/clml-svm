;;; -*- coding:utf-8; mode: lisp; -*-

(in-package :cl-user)

(defpackage clml-svm-asd (:use :cl :asdf))

(in-package :clml-svm-asd)

(defsystem "clml-svm"
  :description "A Soft Margin SVM library which picked out from Common Lisp Machine Learning(clml)."
  :licence "LLGPL"
  :encoding :utf-8
  :depends-on (:cl-ppcre
	       :parse-number
	       #+sbcl :sb-cltl2)
  :components ((:module "src"
			:components
			((:file "utils")
			 (:file "vector" :depends-on ("utils"))
			 (:file "wss3-svm" :depends-on ("utils" "vector"))
			 )))
  :long-description
  #.(with-open-file (stream (merge-pathnames
                             #p"README.org"
                             (or *load-pathname* *compile-file-pathname*))
                            :if-does-not-exist nil
                            :direction :input)
      (when stream
        (let ((seq (make-array (file-length stream)
                               :element-type 'character
                               :fill-pointer t)))
          (setf (fill-pointer seq) (read-sequence seq stream))
          seq)))
  :in-order-to ((test-op (test-op clml-svm-test))))
