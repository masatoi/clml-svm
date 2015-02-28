(defpackage clml-svm-asd (:use :cl :asdf))
(in-package :clml-svm-asd)

(defsystem "clml-svm"
  :description "A Soft Margin SVM library which picked out from Common Lisp Machine Learning(clml)."
  :licence "LLGPL"
  :encoding :utf-8
  :components ((:module "src"
			:components
			((:file "utils")
			 (:file "vector" :depends-on ("utils"))
			 (:file "wss3-svm" :depends-on ("utils" "vector"))
			 ))))
