(declaim (optimize speed))

(defsystem "clml-svm"
  :description "A Soft Margin SVM library which picked out from Common Lisp Machine Learning(clml)."
  :licence "LGPL"
  :encoding :utf-8
  :depends-on (:lapack :iterate :wiz-util)
  :components (
	       (:file "src/utils" :depends-on ())
	       (:file "src/vector" :depends-on ("src/utils"))
	       (:file "src/matrix" :depends-on ("src/vector"))
	       ;; Soft Margin SVM
	       (:file "src/wss3-svm" :depends-on ("src/utils" "src/vector" "src/matrix"))
	       ))
