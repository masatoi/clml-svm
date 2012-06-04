(declaim (optimize speed))

(defsystem "clml-svm"
  :description
  "A SVM library which picked out from Common Lisp Machine Learning(clml)."
  :depends-on (:lapack :iterate)
  :components (
	       (:file "src/utils" :depends-on ())
	       (:file "src/vector" :depends-on ("src/utils"))
	       (:file "src/matrix" :depends-on ("src/vector"))

	       ;; for Hard Margin SVM	       
	       (:file "src/svm" :depends-on ("src/utils"))
	       ;; for Soft Margin SVM
	       (:file "src/wss3-svm" :depends-on ("src/utils" "src/vector" "src/matrix"))
	       ))
