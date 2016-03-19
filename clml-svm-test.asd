#|
  This file is a part of clml-svm project.
|#

(in-package :cl-user)
(defpackage clml-svm-test-asd
  (:use :cl :asdf))
(in-package :clml-svm-test-asd)

(defsystem clml-svm-test
  :author "Satoshi Imai"
  :license "LLGPL"
  :depends-on (:clml-svm
               :prove)
  :components ((:module "t"
                :components
                ((:test-file "clml-svm"))))

  :defsystem-depends-on (:prove-asdf)
  :perform (test-op :after (op c)
                    (funcall (intern #.(string :run-test-system) :prove-asdf) c)
                    (asdf:clear-system c)))
