;; (unless (featurep 'autoinsert)
;;   (require 'autoinsert)
;;   (add-hook 'after-init-hook (lambda () (auto-insert-mode t)))
;;   (setq auto-insert-directory "~/.emacs.d/templates/")
;;   (setq auto-insert-alist nil))

(defun mylib-autoinsert-move-point ()
  (goto-char (point-min))
  (when (search-forward "%POINT%" nil t)
    (replace-match "")))
(defvar mylib-autoinsert-replacement-alist nil)
(defun mylib-autoinsert-perform-replace (&optional alist)
  (loop for (from . to) in (or alist mylib-autoinsert-replacement-alist) do
        (goto-char (point-min))
        (while (search-forward from nil t)
          (replace-match (eval to) t))))
(add-to-list 'mylib-autoinsert-replacement-alist
             '("%NAME%" . (file-name-sans-extension
                           (file-name-nondirectory (buffer-file-name)))))

(add-to-list 'auto-insert-alist
             '(("\\.java$" . "Java")
               . ["default.java"
                  mylib-autoinsert-perform-replace
                  mylib-autoinsert-move-point]))
(add-to-list 'auto-insert-alist
             '(("/cf/\\(\\sw+/\\)+\\sw+\\.java$" . "Codeforces(Java)")
               . ["codeforces.java"
                  mylib-autoinsert-perform-replace
                  mylib-autoinsert-move-point]))
(add-to-list 'auto-insert-alist
             '(("/cf/\\(\\sw+/\\)*\\sw+\\.cpp$" . "Codeforces(C++)")
               . ["codeforces.cpp"
                  mylib-autoinsert-perform-replace
                  mylib-autoinsert-move-point]))
(add-to-list 'auto-insert-alist
             '(("/atc/\\(\\sw+/\\)*\\(\\sw\\|\\s_\\)+\\.cpp$" . "AtCoder(C++)")
               . ["codeforces.cpp"
                  mylib-autoinsert-perform-replace
                  mylib-autoinsert-move-point]))
(add-to-list 'auto-insert-alist
             '(("/yc/\\(\\sw+/\\)*\\(\\sw\\|\\s_\\)+\\.cpp$" . "yukicoder(C++)")
               . ["codeforces.cpp"
                  mylib-autoinsert-perform-replace
                  mylib-autoinsert-move-point]))
(add-to-list 'auto-insert-alist
             '(("/gcj/.*\\.l$" . "GCJ(Lisp)")
               . ["gcj.l"
                  mylib-autoinsert-perform-replace
                  mylib-autoinsert-move-point]))
(add-to-list 'auto-insert-alist
             '(("/gcj/.*\\.java$" . "GCJ(Java)")
               . ["gcj.java"
                  mylib-autoinsert-perform-replace
                  mylib-autoinsert-move-point]))
(add-to-list 'auto-insert-alist
             '(("/gcj/.*\\.cpp$" . "GCJ(C++)")
               . ["gcj.cpp"
                  mylib-autoinsert-perform-replace
                  mylib-autoinsert-move-point]))

;; (FILENAME . KEYWORD-LIST)
(defun mylib-insert-include ()
  (when (and (eq major-mode 'c++-mode)
             (not buffer-read-only))
    (dolist (x '(("random" "mt19937")
                 ("numeric" "accumulate" "iota")
                 ("iomanip" "setprecision" "setw" "setfill")
                 ("vector" "vector")
                 ("set" "set" "multiset")
                 ("unordered_set" "unordered_set")
                 ("queue" "queue")
                 ("unordered_map" "unordered_map")
                 ("map" "map")
                 ("stack" "stack")
                 ("cassert" "assert")
                 ("chrono" "chrono")
                 ("algorithm" "sort" "max_element" "min_element" "lower_bound" "upper_bound"
                  "reverse" "unique")
                 ("cmath" "pow" "abs" "sqrt" "exp")
                 ("cstdio" "scanf" "printf" "puts" "putchar")
                 ("cstring" "memset" "memcpy")
                 ("cstdlib" "atoi" "atol" "atoll")
                 ("iostream" "cin" "cout" "cerr")
                 ("fstream" "fstream" "ifstream" "ofstream")
                 ("bitset" "bitset")
                 ("functional" "function")
                 ("deque" "deque")
                 ;; ("multiset" "multiset")
                 ))
      (let ((file (car x)) (regexp (concat "\\_<" (regexp-opt (cdr x)) "\\_>")))
        (save-excursion
          (goto-char (point-min))
          (when (looking-at "// -\\*-.*-\\*-")
            (forward-line)
            (while (and (bolp) (eolp) (not (eobp))) (forward-line)))
          (save-restriction
            (narrow-to-region (point) (point-max))
            (when (re-search-forward regexp nil t)
              (goto-char (point-min))
              (unless (re-search-forward (format "^#include *<%s>" file) nil t)
                (insert (format "#include <%s>\n" file))))))))
    (dolist (x '(("M_PI" . (lambda ()
                             (save-excursion
                               (goto-char (point-min))
                               (unless (search-forward "#define _USE_MATH_DEFINES" nil t)
                                 (when (search-forward "#include <cmath>\n" nil t)
                                   (replace-match ""))
                                 (goto-char (point-min))
                                 (when (looking-at "// -\\*-.*-\\*-")
                                   (forward-line)
                                   (while (and (bolp) (eolp) (not (eobp))) (forward-line)))
                                 (insert "#define _USE_MATH_DEFINES\n#include <cmath>\n")))))))
      (let ((regexp (car x)) (fun (cdr x)))
        (save-excursion
          (goto-char (point-min))
          (when (re-search-forward regexp nil t)
            (funcall fun)))))))

(add-hook 'c++-mode-hook
  (lambda ()
    (add-hook 'before-save-hook 'mylib-insert-include)
    (setq c-basic-offset 4)))

;; (add-hook 'c++-mode-hook
;;   (lambda ()
;;     (or (boundp 'mylib-insert-include-timer)
;;         mylib-insert-include-timer
;;         (setq mylib-insert-include-timer
;;               (run-with-idle-timer 1 t 'mylib-insert-include)))))

(defun mylib-cpp-insert-input ()
  (interactive "*")
  (let ((vs (split-string (read-from-minibuffer "Variable(s): "))))
    (insert (format "%s; cin >> %s;\n"
                    (mapconcat #'identity vs ", ")
                    (mapconcat #'identity vs " >> ")))
    (c-indent-line nil t)))

(defun mylib-cpp-insert-input-list ()
  (interactive "*")
  (let* ((spec (split-string (read-from-minibuffer "Variable(s) and size: ")))
         (size (car (last spec)))
         (vs (butlast spec)))
    (insert (format "for (int i = 0; i < %s; i++) cin%s;\n"
                    size
                    (mapconcat (lambda (v) (format " >> %s[i]" v))
                               vs "")))
    (c-indent-line nil t)))

(defun mylib-cpp-insert-output ()
  (interactive "*")
  (let ((exprs (split-string (read-from-minibuffer "cout (separate by `;'): ")
                             " *; *" t)))
    (insert (format "cerr << %s << endl;\n"
                    (mapconcat #'identity exprs " << ' ' << ")))
    (c-indent-line nil t)))

(eval-after-load "cc-mode"
  '(progn
     (define-key c++-mode-map "\C-ci" 'mylib-cpp-insert-input)
     (define-key c++-mode-map "\C-cI" 'mylib-cpp-insert-input-list)
     (define-key c++-mode-map "\C-co" 'mylib-cpp-insert-output)
     ))
