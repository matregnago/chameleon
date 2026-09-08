;; https://guix.bordeaux.inria.fr/eval/9358486
;; 2026-08-28 17:50:16

(list (channel
       (name 'guix-science)
       (url "https://codeberg.org/guix-science/guix-science.git")
       (branch "master")
       (commit "2ef021249f38ebec0960856d82f20cdcf70bca97")
       (introduction
        (make-channel-introduction
         "b1fe5aaff3ab48e798a4cce02f0212bc91f423dc"
         (openpgp-fingerprint
          "CA4F 8CF4 37D7 478F DA05  5FD4 4213 7701 1A37 8446"))))
      (channel
       (name 'guix-science-nonfree)
       (url "https://codeberg.org/guix-science/guix-science-nonfree.git")
       (branch "master")
       (commit "94dbea2d597139a8bacc0d5fc8a7d2bf234050d3")
       (introduction
        (make-channel-introduction
         "58661b110325fd5d9b40e6f0177cc486a615817e"
         (openpgp-fingerprint
          "CA4F 8CF4 37D7 478F DA05  5FD4 4213 7701 1A37 8446"))))
      (channel
       (name 'guix)
       (url "https://git.guix.gnu.org/guix.git")
       (branch "master")
       (commit "0bc7063da67b286c7faf1ead87a732be81100550")
       (introduction
        (make-channel-introduction
         "9edb3f66fd807b096b48283debdcddccfea34bad"
         (openpgp-fingerprint
          "BBB0 2DDF 2CEA F6A8 0D1D  E643 A2A0 6DF2 A33A 54FA"))))
      (channel
       (name 'guix-past)
       (url "https://codeberg.org/guix-science/guix-past.git")
       (branch "master")
       (commit "1ef1e5172f2631ee21cfc59656a64be3b5f0f8b3")
       (introduction
        (make-channel-introduction
         "0c119db2ea86a389769f4d2b9c6f5c41c027e336"
         (openpgp-fingerprint
          "3CE4 6455 8A84 FDC6 9DB4  0CFB 090B 1199 3D9A EBB5")))))
