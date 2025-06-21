savedcmd_kmod/foo.mod := printf '%s\n'   foo.o | awk '!x[$$0]++ { print("kmod/"$$0) }' > kmod/foo.mod
