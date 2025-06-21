savedcmd_kmod/abc.mod := printf '%s\n'   abc.o | awk '!x[$$0]++ { print("kmod/"$$0) }' > kmod/abc.mod
