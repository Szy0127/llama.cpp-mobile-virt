savedcmd_kmod/exp.mod := printf '%s\n'   exp.o | awk '!x[$$0]++ { print("kmod/"$$0) }' > kmod/exp.mod
