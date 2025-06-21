savedcmd_kmod/demo.mod := printf '%s\n'   demo.o | awk '!x[$$0]++ { print("kmod/"$$0) }' > kmod/demo.mod
