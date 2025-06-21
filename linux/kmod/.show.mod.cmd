savedcmd_kmod/show.mod := printf '%s\n'   show.o | awk '!x[$$0]++ { print("kmod/"$$0) }' > kmod/show.mod
