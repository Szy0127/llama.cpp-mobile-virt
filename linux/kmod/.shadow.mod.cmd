cmd_kmod-guest/shadow.mod := printf '%s\n'   shadow.o | awk '!x[$$0]++ { print("kmod-guest/"$$0) }' > kmod-guest/shadow.mod
