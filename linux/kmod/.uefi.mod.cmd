savedcmd_kmod/uefi.mod := printf '%s\n'   uefi.o | awk '!x[$$0]++ { print("kmod/"$$0) }' > kmod/uefi.mod
