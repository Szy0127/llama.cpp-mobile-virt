savedcmd_kmod/uefi.ko := ld -r -m elf_x86_64 -z noexecstack --build-id=sha1  -T scripts/module.lds -o kmod/uefi.ko kmod/uefi.o kmod/uefi.mod.o
