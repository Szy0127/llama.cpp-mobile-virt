savedcmd_kmod/exp.ko := aarch64-linux-gnu-ld -r  -EL  -maarch64elf -z noexecstack   --build-id=sha1  -T scripts/module.lds -o kmod/exp.ko kmod/exp.o kmod/exp.mod.o
