cmd_kmod-guest/shadow.ko := ld -r -m elf_x86_64 -z noexecstack --build-id=sha1  -T scripts/module.lds -o kmod-guest/shadow.ko kmod-guest/shadow.o kmod-guest/shadow.mod.o;  true
