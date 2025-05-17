#!/usr/bin/python3

fl1 = []
fl2 = []
with open('log','r') as f:
    with open('log1','r') as f1:
        fl1 = f.readlines()
        fl2 = f1.readlines()


flag = True
for i in range(len(fl1)):
    if flag:
        if not fl2[i] in fl1[i]:
            print("wrong", i, fl1[i], fl2[i])
        flag = not flag
        continue
    num = fl1[i].split() 
    num[1] = num[1].split(':')[1]
    num1 = fl2[i].split() 
    num1[1] = num1[1].split(':')[1]

    if num[0] != num1[1] or num[1] != num1[0]:
        print("wrong", i, fl1[i], fl2[i])


    flag = not flag
    

