
U = [0,0,0,1,1,0,1,0,0]
stay = 0
for u in U:
    stay = stay + 1 if u == 0 else 0
    print(u, stay)