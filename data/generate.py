from random import randint, random

RANGE = 100

sizes = [
    # 4,
    8,
    # 300,
    # 800,
    # 2000
]

for size in sizes:
    print("%dx%d" % (size, size))
    with open("A%d.txt" % size, "w") as file:
        print("%d %d" % (size, size+1), file=file)
        step = size // 1000 if size > 1000 else 1
        for i in range(size):
            for j in range(size):
                if i != j:
                    print("%4d " % randint(1, RANGE), end="", file=file)
                else:
                    print("%4d " % (randint(1, RANGE) + RANGE * size), end="", file=file)
            print("%4d" % randint(1, RANGE), file=file)
            if i % step == 0:
                print("\r%.1f%%" % (i / size * 100), end="")

    with open("B%d.txt" % size, "w") as file:
        print("%d" % size, file=file)
        for _ in range(size):
            print("%f" % random(), file=file)
        print("\r%.1f%%" % 100)
