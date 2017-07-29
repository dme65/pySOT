from pySOT.experimental_design import *


def main():
    print("========================= LHD =======================")
    lhs = LatinHypercube(4, 10, criterion='c')
    print(lhs.generate_points())

    print("\n========================= SLHD =======================")
    slhd = SymmetricLatinHypercube(3, 10)
    print(slhd.generate_points())

    # print("\n========================= 2-Factorial =======================")
    # twofact = TwoFactorial(3)
    # print(twofact.generate_points())
    # print(twofact.npts)

    print("\n========================= Box-Behnken =======================")
    bb = BoxBehnken(3)
    print(bb.generate_points())
    print(bb.npts)


def test_answer():
    main()
