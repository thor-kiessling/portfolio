import bitmex_sniper

# print(bitmex_sniper.__file__)
# print(bitmex_sniper.bitmex.__file__)
while True:
    try:
        bitmex_sniper.main()
    except KeyboardInterrupt or SystemExit:
        break
    except BaseException as e:
        print(e)
        for arg in e.args:
            print(arg)
        # print(e.__traceback__)
        # print(e.__cause__)
#