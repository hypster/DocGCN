import argparse


def bar(hello):
    print(hello)
if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument()
    d = {'hello': 1, 'world': 2}
    bar(**d)