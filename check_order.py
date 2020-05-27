
def main(path):
    """Check if menus are alphabetically sorted."""
    data = list()
    with open(path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        if line.startswith(('    ', '* [')):
            data.append(line)
        elif line.startswith(('- [', '## ')):
            if data != sorted(data, key=str.casefold):
                raise Exception('The content is not alphabetically sorted!')
            data = list()


if __name__ == '__main__':
    main('README.md')
