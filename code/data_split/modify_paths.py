

if __name__ == '__main__':
    fin = open('test_ids.txt', 'r')
    fout = open('test_ids2.txt', 'w')
    for line in fin:
        l = line.strip().split('/')
        fout.write('/'.join(l[-3:]) + '\n')

