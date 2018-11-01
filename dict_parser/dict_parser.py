dir_path = './dict/'


def parse_user_dict(read_file_name, write_file_name):
    # csv_reader = csv.reader()
    read_file = open(dir_path + read_file_name, 'r+')
    write_file = open(dir_path + write_file_name, 'w+')

    while True:
        line = read_file.readline()
        if not line:
            break

        info = line.split(",")
        word = info[0]
        offsets = info[1].strip()

        line = word + " "

        pos = 0
        # if len(offsets) == 1:
        #     line += word + " "
        # else:
        #     if offsets[0] == '0':
        #         line += word + " "
        #         # pos += len(word)

        for offset in offsets:
            if offset in '123456789':
                line += word[pos:pos+int(offset)]+" "
                pos += int(offset)
            else:
                if offset == 'C' or offset == 'J':
                    pos += 1
                elif offset == 'X':
                    line += word[pos]
                    pos += 1
                elif offset == 'Y':
                    line = line[:-1]
                    line += word[pos] + ' '
                    pos += 1
                elif offset == 'a' or offset == 'b':
                    num = len(word)
                    for _ in offsets:
                        if _ == offset:
                            continue
                        num -= int(_)
                    line += word[pos:pos + num] + " "
                    pos += num
                elif offset == 'B':
                    line += word[pos:pos + 1] + " "
                    pos += 1
                # print(offset)

        if line[-1] == ' ':
            line = line[:-1]
        # print(line)
        write_file.write(line+'\n')

    read_file.close()
    write_file.close()


def parse_synonym_dict(read_file_name, write_file_name):
    read_file = open(dir_path + read_file_name, 'r+')
    write_file = open(dir_path + write_file_name, 'w+')

    while True:
        line = read_file.readline()
        if not line:
            break

        info = line.split("\t")
        word = info[1]
        synonyms = info[2]
        line = word + "," + synonyms + " => " + word + "," + synonyms

        write_file.write(line+'\n')

    read_file.close()
    write_file.close()
