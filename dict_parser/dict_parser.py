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
        offsets = info[1]

        line = word + ' => '
        pos = 0
        if len(offsets) == 1:
            line += word
        else:
            if offsets[0] == '0':
                line += word
                # pos += len(word)

            for offset in offsets:
                if offset == 'C':
                    pos += 1
                    continue
                elif offset == 'X':
                    line += word[pos]
                    pos += 1
                    continue
                elif offset == 'Y':
                    line = line[:-1]
                    line += word[pos]+','
                    pos += 1
                    continue

                line += word[pos:pos+int(offset)]+","
                pos += int(offset)

        if line[-1] == ',':
            line = line[:-1]
        print(line)
        write_file.write(line)

    read_file.close()
    write_file.close()


def parse_synonym_dict(read_file_name, write_file_name):
    # csv_reader = csv.reader()
    read_file = open(dir_path + read_file_name, 'r+')
    write_file = open(dir_path + write_file_name, 'w+')

    while True:
        line = read_file.readline()
        if not line:
            break

        info = line.split("\t")
        word = info[1]
        synonyms = info[2]

        line = synonyms + ' => ' + word

        print(line)
        write_file.write(line)

    read_file.close()
    write_file.close()
