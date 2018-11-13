


def main():
    current_id = None
    rows = []
    for line in open('rows'):
        obj_id = line.split()[3]

        if obj_id != current_id:
            if len(rows) > 100:
                with open(f'obj_data/{obj_id}', 'w') as f:
                    for row in rows:
                        f.write(row)
            rows = []
            current_id = obj_id
        rows.append(line)


if __name__ == '__main__':
    main()
