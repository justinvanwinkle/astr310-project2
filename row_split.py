


def main():
    current_id = None
    rows = []
    for line in open('rows'):
        obj_id = line.split()[3]

        if obj_id != current_id:
            if len(rows) > 10:
                with open(f'obj_data/{obj_id}', 'w') as f:
                    print('writing', f'obj_data/{obj_id}')
                    for row in rows:
                        f.write(row)
            rows = []
            current_id = obj_id
        rows.append(line)

    if len(rows) > 10:
        with open(f'obj_data/{obj_id}', 'w') as f:
            print('writing', f'obj_data/{obj_id}')
            for row in rows:
                f.write(row)


if __name__ == '__main__':
    main()
