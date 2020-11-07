# coding: utf-8
from enum import Enum
import pandas as pd
import matplotlib.pyplot as plt


class PlayedRace(Enum):
    Protoss = 0
    Zerg = 1
    Terran = 2




# TRAIN.CSV: max = t5770
def get_max_time_frame(data):
    max = 0
    for index, row in data.iterrows():
        for column in row.keys():
            if not row[column]:
                continue
            if row[column][0] == 't':
                temp = int(row[column].split('t')[1])
                if temp > max:
                    max = temp
    print(f"Max : t{max}")


def read_csv(path_to_file: str):
    content = []
    with open(path_to_file, 'r') as csv_file:
        for line in csv_file.read().splitlines():
            content.append(line.split(','))

    return content


def get_numbers_ids_reference(path_to_file: str):
    data = read_csv(path_to_file)
    ids = {}
    counter = 0
    for play in data:
        if play[0] not in ids:
            ids[play[0]] = counter
            counter += 1

    return ids

def train_csv_to_data_matrix(path_to_file: str):
    with open(path_to_file, 'r') as csv_file:
        data = pd.read_csv(csv_file, header=None, sep='\n')
        data = data[0].str.split(',', expand=True)
        data = data.set_index(0)
        return data


def get_all_counts_actions_timed(data, train: bool):
    columns = []
    elements = ["id", "PlayedRace"]
    secondary_elements = ["Base", "s", "SingleMineral"]
    max = 2000+5  # +5 for the range()
    for elem in elements:
        columns.append(elem)

    for x in range(0, max, 5):
        for elem in secondary_elements:
            columns.append(f"{elem}:{x}")
        for i in range(0, 10):
            for j in range(0, 3):
                key = f"hotkey{i}{j}:{x}"
                columns.append(key)

    df = pd.DataFrame(columns=columns)
    print("Df initialized")
    # Fill the dataframe
    counter = 0
    for index, row in data.iterrows():
        new_line = dict.fromkeys(columns, 0)
        new_line['id'] = index
        ts_under = 0
        if train:
            new_line["PlayedRace"] = PlayedRace[row[1]].value
        else:
            new_line["PlayedRace"] = PlayedRace[index].value
        for column in row.keys():
            if not row[column] or column == 0:
                continue
            if row[column][0] == 't':
                ts_under = int(row[column].split('t')[1])
                if ts_under > 500:
                    break
            if "hotkey" in row[column]:
                new_line[f"{row[column]}:{ts_under}"] += 1
            elif "s" == row[column]:
                new_line[f"s:{ts_under}"] += 1
            elif "SingleMineral" == row[column]:
                new_line[f"SingleMineral:{ts_under}"] += 1
            elif "Base" == row[column]:
                new_line[f"Base:{ts_under}"] += 1

        # Append new line to the dataframe
        df = df.append(new_line, ignore_index=True)
        print(counter)
        counter += 1
    return df


def get_counts_all_actions(data, train: bool):
    columns = []
    elements = ["id", "PlayedRace", 'Base', 's', 'SingleMineral']
    for elem in elements:
        columns.append(elem)
    for i in range(0, 10):
        for j in range(0, 3):
            key = f"hotkey{i}{j}"
            columns.append(key)

    df = pd.DataFrame(columns=columns)

    # Fill the dataframe
    for index, row in data.iterrows():
        new_line = dict.fromkeys(columns, 0)
        new_line['id'] = index
        if train:
            new_line["PlayedRace"] = PlayedRace[row[1]].value
        else:
            new_line["PlayedRace"] = PlayedRace[index].value
        for column in row.keys():
            if not row[column] or column == 0:
                continue
            if "hotkey" in row[column]:
                new_line[row[column]] += 1
            elif "s" == row[column]:
                new_line['s'] += 1
            elif "SingleMineral" == row[column]:
                new_line["SingleMineral"] += 1
            elif "Base" == row[column]:
                new_line["Base"] += 1
        # Append new line to the dataframe
        df = df.append(new_line, ignore_index=True)
    return df


def get_counts_timed_no_pandas(data, train: bool):
    if train:
        labels = ['id', 'PlayedRace']
    else:
        labels = ['PlayedRace']

    for x in range(0, 505, 5):
        labels.append(f"s:T{x}")
        labels.append(f"Base:T{x}")
        labels.append(f"SingleMineral:T{x}")
        for i in range(0, 10):
            for j in range(0, 3):
                labels.append(f"hotkey{i}{j}:T{x}")

    featured_data = []

    for play in data:
        new_line = [0]*len(labels)
        t_stamp = 0
        for index, elem in enumerate(play):
            if t_stamp >= 505:
                break
            if index == 0:
                if train:
                    new_line[0] = elem
                else:
                    new_line[0] = PlayedRace[elem].value
                continue
            if index == 1 and train:
                new_line[labels.index("PlayedRace")] = PlayedRace[elem].value
                continue

            if 't' != elem[0]:
                new_line[labels.index(f"{elem}:T{t_stamp}")] += 1
            else:
                t_stamp = int(elem.split('t')[1])

        featured_data.append(new_line)

    return featured_data


def get_freq_all_actions_no_pandas(data, train: bool):
    if train:
        labels = ['id', 'PlayedRace', 'created', 'updated', 'used', 'Base', 's', 'SingleMineral']
    else:
        labels = ['PlayedRace', 'created', 'updated', 'used', 'Base', 's', 'SingleMineral']
    for i in range(0, 10):
        for j in range(0, 3):
            key = f"hotkey{i}{j}"
            labels.append(key)

    # Actions under 15sec
    to_add = []
    for i, elem in enumerate(labels):
        if train and i < 2:
            continue
        elif not train and i < 1:
            continue
        
        to_add.append(f"{elem}:T15")
    labels.extend(to_add)

    # Action per minute (first min only)
    labels.append('apm')
    featured_data = []

    for play in data:
        t_party = 0
        count_hotkey = 0
        if len(play) < 3:
            continue
        new_line = [0]*len(labels)
        if train:
            number_of_actions = len(play) - 2
        else:
            number_of_actions = len(play) - 1

        for index, elem in enumerate(play):
            if index == 0:
                if train:
                    new_line[0] = elem
                else:
                    new_line[0] = PlayedRace[elem].value
                continue
            if index == 1 and train:
                new_line[labels.index("PlayedRace")] = PlayedRace[elem].value
                continue

            if 't' != elem[0]:
                new_line[labels.index(elem)] += 1
                if t_party < 15:
                    new_line[labels.index(f"{elem}:T15")] += 1
                if "hotkey" in elem:
                    count_hotkey += 1
                    status = int(elem[-1])
                    if status == 0:
                        new_line[labels.index("created")] += 1
                    elif status == 1:
                        new_line[labels.index("updated")] += 1
                    elif status == 2:
                        new_line[labels.index("used")] += 1
            
            else:
                t_party = int(elem.split('t')[1])
                number_of_actions -= 1

        for i in range(0, len(new_line)):
            if ":T15" not in labels[i]:
                if train and i > 4:
                    new_line[i] = new_line[i]/number_of_actions
                elif not train and i > 3:
                    new_line[i] = new_line[i]/number_of_actions

        # Freq Hotkeys
        new_line[labels.index("created")] = new_line[labels.index("created")]/count_hotkey
        new_line[labels.index("updated")] = new_line[labels.index("updated")]/count_hotkey
        new_line[labels.index("used")] = new_line[labels.index("used")]/count_hotkey
        # APM
        new_line.append(number_of_actions/((t_party+5)/60))

        featured_data.append(new_line)

    print(featured_data[0])
    return featured_data


def get_counts_all_actions_no_pandas(data, train: bool):
    if train:
        labels = ['id', 'PlayedRace', 'Base', 's', 'SingleMineral']
    else:
        labels = ['PlayedRace', 'Base', 's', 'SingleMineral']
    for i in range(0, 10):
        for j in range(0, 3):
            key = f"hotkey{i}{j}"
            labels.append(key)

    labels.append('Base:T5')
    labels.append('s:T5')
    labels.append('SingleMineral:T5')
    for i in range(0, 10):
        for j in range(0, 3):
            key = f"hotkey{i}{j}:T5"
            labels.append(key)

    labels.append('Base:T10')
    labels.append('s:T10')
    labels.append('SingleMineral:T10')
    for i in range(0, 10):
        for j in range(0, 3):
            key = f"hotkey{i}{j}:T10"
            labels.append(key)

    labels.append('Base:T15')
    labels.append('s:T15')
    labels.append('SingleMineral:T15')
    for i in range(0, 10):
        for j in range(0, 3):
            key = f"hotkey{i}{j}:T15"
            labels.append(key)

    featured_data = []

    for play in data:
        new_line = [0]*len(labels)
        t_stamp = 0
        for index, elem in enumerate(play):
            if index == 0:
                if train:
                    new_line[0] = elem
                else:
                    new_line[0] = PlayedRace[elem].value
                continue
            if index == 1 and train:
                new_line[labels.index("PlayedRace")] = PlayedRace[elem].value
                continue

            if 't' != elem[0]:
                new_line[labels.index(elem)] += 1
                if t_stamp < 5:
                    new_line[labels.index(f"{elem}:T5")] += 1
                elif t_stamp < 10:
                    new_line[labels.index(f"{elem}:T10")] += 1
                elif t_stamp < 15:
                    new_line[labels.index(f"{elem}:T15")] += 1
            else:
                t_stamp = int(elem.split('t')[1])

        featured_data.append(new_line)

    return featured_data


def get_counts_hotkeys(data):
    columns = ["id"]
    for i in range(0, 10):
        for j in range(0, 3):
            key = f"hotkey{i}{j}"
            columns.append(key)

    df = pd.DataFrame(columns=columns)

    # Fill the dataframe
    for index, row in data.iterrows():
        new_line = dict.fromkeys(columns, 0)
        new_line['id'] = index
        for column in row.keys():
            if not row[column]:
                continue
            if "hotkey" in row[column]:
                new_line[row[column]] += 1
        # Append new line to the dataframe
        df = df.append(new_line, ignore_index=True)

    return df


def less_dimensions_counts(data, train: bool):
    if train:
        labels = ['id', 'PlayedRace', 'Base', 's', 'SingleMineral']
    else:
        labels = ['PlayedRace', 'Base', 's', 'SingleMineral']

    for i in range(0, 10):
        for j in range(0, 3):
            key = f"hotkey{i}{j}"
            labels.append(key)

    featured_data = []

    for play in data:
        new_line = [0]*len(labels)
        t_stamp = 0
        for index, elem in enumerate(play):
            if index == 0:
                if train:
                    new_line[0] = elem
                else:
                    new_line[0] = PlayedRace[elem].value
                continue
            if index == 1 and train:
                new_line[labels.index("PlayedRace")] = PlayedRace[elem].value
                continue

            if t_stamp < 30 and "hotkey" in elem:
                new_line[labels.index(elem)] += 1
            elif 't' != elem[0]:
                new_line[labels.index(elem)] += 1
            else:
                t_stamp = int(elem.split('t')[1])

        featured_data.append(new_line)

    return featured_data


def get_features(data, train: bool):
    if train:
        labels = ['id', 'PlayedRace', 'apm', 'duration', 'created', 'updated', 'used', 'Base', 's', 'SingleMineral']
    else:
        labels = ['PlayedRace', 'apm', 'duration', 'created', 'updated', 'used', 'Base', 's', 'SingleMineral']
    for i in range(0, 10):
        for j in range(0, 3):
            key = f"hotkey{i}{j}"
            labels.append(key)

    # Actions under 20sec
    to_add = []
    for i, elem in enumerate(labels):
        if train and i < 4:
            continue
        elif not train and i < 3:
            continue
        to_add.append(f"{elem}:T20")
    labels.extend(to_add)
    
    # Actions under 40sec
    to_add = []
    for i, elem in enumerate(labels):
        if train and i < 4:
            continue
        elif not train and i < 3:
            continue
        to_add.append(f"{elem}:T40")
    labels.extend(to_add)

    # Actions under 60sec
    to_add = []
    for i, elem in enumerate(labels):
        if train and i < 4:
            continue
        elif not train and i < 3:
            continue
        to_add.append(f"{elem}:T60")
    labels.extend(to_add)

    # Action per minute and duration
    # labels.append('apm')
    # labels.append('duration')

    featured_data = []

    for play in data:
        t_party = 0
        count_hotkey = 0
        if len(play) < 3:
            continue
        new_line = [0]*len(labels)
        if train:
            number_of_actions = len(play) - 2
        else:
            number_of_actions = len(play) - 1

        for index, elem in enumerate(play):
            if index == 0:
                if train:
                    new_line[0] = elem
                else:
                    new_line[0] = PlayedRace[elem].value
                continue

            if index == 1 and train:
                new_line[labels.index("PlayedRace")] = PlayedRace[elem].value
                continue

            if 't' != elem[0]:
                new_line[labels.index(elem)] += 1

                if t_party < 20:
                    new_line[labels.index(f"{elem}:T20")] += 1
                if t_party < 40:
                    new_line[labels.index(f"{elem}:T40")] += 1
                if t_party < 60:
                    new_line[labels.index(f"{elem}:T60")] += 1

                if "hotkey" in elem:
                    count_hotkey += 1
                    status = int(elem[-1])
                    if status == 0:
                        new_line[labels.index("created")] += 1
                    elif status == 1:
                        new_line[labels.index("updated")] += 1
                    elif status == 2:
                        new_line[labels.index("used")] += 1

            else:
                t_party = int(elem.split('t')[1])
                number_of_actions -= 1

        for i in range(0, len(new_line)):
            if ":T20" not in labels[i] and ":T40" not in labels[i] and ":T60" not in labels[i] :
                if train and i > 6:
                    new_line[i] = new_line[i]/number_of_actions
                elif not train and i > 5:
                    new_line[i] = new_line[i]/number_of_actions

        # Freq Hotkeys
        new_line[labels.index("created")] = new_line[labels.index("created")]/count_hotkey
        new_line[labels.index("updated")] = new_line[labels.index("updated")]/count_hotkey
        new_line[labels.index("used")] = new_line[labels.index("used")]/count_hotkey
        # APM
        new_line[labels.index("apm")] = (number_of_actions/((t_party+5)/60))
        # Duration
        new_line[labels.index("duration")] = t_party+5

        featured_data.append(new_line)

    # print(featured_data[0])
    # print(labels)
    return featured_data


def get_informations_data(path_to_data: str):
    data = read_csv(path_to_data)

    # Number of games per player

    count_empty_plays = 0
    count_plays_per_player = {}

    for play in data:
        if len(play) < 3:
            count_empty_plays += 1

        if play[0] in count_plays_per_player:
            count_plays_per_player[play[0]] += 1
        else:
            count_plays_per_player[play[0]] = 1

    print(f"Empty plays : {count_empty_plays}")
    print(f"Number of players : {len(count_plays_per_player.keys())}")

    sorted_list = sorted(
        count_plays_per_player.items(),
        key=lambda item: item[1],
        reverse=True
    )
    sorted_x = []
    sorted_y = []
    for elem in sorted_list:
        sorted_x.append(elem[0].split('/')[-2])
        sorted_y.append(elem[1])
    plt.bar(sorted_x, sorted_y, width=0.5, color='r', align='edge')
    # Hide the x labels because unreadable
    plt.xticks([])
    plt.show()

    # Number of actions per play

    count_actions_per_play = {}
    for index_play, play in enumerate(data):
        nb_actions = 0
        # t_stamp = 0
        for index, elem in enumerate(play):
            if index < 2:
                continue
            """
            if t_stamp > 30:
                break

            if 't' in elem and "hotkey" not in elem:
                t_stamp = int(elem.split('t')[1])
            else:
                nb_actions += 1
            """
            if 't' in elem and "hotkey" not in elem:
                continue

            nb_actions += 1

        count_actions_per_play[f"play_{index_play}"] = nb_actions

    sorted_list = sorted(
        count_actions_per_play.items(),
        key=lambda item: item[1],
        reverse=True
    )
    sorted_x = []
    sorted_y = []
    for elem in sorted_list:
        sorted_x.append(elem[0])
        sorted_y.append(elem[1])

    plt.bar(sorted_x, sorted_y, width=0.5, color='r', align='edge')
    plt.xticks([])
    plt.show()

    # Average number of actions per player
    count_actions_per_player = {}
    count_games_per_player = {}
    for index_play, play in enumerate(data):
        id_player = play[0]
        if id_player not in count_games_per_player:
            count_games_per_player[id_player] = 0
            count_actions_per_player[id_player] = 0
        else:
            count_games_per_player[id_player] += 1

        for index, elem in enumerate(play):
            if index < 2:
                continue

            if 't' in elem and "hotkey" not in elem:
                continue

            count_actions_per_player[id_player] += 1

    for player in count_games_per_player:
        count_actions_per_player[player] = count_actions_per_player[player] / count_games_per_player[player] 

    sorted_list = sorted(
        count_actions_per_player.items(),
        key=lambda item: item[1],
        reverse=True
    )
    sorted_x = []
    sorted_y = []
    for elem in sorted_list:
        sorted_x.append(elem[0])
        sorted_y.append(elem[1])

    plt.bar(sorted_x, sorted_y, width=0.5, color='r', align='edge')
    plt.xticks([])
    plt.show()

    # APM per player ?
    # Proportion des races jouées ? 
    # Durée moyenne des parties d'un joueur ?
