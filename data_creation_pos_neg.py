from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as     np
import random
import pickle
from collections import Counter

lemmatizer = WordNetLemmatizer()
max_lines = 10000000

'''
OrderedDict([(1, 9162), (2, 2745), (3, 1506), (4, 919), (5, 639), (6, 495), (7, 395), (8, 296), (9, 237), (10, 206), 
(11, 175), (12, 134), (13, 126), (14, 96), (15, 97), (16, 73), (17, 79), (18, 60), (19, 55), (20, 68), (21, 56), 
(22, 45), (23, 36), (24, 42), (25, 47), (26, 31), (27, 23), (28, 29), (29, 25), (30, 27), (31, 25), (32, 23), (33, 20), 
(34, 23), (35, 14), (36, 17), (37, 15), (38, 16), (39, 9), (40, 12), (41, 11), (42, 12), (43, 10), (44, 12), (45, 9), 
(46, 5), (47, 10), (48, 13), (49, 7), (50, 10), (51, 6), (52, 16), (53, 9), (54, 13), (55, 6), (56, 6), (57, 4), 
(58, 9), (59, 4), (60, 7), (61, 6), (62, 4), (63, 5), (64, 3), (65, 3), (66, 4), (67, 6), (68, 1), (69, 5), (70, 5), 
(71, 6), (72, 2), (73, 6), (74, 2), (75, 3), (76, 5), (77, 3), (78, 4), (79, 7), (80, 2), (81, 2), (82, 3), (83, 4), 
(84, 2), (85, 2), (86, 2), (87, 2), (88, 3), (89, 3), (90, 1), (91, 1), (92, 1), (93, 2), (94, 4), (96, 2), (97, 1), 
(98, 3), (100, 2), (101, 1), (103, 2), (104, 2), (105, 1), (106, 4), (107, 4), (108, 3), (110, 1), (111, 4), (113, 1), 
(114, 2), (115, 4), (116, 3), (117, 3), (118, 5), (119, 4), (120, 2), (121, 2), (123, 4), (124, 2), (125, 1), (126, 3), 
(127, 2), (128, 1), (129, 1), (132, 1), (133, 1), (135, 2), (137, 1), (138, 2), (139, 2), (140, 2), (141, 1), (142, 1), 
(143, 1), (144, 3), (145, 1), (146, 4), (147, 1), (149, 1), (151, 1), (152, 1), (153, 2), (154, 1), (156, 2), (158, 2), 
(160, 1), (161, 2), (165, 3), (167, 2), (168, 1), (169, 3), (170, 1), (173, 1), (175, 1), (176, 2), (178, 1), (180, 1), 
(181, 2), (183, 1), (185, 1), (186, 1), (188, 1), (189, 1), (191, 1), (193, 1), (194, 1), (196, 1), (197, 3), (200, 2), 
(206, 2), (212, 1), (214, 1), (217, 1), (218, 1), (220, 1), (222, 1), (226, 1), (227, 2), (228, 1), (231, 1), (234, 2), 
(235, 1), (238, 1), (240, 1), (241, 1), (245, 2), (253, 1), (262, 2), (264, 2), (265, 1), (266, 1), (267, 1), (268, 1), 
(269, 1), (270, 1), (271, 2), (274, 1), (282, 1), (284, 1), (291, 1), (292, 1), (302, 1), (308, 1), (312, 1), (325, 1), 
(326, 1), (334, 1), (336, 1), (363, 1), (377, 2), (381, 1), (382, 1), (386, 1), (389, 2), (400, 1), (402, 1), (417, 2), 
(438, 1), (442, 1), (452, 1), (459, 1), (461, 1), (468, 1), (483, 1), (520, 1), (530, 1), (531, 1), (536, 1), (538, 1), 
(555, 1), (583, 1), (630, 1), (640, 1), (642, 1), (656, 1), (666, 1), (670, 1), (673, 1), (707, 1), (722, 1), (732, 1), 
(733, 3), (763, 1), (795, 1), (831, 1), (896, 1), (939, 1), (940, 1), (1179, 1), (1326, 1), (1437, 1), (1443, 1), 
(1547, 1), (1564, 1), (1641, 1), (1770, 1), (2631, 1), (2657, 1), (3537, 1), (3559, 1), (4234, 1), (4760, 1), (6063, 1), 
(6202, 1), (9110, 1), (10037, 1), (10113, 1), (14010, 1)])
'''
max_freq = 1000
min_freq = 50


def plot_graph(x, y):
    # importing the required module
    import matplotlib.pyplot as plt

    # plotting the points
    plt.plot(x, y)

    # naming the x axis
    plt.xlabel('x - axis')
    # naming the y axis
    plt.ylabel('y - axis')

    # function to show the plot
    plt.show()


def create_lexicon(list_of_files):
    lexicon = []

    for files in list_of_files:
        with open(files, 'r') as f:
            contents = f.readlines()

            for l in contents[:max_lines]:
                all_words = word_tokenize(l.lower())
                lexicon += list(all_words)

    lexicon = [lemmatizer.lemmatize(i) for i in lexicon]

    w_counts = Counter(lexicon)

    final_lex = []

    for w in w_counts:
        if w_counts[w] < max_freq or w_counts[w] > min_freq:
            final_lex.append(w)
    return final_lex


def sample_handling(sample, lexicon, classification):
    feature_set = []

    with open(sample, 'r') as f:
        contents = f.readlines()

        for l in contents[:max_lines]:
            current_words = word_tokenize(l.lower())

            current_words = [lemmatizer.lemmatize(i) for i in current_words]

            features = np.zeros(len(lexicon))

            for word in current_words:
                if word in lexicon:
                    index_value = lexicon.index(word)
                    features[index_value] += 1

            feature_set.append([features, classification])

    return feature_set


def create_feature_set_and_labels(pos, neg, test_size=0.1):
    final_lex = create_lexicon([pos, neg])

    features = []
    features += sample_handling(sample=pos, lexicon=final_lex, classification=[1, 0])
    features += sample_handling(sample=neg, lexicon=final_lex, classification=[0, 1])

    random.shuffle(features)

    features = np.array(features)

    testing_size = int(test_size * len(features))

    print("-- Testing Size: ", testing_size)
    train_x = features[:, 0][:-testing_size]
    train_y = features[:, 1][:-testing_size]

    print("-- Train : ", train_x)
    print("-- Train : ", train_y)

    test_x = features[:, 0][-testing_size:]
    test_y = features[:, 1][-testing_size:]

    print("-- Test : ", test_x)
    print("-- Test : ", test_y)

    return train_x, train_y, test_x, test_y


if __name__ == '__main__':
    pos = 'pos.txt'
    neg = 'neg.txt'

    train_x, train_y, test_x, test_y = create_feature_set_and_labels(pos=pos, neg=neg)

    with open('sentiment_set.pickle', 'wb') as f:
        pickle.dump([train_x, train_y, test_x, test_y], f)
